import math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV


# ===== BEGIN base_model.py =====


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def inverse_sigmoid(p: float) -> float:
    """Safe-ish inverse of sigmoid, with clipping."""
    eps = 1e-9
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class BaseSubModel:
    """
    Interface / example for your submodels.
    Each concrete model should implement:
      - get_input_data(home_id, away_id, season, date) -> list[float] | None
      - pre_add_game(game_row, players_df)  (optional)
      - add_game(game_row, players_df)
    """

    def get_input_data(self, home_id, away_id, season, date):
        # Return a list of numeric features, or None if not available.
        raise NotImplementedError

    def pre_add_game(self, game_row: pd.Series):
        # Optional: called before add_game, can be a no-op.
        pass

    def add_game(self, game_row: pd.Series):
        # Update internal state with a finished game.
        raise NotImplementedError
# ===== END base_model.py =====

# ===== BEGIN elo_model.py =====

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class EloModel:
    def __init__(self, k_factor=0.15, home_factor=0.5):
        self.k_factor = k_factor
        self.home_factor = home_factor

        self.elo_map = defaultdict(float)

    def pre_add_game(self, current):
        pass

    def add_game(self, current):
        home_id = current['HID']
        away_id = current['AID']

        elo_prediction = sigmoid(self.get_input_data(home_id, away_id, 0, 0)[0])

        self.elo_map[home_id] += self.k_factor * (current['H'] - elo_prediction)
        self.elo_map[away_id] += self.k_factor * (current['A'] - (1 - elo_prediction))

    def get_input_data(self, home_id, away_id, season, date):
        return [
            self.elo_map[home_id] - self.elo_map[away_id] + self.home_factor
        ]
# ===== END elo_model.py =====

# ===== BEGIN exhaustion_model.py =====

class Exhaustion:
    def __init__(self):
        self.team_rest = {}

    def pre_add_game(self, current):
        pass

    def add_game(self, current):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']

        self.team_rest[home_id] = date
        self.team_rest[away_id] = date

    def get_input_data(self, home_id, away_id, season, date):
        if home_id not in self.team_rest or away_id not in self.team_rest:
            return None

        home_days = (date - self.team_rest[home_id]).days
        away_days = (date - self.team_rest[away_id]).days

        factor = 0.0

        if home_days <= 1:
            factor += 1.0
        if away_days <= 1:
            factor -= 1.0

        return [
            factor
        ]# ===== END exhaustion_model.py =====

# ===== BEGIN four_factor_model.py =====

def inverse_sigmoid(p: float) -> float:
    eps = 1e-4
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class FourFactor:
    """
    Hockey 'four factors' style model, updated to use shot-based analytics:

      - Corsi-like shot attempt share (possession proxy)
      - Fenwick-like unblocked shot share (chance quality proxy)
      - PDO-style luck/goaltending metric
      - Physical / discipline & faceoff factors

    For each team, we keep exponentially-weighted historical stats for:
      - team_stats_average[team_id]: team's own stats
      - opponent_stats_average[team_id]: opponent stats faced by this team

    Then:
      - We compute per-team feature vectors from these analytics-style stats
      - Train a logistic regression to predict win probability
      - For a match, we return logit(p_home) - logit(p_away) as one feature
        for the meta-model.
    """

    def __init__(self, samples_required, retrain_every=1000):
        # How many (feature, label) samples before we start training LR
        self.lr_required_n = samples_required

        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)

        # [*features, win_flag]
        self.past_pred = []
        self.lr: LogisticRegression | None = None
        self.retrain_every = retrain_every
        self.lr_retrain = 0  # countdown to next retrain

    @staticmethod
    def _nz(x):
        """Convert NaN / inf / None to 0.0."""
        if x is None:
            return 0.0
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(v):
            return 0.0
        return v
    
    @staticmethod
    def _safe_div(num, den):
        try:
            num = float(num)
            den = float(den)
        except (TypeError, ValueError):
            return 0.0

        if den <= 0.0 or not np.isfinite(den) or not np.isfinite(num):
            return 0.0
        return num / den

    def _get_stats(self, date, stats_list):
        """
        Exponentially-weighted totals over a list of per-game stat dicts.
        """
        totals = {
            "PEN": 0.0,     # minor penalties taken
            "MAJ": 0.0,     # major penalties
            "PPG": 0.0,     # powerplay goals
            "SHG": 0.0,     # shorthanded goals
            "SV": 0.0,      # saves
            "PIM": 0.0,     # penalty minutes
            "SOG": 0.0,     # shots on goal
            "BLK_S": 0.0,   # blocked shots (shots by this team that get blocked OR shots this team blocks – depends on feed)
            "BLK": 0.0,     # generic blocks
            "HIT": 0.0,     # hits
            "FO": 0.0,      # faceoff wins
            "Goals": 0.0,   # goals scored
            "Win": 0.0,     # 1 if win, 0 otherwise
            "Weight": 0.0,
        }

        for stat in stats_list:
            if stat["Date"] >= date:
                continue
            # Slightly discount older games
            delta_days = abs((date - stat["Date"]).days)
            weight = 0.994 ** delta_days

            totals["PEN"] += self._nz(stat.get("PEN")) * weight
            totals["MAJ"] += self._nz(stat.get("MAJ")) * weight
            totals["PPG"] += self._nz(stat.get("PPG")) * weight
            totals["SHG"] += self._nz(stat.get("SHG")) * weight
            totals["SV"] += self._nz(stat.get("SV")) * weight
            totals["PIM"] += self._nz(stat.get("PIM")) * weight
            totals["SOG"] += self._nz(stat.get("SOG")) * weight
            totals["BLK_S"] += self._nz(stat.get("BLK_S")) * weight
            totals["BLK"] += self._nz(stat.get("BLK")) * weight
            totals["HIT"] += self._nz(stat.get("HIT")) * weight
            totals["FO"] += self._nz(stat.get("FO")) * weight
            totals["Goals"] += self._nz(stat.get("Goals")) * weight
            totals["Win"] += self._nz(stat.get("Win")) * weight
            totals["Weight"] += weight

        return totals
    
    def pre_add_game(self, current):
        pass

    def add_game(self, current):
        """
        Update model with a finished game.

        current: row from the games DataFrame with columns:
          HID, AID, Date, H, A, HS, AS, ...
        """

        # 1) Build input data before adding THIS game to history
        val_home = self._get_team_input_data(current["HID"], current["Date"])
        val_away = self._get_team_input_data(current["AID"], current["Date"])

        if val_home is not None:
            self.past_pred.append([*val_home, float(current["H"])])
            self.lr_retrain -= len(val_home)

        if val_away is not None:
            self.past_pred.append([*val_away, float(current["A"])])
            self.lr_retrain -= len(val_away)

        # 2) Store this game's stats for both teams and their opponents

        # Home team stats (for)
        self.team_stats_average[current["HID"]].append(
            {
                "Date": current["Date"],
                "PEN": current["H_PEN"],
                "MAJ": current["H_MAJ"],
                "PPG": current["H_PPG"],
                "SHG": current["H_SHG"],
                "SV": current["H_SV"],
                "PIM": current["H_PIM"],
                "SOG": current["H_SOG"],
                "BLK_S": current["H_BLK_S"],
                "BLK": current["H_BLK"],
                "HIT": current["H_HIT"],
                "FO": current["H_FO"],
                "Goals": current["HS"],
                "Win": float(current["H"]),
            }
        )

        # Away team stats (for)
        self.team_stats_average[current["AID"]].append(
            {
                "Date": current["Date"],
                "PEN": current["A_PEN"],
                "MAJ": current["A_MAJ"],
                "PPG": current["A_PPG"],
                "SHG": current["A_SHG"],
                "SV": current["A_SV"],
                "PIM": current["A_PIM"],
                "SOG": current["A_SOG"],
                "BLK_S": current["A_BLK_S"],
                "BLK": current["A_BLK"],
                "HIT": current["A_HIT"],
                "FO": current["A_FO"],
                "Goals": current["AS"],
                "Win": float(current["A"]),
            }
        )

        # Opponent stats: what each team *faced*
        # For the home team, opponent is the away team
        self.opponent_stats_average[current["HID"]].append(
            {
                "Date": current["Date"],
                "PEN": current["A_PEN"],
                "MAJ": current["A_MAJ"],
                "PPG": current["A_PPG"],
                "SHG": current["A_SHG"],
                "SV": current["A_SV"],
                "PIM": current["A_PIM"],
                "SOG": current["A_SOG"],
                "BLK_S": current["A_BLK_S"],
                "BLK": current["A_BLK"],
                "HIT": current["A_HIT"],
                "FO": current["A_FO"],
                "Goals": current["AS"],
            }
        )
        # For the away team, opponent is the home team
        self.opponent_stats_average[current["AID"]].append(
            {
                "Date": current["Date"],
                "PEN": current["H_PEN"],
                "MAJ": current["H_MAJ"],
                "PPG": current["H_PPG"],
                "SHG": current["H_SHG"],
                "SV": current["H_SV"],
                "PIM": current["H_PIM"],
                "SOG": current["H_SOG"],
                "BLK_S": current["H_BLK_S"],
                "BLK": current["H_BLK"],
                "HIT": current["H_HIT"],
                "FO": current["H_FO"],
                "Goals": current["HS"],
            }
        )

    def _get_team_input_data(self, team_id, date):
        """
        Build a feature vector for a single team on a given date.

        Uses up to the last 100 games, with exponential time decay,
        and returns analytics-style percentages/ratios:

          - Corsi-style shot attempt share & differential
          - Fenwick-style shot share & differential
          - PDO (shooting% + save%)
          - Shooting% for/against, save%
          - Faceoff share
          - Penalty, hits, blocks effects

        IMPORTANT: this function does NOT divide any stats by Weight.
        All features are constructed from ratios of raw (time-decayed) totals.
        """
        if len(self.team_stats_average[team_id]) <= 5:
            return None
        if len(self.opponent_stats_average[team_id]) <= 5:
            return None

        stats = self._get_stats(date, self.team_stats_average[team_id][-100:])
        opp_stats = self._get_stats(date, self.opponent_stats_average[team_id][-100:])

        # --- Basic totals (no Weight normalization here) --------------------
        shots_for = self._nz(stats["SOG"])
        shots_against = self._nz(opp_stats["SOG"])

        # Interpretation (based on common NHL stat feeds):
        #   BLK_S: shots by this team that were blocked by the opponent
        #   BLK:   shots by the opponent that were blocked by this team
        blocks_for = self._nz(stats["BLK"])
        blocks_against = self._nz(opp_stats["BLK"])
        shots_blocked_by_opp = self._nz(stats["BLK_S"])
        shots_blocked_by_team = self._nz(opp_stats["BLK_S"])

        goals_for = self._nz(stats["Goals"])
        goals_against = self._nz(opp_stats["Goals"])

        saves = self._nz(stats["SV"])
        # shots on goal against ≈ saves + goals against
        shots_on_goal_against = saves + goals_against

        hits_for = self._nz(stats["HIT"])
        hits_against = self._nz(opp_stats["HIT"])

        faceoffs_for = self._nz(stats["FO"])
        faceoffs_against = self._nz(opp_stats["FO"])

        pen_minor_for = self._nz(stats["PEN"])
        pen_minor_against = self._nz(opp_stats["PEN"])
        pen_major_for = self._nz(stats["MAJ"])
        pen_major_against = self._nz(opp_stats["MAJ"])

        ppg_for = self._nz(stats["PPG"])
        shg_for = self._nz(stats["SHG"])
        ppg_against = self._nz(opp_stats["PPG"])
        shg_against = self._nz(opp_stats["SHG"])

        # --- Corsi (SAT) style metrics --------------------------------------
        # Corsi For: shots on goal + blocked shots for (we don't have misses)
        cf_for = shots_for + shots_blocked_by_opp
        # Corsi Against: shots against + blocked shots against
        cf_against = shots_against + shots_blocked_by_team
        cf_total = cf_for + cf_against

        corsi_for_pct = self._safe_div(cf_for, cf_total)      # CF%
        corsi_diff = cf_for - cf_against                      # CF differential

        # --- Fenwick (USAT) style metrics -----------------------------------
        # Without explicit missed-shot counts, approximate Fenwick as SOG-only.
        ff_for = shots_for
        ff_against = shots_against
        ff_total = ff_for + ff_against

        fenwick_for_pct = self._safe_div(ff_for, ff_total)    # FF%
        fenwick_diff = ff_for - ff_against                    # FF differential

        # --- Shooting %, save %, PDO ----------------------------------------
        shooting_pct_for = self._safe_div(goals_for, shots_for)
        shooting_pct_against = self._safe_div(goals_against, shots_against)

        save_pct = self._safe_div(saves, shots_on_goal_against)

        # PDO = (team shooting% + team save%) * 100
        pdo = 100.0 * (shooting_pct_for + save_pct)

        # --- Discipline / physical play -------------------------------------
        # Penalty load (weight majors a bit more)
        penalty_load_for = pen_minor_for + 2.0 * pen_major_for
        penalty_load_against = pen_minor_against + 2.0 * pen_major_against
        penalty_diff = penalty_load_for - penalty_load_against

        # Hits: share & differential (physicality)
        hits_total = hits_for + hits_against
        hit_share = self._safe_div(hits_for, hits_total)
        hit_diff = hits_for - hits_against

        # Blocks: share (defensive shot suppression)
        blocks_total = blocks_for + blocks_against
        block_share = self._safe_div(blocks_for, blocks_total)

        # --- Faceoffs (proxy for possession / zone starts) ------------------
        fo_total = faceoffs_for + faceoffs_against
        faceoff_share = self._safe_div(faceoffs_for, fo_total)

        # --- Special teams efficiency ---------------------------------------
        # Share of total PP goals, and net special-teams goals differential.
        pp_goals_total = ppg_for + ppg_against
        pp_goal_share = self._safe_div(ppg_for, pp_goals_total)

        special_teams_net = (ppg_for + shg_for) - (ppg_against + shg_against)

        # Feature vector: keep order fixed.
        # These correspond to:
        #   - Corsi-style possession (CF%, CF diff)
        #   - Fenwick-style chance share (FF%, FF diff)
        #   - PDO & its components (shooting%, save%)
        #   - Disicpline/physical (penalties, hits, blocks)
        #   - Faceoff share & special teams
        return [
            corsi_for_pct,         # 0: CF%
            corsi_diff,            # 1: CF differential
            fenwick_for_pct,       # 2: FF%
            fenwick_diff,          # 3: FF differential

            pdo,                   # 4: PDO (shooting% + save%) * 100
            shooting_pct_for,      # 5: team shooting%
            save_pct,              # 6: team save%
            shooting_pct_against,  # 7: opp shooting%

            faceoff_share,         # 8: faceoff win share
            pp_goal_share,         # 9: PP goal share
            special_teams_net,     # 10: net PP+SH goals

            penalty_diff,          # 11: penalty load differential
            hit_share,             # 12: hits share
            hit_diff,              # 13: hits differential
            block_share,           # 14: block share
        ]

    def get_input_data(self, home_id, away_id, season, date):
        """
        For the ensemble: return a single feature
        = logit(p_home_win) - logit(p_away_win) from this submodel.

        If we don't have enough data, returns None.
        """
        val_home = self._get_team_input_data(home_id, date)
        val_away = self._get_team_input_data(away_id, date)

        if val_home is None or val_away is None:
            return None

        if len(self.past_pred) < self.lr_required_n:
            return None

        # Retrain logistic regression occasionally with time-decayed weights
        if self.lr is None or self.lr_retrain <= 0:
            self.lr_retrain = self.retrain_every

            np_array = np.array(self.past_pred, dtype=float)
            X = np_array[:, :-1]
            y = np_array[:, -1]

            sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
            sample_weights = sample_weights[::-1]  # most recent gets highest weight

            self.lr = Pipeline([
                # ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=10000))
            ])
            self.lr.fit(X, y, logreg__sample_weight=sample_weights)

        # Predict win probabilities for home and away team
        ph = float(self.lr.predict_proba(np.array([val_home]))[0, 1])
        pa = float(self.lr.predict_proba(np.array([val_away]))[0, 1])

        # Return log-odds difference as meta-model feature
        return [inverse_sigmoid(ph) - inverse_sigmoid(pa)]
# ===== END four_factor_model.py =====

# ===== BEGIN fte_model.py =====



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def inverse_sigmoid(p: float) -> float:
    """Safe-ish inverse of sigmoid, with clipping."""
    eps = 1e-9
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class EloNHLSubModel:
    """
    FiveThirtyEight-style Elo model for NHL.

    Key mechanics (summarized from the description):

    - Each team has an Elo rating (league average ~1500).
    - Game win probability (for team A vs B) is:

        Pr(A wins) = 1 / (10^(-EloDiff / 400) + 1)

      where EloDiff is A's pregame Elo minus B's, plus adjustments:
        * +50 points for home-ice advantage
        * Multiply EloDiff by 1.25 for playoff games.

    - After the game, Elo ratings shift:

        shift = K * MOV_multiplier * AutocorrAdjustment * (TeamWin - TeamWinProb)

      with:
        K = 6
        MOV_multiplier = 0.6686 * ln(MOV) + 0.8048, MOV = margin of victory (>=1)
        AutocorrAdjustment = 2.05 / (WinnerEloDiff * 0.001 + 2.05)
          (WinnerEloDiff > 0 => big favorite => smaller adj)

      shift is added to the **winner's** Elo and subtracted from the loser's.

    - Preseason ratings:
        * Teams retain 70% of previous end-of-season rating,
          and 30% revert toward league average (1505):

              Elo_preseason = 0.7 * Elo_prev + 0.3 * 1505

        * Expansion teams:
            - Pre-salary-cap era: Elo = 1380
            - Salary-cap era (season >= 2005): Elo = 1490

    Implementation notes / assumptions:

    - We assume games are processed in chronological order.
    - 'Season' is treated as an opaque label for equality
      and as int when comparing to salary-cap year (2005).
    - For add_game, we look for optional columns:
        * 'IsPlayoff' or 'Playoff'    -> playoff flag
        * 'Neutral' or 'NeutralSite'  -> neutral-site flag
      If missing, we assume regular-season, non-neutral.
    - Ties (if any) are treated as 0.5 win for each side with MOV=1.
    """

    def __init__(
        self,
        k_factor: float = 6.0,
        home_advantage: float = 50.0,
        playoff_multiplier: float = 1.25,
        league_avg_elo: float = 1505.0,
        preseason_retention: float = 0.7,
        expansion_elo_precap: float = 1380.0,
        expansion_elo_cap: float = 1490.0,
        salary_cap_start_year: int = 2005,
    ):
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.playoff_multiplier = float(playoff_multiplier)
        self.league_avg_elo = float(league_avg_elo)
        self.preseason_retention = float(preseason_retention)
        self.expansion_elo_precap = float(expansion_elo_precap)
        self.expansion_elo_cap = float(expansion_elo_cap)
        self.salary_cap_start_year = int(salary_cap_start_year)

        # Current Elo rating per team_id
        self.team_elo: Dict[str, float] = {}

        # Current season per team_id (for preseason reversion / expansions)
        self.team_current_season: Dict[str, object] = {}

    # --------- small helpers ---------

    @staticmethod
    def _nz(x, default: float = 0.0) -> float:
        try:
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return default
            v = float(x)
            if not math.isfinite(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _season_year(season) -> Optional[int]:
        """
        Try to extract an integer "year" from Season.
        We support:
          - integer seasons
          - strings like '2019', '2019-20', '2019/20'
        """
        if season is None:
            return None
        if isinstance(season, int):
            return season
        s = str(season)
        # Extract leading int
        num = ""
        for ch in s:
            if ch.isdigit():
                num += ch
            else:
                break
        try:
            return int(num) if num else None
        except ValueError:
            return None

    # --------- preseason / expansion logic ---------

    def _ensure_team_season(self, team_id: str, season):
        """
        Ensure that team_id has an Elo rating appropriate for this season.
        Handles expansion teams and preseason reversion.

        Called from both get_input_data and add_game, so it's robust to
        whichever is called first in a season.
        """
        team_id = str(team_id)
        current_season = self.team_current_season.get(team_id)

        # If same season already set, nothing to do.
        if current_season == season:
            return

        # First time we ever see this team
        if current_season is None:
            season_year = self._season_year(season)
            if season_year is not None and season_year >= self.salary_cap_start_year:
                base_elo = self.expansion_elo_cap
            else:
                base_elo = self.expansion_elo_precap
            self.team_elo[team_id] = base_elo
            self.team_current_season[team_id] = season
            return

        # Existing team entering a new season: apply preseason reversion
        old_rating = self.team_elo.get(team_id, self.league_avg_elo)
        retain = self.preseason_retention
        new_rating = retain * old_rating + (1.0 - retain) * self.league_avg_elo
        self.team_elo[team_id] = new_rating
        self.team_current_season[team_id] = season

    # --------- Elo core functions ---------

    def _win_prob_from_diff(self, elo_diff: float) -> float:
        """
        Pr(A wins) = 1 / (10^(-EloDiff / 400) + 1)
        """
        try:
            exponent = -elo_diff / 400.0
            denom = (10.0 ** exponent) + 1.0
            if denom <= 0.0 or not math.isfinite(denom):
                return 0.5
            return 1.0 / denom
        except OverflowError:
            # If exponent is huge negative, Team A is basically certain to win
            return 1.0 if elo_diff > 0 else 0.0

    def _mov_multiplier(self, margin: float) -> float:
        """
        Margin-of-victory multiplier:
            MOV_mult = 0.6686 * ln(MOV) + 0.8048
        where MOV >= 1.
        """
        margin = max(1.0, margin)
        return 0.6686 * math.log(margin) + 0.8048

    def _autocorr_adjustment(self, winner_elo_diff: float) -> float:
        """
        Autocorrelation adjustment:
            AutocorrAdj = 2.05 / (WinnerEloDiff * 0.001 + 2.05)

        If WinnerEloDiff > 0 (big favorite), denominator grows -> smaller adj.
        If WinnerEloDiff < 0 (underdog), denominator shrinks -> larger adj.
        """
        denom = winner_elo_diff * 0.001 + 2.05
        # Safety clamp to avoid division by zero or sign flip extremes
        if denom <= 0.1:
            denom = 0.1
        return 2.05 / denom

    # --------- BaseSubModel methods ---------

    def pre_add_game(self, game_row: pd.Series):
        """
        No special pre-processing needed; kept for interface.
        """
        pass

    def add_game(self, game_row: pd.Series, players_df: Optional[pd.DataFrame] = None):
        """
        Update Elo ratings with a finished game.

        Expected columns:
          - 'Season'
          - 'Date'
          - 'HID', 'AID'  (home/away team IDs)
          - 'HS', 'AS'    (home/away goals scored)
        Optional columns:
          - 'IsPlayoff' or 'Playoff'  -> playoff flag
          - 'Neutral' or 'NeutralSite' -> neutral-site flag
        """
        season = game_row["Season"]
        home_id = str(game_row["HID"])
        away_id = str(game_row["AID"])

        # Ensure both teams have correct preseason/expansion Elo
        self._ensure_team_season(home_id, season)
        self._ensure_team_season(away_id, season)

        home_elo_pre = float(self.team_elo[home_id])
        away_elo_pre = float(self.team_elo[away_id])

        # Flags
        is_playoff = bool(game_row.get("IsPlayoff", False) or game_row.get("Playoff", False))
        is_neutral = bool(game_row.get("Neutral", False) or game_row.get("NeutralSite", False))

        # Home-ice advantage (ignored for neutral-site games)
        hfa = 0.0 if is_neutral else self.home_advantage

        # Base Elo difference from home perspective (before playoff multiplier)
        elo_diff = (home_elo_pre + hfa) - away_elo_pre

        # Apply playoff multiplier to EloDiff when computing win probs
        if is_playoff:
            elo_diff_adj = elo_diff * self.playoff_multiplier
        else:
            elo_diff_adj = elo_diff

        # Pregame win probability for the home team
        p_home = self._win_prob_from_diff(elo_diff_adj)

        # Final score
        hs = self._nz(game_row.get("HS"))
        away_score = self._nz(game_row.get("AS"))

        # Determine outcome
        if hs > away_score:
            home_win = True
            away_win = False
            margin = hs - away_score
        elif away_score > hs:
            home_win = False
            away_win = True
            margin = away_score - hs
        else:
            # Tie (historical seasons). Treat each as 0.5 win, MOV=1.
            home_win = False
            away_win = False
            margin = 1.0

        # Margin-of-victory multiplier (base)
        mov_mult_base = self._mov_multiplier(margin)

        if home_win or away_win:
            # Winner perspective
            if home_win:
                winner_id = home_id
                loser_id = away_id
                winner_elo_pre = home_elo_pre
                loser_elo_pre = away_elo_pre
                winner_prob = p_home
                # Use adjusted diff from winner's perspective
                winner_elo_diff = elo_diff_adj
            else:
                winner_id = away_id
                loser_id = home_id
                winner_elo_pre = away_elo_pre
                loser_elo_pre = home_elo_pre
                winner_prob = 1.0 - p_home
                winner_elo_diff = -elo_diff_adj

            # Autocorrelation adjustment
            autocorr_adj = self._autocorr_adjustment(winner_elo_diff)

            mov_mult = mov_mult_base * autocorr_adj

            # Pregame favorite multiplier: TeamWin (1) - TeamWinProb
            pregame_fav_mult = 1.0 - winner_prob

            # Elo shift for winner
            shift = self.k_factor * mov_mult * pregame_fav_mult

            # Apply updates: zero-sum
            self.team_elo[winner_id] = winner_elo_pre + shift
            self.team_elo[loser_id] = loser_elo_pre - shift

        else:
            # Tie: treat both as 0.5 win; no autocorr adjustment
            teamwin_home = 0.5
            teamwin_away = 0.5

            # MOV multiplier only (no winner-specific autocorr)
            mov_mult = mov_mult_base

            # Home shift
            shift_home = self.k_factor * mov_mult * (teamwin_home - p_home)
            # Away win prob is 1 - p_home
            shift_away = self.k_factor * mov_mult * (teamwin_away - (1.0 - p_home))

            # To keep total roughly zero-sum, average the two shifts
            # (this is a simple compromise for tie handling).
            avg_shift = 0.5 * (shift_home - shift_away)

            self.team_elo[home_id] = home_elo_pre + avg_shift
            self.team_elo[away_id] = away_elo_pre - avg_shift

    def get_input_data(self, home_id, away_id, season, date):
        """
        Return a single feature:
            [logit(Pr_home_win)]

        computed from current Elo ratings and home advantage.
        We do NOT apply playoff multiplier here because the interface
        doesn't tell us if this particular game is a playoff game.
        (If you know it is, you could wrap this model or extend the
        interface to pass that in.)
        """
        home_id = str(home_id)
        away_id = str(away_id)

        # Ensure both teams have season-appropriate Elo (for new seasons).
        self._ensure_team_season(home_id, season)
        self._ensure_team_season(away_id, season)

        if home_id not in self.team_elo or away_id not in self.team_elo:
            return None

        home_elo = float(self.team_elo[home_id])
        away_elo = float(self.team_elo[away_id])

        # Assume regular-season, non-neutral for pregame prediction.
        elo_diff = (home_elo + self.home_advantage) - away_elo
        p_home = self._win_prob_from_diff(elo_diff)

        # Return log-odds (logit) of the home-win probability
        return [inverse_sigmoid(p_home)]
# ===== END fte_model.py =====


# ===== BEGIN gradient_descent_model.py =====


class PoissonModel:
    """
    Poisson Regression model for Hockey.
    Calculates expected goals for Home and Away, then converts to Win/Loss/Draw probabilities.
    """

    def __init__(self, decay=0.995, min_games=10):
        self.decay = decay
        self.min_games = min_games

        # Stores [Goals_For_Weighted_Sum, Goals_Against_Weighted_Sum, Game_Count_Weighted]
        # We track weighted sums so we can calculate rolling averages easily
        self.team_stats = defaultdict(lambda: [1.0, 1.0, 0.1])
        self.league_stats = [1.0, 1.0, 0.1]  # [League_GF, League_GA, League_Games]

    def pre_add_game(self, current):
        pass

    def add_game(self, current):
        hid = current["HID"]
        aid = current["AID"]
        hs = float(current["HS"])
        as_ = float(current["AS"])

        # Update League Average (Weighted)
        self.league_stats[0] = self.league_stats[0] * self.decay + hs
        self.league_stats[1] = self.league_stats[1] * self.decay + as_
        self.league_stats[2] = self.league_stats[2] * self.decay + 1.0

        # Update Home Team
        h_stats = self.team_stats[hid]
        h_stats[0] = h_stats[0] * self.decay + hs  # GF
        h_stats[1] = h_stats[1] * self.decay + as_  # GA
        h_stats[2] = h_stats[2] * self.decay + 1.0  # Games

        # Update Away Team
        a_stats = self.team_stats[aid]
        a_stats[0] = a_stats[0] * self.decay + as_  # GF
        a_stats[1] = a_stats[1] * self.decay + hs  # GA
        a_stats[2] = a_stats[2] * self.decay + 1.0  # Games

    def get_input_data(self, home_id, away_id, season, date):
        h_stats = self.team_stats[home_id]
        a_stats = self.team_stats[away_id]
        l_stats = self.league_stats

        # Need minimum history
        if h_stats[2] < self.min_games or a_stats[2] < self.min_games:
            return None

        # Calculate averages
        avg_league_goals = l_stats[0] / l_stats[2]

        # Home Attack Strength = (Home Avg Goals For) / (League Avg Goals)
        h_att = (h_stats[0] / h_stats[2]) / avg_league_goals
        # Home Defense Strength = (Home Avg Goals Against) / (League Avg Goals)
        h_def = (h_stats[1] / h_stats[2]) / avg_league_goals

        a_att = (a_stats[0] / a_stats[2]) / avg_league_goals
        a_def = (a_stats[1] / a_stats[2]) / avg_league_goals

        # Expected Goals
        # Home Exp Goals = Home Attack * Away Defense * League Avg
        exp_goals_h = h_att * a_def * avg_league_goals
        # Away Exp Goals = Away Attack * Home Defense * League Avg
        exp_goals_a = a_att * h_def * avg_league_goals

        # Calculate Probabilities using Poisson Mass Function
        # We sum probabilities for scores up to 10 goals
        prob_h, prob_a, prob_d = 0.0, 0.0, 0.0

        for i in range(10):  # Home goals
            for j in range(10):  # Away goals
                p = poisson.pmf(i, exp_goals_h) * poisson.pmf(j, exp_goals_a)
                if i > j:
                    prob_h += p
                elif j > i:
                    prob_a += p
                else:
                    prob_d += p

        # Feature: Logit difference (Probability of Home Win vs Away Win)
        # Using a safe inverse sigmoid logic
        diff = prob_h - prob_a
        return [diff]# ===== END poisson_model.py =====

# ===== BEGIN pythagorean_model.py =====


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def inverse_sigmoid(p: float) -> float:
    """Safe-ish inverse of sigmoid, with clipping."""
    eps = 1e-9
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class Pythagorean:
    # https://web.williams.edu/Mathematics/sjmiller/public_html/math/papers/DayaratnaMiller_HockeyFinal.pdf
    def __init__(
        self,
        gamma: float = 5.0,
        translation: float = 0.5,
        min_games: int = 10,
    ):
        self.gamma = float(gamma)
        self.translation = float(translation)
        self.min_games = int(min_games)

        self.team_games: Dict[Tuple, List[Dict]] = defaultdict(list)

    # --------- internal helpers ---------

    @staticmethod
    def _nz(x, default: float = 0.0) -> float:
        """Convert value to finite float, else default."""
        try:
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return default
            v = float(x)
            if not math.isfinite(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    def _team_totals_before(
        self,
        team_id,
        season,
        date,
    ) -> Optional[Tuple[float, float, int]]:
        key = (season, team_id)
        games = self.team_games.get(key, [])
        gf = 0.0
        ga = 0.0
        n = 0

        for g in games:
            if g["Date"] >= date:
                continue
            gf += g["GF"]
            ga += g["GA"]
            n += 1

        if n < self.min_games:
            return None
        return gf, ga, n

    def _pythag_wp(self, gf: float, ga: float) -> float:
        gf_t = gf + self.translation
        ga_t = ga + self.translation

        gf_t = max(gf_t, 0.0)
        ga_t = max(ga_t, 0.0)

        num = gf_t ** self.gamma
        den = num + ga_t ** self.gamma
        if den <= 0.0 or not math.isfinite(den):
            return 0.5
        return num / den

    # --------- BaseSubModel interface ---------

    def pre_add_game(self, game_row: pd.Series):
        pass

    def add_game(self, game_row: pd.Series):
        """
        Update internal state with a finished game.

        Expected columns in `game_row`:
          - 'Season'
          - 'Date'
          - 'HID', 'AID'  (home/away team IDs)
          - 'HS', 'AS'    (home/away goals scored)
        """
        season = game_row["Season"]
        date = game_row["Date"]
        home_id = game_row["HID"]
        away_id = game_row["AID"]

        hs = self._nz(game_row.get("HS"))
        away_score = self._nz(game_row.get("AS"))

        # Store from home-team viewpoint
        self.team_games[(season, home_id)].append(
            {"Date": date, "GF": hs, "GA": away_score}
        )

        # Store from away-team viewpoint
        self.team_games[(season, away_id)].append(
            {"Date": date, "GF": away_score, "GA": hs}
        )

    def get_input_data(self, home_id, away_id, season, date):
        """
        Return [logit(P_home) - logit(P_away)] based on season-to-date
        Pythagorean win percentages, or None if not enough data.
        """
        home_totals = self._team_totals_before(home_id, season, date)
        away_totals = self._team_totals_before(away_id, season, date)

        if home_totals is None or away_totals is None:
            return None

        gf_h, ga_h, _ = home_totals
        gf_a, ga_a, _ = away_totals

        p_home = self._pythag_wp(gf_h, ga_h)
        p_away = self._pythag_wp(gf_a, ga_a)

        # Use log-odds difference as single meta-feature
        return [inverse_sigmoid(p_home)]
    
    def get_prob(self, home_id, away_id, season, date):
        home_totals = self._team_totals_before(home_id, season, date)
        away_totals = self._team_totals_before(away_id, season, date)

        if home_totals is None or away_totals is None:
            return None

        gf_h, ga_h, _ = home_totals
        gf_a, ga_a, _ = away_totals

        p_home = self._pythag_wp(gf_h, ga_h)
        p_away = self._pythag_wp(gf_a, ga_a)

        # Use log-odds difference as single meta-feature
        return p_home# ===== END pythagorean_model.py =====

# ===== BEGIN rf_model.py =====


# Reuse your helpers
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def inverse_sigmoid(p: float) -> float:
    eps = 1e-9
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))

class NHLRandomForestSubModel:
    """
    Random-forest NHL model based on the 'NHL Game Prediction and Season Simulation'
    thesis feature set and model choice.

    - Features per team:
        * Recent form: wins in previous 3 / 8 / 15 games.
        * Current-season aggregates up to date: win%, goal diff per game,
          shots for / against per game.
        * Previous-season aggregates: win%, SRS-like rating (goal diff / game),
          shots against per game.
    - Match feature vector:
        [home_team_features..., away_team_features...]

    - Internal model:
        RandomForestClassifier(class_weight='balanced', ...)
        trained to predict P(home win).

    - get_input_data(home_id, away_id, season, date):
        returns [logit(P_home_win)] or None if the RF is not trained
        or if we can't build features.
    """

    def __init__(self,
                 min_samples: int = 500,
                 retrain_every: int = 500,
                 random_state: int = 42):
        self.min_samples = min_samples
        self.retrain_every = retrain_every
        self.random_state = random_state

        # History of games per team
        # team_id -> list of game dicts (one per past game)
        self.games_by_team: Dict[str, List[Dict]] = defaultdict(list)

        # Training data for RF
        self._X: List[List[float]] = []
        self._y: List[int] = []

        self._n_games_seen = 0
        self.rf: Optional[RandomForestClassifier] = None

        # Fixed feature order
        self.team_feature_names: List[str] = [
            # recent form
            "wins_last_3",
            "wins_last_8",
            "wins_last_15",
            # current-season aggregates
            "season_win_pct",
            "season_goal_diff_pg",
            "season_shots_for_pg",
            "season_shots_against_pg",
            # previous-season aggregates
            "prev_season_win_pct",
            "prev_season_srs",
            "prev_season_shots_against_pg",
        ]

    # ---------- Utility helpers ----------

    @staticmethod
    def _nz(x, default: float = 0.0) -> float:
        try:
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return default
            v = float(x)
            if not math.isfinite(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _ratio(num, den, default: float = 0.0) -> float:
        num = NHLRandomForestSubModel._nz(num)
        den = NHLRandomForestSubModel._nz(den)
        if den <= 0.0:
            return default
        return num / den

    @staticmethod
    def _toi_to_seconds(toi_value) -> Optional[float]:
        """
        Convert TOI/60 representation to seconds.

        - If format is 'MM:SS', parse as minutes:seconds.
        - If already numeric, assume seconds.
        """
        if toi_value is None or (isinstance(toi_value, float) and math.isnan(toi_value)):
            return None

        # String format 'MM:SS'
        if isinstance(toi_value, str):
            if ":" not in toi_value:
                try:
                    return float(toi_value)
                except ValueError:
                    return None
            parts = toi_value.split(":")
            if len(parts) != 2:
                return None
            try:
                m = float(parts[0])
                s = float(parts[1])
            except ValueError:
                return None
            return m * 60.0 + s

        # Numeric
        try:
            return float(toi_value)
        except (TypeError, ValueError):
            return None

    # ---------- Team history & feature construction ----------

    def _update_team_history(self, game_row: pd.Series):
        """
        Store basic per-game stats for each team, for later rolling features.
        Assumes columns:
            - 'Season', 'Date', 'HID', 'AID', 'H', 'HS', 'AS',
              'H_SOG', 'A_SOG'
        """
        season = game_row["Season"]
        date = game_row["Date"]
        home_id = str(game_row["HID"])
        away_id = str(game_row["AID"])

        # Home perspective
        self.games_by_team[home_id].append({
            "Season": season,
            "Date": date,
            "is_home": True,
            "goals_for": self._nz(game_row.get("HS")),
            "goals_against": self._nz(game_row.get("AS")),
            "shots_for": self._nz(game_row.get("H_SOG")),
            "shots_against": self._nz(game_row.get("A_SOG")),
            "win": 1 if self._nz(game_row.get("H")) > 0.5 else 0,
        })

        # Away perspective
        self.games_by_team[away_id].append({
            "Season": season,
            "Date": date,
            "is_home": False,
            "goals_for": self._nz(game_row.get("AS")),
            "goals_against": self._nz(game_row.get("HS")),
            "shots_for": self._nz(game_row.get("A_SOG")),
            "shots_against": self._nz(game_row.get("H_SOG")),
            "win": 1 if self._nz(game_row.get("A")) > 0.5 else 0,
        })

    def _get_team_games_before(self, team_id: str, season, date) -> List[Dict]:
        """
        All games for a given team in a given season strictly before 'date',
        sorted descending by date (most recent first).
        """
        history = self.games_by_team.get(team_id, [])
        out = [g for g in history if g["Season"] == season and g["Date"] < date]
        # Sort newest first
        out.sort(key=lambda g: g["Date"], reverse=True)
        return out

    def _aggregate_current_season(self, games: List[Dict]) -> Dict[str, float]:
        """
        Aggregate current-season stats over all games in 'games'.
        """
        if not games:
            return {
                "season_win_pct": 0.0,
                "season_goal_diff_pg": 0.0,
                "season_shots_for_pg": 0.0,
                "season_shots_against_pg": 0.0,
            }

        n = len(games)
        wins = sum(g["win"] for g in games)
        gf = sum(g["goals_for"] for g in games)
        ga = sum(g["goals_against"] for g in games)
        sf = sum(g["shots_for"] for g in games)
        sa = sum(g["shots_against"] for g in games)

        return {
            "season_win_pct": self._ratio(wins, n),
            "season_goal_diff_pg": self._ratio(gf - ga, n),
            "season_shots_for_pg": self._ratio(sf, n),
            "season_shots_against_pg": self._ratio(sa, n),
        }

    def _aggregate_previous_season(self, team_id: str, season) -> Dict[str, float]:
        """
        Aggregate previous-season stats for team (season-1).
        """
        # Try to interpret season as int
        prev_season = None
        try:
            prev_season = int(season) - 1
        except (TypeError, ValueError):
            # If we can't, we simply don't use previous season info
            prev_season = None

        out = {
            "prev_season_win_pct": 0.0,
            "prev_season_srs": 0.0,  # approximated as goal_diff per game
            "prev_season_shots_against_pg": 0.0,
            "ews60_staying": 0.0,
            "ews60_left": 0.0,
            "ews60_new": 0.0,
            "ews60_overall": 0.0,
        }

        if prev_season is None:
            return out

        # Team's previous-season games
        history = self.games_by_team.get(team_id, [])
        prev_games = [g for g in history if g["Season"] == prev_season]
        if prev_games:
            n = len(prev_games)
            wins = sum(g["win"] for g in prev_games)
            gf = sum(g["goals_for"] for g in prev_games)
            ga = sum(g["goals_against"] for g in prev_games)
            sa = sum(g["shots_against"] for g in prev_games)

            out["prev_season_win_pct"] = self._ratio(wins, n)
            # SRS-like: goal diff per game (we're not re-implementing full SRS here)
            out["prev_season_srs"] = self._ratio(gf - ga, n)
            out["prev_season_shots_against_pg"] = self._ratio(sa, n)

        return out

    def _team_feature_vector(self, team_id: str, season, date) -> List[float]:
        """
        Construct the team feature vector matching self.team_feature_names.
        """
        team_id = str(team_id)
        # Current season games before date
        cur_games = self._get_team_games_before(team_id, season, date)

        # Recent form windowed wins
        def wins_in_last_n(n: int) -> float:
            return float(sum(g["win"] for g in cur_games[:n]))

        rec = {
            "wins_last_3": wins_in_last_n(3),
            "wins_last_8": wins_in_last_n(8),
            "wins_last_15": wins_in_last_n(15),
        }

        cur_agg = self._aggregate_current_season(cur_games)
        prev_agg = self._aggregate_previous_season(team_id, season)

        # Merge all
        feats = {**rec, **cur_agg, **prev_agg}

        # Build final vector in fixed order
        return [float(feats.get(name, 0.0)) for name in self.team_feature_names]

    def _build_match_features(self, home_id, away_id, season, date) -> Optional[List[float]]:
        """
        Combine home & away team vectors into a single match feature vector.
        """
        home_vec = self._team_feature_vector(str(home_id), season, date)
        away_vec = self._team_feature_vector(str(away_id), season, date)

        if home_vec is None or away_vec is None:
            return None

        # Concatenate [home_features..., away_features...]
        return list(home_vec) + list(away_vec)

    # ---------- Model fitting ----------

    def _maybe_fit_rf(self):
        """
        Fit or refit the RandomForestClassifier if we have enough samples.
        """
        if len(self._y) < self.min_samples:
            return

        X = np.asarray(self._X, dtype=float)
        y = np.asarray(self._y, dtype=int)

        # Hyperparameters inspired by thesis (they used grid-search +
        # cross-validation and ended with ~55.6% accuracy).
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            max_features="sqrt",     # akin to limiting number of features per split
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",  # handle home/away imbalance
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf.fit(X, y)

    # ---------- BaseSubModel interface ----------

    def pre_add_game(self, game_row: pd.Series):
        pass

    def add_game(self, game_row: pd.Series):
        season = game_row["Season"]
        date = game_row["Date"]
        home_id = str(game_row["HID"])
        away_id = str(game_row["AID"])

        x = self._build_match_features(home_id, away_id, season, date)
        if x is not None:
            label = 1 if self._nz(game_row.get("H")) > 0.5 else 0
            self._X.append(x)
            self._y.append(label)
            self._n_games_seen += 1

            # Initial fitting and occasional refits
            if (self.rf is None and len(self._y) >= self.min_samples) or \
               (self.rf is not None and self._n_games_seen % self.retrain_every == 0):
                self._maybe_fit_rf()

        # Now update team histories including this game
        self._update_team_history(game_row)

    def get_input_data(self, home_id, away_id, season, date):
        """
        Return a single feature: logit(P_home_win) from the random forest.

        If the forest isn't trained yet or we can't build features, return None.
        """
        if self.rf is None:
            return None

        x = self._build_match_features(home_id, away_id, season, date)
        if x is None:
            return None

        X = np.asarray([x], dtype=float)
        p_home = float(self.rf.predict_proba(X)[0, 1])

        # Provide log-odds as the feature for the ensemble/meta-model.
        return [inverse_sigmoid(p_home)]

    def stand_predict(self, home_id, away_id, season, date):
        """
        Return a single feature: logit(P_home_win) from the random forest.

        If the forest isn't trained yet or we can't build features, return None.
        """
        if self.rf is None:
            return None

        x = self._build_match_features(home_id, away_id, season, date)
        if x is None:
            return None

        X = np.asarray([x], dtype=float)
        p_home = float(self.rf.predict_proba(X)[0, 1])

        # Provide log-odds as the feature for the ensemble/meta-model.
        return p_home

# ===== END rf_model.py =====

# ===== BEGIN score_diff_ridge_model.py =====



class ScoreDiffRidgeModel:
    """
    Linear regression with L2 regularization (Ridge) on past score differences.

    - For each game, we build features from the last `n_lags` score differences
      of the home and away teams (before this game).
    - Target is the actual score difference: HSC - ASC.
    - We train a Ridge regression on these examples with exponential time-decay
      sample weights (recent games matter more).
    - At prediction time, we feed the latest lags of both teams into the ridge
      model and return the predicted score difference as a single feature.
    """

    def __init__(
        self,
        samples_required: int = 1500,
        retrain_interval: int = 200,
        n_lags: int = 10,
        alpha: float = 1.0,
        decay_rate: float = 0.003
    ):
        self.samples_required = samples_required
        self.retrain_interval = retrain_interval
        # how many past score diffs per team to use as features
        self.n_lags = n_lags
        self.alpha = alpha

        self.team_diffs = defaultdict(list)

        self.training_examples: list[list[float]] = []

        self.model: Ridge | None = None
        self.retrain_counter: int = 0
        self.decay_rate: float = float(decay_rate)

    # ---- internal helpers ----

    def _get_team_lags(self, team_id, date):
        """
        Return the last n_lags score differences for this team *before* `date`.
        If there aren't enough games, return None.
        """
        history = [entry for entry in self.team_diffs[team_id] if entry[0] < date]

        if len(history) < self.n_lags:
            return None

        last = history[-self.n_lags:]
        return [float(diff) for (_, diff) in last]

    def _maybe_retrain(self):
        if len(self.training_examples) < self.samples_required:
            return

        if self.retrain_counter > 0:
            return

        self.retrain_counter = self.retrain_interval

        arr = np.array(self.training_examples, dtype=float)
        X = arr[:, :-1]
        y = arr[:, -1]

        # exponential decay: recent games have higher weight
        sample_weights = np.exp(-self.decay_rate * np.arange(len(self.training_examples)))
        sample_weights = sample_weights[::-1]

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y, sample_weight=sample_weights)

    def pre_add_game(self, current):
        """
        Build a training row using only *past* score diffs.

        current is a row/Series-like with:
          - 'Date', 'HID', 'AID', 'HSC', 'ASC', etc.
        """
        date = current["Date"]
        home_id = current["HID"]
        away_id = current["AID"]

        home_feats = self._get_team_lags(home_id, date)
        away_feats = self._get_team_lags(away_id, date)

        if home_feats is None or away_feats is None:
            return

        # target = actual score difference (home - away)
        margin = float(current["HS"] - current["AS"])

        diff_feats = [h - a for h, a in zip(home_feats, away_feats)]
        x = diff_feats

        self.training_examples.append([*x, margin])
        self.retrain_counter -= 1

    def add_game(self, current):
        """
        After pre_add_game has possibly used past data, we now record
        this game's score diff into each team's history and maybe retrain.
        """
        date = current["Date"]
        home_id = current["HID"]
        away_id = current["AID"]

        margin = float(current["HS"] - current["AS"])

        self.team_diffs[home_id].append((date, margin))
        self.team_diffs[away_id].append((date, -margin))

        self._maybe_retrain()

    def get_input_data(self, home_id, away_id, season, date):
        """
        Return [predicted_score_diff] or None if we can't produce a prediction yet.
        """
        if self.model is None:
            return None

        home_feats = self._get_team_lags(home_id, date)
        away_feats = self._get_team_lags(away_id, date)

        if home_feats is None or away_feats is None:
            return None

        diff_feats = [h - a for h, a in zip(home_feats, away_feats)]
        x = np.array([diff_feats], dtype=float)
        pred_margin = float(self.model.predict(x)[0])

        return [pred_margin]
# ===== END score_diff_ridge_model.py =====

# ===== BEGIN meta_model.py =====



class Model:

    def __init__(
            self,
            min_samples_for_ensemble: int = 50,
            retrain_interval: int = 50,
            debug_mode: bool = True,
            decay_rate: int = 0.001,
    ):
        self.debug_mode = debug_mode

        self.model_list = [
            EloNHLSubModel(),
            PoissonModel(),
            Pythagorean(),
            FourFactor(samples_required=min_samples_for_ensemble),
            ScoreDiffRidgeModel(samples_required=min_samples_for_ensemble, retrain_interval=retrain_interval),
            Exhaustion(),
        ]

        self.submodel_dims = [1] * len(self.model_list)
        self.min_samples_for_ensemble = min_samples_for_ensemble
        self.retrain_interval = retrain_interval

        base_lr = LogisticRegression(solver='lbfgs', C=0.5, max_iter=1000)

        self.meta_model = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(base_lr, method='sigmoid', cv='prefit')
        )

        self.past_examples: list[list[float]] = []
        self.retrain_counter: int = self.retrain_interval

        self.memory_window = 1200

        self.base_kelly_fraction = 0.15
        self.peak_bankroll = 1000.0

        self.bet_metrics = {'count': 0, 'volume': 0, 'sum_odds': 0}

    def _get_input_features(self, home_id, away_id, season, date) -> list[float] | None:
        input_data: list[float] = []
        for model in self.model_list:
            data = model.get_input_data(home_id, away_id, season, date)
            if data is None: return None
            input_data.extend(data)
        return input_data

    def _consume_finished_game(self, idx, game_row: pd.Series):
        season = game_row["Season"]
        date = game_row["Date"]
        home_id = game_row["HID"]
        away_id = game_row["AID"]

        if game_row["H"] == 1:
            target = 2
        elif game_row["D"] == 1:
            target = 1
        else:
            target = 0

        input_arr = self._get_input_features(home_id, away_id, season, date)

        if input_arr is not None:
            self.past_examples.append([*input_arr, target])
            self.retrain_counter -= 1

        for model in self.model_list:
            model.pre_add_game(game_row)
            model.add_game(game_row)

    def _maybe_retrain_meta_model(self):
        if len(self.past_examples) < self.min_samples_for_ensemble:
            return

        if self.retrain_counter > 0:
            return

        self.retrain_counter = self.retrain_interval

        data = np.array(self.past_examples)

        if len(data) > self.memory_window:
            window_data = data[-self.memory_window:]
        else:
            window_data = data

        X = window_data[:, :-1]
        y = window_data[:, -1]

        self.meta_model.fit(X, y)

    def place_bets(self, summary, opps, inc) -> pd.DataFrame:
        if inc is not None:
            sorted_inc = inc.sort_values("Date")
            for idx in sorted_inc.index:
                self._consume_finished_game(idx, sorted_inc.loc[idx])

        self._maybe_retrain_meta_model()

        bets = pd.DataFrame(0.0, columns=["BetH", "BetA", "BetD"], index=opps.index)

        try:
            if not hasattr(self.meta_model, 'predict_proba'):
                return bets
            self.meta_model.predict_proba(np.zeros((1, len(self.submodel_dims))))
        except:
            return bets

        bankroll = float(summary.iloc[0]["Bankroll"])
        min_bet = float(summary.iloc[0]["Min_bet"])
        hard_max_bet = float(summary.iloc[0]["Max_bet"])

        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll

        drawdown = (self.peak_bankroll - bankroll) / self.peak_bankroll

        risk_multiplier = max(0.2, 1.0 - (drawdown * 2))

        current_kelly = self.base_kelly_fraction * risk_multiplier

        dynamic_cap = bankroll * 0.05
        effective_max_bet = min(hard_max_bet, dynamic_cap)

        for i in opps.index:
            row = opps.loc[i]
            feats = self._get_input_features(row["HID"], row["AID"], row["Season"], row["Date"])
            if feats is None: continue

            probs = self.meta_model.predict_proba([feats])[0]

            p_away = probs[0]
            p_draw = probs[1]
            p_home = probs[2]

            odds_h = float(row["OddsH"])
            odds_a = float(row["OddsA"])

            def get_kelly_fraction(prob, odds):
                if prob <= 0 or odds <= 1: return 0.0
                b = odds - 1
                q = 1.0 - prob
                f = (b * prob - q) / b
                return max(0.0, f)

            f_h = get_kelly_fraction(p_home, odds_h)
            f_a = get_kelly_fraction(p_away, odds_a)

            f_h *= current_kelly
            f_a *= current_kelly

            total_f = f_h + f_a
            if total_f > 0.35:
                factor = 0.35 / total_f
                f_h *= factor
                f_a *= factor

            stake_h = bankroll * f_h
            stake_a = bankroll * f_a

            if stake_h >= min_bet:
                bets.at[i, "BetH"] = min(stake_h, effective_max_bet)

            if stake_a >= min_bet:
                bets.at[i, "BetA"] = min(stake_a, effective_max_bet)

        return bets# ===== END meta_model.py =====

