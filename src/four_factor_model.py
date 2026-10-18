import math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
