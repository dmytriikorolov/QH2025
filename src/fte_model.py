import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


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
