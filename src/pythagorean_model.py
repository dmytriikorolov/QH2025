import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

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
        return p_home