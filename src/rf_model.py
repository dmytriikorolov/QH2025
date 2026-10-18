import math
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

