import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge


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
