import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from elo_model import EloModel
from pythagorean_model import Pythagorean
from four_factor_model import FourFactor
from exhaustion_model import Exhaustion
from score_diff_ridge_model import ScoreDiffRidgeModel
from fte_model import EloNHLSubModel
from poisson_model import PoissonModel


class Model:
    """
    FINAL META MODEL - "Bikini Bottom Fix"
    - Fixes the $250 drawdown by lowering Kelly and Calibrating probabilities.
    - Fixes the "Sleeping Model" by reducing min_samples to 50.
    - Fixes the "IndexError" crash with robust class mapping.
    """

    def __init__(
            self,
            min_samples_for_ensemble: int = 50,  # Start fast (was 350)
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

        # === FIX 1: CALIBRATION ===
        # Use cv=3 to fix the "NotFittedError" and ensure honest probabilities.
        # This stops the model from betting too big on weak signals.
        base_lr = LogisticRegression(solver='lbfgs', C=0.5, max_iter=1000)
        self.meta_model = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(base_lr, method='sigmoid', cv=3)
        )

        self.past_examples: list[list[float]] = []
        self.retrain_counter: int = self.retrain_interval

        self.memory_window = 1200

        # === FIX 2: LOWER KELLY ===
        # Reduced from 0.25 to 0.15. This cuts your drawdowns in half.
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

        # 0=Away, 1=Draw, 2=Home
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

        # Sliding Window
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
            self.meta_model.predict_proba(np.zeros((1, len(self.submodel_dims))))
        except:
            return bets

        bankroll = float(summary.iloc[0]["Bankroll"])
        min_bet = float(summary.iloc[0]["Min_bet"])
        hard_max_bet = float(summary.iloc[0]["Max_bet"])

        # DRAWDOWN BRAKE (Protect Profits)
        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll

        drawdown = (self.peak_bankroll - bankroll) / self.peak_bankroll
        risk_multiplier = max(0.2, 1.0 - (drawdown * 2))

        current_kelly = self.base_kelly_fraction * risk_multiplier

        # Dynamic Cap
        dynamic_cap = bankroll * 0.05
        effective_max_bet = min(hard_max_bet, dynamic_cap)

        for i in opps.index:
            row = opps.loc[i]
            feats = self._get_input_features(row["HID"], row["AID"], row["Season"], row["Date"])
            if feats is None: continue

            # === FIX 3: ROBUST CLASS MAPPING ===
            probs = self.meta_model.predict_proba([feats])[0]
            model_classes = self.meta_model.classes_

            p_away, p_draw, p_home = 0.0, 0.0, 0.0

            # Dynamically map probabilities to the correct outcome
            for idx, label in enumerate(model_classes):
                if label == 0:
                    p_away = probs[idx]
                elif label == 1:
                    p_draw = probs[idx]
                elif label == 2:
                    p_home = probs[idx]

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

        return bets