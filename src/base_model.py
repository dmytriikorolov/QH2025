import math
import numpy as np
import pandas as pd


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
