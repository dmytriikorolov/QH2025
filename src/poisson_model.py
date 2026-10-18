import math
import numpy as np
from collections import defaultdict
from scipy.stats import poisson


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
        return [diff]