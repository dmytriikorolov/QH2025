import math
import numpy as np
from collections import defaultdict


class GlickoModel:
    """
    Glicko-2 implementation adapted for the hackathon interface.

    Key differences from Elo:
      - Tracks Rating (r)
      - Tracks Rating Deviation (RD): certainty of the rating
      - Tracks Volatility (sigma): expected fluctuation

    If a team hasn't played in a while, RD increases (we become less sure).
    """

    def __init__(self, tau=0.35, default_rating=1500, default_rd=350, default_vol=0.06):
        self.tau = tau
        self.default_rating = default_rating
        self.default_rd = default_rd
        self.default_vol = default_vol

        # team_id -> { "r": rating, "rd": deviation, "vol": volatility, "last_date": timestamp }
        self.teams = defaultdict(lambda: {
            "r": self.default_rating,
            "rd": self.default_rd,
            "vol": self.default_vol,
            "last_date": None
        })

        self.last_season = None

        # Glicko-2 scale factor
        self.q = math.log(10) / 400
        self.glicko_scale = 173.7178

    def _g(self, phi):
        return 1.0 / math.sqrt(1.0 + 3.0 * (phi ** 2) / (math.pi ** 2))

    def _E(self, mu, mu_j, phi_j):
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def _win_prob(self, r_home, rd_home, r_away, rd_away):
        """
        Calculates win probability for home team given ratings and deviations.
        """
        # Home advantage (approx 30-50 Elo points is standard, we use 30 here)
        home_adv = 30.0

        rd_sq = rd_home ** 2 + rd_away ** 2
        return 1.0 / (1.0 + 10.0 ** (-((r_home + home_adv) - r_away) / (
                    400.0 * math.sqrt(1 + (3 * math.log(10) ** 2 * rd_sq) / (400 ** 2 * math.pi ** 2)))))

    def _new_season_reset(self):
        """Regress teams toward the mean at the start of a new season."""
        for tid in self.teams:
            # Regress rating to mean (1500)
            self.teams[tid]["r"] = 0.75 * self.teams[tid]["r"] + 0.25 * 1500
            # Increase uncertainty (RD)
            self.teams[tid]["rd"] = math.sqrt(self.teams[tid]["rd"] ** 2 + 50 ** 2)
            # Reset volatility slightly
            self.teams[tid]["vol"] = 0.06

    def pre_add_game(self, current):
        pass

    def add_game(self, current):
        season = current["Season"]
        if self.last_season is not None and season != self.last_season:
            self._new_season_reset()
        self.last_season = season

        hid = current["HID"]
        aid = current["AID"]

        # Determine score (S): 1 for win, 0 for loss, 0.5 for draw
        h_score = 0.0
        if current["H"] == 1:
            h_score = 1.0
        elif current["A"] == 1:
            h_score = 0.0
        else:
            h_score = 0.5  # Draw case

        # Update both teams
        self._update_team(hid, aid, h_score)
        self._update_team(aid, hid, 1.0 - h_score)

    def _update_team(self, player_id, opponent_id, score):
        # 1. Get current values
        p = self.teams[player_id]
        o = self.teams[opponent_id]

        # Convert to Glicko-2 scale
        mu = (p["r"] - 1500) / self.glicko_scale
        phi = p["rd"] / self.glicko_scale
        sigma = p["vol"]

        mu_j = (o["r"] - 1500) / self.glicko_scale
        phi_j = o["rd"] / self.glicko_scale

        # 2. Compute estimated variance (v) and delta
        g_phij = self._g(phi_j)
        E = self._E(mu, mu_j, phi_j)

        v = 1.0 / (g_phij ** 2 * E * (1 - E))
        delta = v * g_phij * (score - E)

        # 3. Update Volatility (Sigma) using iterative algorithm (Illinois algorithm)
        a = math.log(sigma ** 2)

        def f(x):
            ex = math.exp(x)
            num = ex * (delta ** 2 - phi ** 2 - v - ex)
            den = 2 * ((phi ** 2 + v + ex) ** 2)
            return (num / den) - ((x - a) / (self.tau ** 2))

        epsilon = 0.000001
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA = f(A)
        fB = f(B)

        while abs(B - A) > epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC

        new_sigma = math.exp(A / 2.0)

        # 4. Update RD (phi) and Rating (mu)
        phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)
        new_phi = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
        new_mu = mu + (new_phi ** 2) * g_phij * (score - E)

        # 5. Convert back to standard scale
        p["r"] = 1500 + new_mu * self.glicko_scale
        p["rd"] = new_phi * self.glicko_scale
        p["vol"] = new_sigma

    def get_input_data(self, home_id, away_id, season, date):
        # Handle season reset if we are predicting the first game of new season
        if self.last_season is not None and season != self.last_season:
            self._new_season_reset()
            self.last_season = season

        h = self.teams[home_id]
        a = self.teams[away_id]

        # Return Win Probability for Home Team
        # We pass this to meta-model. Meta-model loves probabilities.
        prob = self._win_prob(h["r"], h["rd"], a["r"], a["rd"])

        # To avoid log(0) issues later, clip it safely
        prob = max(0.01, min(0.99, prob))

        # Return as log-odds (logit) to match your other models' style
        return [math.log(prob / (1.0 - prob))]