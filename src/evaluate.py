import pandas as pd
from meta_model import Model
from environment import Environment

# 1. Load the data
games = pd.read_csv("../data/round1_games.csv", index_col=0, parse_dates=["Date", "Open"])

# 2. FILTER: Only keep the "Modern Era" (2015 onwards)
# This removes the broken 1989-2014 data where odds aren't "Open" yet.
games = games[games["Date"] >= "2015-09-01"]

# 3. Force the simulation start date
# We use the first actual game date in the filtered set.
actual_start = games["Date"].min()
print(f"Simulation starting from: {actual_start}")

# 4. Initialize Environment with the filtered data
env = Environment(
    games,
    Model(),
    start_date=actual_start,
    init_bankroll=1000,
    min_bet=5,
    max_bet=100
)

# 5. Run
evaluation = env.run()

print()
print(f"Final bankroll: {env.bankroll:.2f}")

history = env.get_history()
history.to_csv("results_round1.csv")
print("Results saved to results_round1.csv")