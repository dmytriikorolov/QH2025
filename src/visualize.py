import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_performance(filename="results.csv"):
    try:
        # Load the data
        df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'. Run Evaluate.py first!")
        return

    # Create a nice layout
    plt.style.use('bmh')  # 'bmh' is a clean style for financial data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PLOT: Bankroll over time ---
    ax1.plot(df.index, df['Bankroll'], color='#2ca02c', linewidth=2, label='Portfolio Value')

    # Add a baseline (Starting Money)
    start_cash = df['Bankroll'].iloc[0]
    ax1.axhline(y=start_cash, color='red', linestyle='--', alpha=0.5, label='Initial Stake')

    ax1.set_title(f'Performance History (Final: ${df["Bankroll"].iloc[-1]:.2f})', fontsize=14)
    ax1.set_ylabel('Bankroll ($)')
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- BOTTOM PLOT: Drawdown (How much are we down from the peak?) ---
    # This is crucial for checking risk. If this hits -50%, your model is too risky.
    rolling_max = df['Bankroll'].cummax()
    drawdown = (df['Bankroll'] - rolling_max) / rolling_max

    ax2.fill_between(df.index, drawdown, 0, color='#d62728', alpha=0.3, label='Drawdown')
    ax2.plot(df.index, drawdown, color='#d62728', linewidth=1)

    ax2.set_title('Drawdown (Risk Analysis)', fontsize=12)
    ax2.set_ylabel('% Down from Peak')
    ax2.set_xlabel('Date')

    # Format Y-axis as percentage
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    # Format Date Axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every 2 years to avoid clutter

    plt.tight_layout()

    # Save the plot AND show it
    plt.savefig("performance_graph.png")
    print("Graph saved as performance_graph.png")
    plt.show()


if __name__ == "__main__":
    plot_performance()