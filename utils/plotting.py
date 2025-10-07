import matplotlib.pyplot as plt
import pandas as pd

def plot_portfolio(df: pd.DataFrame, title="Portfolio Performance"):
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["total_asset"], label="Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
