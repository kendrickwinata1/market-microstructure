import numpy as np
import pandas as pd

def main():
    # Load price and feature data
    df = pd.read_csv("inputs/historical_labels.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Calculate forward price change (current to next row)
    df["Price_Change"] = df["Price"].diff().shift(-1)

    # Filter rows where BettiCurve_2 is at least 1
    betti_2_filtered = df[df["BettiCurve_2"] > 1]

    # Correlation between BettiCurve_2 and next price change
    correlation = betti_2_filtered["BettiCurve_2"].corr(betti_2_filtered["Price_Change"])
    print(f"Correlation between BettiCurve_2 > 1 and subsequent Price Change: {correlation:.5f}")

    # Analyze correlations with lagged BettiCurve_2 values (from 1 to 19 steps back)
    for lag in range(1, 20):
        df[f"BettiCurve_2_Lag{lag}"] = df["BettiCurve_2"].shift(lag)
        lag_corr = df[f"BettiCurve_2_Lag{lag}"].corr(df["Price_Change"])
        print(f"Lag {lag} Correlation: {lag_corr:.5f}")

if __name__ == "__main__":
    main()
