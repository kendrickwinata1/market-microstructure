import pandas as pd

def calculate_pnl(signal_file="signals.csv"):
    """
    Calculate total PnL from a series of Buy/Sell signals in a CSV file.
    Each 'Buy' is matched to the next available 'Sell'.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(signal_file)

    pnl = 0.0
    positions = []

    for index, row in df.iterrows():
        if row["Signal"] == "Buy":
            positions.append(row["Price"])  # Track buy price
        elif row["Signal"] == "Sell" and positions:
            buy_price = positions.pop(0)   # FIFO: earliest buy price
            sell_price = row["Price"]
            pnl += sell_price - buy_price  # Add profit for this round-trip

    print(f"Total PnL: ${pnl:.2f}")
    return pnl

if __name__ == "__main__":
    calculate_pnl("signals.csv")
