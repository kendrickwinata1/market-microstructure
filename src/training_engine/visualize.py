import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("inputs/historical_labels.csv")

# Ensure 'Timestamp' is in datetime format for plotting
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Handle missing columns defensively
if "BettiCurve_1" not in df.columns or "BettiCurve_2" not in df.columns:
    raise ValueError("DataFrame must include 'BettiCurve_1' and 'BettiCurve_2' columns.")

# Mask BettiCurve_1 values less than 3 (show only significant values)
betti_1_masked = np.where(df["BettiCurve_1"] >= 3, df["BettiCurve_1"], np.nan)

# Create a figure and the first y-axis (Price)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["Timestamp"], df["Price"], "b-", label="Price")
ax1.set_xlabel("Time")
ax1.set_ylabel("Price", color="b")
ax1.tick_params(axis="y", labelcolor="b")

# Second y-axis (BettiCurve_2)
ax2 = ax1.twinx()
ax2.plot(df["Timestamp"], df["BettiCurve_2"], "g-", label="BettiCurve_2")
ax2.set_ylabel("BettiCurve_2", color="g")
ax2.tick_params(axis="y", labelcolor="g")

# Third y-axis (BettiCurve_1, masked), offset outward for clarity
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(df["Timestamp"], betti_1_masked, "r-", label="BettiCurve_1")
ax3.set_ylabel("BettiCurve_1", color="r")
ax3.tick_params(axis="y", labelcolor="r")

# Combine all legends in one location
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
plt.legend(
    lines_1 + lines_2 + lines_3,
    labels_1 + labels_2 + labels_3,
    loc="upper left"
)

# Add a title and improve layout
plt.title("Price and Betti Curves Over Time")
fig.tight_layout()  # Adjust layout so y-axes and labels do not overlap

# Show the plot
plt.show()
