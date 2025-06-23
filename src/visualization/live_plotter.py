import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def live_performance_plot():
    csv_path = "performance_report.csv"
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten() 
    plt.subplots_adjust(hspace=0.4)
    param_list = [
        "wallet_balance", "unrealized_pnl",
        "realized_pnl", "sharpe_ratio",
        "max_drawdown"
    ]

    def animate(i):
        try:
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            timestamps = df['timestamp']
            for idx, param in enumerate(param_list):
                ax = axs.flat[idx]
                ax.clear()
                if param in df.columns:
                    ax.plot(timestamps, df[param], marker='o')
                    ax.set_title(param)
                    ax.set_xlabel("Time")
                    ax.set_ylabel(param)
            fig.suptitle("Live Performance Report")
        except Exception as e:
            print("Plot update error:", e)

    ani = animation.FuncAnimation(fig, animate, interval=2000)  # Update every 2s
    fig.delaxes(axs[5])
    plt.show()




# import datetime
# import tkinter as tk
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import pandas as pd


# class LivePlotter:
#     """
#     Real-time plotting of BTC price and detected peaks/troughs using matplotlib and tkinter.
#     - Receives new data from a strategy object.
#     - Maintains a buffer of recent data for plotting.
#     - Plots live BTC price, peaks, and troughs.
#     """

#     def __init__(self, master, strategy, data_window=60):
#         self.master = master
#         self.strategy = strategy
#         self.data_window = data_window  # seconds of data to display

#         self.fig, self.ax = plt.subplots()
#         self.line, = self.ax.plot([], [], "b-", label="BTC Price")
#         self.peaks_plot, = self.ax.plot([], [], "r^", label="Peaks")
#         self.troughs_plot, = self.ax.plot([], [], "gv", label="Troughs")
#         self.ax.legend()
#         self.ax.grid(True)
#         self.ax.set_xlabel("Time")
#         self.ax.set_ylabel("Price")

#         self.data_buffer = pd.DataFrame(columns=["Timestamp", "Price"])
#         self.last_price = None

#         # Set up animation
#         self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000)

#         # Add Start Plotting button to tkinter window
#         self.plot_button = tk.Button(master, text="Start Plotting", command=self.run_plot)
#         self.plot_button.pack()

#         # Log file for peaks/troughs (optional: remove if not needed)
#         self.file = open("btc_peaks_troughs_log.txt", "w")

#     def update_plot(self, frame):
#         current_time = datetime.datetime.now()
#         new_data = self.strategy.collect_new_data()

#         # Append new data to the buffer
#         if new_data is not None and not new_data.empty:
#             self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)

#         # Remove data outside the time window
#         cutoff_time = current_time - pd.Timedelta(seconds=self.data_window)
#         self.data_buffer = self.data_buffer[self.data_buffer["Timestamp"] >= cutoff_time]

#         if not self.data_buffer.empty:
#             # Update strategy data and run analysis
#             self.strategy.data = self.data_buffer.copy()
#             self.strategy.analyze_data()

#             # Prepare times/prices for plotting
#             times = pd.to_datetime(self.data_buffer["Timestamp"])
#             prices = self.data_buffer["Price"]

#             # Convert detected peaks/troughs to integer index positions
#             peaks_idx = getattr(self.strategy, 'peaks', [])
#             troughs_idx = getattr(self.strategy, 'troughs', [])
#             peaks_idx = [int(p) for p in peaks_idx if p < len(times)]
#             troughs_idx = [int(t) for t in troughs_idx if t < len(times)]

#             # Update plots
#             self.line.set_data(times, prices)
#             self.peaks_plot.set_data(times.iloc[peaks_idx], prices.iloc[peaks_idx])
#             self.troughs_plot.set_data(times.iloc[troughs_idx], prices.iloc[troughs_idx])

#             # Log and print price changes
#             if self.last_price is not None:
#                 diff = prices.iloc[-1] - self.last_price
#                 print(f"Latest BTC Price: ${prices.iloc[-1]:.2f}, Change: {diff:.2f}")
#             self.last_price = prices.iloc[-1]

#             # Logging peaks/troughs (optional)
#             peaks_log = [(str(times.iloc[p]), prices.iloc[p]) for p in peaks_idx]
#             troughs_log = [(str(times.iloc[t]), prices.iloc[t]) for t in troughs_idx]
#             self.file.write(f"Time: {times.iloc[-1]} | Price: {prices.iloc[-1]:.2f}\n")
#             self.file.write(f"Peaks: {peaks_log}\nTroughs: {troughs_log}\n")
#             self.file.flush()

#             # Rescale plot
#             self.ax.relim()
#             self.ax.autoscale_view()

#         return self.line, self.peaks_plot, self.troughs_plot

#     def run_plot(self):
#         plt.show()

#     def close(self):
#         self.file.close()
#         plt.close(self.fig)

#     def __del__(self):
#         self.close()
