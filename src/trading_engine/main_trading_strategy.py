import csv
import joblib
import numpy as np
import pandas as pd
import pywt
from gtda.diagrams import BettiCurve
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SlidingWindow
from hurst import compute_Hc
from pykalman import KalmanFilter
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import datetime
import os
import warnings
import logging

# Ignore pandas FutureWarnings for a cleaner log
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

class TradingStrategy:
    """
    This class implements your trading signal logic:
    - Collects and aggregates streaming tick data
    - Computes technical indicators and engineered features
    - Runs a machine learning model to output trading signals
    """
    def __init__(self, queue):
        self.queue = queue
        self.raw_data = pd.DataFrame(columns=["Timestamp", "Price"])
        self.data = pd.DataFrame(columns=["Timestamp", "Price"])
        self.file_path = "ohlc_seconds.csv"
        self.peaks = []
        self.troughs = []
        self.smoothed_prices = []
        # Load the trained classification/regression model
        self.model = joblib.load("training_engine/outputs/logistic_regression_model_updated.pkl")

    def collect_new_data(self):
        """
        Pulls new tick data from the queue and appends it to the raw_data DataFrame.
        Each tick is expected to be a tuple: (timestamp, price)
        """
        new_rows = []
        count = 0

        while not self.queue.empty():
            tick = self.queue.get()
            logging.info(f"[TradingStrategy] New tick from queue: {tick}")

            # Defensive: handle different tick formats
            if isinstance(tick, dict):
                ts, price = tick.get("Timestamp") or tick.get("datetime"), tick.get("Price") or tick.get("lastprice")
            elif isinstance(tick, (tuple, list)):
                ts, price = tick[0], tick[1]
            else:
                logging.warning(f"[TradingStrategy] Unexpected tick format: {tick}")
                continue

            if price != "" and price is not None:
                try:
                    price = float(price)
                    new_rows.append({"Timestamp": ts, "Price": price})
                    count += 1
                except ValueError:
                    logging.warning(f"[TradingStrategy] Invalid price value: {price}")
            else:
                logging.warning("[TradingStrategy] Empty or invalid price in data stream.")

        if new_rows:
            new_data = pd.DataFrame(new_rows)
            new_data["Timestamp"] = pd.to_datetime(new_data["Timestamp"])
            self.raw_data = pd.concat([self.raw_data, new_data], ignore_index=True)
            # Optional: set index for convenience
            if "Timestamp" not in self.raw_data.index.names:
                self.raw_data.set_index("Timestamp", inplace=True, drop=False)

        logging.info(f"[TradingStrategy] Collected {count} new ticks.")
        logging.info(f"[TradingStrategy] Data buffer length: {len(self.raw_data)}")
        logging.info(f"[TradingStrategy] Sample buffer:\n{self.raw_data.tail()}")

        return self.raw_data


    def aggregate_data(self, freq="1S", save_csv=False):
        """
        Aggregates raw_data into OHLC based on the given frequency.
        Keeps only the last 300 seconds of data for rolling analysis.
        Optionally saves to CSV for debugging.
        """
        if self.raw_data.empty:
            logging.warning("[TradingStrategy] Raw data is empty; cannot aggregate.")
            return

        # Convert timestamp and set index
        df = self.raw_data.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.set_index("Timestamp", inplace=True)
        df = df[pd.to_numeric(df["Price"], errors="coerce").notnull()]
        df["Price"] = df["Price"].astype(float)

        # Keep only last 300 seconds of data
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(seconds=300)
        df = df[df.index >= cutoff_time]

        # Resample to OHLC
        ohlc = df["Price"].resample(freq).ohlc()
        ohlc.dropna(inplace=True)
        ohlc.columns = [col.capitalize() for col in ohlc.columns]

        # Save as self.data for the rest of the pipeline
        self.data = ohlc.reset_index()
        logging.info(f"[TradingStrategy] Aggregated data to {freq} OHLC; shape: {self.data.shape}")
        logging.info(f"[TradingStrategy] Aggregated sample:\n{self.data.tail()}")

        if save_csv:
            self.data.to_csv("ohlc_seconds.csv", index=False)


    def analyze_data(self):
        """
        Analyze market data and return trade signal (Buy/Sell/Hold).
        """
        self.data = pd.read_csv(self.file_path)
        self.data["Date"] = pd.to_datetime(self.data["Timestamp"])
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)

        if self.data[["Open", "High", "Low", "Close"]].isnull().any().any():
            logging.warning("Incomplete data, skipping analysis.")
            return

        self.calculate_daily_percentage_change()
        self.calculate_moving_averages_and_rsi()
        self.calculate_stochastic_oscillator(k_window=5, d_window=3)
        self.construct_kalman_filter()
        self.calculate_first_second_order_derivatives()
        self.detect_rolling_peaks_and_troughs(window_size=3)
        return self.predict()

    def calculate_daily_percentage_change(self):
        """Computes daily % change from OHLC data."""
        self.data["Daily_Change"] = self.data["Close"].pct_change() * 100
        self.data["Daily_Change_Open_to_Close"] = (
            (self.data["Open"] - self.data["Close"].shift(1)) / self.data["Close"].shift(1)
        ) * 100

    def perform_fourier_transform_analysis(self):
        """
        Computes the dominant frequencies/amplitudes in closing price using FFT.
        """
        data_window = self.data
        close_prices = data_window["Close"].to_numpy()
        N = len(close_prices)
        T = 1.0  # 1 second intervals
        close_fft = fft(close_prices)
        fft_freq = np.fft.fftfreq(N, T)
        positive_frequencies = fft_freq[: N // 2]
        positive_fft_values = 2.0 / N * np.abs(close_fft[0 : N // 2])
        amplitude_threshold = 0.1
        significant_peaks, _ = find_peaks(positive_fft_values, height=amplitude_threshold)
        significant_frequencies = positive_frequencies[significant_peaks]
        significant_amplitudes = positive_fft_values[significant_peaks]
        days_per_cycle = 1 / significant_frequencies
        self.fft_features = pd.DataFrame(
            {"Frequency": significant_frequencies, "Amplitude": significant_amplitudes, "MinutesPerCycle": days_per_cycle}
        )

    def calculate_stochastic_oscillator(self, k_window=5, d_window=3):
        low_min = self.data["Low"].rolling(window=k_window).min()
        high_max = self.data["High"].rolling(window=k_window).max()
        range_max_min = high_max - low_min

        # Avoid division by zero by replacing zeros with NaN, then filling forward
        range_max_min = range_max_min.replace(0, np.nan).ffill().bfill()

        self.data["%K"] = 100 * (self.data["Close"] - low_min) / range_max_min
        self.data["%D"] = self.data["%K"].rolling(window=d_window).mean()

        # Fill remaining NaNs (initial rows) with neutral value (50)
        self.data["%K"].fillna(50, inplace=True)
        self.data["%D"].fillna(50, inplace=True)


    def calculate_slow_stochastic_oscillator(self, d_window=3, slow_k_window=3):
        """
        Calculates the slow stochastic oscillator for trend confirmation.
        """
        self.data["Slow %K"] = self.data["%K"].rolling(window=slow_k_window).mean()
        self.data["Slow %D"] = self.data["Slow %K"].rolling(window=d_window).mean()
        self.data["Slow %K"].bfill(inplace=True)
        self.data["Slow %D"].bfill(inplace=True)

    def detect_rolling_peaks_and_troughs(self, window_size=3):
        """
        More sensitive local peak/trough detection.
        """
        self.data["isLocalPeak"] = self.data["Close"] > self.data["Close"].rolling(window_size, center=True).max().shift(-1)
        self.data["isLocalTrough"] = self.data["Close"] < self.data["Close"].rolling(window_size, center=True).min().shift(-1)

        self.data["Label"] = "Hold"
        self.data.loc[self.data["isLocalPeak"], "Label"] = "Sell"
        self.data.loc[self.data["isLocalTrough"], "Label"] = "Buy"


    def calculate_rsi(self, window=5):
        delta = self.data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(to_replace=0).ffill().bfill()
        rsi = 100 - (100 / (1 + rs))
        return rsi.ffill().bfill()


    def calculate_moving_averages_and_rsi(self):
        """
        Computes short/long moving averages and RSI.
        """
        short_window = 5
        long_window = 20
        rsi_period = 14
        self.data["Short_Moving_Avg"] = self.data["Close"].rolling(window=short_window).mean()
        self.data["Long_Moving_Avg"] = self.data["Close"].rolling(window=long_window).mean()
        self.data["RSI"] = self.calculate_rsi(window=rsi_period)

    def construct_kalman_filter(self):
        """
        Applies Kalman filter to closing price for state estimation and noise reduction.
        """
        close_prices = self.data["Close"]
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        state_means, _ = kf.filter(close_prices.values)
        kalman_estimates = pd.Series(state_means.flatten(), index=self.data.index)
        kalman_estimates = pd.DataFrame({"KalmanFilterEst": kalman_estimates})
        self.data = self.data.join(kalman_estimates)

    def calculate_days_since_peaks_and_troughs(self):
        """
        For each row, computes time (in minutes) since last detected peak/trough 
        and price changes from those reference points.
        """
        self.data["MinutesSincePeak"] = 0.0
        self.data["MinutesSinceTrough"] = 0.0
        self.data["PriceChangeSincePeak"] = 0.0
        self.data["PriceChangeSinceTrough"] = 0.0

        last_peak_time = None
        last_trough_time = None
        last_peak_price = None
        last_trough_price = None

        for index, row in self.data.iterrows():
            current_time = pd.to_datetime(row["Date"])
            current_price = row["Open"]

            # Update peak/trough checkpoints
            if row["Label"] == "Sell":
                last_peak_time = current_time
                last_peak_price = current_price
            elif row["Label"] == "Buy":
                last_trough_time = current_time
                last_trough_price = current_price

            # Compute time since last peak/trough
            minutes_since_peak = (current_time - last_peak_time).total_seconds() / 60.0 if last_peak_time else 0
            minutes_since_trough = (current_time - last_trough_time).total_seconds() / 60.0 if last_trough_time else 0

            # Compute price changes
            price_change_peak = current_price - last_peak_price if last_peak_price is not None else 0.0
            price_change_trough = current_price - last_trough_price if last_trough_price is not None else 0.0

            # Set values
            self.data.at[index, "MinutesSincePeak"] = minutes_since_peak
            self.data.at[index, "MinutesSinceTrough"] = minutes_since_trough
            self.data.at[index, "PriceChangeSincePeak"] = price_change_peak
            self.data.at[index, "PriceChangeSinceTrough"] = price_change_trough

        # Fill any remaining NaNs just in case
        self.data.fillna(0, inplace=True)

    def calculate_first_second_order_derivatives(self):
        """
        Calculates first and second derivatives (velocity/acceleration) of key features.
        """
        for feature in ["KalmanFilterEst", "Short_Moving_Avg", "Long_Moving_Avg"]:
            self.data[f"{feature}_1st_Deriv"] = self.data[feature].diff() * 100
            self.data[f"{feature}_2nd_Deriv"] = self.data[f"{feature}_1st_Deriv"].diff() * 100
        self.data.bfill(inplace=True)

    def predict(self):
    
        """
        Improved Predict method with better logging and exception handling.
        """
        feature_set = [
            "Short_Moving_Avg_1st_Deriv",
            "Short_Moving_Avg_2nd_Deriv",
            "Long_Moving_Avg_1st_Deriv",
            "Long_Moving_Avg_2nd_Deriv",
            "MinutesSincePeak",
            "MinutesSinceTrough",
            "%K",
            "%D",
            "KalmanFilterEst_1st_Deriv",
            "KalmanFilterEst_2nd_Deriv",
        ]

        last_row = self.data[feature_set].iloc[-1:]

        logging.info(f"[Predict] Last feature row for prediction:\n{last_row}")

        if last_row.isnull().any().any():
            logging.warning("[Predict] NaN values in features; returning HOLD signal.")
            price = self.data["Close"].iloc[-1]
            return ("Hold", price, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        try:
            prediction = self.model.predict(last_row)[0]
            logging.info(f"[Predict] Model Prediction: {prediction}")
            price = self.data["Close"].iloc[-1]
            return (prediction, price, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            logging.error(f"[Predict] Model prediction error: {e}")
            price = self.data["Close"].iloc[-1]
            return ("Hold", price, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


    # --- Additional methods omitted for brevity, but should follow the same style ---

