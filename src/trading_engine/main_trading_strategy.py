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
        """
        new_rows = []
        while not self.queue.empty():
            data_point = self.queue.get()
            if data_point[1] != "":
                new_row = {"Timestamp": data_point[0], "Price": float(data_point[1])}
                new_rows.append(new_row)
            else:
                print("Warning: Empty or invalid price in data stream.")

        if new_rows:
            new_data = pd.DataFrame(new_rows)
            new_data["Timestamp"] = pd.to_datetime(new_data["Timestamp"])
            if not self.raw_data.empty:
                self.raw_data = pd.concat([self.raw_data, new_data], ignore_index=True)
            else:
                self.raw_data = new_data
            if "Timestamp" not in self.raw_data.index.names:
                self.raw_data.set_index("Timestamp", inplace=True, drop=False)
        return self.raw_data

    def aggregate_data(self):
        """
        Aggregates collected tick data into OHLC bars, resampled by second.
        Only keeps the last 300 seconds of data.
        """
        if self.raw_data.empty:
            return

        self.raw_data["Timestamp"] = pd.to_datetime(self.raw_data["Timestamp"])
        if "Timestamp" not in self.raw_data.index.names:
            self.raw_data.set_index("Timestamp", inplace=True)
        self.raw_data.index = pd.to_datetime(self.raw_data.index)

        # Keep only the last 300 seconds of data
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(seconds=300)
        self.raw_data = self.raw_data[self.raw_data.index >= cutoff_time]

        try:
            ohlc = self.raw_data["Price"].resample("S").ohlc()
        except:
            return

        ohlc.columns = [col.capitalize() for col in ohlc.columns]
        print("initializing..")
        ohlc.to_csv("ohlc_seconds.csv")

    def analyze_data(self):
        """
        Main feature engineering and inference pipeline:
        - Loads OHLC data
        - Calculates features
        - Returns model action (Buy/Sell/Hold + price)
        """
        self.data = pd.read_csv(self.file_path)
        self.data["Date"] = pd.to_datetime(self.data["Timestamp"])
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)

        # Skip if OHLC data is incomplete
        if self.data[["Open", "High", "Low", "Close"]].isnull().any().any():
            print("Incomplete data, skipping analysis.")
            return

        # Feature engineering: indicators, oscillators, transforms, peaks, etc.
        self.calculate_daily_percentage_change()
        self.perform_fourier_transform_analysis()
        self.calculate_stochastic_oscillator()
        self.calculate_slow_stochastic_oscillator()
        self.construct_kalman_filter()
        self.detect_rolling_peaks_and_troughs()
        self.calculate_moving_averages_and_rsi()
        self.calculate_days_since_peaks_and_troughs()
        self.calculate_first_second_order_derivatives()

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

    def calculate_stochastic_oscillator(self, k_window=14, d_window=3, slow_k_window=3):
        """
        Calculates the Stochastic Oscillator %K and %D for momentum/overbought/oversold detection.
        """
        low_min = self.data["Low"].rolling(window=k_window).min()
        high_max = self.data["High"].rolling(window=k_window).max()
        self.data["%K"] = 100 * (self.data["Close"] - low_min) / (high_max - low_min)
        self.data["%D"] = self.data["%K"].rolling(window=d_window).mean()
        self.data["%K"].bfill(inplace=True)
        self.data["%D"].bfill(inplace=True)

    def calculate_slow_stochastic_oscillator(self, d_window=3, slow_k_window=3):
        """
        Calculates the slow stochastic oscillator for trend confirmation.
        """
        self.data["Slow %K"] = self.data["%K"].rolling(window=slow_k_window).mean()
        self.data["Slow %D"] = self.data["Slow %K"].rolling(window=d_window).mean()
        self.data["Slow %K"].bfill(inplace=True)
        self.data["Slow %D"].bfill(inplace=True)

    def detect_rolling_peaks_and_troughs(self, window_size=5):
        """
        Identifies local peaks and troughs in the closing price using a rolling window.
        Labels each row as Buy/Sell/Hold accordingly.
        """
        self.data["isLocalPeak"] = False
        self.data["isLocalTrough"] = False
        for end_idx in range(window_size, len(self.data)):
            start_idx = max(0, end_idx - window_size)
            window_data = self.data["Close"][start_idx:end_idx]
            peaks, _ = find_peaks(window_data)
            peaks_global_indices = [start_idx + p for p in peaks]
            self.data.loc[peaks_global_indices, "isLocalPeak"] = True
            troughs, _ = find_peaks(-window_data)
            troughs_global_indices = [start_idx + t for t in troughs]
            self.data.loc[troughs_global_indices, "isLocalTrough"] = True
        self.data["Label"] = "Hold"
        self.data.loc[self.data["isLocalPeak"], "Label"] = "Sell"
        self.data.loc[self.data["isLocalTrough"], "Label"] = "Buy"

    def calculate_rsi(self, window=14):
        """Manually calculates Relative Strength Index (RSI) indicator."""
        delta = self.data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

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
        For each row, computes time since last detected peak/trough and price changes from those points.
        """
        self.data["MinutesSincePeak"] = 0
        self.data["MinutesSinceTrough"] = 0
        self.data["PriceChangeSincePeak"] = 0
        self.data["PriceChangeSinceTrough"] = 0

        checkpoint_date_bottom = None
        checkpoint_date_top = None
        checkpoint_price_bottom = None
        checkpoint_price_top = None
        price_change_since_bottom = 0
        price_change_since_peak = 0

        for index, row in self.data.iterrows():
            current_price = row["Open"]
            today_date = pd.to_datetime(row["Date"])
            if row["Label"] == "Buy":
                checkpoint_date_bottom = today_date
                checkpoint_price_bottom = current_price
            if row["Label"] == "Sell":
                checkpoint_date_top = today_date
                checkpoint_price_top = current_price
            days_since_bottom = (today_date - checkpoint_date_bottom).seconds if checkpoint_date_bottom else 0
            days_since_peak = (today_date - checkpoint_date_top).seconds if checkpoint_date_top else 0
            if checkpoint_price_bottom is not None:
                price_change_since_bottom = current_price - checkpoint_price_bottom
            if checkpoint_price_top is not None:
                price_change_since_peak = current_price - checkpoint_price_top
            self.data.at[index, "MinutesSincePeak"] = days_since_peak
            self.data.at[index, "MinutesSinceTrough"] = days_since_bottom
            self.data["PriceChangeSincePeak"] = self.data["PriceChangeSincePeak"].astype(float)
            self.data["PriceChangeSinceTrough"] = self.data["PriceChangeSinceTrough"].astype(float)
            self.data.at[index, "PriceChangeSincePeak"] = float(price_change_since_peak)
            self.data.at[index, "PriceChangeSinceTrough"] = float(price_change_since_bottom)

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
        Applies the loaded model to the most recent feature vector to generate a trading action.
        Returns: (label, price, timestamp)
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
        test_data = self.data[feature_set]
        print("check last data: ", self.data[-5:])
        test_data.to_csv("test_data.csv")

        # Only keep last row for prediction
        last_row = test_data.tail(1)
        # Check for NaN in the last row
        if last_row.isnull().any(axis=1).iloc[0]:
            print("NaN detected in last row of features, return Hold")
            price = self.data["Open"][-1:].iloc[-1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            return ("Hold", price, formatted_datetime)

        try:
            output = self.model.predict(last_row)
            price = self.data["Open"][-1:].iloc[-1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            action = (output[0], price, formatted_datetime)
            print("action: ", action)
            return action
        except Exception as e:
            print("model err, just ignore and hold!", e)
            price = self.data["Open"][-1:].iloc[-1]
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            return ("Hold", price, formatted_datetime)

    # --- Additional methods omitted for brevity, but should follow the same style ---

