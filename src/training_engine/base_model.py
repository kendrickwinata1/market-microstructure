import numpy as np
import pandas as pd
# import talib
from hurst import compute_Hc
from pykalman import KalmanFilter
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

class BaseModel:
    """
    BaseModel for feature extraction, preprocessing, and as a template for trading model training/prediction.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.split_idx = None
        self.model = None
        self.scaler = StandardScaler()
        self.criterion = None
        self.optimizer = None
        self.fft_features = None
        self.predicted_categories = None

    def load_preprocess_data(self):
        """
        Loads and preprocesses raw CSV data, extracting all engineered features.
        """
        self.data = pd.read_csv(self.file_path)
        self.data["Date"] = pd.to_datetime(self.data["Timestamp"])
        self.perform_fourier_transform_analysis()
        self.calculate_stochastic_oscillator()
        self.calculate_slow_stochastic_oscillator()
        self.construct_kalman_filter()
        self.detect_rolling_peaks_and_troughs()
        self.calculate_moving_averages_and_rsi()
        self.calculate_days_since_peaks_and_troughs()
        self.calculate_first_second_order_derivatives()
        self.preprocess_data()

    def calculate_daily_percentage_change(self):
        """
        Adds daily percentage change columns.
        """
        self.data["Daily_Change"] = self.data["Close"].pct_change() * 100
        self.data["Daily_Change_Open_to_Close"] = (
            (self.data["Open"] - self.data["Close"].shift(1)) / self.data["Close"].shift(1)
        ) * 100

    def perform_fourier_transform_analysis(self):
        """
        Performs FFT on close prices and extracts significant frequencies.
        """
        close_prices = self.data["Close"].to_numpy()
        mean_value = np.nanmean(close_prices)
        close_prices[np.isnan(close_prices)] = mean_value

        N = len(close_prices)
        T = 1.0  # Sampling period
        close_fft = fft(close_prices)
        fft_freq = np.fft.fftfreq(N, T)
        positive_frequencies = fft_freq[: N // 2]
        positive_fft_values = 2.0 / N * np.abs(close_fft[0 : N // 2])

        amplitude_threshold = 1.0  # configurable
        significant_peaks, _ = find_peaks(positive_fft_values, height=amplitude_threshold)
        significant_frequencies = positive_frequencies[significant_peaks]
        significant_amplitudes = positive_fft_values[significant_peaks]
        days_per_cycle = 1 / significant_frequencies

        self.fft_features = pd.DataFrame({
            "Frequency": significant_frequencies,
            "Amplitude": significant_amplitudes,
            "SecondsPerCycle": days_per_cycle,
        })

    def calculate_stochastic_oscillator(self, k_window=14, d_window=3, slow_k_window=3):
        """
        Calculates Stochastic Oscillator %K and %D.
        """
        low_min = self.data["Low"].rolling(window=k_window).min()
        high_max = self.data["High"].rolling(window=k_window).max()
        self.data["%K"] = 100 * (self.data["Close"] - low_min) / (high_max - low_min)
        self.data["%D"] = self.data["%K"].rolling(window=d_window).mean()
        self.data["%K"].bfill(inplace=True)
        self.data["%D"].bfill(inplace=True)

    def calculate_slow_stochastic_oscillator(self, d_window=3, slow_k_window=3):
        """
        Calculates Slow Stochastic Oscillator (smoothed %K and %D).
        """
        self.data["Slow %K"] = self.data["%K"].rolling(window=slow_k_window).mean()
        self.data["Slow %D"] = self.data["Slow %K"].rolling(window=d_window).mean()
        self.data["Slow %K"].bfill(inplace=True)
        self.data["Slow %D"].bfill(inplace=True)

    def detect_fourier_signals(self):
        """
        Example: Tag periods that match dominant Fourier cycles (for Buy/Sell signals).
        """
        print("check fft_features: ", self.fft_features[:30])
        dominant_period_lengths = sorted(
            set((self.fft_features["SecondsPerCycle"].values / 2).astype(int)), reverse=True
        )
        dominant_period_lengths = [i for i in dominant_period_lengths if i < 15]
        dominant_period_lengths = [15, 7, 5]  # hardcoded override
        print("check dominant_period_lengths: ", dominant_period_lengths)
        self.data["FourierSignalSell"] = self.data["MinutesSinceTrough"].isin(dominant_period_lengths)
        self.data["FourierSignalBuy"] = self.data["MinutesSincePeak"].isin(dominant_period_lengths)
        print("FourierSignalSell: ", self.data["FourierSignalSell"])
        print("FourierSignalBuy: ", self.data["FourierSignalBuy"])

    def detect_rolling_peaks_and_troughs(self, window_size=5):
        """
        Detects local peaks/troughs in Close price over a rolling window.
        """
        self.data["isLocalPeak"] = False
        self.data["isLocalTrough"] = False
        for end_idx in range(window_size, len(self.data)):
            start_idx = max(0, end_idx - window_size)
            window_data = self.data["Close"][start_idx:end_idx]
            peaks, _ = find_peaks(window_data)
            troughs, _ = find_peaks(-window_data)
            peaks_global_indices = [start_idx + p for p in peaks]
            troughs_global_indices = [start_idx + t for t in troughs]
            self.data.loc[peaks_global_indices, "isLocalPeak"] = True
            self.data.loc[troughs_global_indices, "isLocalTrough"] = True
        # Label: Buy/Sell/Hold
        self.data["Label"] = "Hold"
        self.data.loc[self.data["isLocalPeak"], "Label"] = "Sell"
        self.data.loc[self.data["isLocalTrough"], "Label"] = "Buy"

    def calculate_rsi(self, window=14):
        """
        Calculate Relative Strength Index (RSI) for given window.
        """
        delta = self.data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_moving_averages_and_rsi(self):
        """
        Add moving averages and RSI to the data.
        """
        short_window = 5
        long_window = 20
        rsi_period = 14
        self.data["Short_Moving_Avg"] = self.data["Close"].rolling(window=short_window).mean()
        self.data["Long_Moving_Avg"] = self.data["Close"].rolling(window=long_window).mean()
        self.data["RSI"] = self.calculate_rsi(window=rsi_period)

    def construct_kalman_filter(self):
        """
        Adds a Kalman filter estimate column to data.
        """
        close_prices = self.data["Close"]
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        state_means, _ = kf.filter(close_prices.values)
        kalman_estimates = pd.Series(state_means.flatten(), index=self.data.index)
        kalman_estimates = pd.DataFrame({"KalmanFilterEst": kalman_estimates})
        self.data = self.data.join(kalman_estimates)

    def estimate_hurst_exponent(self, window_size=100, step_size=1):
        """
        Calculates the Hurst exponent over sliding windows and appends to DataFrame.
        """
        hurst_exponents = []
        for i in range(0, len(self.data) - window_size + 1, step_size):
            window = self.data["Close"].iloc[i : i + window_size]
            H, _, _ = compute_Hc(window, kind="price", simplified=True)
            hurst_exponents.append({"Date": self.data["Date"].iloc[i], "HurstExponent": H})
        hurst_exponents = pd.DataFrame(hurst_exponents)
        self.data = self.data.merge(hurst_exponents, on="Date", how="left")

    def calculate_days_since_peaks_and_troughs(self):
        """
        Calculates time and price change since last peak/trough for each row.
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

            days_since_bottom = (
                (today_date - checkpoint_date_bottom).seconds if checkpoint_date_bottom else 0
            )
            days_since_peak = (
                (today_date - checkpoint_date_top).seconds if checkpoint_date_top else 0
            )
            if checkpoint_price_bottom is not None:
                price_change_since_bottom = current_price - checkpoint_price_bottom
            if checkpoint_price_top is not None:
                price_change_since_peak = current_price - checkpoint_price_top

            self.data.at[index, "MinutesSincePeak"] = days_since_peak
            self.data.at[index, "MinutesSinceTrough"] = days_since_bottom
            self.data.at[index, "PriceChangeSincePeak"] = price_change_since_peak
            self.data.at[index, "PriceChangeSinceTrough"] = price_change_since_bottom

    def calculate_first_second_order_derivatives(self):
        """
        Calculates first and second derivatives for key features.
        """
        for feature in ["KalmanFilterEst", "Short_Moving_Avg", "Long_Moving_Avg"]:
            self.data[f"{feature}_1st_Deriv"] = self.data[feature].diff() * 100
            self.data[f"{feature}_2nd_Deriv"] = self.data[f"{feature}_1st_Deriv"].diff() * 100
        self.data.bfill(inplace=True)

    def preprocess_data(self):
        """
        Saves and prunes missing values in feature data.
        """
        print("12345: ", self.data)
        self.data.to_csv("processed.csv")
        self.data.dropna(inplace=True)

    def train_test_split_time_series(self):
        """
        Splits data into training and test sets in chronological order (no shuffling).
        """
        print("SPLITTTTTTT")
        print(self.data)
        self.data.sort_values("Date", inplace=True)
        split_ratio = 2 / 3
        split_index = int(len(self.data) * split_ratio)
        print("SPLIT INDEX: ", split_index)
        self.train_data = self.data.iloc[:split_index].copy()
        self.test_data = self.data.iloc[split_index:].copy()
        self.train_data.sort_values("Date", inplace=True)
        self.train_data.to_csv("inspect_training_set.csv")
        self.test_data.to_csv("inspect_testing_set.csv")

        feature_set = [
            "Short_Moving_Avg_1st_Deriv", "Short_Moving_Avg_2nd_Deriv",
            "Long_Moving_Avg_1st_Deriv", "Long_Moving_Avg_2nd_Deriv",
            "MinutesSincePeak", "MinutesSinceTrough",
            "%K", "%D",
            "KalmanFilterEst_1st_Deriv", "KalmanFilterEst_2nd_Deriv",
        ]
        self.X_train = self.train_data[feature_set]
        self.X_test = self.test_data[feature_set]
        self.y_train = self.train_data["Label"]
        self.y_test = self.test_data["Label"]

        print("len X train: ", len(self.X_train))
        print("len X test: ", len(self.X_test))
        print("len y train: ", len(self.y_train))
        print("len y test: ", len(self.y_test))

    def retrieve_test_set(self):
        """
        Returns test data set (for analysis or evaluation).
        """
        return self.test_data

    def train(self):
        """
        Placeholder for training logic, to be implemented in child classes.
        """
        pass

    def predict(self):
        """
        Placeholder for prediction logic, to be implemented in child classes.
        """
        pass

    def evaluate(self, X, y):
        """
        Placeholder for evaluation logic, to be implemented in child classes.
        """
        pass
