import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import warnings
import json


warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


try:
    if tf.config.list_physical_devices("GPU"):
        print("GPU detected! Setting up for GPU acceleration...")

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected, using CPU...")
except Exception as e:
    print(f"GPU setup failed, falling back to CPU: {e}")

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (14, 8)
plt.ioff()

os.makedirs("results/lstm", exist_ok=True)


class BRLUSDLSTMModel:
    def __init__(self, sequence_length=60, forecast_horizon=1, random_state=42):
        """
        LSTM Model for BRL/USD Futures Price Forecasting

        Args:
            sequence_length (int): Number of time steps to look back
            forecast_horizon (int): Number of steps to forecast ahead
            random_state (int): Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.history = None

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def load_and_preprocess_data(self, file_path="./data/Dolfut.csv"):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")

        df = pd.read_csv(file_path, index_col=0)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        df = df.fillna(method="ffill").fillna(method="bfill")

        df["Price_MA_5"] = df["Close"].rolling(window=5).mean()
        df["Price_MA_20"] = df["Close"].rolling(window=20).mean()
        df["Price_Std_20"] = df["Close"].rolling(window=20).std()
        df["RSI"] = self._calculate_rsi(df["Close"])
        df["Volume_MA_10"] = df["Volume"].rolling(window=10).mean()
        df["High_Low_Ratio"] = df["High"] / df["Low"]
        df["Close_Open_Ratio"] = df["Close"] / df["Open"]

        df = df.dropna()

        print(f"Dataset shape after preprocessing: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        print("Creating sequences for LSTM...")

        feature_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Returns",
            "Price_MA_5",
            "Price_MA_20",
            "Price_Std_20",
            "RSI",
            "Volume_MA_10",
            "High_Low_Ratio",
            "Close_Open_Ratio",
        ]

        target_column = "Close"

        features = data[feature_columns].values
        target = data[target_column].values

        self.feature_scaler = StandardScaler()
        self.scaler = MinMaxScaler()

        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(
            self.sequence_length, len(features_scaled) - self.forecast_horizon + 1
        ):
            X.append(features_scaled[i - self.sequence_length : i])
            y.append(target_scaled[i + self.forecast_horizon - 1])

        X, y = np.array(X), np.array(y)

        print(f"Sequences created - X shape: {X.shape}, y shape: {y.shape}")

        return X, y, data.index[self.sequence_length :]

    def train_test_split(self, X, y, dates, test_years=1):
        """Split data into train and test sets using last N years as test"""

        test_start_date = dates.max() - pd.DateOffset(years=test_years)
        split_idx = None

        for i, date in enumerate(dates):
            if date >= test_start_date:
                split_idx = i
                break

        if split_idx is None:
            split_idx = int(len(X) * 0.8)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train = dates[:split_idx]
        dates_test = dates[split_idx:]

        print(
            f"Train set: {X_train.shape[0]} samples (up to {dates_train.max().strftime('%Y-%m-%d')})"
        )
        print(
            f"Test set: {X_test.shape[0]} samples (from {dates_test.min().strftime('%Y-%m-%d')} to {dates_test.max().strftime('%Y-%m-%d')})"
        )

        return X_train, X_test, y_train, y_test, dates_train, dates_test

    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        print("Building LSTM model...")

        model = Sequential(
            [
                LSTM(100, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        print("Model architecture:")
        model.summary()

        return model

    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the LSTM model"""
        print("Training LSTM model...")

        device_name = tf.config.list_physical_devices("GPU")
        if device_name:
            print(f"Training on GPU: {device_name}")
        else:
            print("Training on CPU")

        with tf.device(
            "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        ):
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6, verbose=1
            ),
        ]

        with tf.device(
            "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        ):
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )

        print("Training completed!")

        return self.history

    def evaluate_model(self, X_test, y_test, dates_test):
        """Evaluate model performance"""
        print("\nEvaluating model performance...")

        y_pred_scaled = self.model.predict(X_test)

        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(f"Model Performance Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

        with open("results/lstm/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("Metrics saved to results/lstm/metrics.json")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "predictions": y_pred.flatten(),
            "actual": y_true.flatten(),
            "dates": dates_test,
        }

    def forecast_future(self, data, n_steps=30):
        """Forecast future prices"""
        print(f"\nForecasting next {n_steps} days...")

        feature_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Returns",
            "Price_MA_5",
            "Price_MA_20",
            "Price_Std_20",
            "RSI",
            "Volume_MA_10",
            "High_Low_Ratio",
            "Close_Open_Ratio",
        ]

        last_sequence = data[feature_columns].iloc[-self.sequence_length :].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)

        forecasts = []
        current_sequence = last_sequence_scaled.copy()

        for _ in range(n_steps):
            pred_scaled = self.model.predict(
                current_sequence.reshape(1, self.sequence_length, -1)
            )
            pred_price = self.scaler.inverse_transform(pred_scaled)[0, 0]
            forecasts.append(pred_price)

            new_row = current_sequence[-1].copy()
            new_row[3] = pred_scaled[0, 0]

            current_sequence = np.vstack([current_sequence[1:], new_row])

        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=n_steps, freq="D"
        )

        return forecasts, forecast_dates

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history["loss"], label="Training Loss")
        ax1.plot(self.history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss During Training")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history["mae"], label="Training MAE")
        ax2.plot(self.history.history["val_mae"], label="Validation MAE")
        ax2.set_title("Model MAE During Training")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/lstm/training_history.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_predictions(self, data, results, dates_train, dates_test, n_show=500):
        """Plot training data, test true values, predicted curves, and residuals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        train_end_idx = min(len(dates_train), n_show)
        train_dates = dates_train[-train_end_idx:]
        train_prices = data.loc[train_dates, "Close"]

        test_dates = results["dates"]
        test_actual = results["actual"]
        test_pred = results["predictions"]

        ax1.plot(
            train_dates,
            train_prices,
            label="Training Data",
            linewidth=1.5,
            alpha=0.7,
            color="blue",
        )

        ax1.plot(
            test_dates,
            test_actual,
            label="Test True Values",
            linewidth=1.5,
            alpha=0.8,
            color="green",
        )

        ax1.plot(
            test_dates,
            test_pred,
            label="Predicted Values",
            linewidth=1.5,
            alpha=0.8,
            color="red",
            linestyle="--",
        )

        ax1.set_title("LSTM Model: Training Data, Test True Values, and Predictions")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (BRL)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        residuals = test_actual - test_pred
        ax2.plot(test_dates, residuals, color="red", alpha=0.7, linewidth=1)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_title("Prediction Residuals")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("results/lstm/predictions.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_forecast(self, data, forecasts, forecast_dates, n_history=100):
        """Plot future forecasts"""
        fig, ax = plt.subplots(figsize=(15, 8))

        historical_dates = data.index[-n_history:]
        historical_prices = data["Close"].iloc[-n_history:]

        ax.plot(
            historical_dates,
            historical_prices,
            label="Historical Prices",
            linewidth=2,
            color="blue",
        )

        ax.plot(
            forecast_dates,
            forecasts,
            label="Forecasted Prices",
            linewidth=2,
            color="red",
            linestyle="--",
        )

        ax.plot(
            [historical_dates[-1], forecast_dates[0]],
            [historical_prices.iloc[-1], forecasts[0]],
            color="orange",
            linewidth=2,
            alpha=0.7,
        )

        ax.set_title("BRL/USD Futures: Historical Data and LSTM Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (BRL)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("results/lstm/future_forecast.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_lstm_baseline():
    """Run the complete LSTM baseline model pipeline"""
    print("=" * 60)
    print("BRL/USD FUTURES - LSTM BASELINE MODEL")
    print("=" * 60)

    print("Device Configuration:")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Available GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print("-" * 40)

    model = BRLUSDLSTMModel(sequence_length=60, forecast_horizon=1)

    data = model.load_and_preprocess_data()

    X, y, dates = model.create_sequences(data)

    X_train, X_test, y_train, y_test, dates_train, dates_test = model.train_test_split(
        X, y, dates, test_years=1
    )

    history = model.train_model(
        X_train, y_train, X_test, y_test, epochs=100, batch_size=32
    )

    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_mae = float(history.history["mae"][-1])
    final_val_mae = float(history.history["val_mae"][-1])

    training_metrics = {
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        "final_mae": final_mae,
        "final_val_mae": final_val_mae,
    }

    with open("results/lstm/training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    print(f"Training metrics saved to results/lstm/training_metrics.json")

    results = model.evaluate_model(X_test, y_test, dates_test)

    forecasts, forecast_dates = model.forecast_future(data, n_steps=30)

    model.plot_training_history()
    model.plot_predictions(data, results, dates_train, dates_test)
    model.plot_forecast(data, forecasts, forecast_dates)

    print("\n" + "=" * 60)
    print("LSTM BASELINE MODEL - SUMMARY")
    print("=" * 60)
    print(f"Model RMSE: {results['rmse']:.2f} BRL")
    print(f"Model MAE: {results['mae']:.2f} BRL")
    print(f"Model R²: {results['r2']:.4f}")
    print(f"Model MAPE: {results['mape']:.2f}%")
    print(
        f"\nNext 30-day forecast range: {min(forecasts):.2f} - {max(forecasts):.2f} BRL"
    )
    print(f"Average forecasted price: {np.mean(forecasts):.2f} BRL")
    print(f"Current price: {data['Close'].iloc[-1]:.2f} BRL")
    print(
        f"Forecasted change: {((np.mean(forecasts) - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%"
    )
    print("\nAll plots saved in 'results/lstm/' directory")

    return model, results, forecasts, forecast_dates


if __name__ == "__main__":
    model, results, forecasts, forecast_dates = run_lstm_baseline()
