import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import warnings
import os
import json

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (14, 8)
plt.ioff()

os.makedirs("results/arima", exist_ok=True)


class BRLUSDArimaModel:
    def __init__(self, random_state=42):
        """
        ARIMA Model for BRL/USD Futures Price Forecasting

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.original_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None

        np.random.seed(random_state)

    def load_and_preprocess_data(
        self, file_path="./data/Dolfut.csv", target_column="Close"
    ):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data for ARIMA...")

        df = pd.read_csv(file_path, index_col=0)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        df = df.fillna(method="ffill").fillna(method="bfill")

        self.original_data = df[target_column].copy()

        print(f"Dataset shape: {df.shape}")
        print(f"Target series length: {len(self.original_data)}")
        print(
            f"Date range: {self.original_data.index.min()} to {self.original_data.index.max()}"
        )

        return df

    def test_stationarity(self, series, title="Time Series"):
        """Test stationarity using ADF and KPSS tests"""
        print(f"\n{'=' * 50}")
        print(f"STATIONARITY TESTS FOR {title.upper()}")
        print(f"{'=' * 50}")

        print("Augmented Dickey-Fuller Test:")
        adf_result = adfuller(series.dropna())
        print(f"ADF Statistic: {adf_result[0]:.6f}")
        print(f"p-value: {adf_result[1]:.6f}")
        print("Critical Values:")
        for key, value in adf_result[4].items():
            print(f"\t{key}: {value:.3f}")

        if adf_result[1] <= 0.05:
            print("ADF Test: Series is stationary (reject null hypothesis)")
            adf_stationary = True
        else:
            print("ADF Test: Series is non-stationary (fail to reject null hypothesis)")
            adf_stationary = False

        print("\nKwiatkowski-Phillips-Schmidt-Shin Test:")
        kpss_result = kpss(series.dropna(), regression="c")
        print(f"KPSS Statistic: {kpss_result[0]:.6f}")
        print(f"p-value: {kpss_result[1]:.6f}")
        print("Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"\t{key}: {value:.3f}")

        if kpss_result[1] >= 0.05:
            print("KPSS Test: Series is stationary (fail to reject null hypothesis)")
            kpss_stationary = True
        else:
            print("KPSS Test: Series is non-stationary (reject null hypothesis)")
            kpss_stationary = False

        print(f"\nCOMBINED RESULT:")
        if adf_stationary and kpss_stationary:
            print("Both tests indicate the series is STATIONARY")
            is_stationary = True
        elif not adf_stationary and not kpss_stationary:
            print("Both tests indicate the series is NON-STATIONARY")
            is_stationary = False
        else:
            print("⚠ Tests give conflicting results - further investigation needed")
            is_stationary = False

        return {
            "adf_statistic": adf_result[0],
            "adf_pvalue": adf_result[1],
            "kpss_statistic": kpss_result[0],
            "kpss_pvalue": kpss_result[1],
            "is_stationary": is_stationary,
        }

    def make_stationary(self, series, max_diff=2):
        """Make series stationary through differencing"""
        print(f"\n{'=' * 40}")
        print("MAKING SERIES STATIONARY")
        print(f"{'=' * 40}")

        differenced_series = series.copy()
        d = 0

        for i in range(max_diff + 1):
            stationarity_result = self.test_stationarity(
                differenced_series, f"Differenced Series (d={i})"
            )

            if stationarity_result["is_stationary"]:
                print(f"\nSeries became stationary after {i} differencing steps")
                d = i
                break

            if i < max_diff:
                print(f"\nApplying differencing (step {i + 1})...")
                differenced_series = differenced_series.diff().dropna()

        if not stationarity_result["is_stationary"]:
            print(
                f"\n⚠ Warning: Series may still not be stationary after {max_diff} differencing steps"
            )
            d = max_diff

        self.processed_data = differenced_series
        return differenced_series, d

    def plot_series_analysis(self, original_series, differenced_series=None, d=0):
        """Plot time series analysis including decomposition and ACF/PACF"""
        fig = plt.figure(figsize=(16, 12))

        if differenced_series is not None:
            ax1 = plt.subplot(3, 2, 1)
            original_series.plot(title=f"Original Series - BRL/USD Close Price", ax=ax1)
            ax1.grid(True, alpha=0.3)

            ax2 = plt.subplot(3, 2, 2)
            differenced_series.plot(
                title=f"Differenced Series (d={d})", ax=ax2, color="red"
            )
            ax2.grid(True, alpha=0.3)

            ax3 = plt.subplot(3, 2, 3)
            plot_acf(
                differenced_series.dropna(),
                lags=40,
                ax=ax3,
                title="ACF - Differenced Series",
            )

            ax4 = plt.subplot(3, 2, 4)
            plot_pacf(
                differenced_series.dropna(),
                lags=40,
                ax=ax4,
                title="PACF - Differenced Series",
            )

            ax5 = plt.subplot(3, 2, 5)
            differenced_series.hist(bins=50, ax=ax5, alpha=0.7)
            ax5.set_title("Distribution of Differenced Series")
            ax5.grid(True, alpha=0.3)

            ax6 = plt.subplot(3, 2, 6)
            from scipy import stats

            stats.probplot(differenced_series.dropna(), dist="norm", plot=ax6)
            ax6.set_title("Q-Q Plot - Differenced Series")
            ax6.grid(True, alpha=0.3)
        else:
            ax1 = plt.subplot(2, 2, 1)
            original_series.plot(title="Original Series - BRL/USD Close Price", ax=ax1)
            ax1.grid(True, alpha=0.3)

            ax2 = plt.subplot(2, 2, 2)
            plot_acf(
                original_series.dropna(), lags=40, ax=ax2, title="ACF - Original Series"
            )

            ax3 = plt.subplot(2, 2, 3)
            plot_pacf(
                original_series.dropna(),
                lags=40,
                ax=ax3,
                title="PACF - Original Series",
            )

            ax4 = plt.subplot(2, 2, 4)
            original_series.hist(bins=50, ax=ax4, alpha=0.7)
            ax4.set_title("Distribution of Original Series")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/arima/series_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def seasonal_decomposition(self, series, period=252):
        """Perform seasonal decomposition"""
        print(f"\n{'=' * 40}")
        print("SEASONAL DECOMPOSITION")
        print(f"{'=' * 40}")

        try:
            decomposition = seasonal_decompose(
                series.dropna(), model="multiplicative", period=period
            )

            fig, axes = plt.subplots(4, 1, figsize=(15, 12))

            decomposition.observed.plot(ax=axes[0], title="Original Series")
            axes[0].grid(True, alpha=0.3)

            decomposition.trend.plot(ax=axes[1], title="Trend Component")
            axes[1].grid(True, alpha=0.3)

            decomposition.seasonal.plot(ax=axes[2], title="Seasonal Component")
            axes[2].grid(True, alpha=0.3)

            decomposition.resid.plot(ax=axes[3], title="Residual Component")
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "results/arima/seasonal_decomposition.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print("Seasonal decomposition completed and saved")
            return decomposition

        except Exception as e:
            print(f"Seasonal decomposition failed: {e}")
            return None

    def find_optimal_parameters(self, series, max_p=5, max_d=2, max_q=5):
        """Find optimal ARIMA parameters using grid search"""
        print(f"\n{'=' * 40}")
        print("ARIMA PARAMETER OPTIMIZATION")
        print(f"{'=' * 40}")

        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)

        param_combinations = list(itertools.product(p_values, d_values, q_values))

        print(f"Testing {len(param_combinations)} parameter combinations...")

        best_aic = float("inf")
        best_bic = float("inf")
        best_params = None
        results = []

        for params in param_combinations:
            try:
                model = ARIMA(series, order=params)
                fitted_model = model.fit()

                aic = fitted_model.aic
                bic = fitted_model.bic

                results.append(
                    {
                        "params": params,
                        "aic": aic,
                        "bic": bic,
                        "converged": fitted_model.mle_retvals["converged"],
                    }
                )

                if aic < best_aic and fitted_model.mle_retvals["converged"]:
                    best_aic = aic
                    best_params = params

                print(
                    f"ARIMA{params}: AIC={aic:.2f}, BIC={bic:.2f}, Converged={fitted_model.mle_retvals['converged']}"
                )

            except Exception as e:
                print(f"ARIMA{params}: Failed - {str(e)[:50]}")
                continue

        if best_params is None:
            print("No valid ARIMA model found!")
            return None

        print(f"\nBest ARIMA model: {best_params}")
        print(f"Best AIC: {best_aic:.2f}")

        self.best_params = best_params

        results_df = pd.DataFrame(results)
        results_df = (
            results_df[results_df["converged"] == True].sort_values("aic").head(10)
        )

        print("\nTop 10 Models by AIC:")
        print(results_df.to_string(index=False))

        self.parameter_results = results_df

        return best_params, results_df

    def train_test_split(self, series, test_years=1):
        """Split data into train and test sets using last N years as test"""

        test_start_date = series.index.max() - pd.DateOffset(years=test_years)
        split_idx = None

        for i, date in enumerate(series.index):
            if date >= test_start_date:
                split_idx = i
                break

        if split_idx is None:
            split_idx = int(len(series) * 0.8)

        train_data = series[:split_idx]
        test_data = series[split_idx:]

        self.train_data = train_data
        self.test_data = test_data

        print(
            f"\nTrain set: {len(train_data)} observations (up to {train_data.index[-1].strftime('%Y-%m-%d')})"
        )
        print(
            f"Test set: {len(test_data)} observations (from {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})"
        )

        return train_data, test_data

    def fit_arima_model(self, train_data, order=None):
        """Fit ARIMA model"""
        if order is None:
            order = self.best_params if self.best_params else (1, 1, 1)

        print(f"\n{'=' * 40}")
        print(f"FITTING ARIMA{order} MODEL")
        print(f"{'=' * 40}")

        try:
            self.model = ARIMA(train_data, order=order)
            self.fitted_model = self.model.fit()

            print("Model fitted successfully!")
            print(f"AIC: {self.fitted_model.aic:.2f}")
            print(f"BIC: {self.fitted_model.bic:.2f}")
            print(f"Log-likelihood: {self.fitted_model.llf:.2f}")

            print("\nModel Summary:")
            print(self.fitted_model.summary())

            return self.fitted_model

        except Exception as e:
            print(f"Model fitting failed: {e}")
            return None

    def evaluate_model(self, test_data):
        """Evaluate ARIMA model performance using rolling forecasts"""
        if self.fitted_model is None:
            print("No fitted model available!")
            return None

        print(f"\n{'=' * 40}")
        print("MODEL EVALUATION - ROLLING FORECASTS")
        print(f"{'=' * 40}")

        try:
            print("Performing one-step-ahead rolling forecasts...")

            full_series = pd.concat([self.train_data, test_data])

            extended_model = ARIMA(
                full_series, order=self.fitted_model.specification["order"]
            )
            extended_fitted = extended_model.fit()

            train_size = len(self.train_data)
            pred_values = []

            print(f"Generating {len(test_data)} one-step-ahead forecasts...")

            for i in range(len(test_data)):
                end_idx = train_size + i
                if end_idx < len(full_series):
                    forecast = extended_fitted.get_prediction(
                        start=end_idx, end=end_idx
                    )
                    pred_value = forecast.predicted_mean.iloc[0]
                    pred_values.append(pred_value)

                    if (i + 1) % 50 == 0:
                        print(f"Completed {i + 1}/{len(test_data)} forecasts...")

            test_values = test_data.values
            pred_values = np.array(pred_values)

            min_len = min(len(test_values), len(pred_values))
            test_values = test_values[:min_len]
            pred_values = pred_values[:min_len]

            mse = mean_squared_error(test_values, pred_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_values, pred_values)

            non_zero_mask = test_values != 0
            if np.sum(non_zero_mask) > 0:
                mape = (
                    np.mean(
                        np.abs(
                            (test_values[non_zero_mask] - pred_values[non_zero_mask])
                            / test_values[non_zero_mask]
                        )
                    )
                    * 100
                )
            else:
                mape = float("inf")

            try:
                r2 = r2_score(test_values, pred_values)
            except:
                r2 = float("-inf")

            print(f"Model Performance Metrics:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"R²: {r2:.4f}")

            metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mape": float(mape),
            }

            with open("results/arima/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print("Metrics saved to results/arima/metrics.json")

            predictions_series = pd.Series(pred_values, index=test_data.index[:min_len])
            residuals = test_data.iloc[:min_len] - predictions_series

            try:
                lb_test = acorr_ljungbox(
                    residuals.dropna(),
                    lags=min(10, len(residuals) // 4),
                    return_df=True,
                )
                print(
                    f"\nLjung-Box Test (p-value): {lb_test['lb_pvalue'].iloc[-1]:.4f}"
                )
                if lb_test["lb_pvalue"].iloc[-1] > 0.05:
                    print("Residuals appear to be white noise")
                else:
                    print("Residuals may have autocorrelation")
            except Exception as lb_e:
                print(f"⚠ Could not perform Ljung-Box test: {lb_e}")

            return {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "predictions": predictions_series,
                "residuals": residuals,
                "test_data": test_data.iloc[:min_len],
            }

        except Exception as e:
            print(f"Model evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def forecast_future(self, n_steps=30):
        """Forecast future values"""
        if self.fitted_model is None:
            print("No fitted model available!")
            return None, None

        print(f"\n{'=' * 40}")
        print(f"FORECASTING NEXT {n_steps} PERIODS")
        print(f"{'=' * 40}")

        try:
            forecast_result = self.fitted_model.forecast(steps=n_steps)

            forecast_values = np.array(forecast_result)

            forecast_ci = self.fitted_model.get_forecast(steps=n_steps).conf_int()

            last_date = self.original_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=n_steps, freq="D"
            )

            print(f"Forecast generated successfully")
            print(
                f"Forecast range: {forecast_values.min():.2f} - {forecast_values.max():.2f}"
            )
            print(f"Average forecast: {forecast_values.mean():.2f}")

            return forecast_values, forecast_dates, forecast_ci

        except Exception as e:
            print(f"Forecasting failed: {e}")
            return None, None, None

    def plot_training_history(self):
        """Plot training history (parameter optimization results)"""
        if not hasattr(self, "parameter_results") or self.parameter_results is None:
            print("No parameter optimization results available!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        results_df = self.parameter_results.head(20)
        param_labels = [str(params) for params in results_df["params"]]

        ax1.bar(range(len(results_df)), results_df["aic"], alpha=0.7)
        ax1.set_title("AIC Values for Parameter Combinations")
        ax1.set_xlabel("Parameter Combination")
        ax1.set_ylabel("AIC")
        ax1.tick_params(axis="x", rotation=45, labelsize=8)
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(param_labels)
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(len(results_df)), results_df["bic"], alpha=0.7, color="orange")
        ax2.set_title("BIC Values for Parameter Combinations")
        ax2.set_xlabel("Parameter Combination")
        ax2.set_ylabel("BIC")
        ax2.tick_params(axis="x", rotation=45, labelsize=8)
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels(param_labels)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/arima/training_history.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_predictions(self, data, evaluation_results, n_show=500):
        """Plot training data, test true values, predicted curves, and residuals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        train_data = self.train_data
        test_data = evaluation_results["test_data"]
        predictions = evaluation_results["predictions"]

        train_end_idx = min(len(train_data), n_show)
        train_dates = train_data.index[-train_end_idx:]
        train_prices = train_data.iloc[-train_end_idx:]

        ax1.plot(
            train_dates,
            train_prices.values,
            label="Training Data",
            linewidth=1.5,
            alpha=0.7,
            color="blue",
        )
        ax1.plot(
            test_data.index,
            test_data.values,
            label="Test True Values",
            linewidth=1.5,
            alpha=0.8,
            color="green",
        )
        ax1.plot(
            test_data.index,
            predictions.values,
            label="Predicted Values",
            linewidth=1.5,
            alpha=0.8,
            color="red",
            linestyle="--",
        )

        ax1.set_title("ARIMA Model: Training Data, Test True Values, and Predictions")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (BRL)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        residuals = evaluation_results["residuals"]
        ax2.plot(test_data.index, residuals.values, color="red", alpha=0.7, linewidth=1)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_title("Prediction Residuals")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("results/arima/predictions.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_results(
        self,
        evaluation_results,
        forecast_values=None,
        forecast_dates=None,
        forecast_ci=None,
    ):
        """Plot model results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        test_data = evaluation_results["test_data"]
        predictions = evaluation_results["predictions"]

        axes[0].plot(test_data.index, test_data.values, label="Actual", linewidth=2)
        axes[0].plot(
            test_data.index, predictions, label="Predicted", linewidth=2, alpha=0.8
        )
        axes[0].set_title("ARIMA Model: Predicted vs Actual Prices")
        axes[0].set_ylabel("Price (BRL)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        residuals = evaluation_results["residuals"]
        axes[1].plot(test_data.index, residuals, color="red", alpha=0.7)
        axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_title("Prediction Residuals")
        axes[1].set_ylabel("Residual")
        axes[1].grid(True, alpha=0.3)

        if forecast_values is not None and forecast_dates is not None:
            historical_data = self.original_data.iloc[-100:]
            axes[2].plot(
                historical_data.index,
                historical_data.values,
                label="Historical",
                linewidth=2,
                color="blue",
            )

            axes[2].plot(
                forecast_dates,
                forecast_values,
                label="Forecast",
                linewidth=2,
                color="red",
                linestyle="--",
            )

            if forecast_ci is not None:
                axes[2].fill_between(
                    forecast_dates,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1],
                    color="red",
                    alpha=0.2,
                    label="Confidence Interval",
                )

            if hasattr(forecast_values, "iloc"):
                first_forecast = forecast_values.iloc[0]
            else:
                first_forecast = forecast_values[0]

            axes[2].plot(
                [historical_data.index[-1], forecast_dates[0]],
                [historical_data.values[-1], first_forecast],
                color="orange",
                linewidth=2,
                alpha=0.7,
            )

            axes[2].set_title("Historical Data and ARIMA Forecast")
            axes[2].set_ylabel("Price (BRL)")
            axes[2].legend()
        else:
            axes[2].plot(
                self.original_data.index,
                self.original_data.values,
                label="Full Historical Data",
                linewidth=1,
                color="blue",
            )
            axes[2].set_title("Full Historical Price Data")
            axes[2].set_ylabel("Price (BRL)")
            axes[2].legend()

        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/arima/model_results.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_arima_model():
    """Run the complete ARIMA model pipeline"""
    print("=" * 60)
    print("BRL/USD FUTURES - ARIMA MODEL")
    print("=" * 60)

    try:
        arima_model = BRLUSDArimaModel()

        data = arima_model.load_and_preprocess_data()

        stationarity_result = arima_model.test_stationarity(
            arima_model.original_data, "Original Series"
        )

        if not stationarity_result["is_stationary"]:
            differenced_series, d = arima_model.make_stationary(
                arima_model.original_data
            )
        else:
            differenced_series = arima_model.original_data
            d = 0

        arima_model.plot_series_analysis(
            arima_model.original_data, differenced_series, d
        )

        decomposition = arima_model.seasonal_decomposition(arima_model.original_data)

        train_data, test_data = arima_model.train_test_split(
            arima_model.original_data, test_years=1
        )

        optimal_result = arima_model.find_optimal_parameters(train_data)
        if optimal_result is not None:
            best_params, results_df = optimal_result
        else:
            best_params = None
            results_df = None

        if best_params is None:
            print("Could not find optimal parameters. Using default (1,1,1)")
            best_params = (1, 1, 1)

        fitted_model = arima_model.fit_arima_model(train_data, best_params)

        if fitted_model is None:
            print("Model fitting failed!")
            return None, None, None, None

        evaluation_results = arima_model.evaluate_model(test_data)

        if evaluation_results is None:
            print("Model evaluation failed!")
            return arima_model, None, None, None

        forecast_result = arima_model.forecast_future(n_steps=30)
        if forecast_result is not None and len(forecast_result) == 3:
            forecast_values, forecast_dates, forecast_ci = forecast_result
        else:
            forecast_values, forecast_dates, forecast_ci = None, None, None

        arima_model.plot_training_history()
        arima_model.plot_predictions(data, evaluation_results)

        print("\n" + "=" * 60)
        print("ARIMA MODEL - SUMMARY")
        print("=" * 60)
        print(f"Best ARIMA parameters: {best_params}")
        print(f"Model RMSE: {evaluation_results['rmse']:.2f} BRL")
        print(f"Model MAE: {evaluation_results['mae']:.2f} BRL")
        print(f"Model MAPE: {evaluation_results['mape']:.2f}%")
        print(f"Model R²: {evaluation_results['r2']:.4f}")

        if forecast_values is not None:
            print(
                f"\nNext 30-day forecast range: {forecast_values.min():.2f} - {forecast_values.max():.2f} BRL"
            )
            print(f"Average forecasted price: {forecast_values.mean():.2f} BRL")
            print(f"Current price: {arima_model.original_data.iloc[-1]:.2f} BRL")
            change_pct = (
                (forecast_values.mean() - arima_model.original_data.iloc[-1])
                / arima_model.original_data.iloc[-1]
                * 100
            )
            print(f"Forecasted change: {change_pct:.2f}%")

        print("\nAll plots saved in 'results/arima/' directory")

        return arima_model, evaluation_results, forecast_values, forecast_dates

    except Exception as e:
        print(f"ARIMA model pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    try:
        result = run_arima_model()
        if result is not None:
            model, results, forecasts, forecast_dates = result
            print("ARIMA model completed successfully!")
        else:
            print("ARIMA model failed to complete")
    except Exception as e:
        print(f"Error running ARIMA model: {e}")
