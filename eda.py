import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.ioff()

os.makedirs("eda_plots", exist_ok=True)


def load_data(file_path="./data/Dolfut.csv"):
    """Load and prepare the BRL/USD futures dataset"""
    df = pd.read_csv(file_path, index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.sort_index()
    return df


def basic_info(df):
    """Display basic information about the dataset"""
    print("=" * 60)
    print("BRAZILIAN REAL/USD FUTURES - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape}")
    print(
        f"Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
    )
    print(f"Total Trading Days: {len(df)}")
    print(f"Years Covered: {df.index.max().year - df.index.min().year + 1}")

    print("\nColumn Information:")
    print(df.info())

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")


def analyze_price_patterns(df):
    """Analyze price patterns and trends"""
    print("\n" + "=" * 40)
    print("PRICE PATTERN ANALYSIS")
    print("=" * 40)

    print(f"Close Price Range: {df['Close'].min():.2f} - {df['Close'].max():.2f} BRL")
    print(f"Average Daily Volume: {df['Volume'].mean():,.0f}")
    print(f"Average Daily Return: {df['Returns'].mean():.4f}%")
    print(f"Return Volatility (std): {df['Returns'].std():.4f}%")

    print(f"\nLargest Daily Gain: {df['Returns'].max():.2f}%")
    print(f"Largest Daily Loss: {df['Returns'].min():.2f}%")

    df_monthly = (
        df.groupby(df.index.month)
        .agg({"Close": "mean", "Returns": ["mean", "std"], "Volume": "mean"})
        .round(4)
    )

    print("\nMonthly Patterns (Average):")
    print(df_monthly)


def plot_time_series(df):
    """Create time series visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(df.index, df["Close"], linewidth=1, color="blue")
    axes[0, 0].set_title(
        "BRL/USD Futures Price Evolution", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_ylabel("Price (BRL per USD)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df.index, df["Returns"], linewidth=0.8, color="red", alpha=0.7)
    axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Daily Returns (%)", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylabel("Returns (%)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df.index, df["Volume"], linewidth=0.8, color="green", alpha=0.7)
    axes[1, 0].set_title("Trading Volume", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("Volume")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(df["Volume"], df["Close"], alpha=0.5, s=1)
    axes[1, 1].set_title("Price vs Volume Relationship", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Volume")
    axes[1, 1].set_ylabel("Close Price (BRL)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_plots/time_series_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_distributions(df):
    """Plot distribution analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].hist(df["Returns"], bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0, 0].axvline(
        df["Returns"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['Returns'].mean():.3f}%",
    )
    axes[0, 0].set_title(
        "Distribution of Daily Returns", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Returns (%)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    from scipy import stats

    stats.probplot(df["Returns"], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(
        "Q-Q Plot: Returns vs Normal Distribution", fontsize=14, fontweight="bold"
    )
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(df["Close"], bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[1, 0].axvline(
        df["Close"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['Close'].mean():.2f}",
    )
    axes[1, 0].set_title("Distribution of Close Prices", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Close Price (BRL)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(df["Volume"], bins=30, alpha=0.7, color="orange", edgecolor="black")
    axes[1, 1].axvline(
        df["Volume"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['Volume'].mean():.0f}",
    )
    axes[1, 1].set_title(
        "Distribution of Trading Volume", fontsize=14, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Volume")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_plots/distributions_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def seasonal_analysis(df):
    """Analyze seasonal patterns"""
    print("\n" + "=" * 40)
    print("SEASONAL ANALYSIS")
    print("=" * 40)

    df_seasonal = df.copy()
    df_seasonal["Year"] = df_seasonal.index.year
    df_seasonal["Month"] = df_seasonal.index.month
    df_seasonal["DayOfWeek"] = df_seasonal.index.dayofweek
    df_seasonal["Quarter"] = df_seasonal.index.quarter

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    monthly_returns = df_seasonal.groupby("Month")["Returns"].mean()
    axes[0, 0].bar(
        range(1, 13), monthly_returns.values, color="skyblue", edgecolor="navy"
    )
    axes[0, 0].set_title("Average Returns by Month", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Average Return (%)")
    axes[0, 0].set_xticks(range(1, 13))
    axes[0, 0].grid(True, alpha=0.3)

    dow_returns = df_seasonal.groupby("DayOfWeek")["Returns"].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    weekday_returns = dow_returns.iloc[:5] if len(dow_returns) > 5 else dow_returns
    axes[0, 1].bar(
        range(len(weekday_returns)),
        weekday_returns.values,
        color="lightgreen",
        edgecolor="darkgreen",
    )
    axes[0, 1].set_title(
        "Average Returns by Day of Week", fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Day of Week")
    axes[0, 1].set_ylabel("Average Return (%)")
    axes[0, 1].set_xticks(range(len(weekday_returns)))
    axes[0, 1].set_xticklabels(dow_names[: len(weekday_returns)])
    axes[0, 1].grid(True, alpha=0.3)

    yearly_stats = df_seasonal.groupby("Year").agg(
        {"Close": "mean", "Returns": ["mean", "std"], "Volume": "mean"}
    )

    axes[1, 0].plot(
        yearly_stats.index,
        yearly_stats["Close"]["mean"],
        marker="o",
        linewidth=2,
        color="blue",
    )
    axes[1, 0].set_title("Average Annual Price", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Average Price (BRL)")
    axes[1, 0].grid(True, alpha=0.3)

    rolling_vol = df["Returns"].rolling(window=30).std()
    axes[1, 1].plot(df.index, rolling_vol, linewidth=1, color="red")
    axes[1, 1].set_title("30-Day Rolling Volatility", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Volatility (%)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_plots/seasonal_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def volatility_analysis(df):
    """Analyze volatility patterns"""
    print("\n" + "=" * 40)
    print("VOLATILITY ANALYSIS")
    print("=" * 40)

    df_vol = df.copy()
    df_vol["Daily_Range"] = ((df_vol["High"] - df_vol["Low"]) / df_vol["Close"]) * 100
    df_vol["Price_Change"] = abs(df_vol["Close"] - df_vol["Open"])
    df_vol["Vol_20d"] = df_vol["Returns"].rolling(20).std()
    df_vol["Vol_60d"] = df_vol["Returns"].rolling(60).std()

    print(f"Average Daily Range: {df_vol['Daily_Range'].mean():.2f}%")
    print(f"20-day Average Volatility: {df_vol['Vol_20d'].mean():.4f}%")
    print(f"60-day Average Volatility: {df_vol['Vol_60d'].mean():.4f}%")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(df_vol.index, df_vol["Vol_20d"], label="20-day Volatility", alpha=0.8)
    axes[0].plot(df_vol.index, df_vol["Vol_60d"], label="60-day Volatility", alpha=0.8)
    axes[0].set_title("Rolling Volatility Measures", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Volatility (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_vol.index, df_vol["Daily_Range"], alpha=0.7, color="red")
    axes[1].set_title("Daily Price Range (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Range (%)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_plots/volatility_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_complete_eda():
    """Run complete EDA analysis"""

    df = load_data()

    basic_info(df)

    analyze_price_patterns(df)

    plot_time_series(df)

    plot_distributions(df)

    seasonal_analysis(df)

    volatility_analysis(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETE - KEY INSIGHTS FOR TIME SERIES MODELING:")
    print("=" * 60)
    print("1. Check for stationarity in the price series")
    print("2. Consider log transformation for price modeling")
    print("3. Returns appear to have fat tails (check Q-Q plot)")
    print("4. Volume patterns may provide additional signals")
    print("5. Seasonal patterns exist - incorporate in models")
    print("6. Volatility clustering is evident - consider GARCH models")
    print("7. Monthly contract rollovers may create structural breaks")
    print("\nAll plots saved in 'eda_plots/' directory")


if __name__ == "__main__":
    run_complete_eda()
