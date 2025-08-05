# BRL/USD Forecasting

Machine learning project for forecasting Brazilian Real to USD exchange rates with time series analysis.

## Dataset

- **Source**: BRL/USD Futures (`data/Dolfut.csv`)
- **Features**: OHLCV + technical indicators
- **Period**: Multi-year daily data

## Models

### LSTM Baseline
- 60-day sequences
- 3-layer LSTM with dropout
- Technical indicators: MA, RSI, volatility ratios

*Additional models will be evaluated in future iterations.*

## Results

```
RMSE: 75.51 BRL
MAE: 59.77 BRL
R²: 0.8515
MAPE: 1.16%
```

## Structure

```
├── data/Dolfut.csv
├── eda.py
├── lstm_baseline_model.py
├── results/lstm/
├── requirements.txt
└── README.md
```

## Key Insights

- High accuracy (R² > 0.85, MAPE 1.16%)
