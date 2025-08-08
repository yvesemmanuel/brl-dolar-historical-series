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

### ARIMA Model
- Automatic parameter optimization (p,d,q)
- Rolling one-step-ahead forecasts
- Stationarity testing and differencing
- Best model: ARIMA(2,1,3)

## Model Performance Comparison

| Metric | LSTM | ARIMA | Winner |
|--------|------|-------|--------|
| **RMSE** | 75.51 BRL | 60.49 BRL | 🏆 ARIMA |
| **MAE** | 59.77 BRL | 46.79 BRL | 🏆 ARIMA |
| **R²** | 0.8515 | 0.9047 | 🏆 ARIMA |
| **MAPE** | 1.16% | 0.91% | 🏆 ARIMA |

The ARIMA model significantly outperforms the LSTM baseline across all metrics, showing superior accuracy with lower error rates and higher R² score.

## Structure

```
├── data/Dolfut.csv
├── eda.py
├── lstm_baseline_model.py
├── arima_model.py
├── results/
│   ├── lstm/
│   │   ├── metrics.json
│   │   ├── predictions.png
│   │   └── training_history.png
│   └── arima/
│       ├── metrics.json
│       ├── predictions.png
│       ├── training_history.png
│       ├── series_analysis.png
│       └── seasonal_decomposition.png
├── requirements.txt
└── README.md
```

## Key Insights

- **ARIMA** achieves superior performance (R² = 0.9047, MAPE = 0.91%)
- **LSTM** provides competitive baseline (R² = 0.8515, MAPE = 1.16%)
- Rolling forecasts eliminate straight-line predictions in ARIMA
- Both models demonstrate strong predictive capability for BRL/USD futures
