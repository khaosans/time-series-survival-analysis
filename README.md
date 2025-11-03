# Specialized Models: Time Series and Survival Analysis

## IBM Machine Learning Assignment - Cryptocurrency Price Prediction using Time Series Models

This project implements specialized time series forecasting models (ARIMA, Prophet, LSTM) to predict cryptocurrency prices using historical data from Kaggle. The project demonstrates advanced time series analysis techniques for financial market prediction.

## Dataset

- **Source**: Kaggle - Cryptocurrency Historical Prices Top 100 (2025)
- **Dataset ID**: `isaaclopgu/cryptocurrency-historical-prices-top-100-2025`
- **URL**: https://www.kaggle.com/datasets/isaaclopgu/cryptocurrency-historical-prices-top-100-2025

## Project Structure

```
IBM-Machine-Learning/
├── data/
│   └── crypto_data/          # Downloaded Kaggle dataset
├── notebooks/
│   └── crypto_analysis.ipynb  # Main Jupyter notebook
├── src/
│   ├── data_loader.py         # Dataset download and loading utilities
│   ├── preprocessing.py       # Data cleaning and preprocessing
│   ├── models/
│   │   ├── arima_model.py     # ARIMA implementation
│   │   ├── lstm_model.py      # LSTM implementation
│   │   └── prophet_model.py   # Prophet implementation
│   ├── visualization.py       # Plotting utilities
│   └── utils.py               # Helper functions
├── results/
│   ├── plots/                 # Saved visualizations
│   └── models/                # Saved trained models
├── reports/
│   └── final_report.md        # Summary report with findings
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Kaggle API credentials configured at `~/.kaggle/kaggle.json`

### Installation

1. Clone or navigate to this repository
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure Kaggle API is configured:
   ```bash
   # Verify kaggle.json exists at ~/.kaggle/kaggle.json
   ls -la ~/.kaggle/kaggle.json
   ```
   
   If not configured, place your `kaggle.json` file in `~/.kaggle/` with permissions 600:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Usage

1. **Download Dataset**:
   ```python
   from src.data_loader import download_dataset
   download_dataset('isaaclopgu/cryptocurrency-historical-prices-top-100-2025')
   ```

2. **Run Analysis**:
   - Open `notebooks/crypto_analysis.ipynb` in Jupyter
   - Execute cells sequentially to perform complete analysis

3. **View Results**:
   - Check `results/plots/` for all visualizations
   - Read `reports/final_report.md` for comprehensive findings

## Models Implemented

1. **ARIMA** - AutoRegressive Integrated Moving Average for time series forecasting
2. **Prophet** - Facebook's Prophet for forecasting with seasonality
3. **LSTM** - Long Short-Term Memory neural network for deep learning approach

## Metrics

All models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

## Authors

- IBM Machine Learning Course Assignment

## License

This project is for educational purposes as part of the IBM Machine Learning course.

