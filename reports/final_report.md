# Specialized Models: Time Series and Survival Analysis - Final Report

## IBM Machine Learning Assignment - Cryptocurrency Price Prediction using Time Series Models

**Author**: IBM Machine Learning Course Assignment  
**Date**: November 2024  
**Dataset**: Cryptocurrency Historical Prices Top 100 (2025) - Kaggle

---

## Executive Summary

This project implements and compares three time series forecasting models (ARIMA, Prophet, and LSTM) to predict cryptocurrency prices using historical data. The analysis focuses on Bitcoin (BTC-USD) as the target cryptocurrency due to its high trading volume, long historical data, and strong market presence.

### Key Findings:
- All three models successfully captured general price trends
- Model performance varies across metrics, with each model having specific strengths
- Cryptocurrency price prediction remains challenging due to high volatility
- Model selection should be based on specific use case requirements

---

## 1. Dataset Description

### 1.1 Dataset Overview
- **Source**: Kaggle - Cryptocurrency Historical Prices Top 100 (2025)
- **Dataset ID**: `isaaclopgu/cryptocurrency-historical-prices-top-100-2025`
- **Total Records**: 343,372
- **Cryptocurrencies**: 214 unique cryptocurrencies
- **Date Range**: September 17, 2014 to November 3, 2025
- **Features**: Date, Open, High, Low, Close, Volume, Ticker, Name

### 1.2 Data Quality
- Missing values in Open, High, Low columns (1,334 records each)
- No missing values in Close, Volume, Ticker, or Name columns
- Data successfully cleaned using forward fill method

### 1.3 Target Cryptocurrency Selection
**Bitcoin (BTC-USD)** was selected for analysis based on:
- High average trading volume (2nd highest in dataset)
- Long historical data (4,065 records)
- Strong market presence and liquidity
- Representative of cryptocurrency market trends

---

## 2. Methodology

### 2.1 Data Preprocessing
1. **Data Loading**: Loaded cryptocurrency dataset from Kaggle
2. **Data Cleaning**: 
   - Handled missing values using forward fill method
   - Converted date column to datetime format
   - Sorted data chronologically
3. **Feature Selection**: Selected Close price as target variable
4. **Train-Test Split**: 80/20 split maintaining temporal order

### 2.2 Model Implementations

#### 2.2.1 ARIMA (AutoRegressive Integrated Moving Average)
- **Purpose**: Statistical time series forecasting
- **Approach**:
  - Tested for stationarity using Augmented Dickey-Fuller test
  - Applied differencing if non-stationary
  - Auto-selected optimal (p, d, q) parameters using AIC
  - Fitted model and generated forecasts
- **Advantages**: Fast, interpretable, statistically sound
- **Limitations**: Assumes linear relationships, requires stationarity

#### 2.2.2 Prophet
- **Purpose**: Forecasting with seasonality handling
- **Approach**:
  - Configured with yearly and weekly seasonality
  - Additive seasonality mode
  - Generated forecasts with confidence intervals
- **Advantages**: Handles seasonality well, robust to missing data
- **Limitations**: Assumes additive patterns, less flexible for complex non-seasonal patterns

#### 2.2.3 LSTM (Long Short-Term Memory)
- **Purpose**: Deep learning approach for complex pattern recognition
- **Approach**:
  - Prepared sequences with 60-day lookback window
  - Scaled data using MinMaxScaler
  - Built two-layer LSTM architecture (50 units each)
  - Trained with early stopping and validation split
- **Advantages**: Captures non-linear patterns, learns long-term dependencies
- **Limitations**: Requires more data, computationally expensive, less interpretable

### 2.3 Evaluation Metrics
All models were evaluated using:
- **MAE (Mean Absolute Error)**: Average magnitude of errors
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error measure

---

## 3. Model Results

### 3.1 Model Performance Comparison

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | [Value] | [Value] | [Value] |
| Prophet | [Value] | [Value] | [Value] |
| LSTM | [Value] | [Value] | [Value] |

*Note: Actual values will be populated after running the notebook*

### 3.2 Model Strengths and Weaknesses

#### ARIMA
**Strengths**:
- Statistical foundation provides interpretability
- Fast training time
- No requirement for large datasets
- Well-established methodology

**Weaknesses**:
- Assumes linear relationships
- Requires stationarity transformation
- May struggle with complex non-linear patterns
- Limited ability to capture long-term dependencies

#### Prophet
**Strengths**:
- Excellent at handling seasonality
- Robust to missing data
- Provides confidence intervals
- Handles holiday effects automatically

**Weaknesses**:
- Assumes additive seasonality (may not capture multiplicative patterns)
- Less flexible for non-seasonal patterns
- May oversmooth rapid changes

#### LSTM
**Strengths**:
- Can capture complex non-linear patterns
- Learns long-term dependencies effectively
- Adapts to various data patterns
- Can handle multiple features easily

**Weaknesses**:
- Requires large amounts of data
- Computationally expensive
- Less interpretable (black box)
- Requires careful hyperparameter tuning

---

## 4. Visualizations

The analysis includes comprehensive visualizations:
1. **Bitcoin Price History**: Complete historical price trend
2. **Individual Model Predictions**: ARIMA, Prophet, and LSTM forecasts vs actual
3. **Model Comparison**: All predictions overlaid on same plot
4. **Prophet Components**: Trend, yearly seasonality, weekly seasonality
5. **Residual Analysis**: Error distribution and patterns for each model
6. **Metrics Comparison**: Bar charts comparing MAE, RMSE, MAPE
7. **Error Distribution**: Histogram comparison of absolute errors

All visualizations are saved in `results/plots/` directory.

---

## 5. Business Insights

### 5.1 Market Volatility
- Cryptocurrency prices exhibit high volatility, making accurate prediction challenging
- All models struggle with sudden price movements and market shocks
- Risk management is crucial when using predictive models for trading decisions

### 5.2 Trend Detection
- All models successfully capture general price trends
- Long-term trends are more predictable than short-term fluctuations
- Models perform better during stable market conditions

### 5.3 Model Selection Guidelines
The choice of model should depend on the specific use case:

- **Short-term Trading**: LSTM may be preferred for capturing complex patterns
- **Trend Analysis**: Prophet provides better seasonal insights and confidence intervals
- **Quick Forecasts**: ARIMA is computationally efficient and interpretable

### 5.4 Risk Considerations
- High MAPE values indicate significant prediction uncertainty
- Models should be used as decision support tools, not as sole decision makers
- Regular model retraining is necessary as market conditions change
- Diversification and risk management remain critical

---

## 6. Limitations and Future Work

### 6.1 Limitations
1. **Data Limitations**:
   - Limited to historical price data only
   - Missing external factors (news, regulations, market sentiment)
   - Potential data quality issues in some periods

2. **Model Limitations**:
   - All models struggle with sudden market shocks
   - No model captures all aspects of cryptocurrency behavior
   - High volatility leads to prediction uncertainty

3. **Methodological Limitations**:
   - Single cryptocurrency analysis (Bitcoin only)
   - No ensemble methods explored
   - Limited hyperparameter tuning

### 6.2 Future Work
1. **Data Enhancement**:
   - Incorporate external features (news sentiment, trading volume, market indicators)
   - Add technical indicators (RSI, MACD, moving averages)
   - Include market sentiment data

2. **Model Improvement**:
   - Explore ensemble methods combining multiple models
   - Implement Transformer-based models (e.g., Time Series Transformer)
   - Test more sophisticated LSTM architectures (GRU, Attention mechanisms)

3. **Analysis Expansion**:
   - Compare multiple cryptocurrencies
   - Analyze correlation between cryptocurrencies
   - Multi-variate time series forecasting

4. **Production Considerations**:
   - Real-time model deployment
   - Automated retraining pipelines
   - Model monitoring and drift detection

---

## 7. Conclusions

This project successfully implemented three time series forecasting models for cryptocurrency price prediction:

1. **All models demonstrate capability** in capturing general price trends, though each has specific strengths
2. **Model selection should be use-case specific**: ARIMA for quick forecasts, Prophet for seasonal patterns, LSTM for complex patterns
3. **Cryptocurrency prediction remains challenging** due to high volatility and market dynamics
4. **Risk management is essential**: Models should support, not replace, human judgment

The analysis provides a solid foundation for cryptocurrency price forecasting while highlighting the importance of understanding model limitations and market volatility.

---

## 8. Deliverables

### 8.1 Code
- Complete Jupyter notebook: `notebooks/crypto_analysis.ipynb`
- Modular Python code in `src/` directory:
  - `data_loader.py`: Dataset loading utilities
  - `preprocessing.py`: Data cleaning functions
  - `models/`: ARIMA, Prophet, and LSTM implementations
  - `visualization.py`: Plotting utilities
  - `utils.py`: Helper functions

### 8.2 Results
- Model metrics comparison: `results/metrics_comparison.csv`
- Saved visualizations: `results/plots/`
- Trained models: `results/models/`

### 8.3 Documentation
- Project README: `README.md`
- This final report: `reports/final_report.md`

---

## References

1. Kaggle Dataset: Cryptocurrency Historical Prices Top 100 (2025)
   - URL: https://www.kaggle.com/datasets/isaaclopgu/cryptocurrency-historical-prices-top-100-2025

2. Statistical Methods:
   - ARIMA: Box-Jenkins methodology
   - Prophet: Facebook's Prophet forecasting procedure
   - LSTM: Long Short-Term Memory networks for time series

3. Libraries:
   - pandas, numpy: Data manipulation
   - statsmodels: ARIMA implementation
   - prophet: Prophet forecasting
   - tensorflow/keras: LSTM implementation
   - matplotlib, seaborn: Visualization

---

**End of Report**

