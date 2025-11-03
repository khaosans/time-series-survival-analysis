# BTC-USD Time Series Forecasting with Prophet

A comprehensive time-series forecasting project using Facebook's Prophet library to predict BTC-USD closing prices.

## 📋 Project Overview

This project demonstrates the application of Prophet, a powerful forecasting tool developed by Facebook, to cryptocurrency price prediction. The analysis includes:

- Data preparation and cleaning
- Exploratory data analysis with trend and seasonality decomposition
- Training and comparison of three Prophet model variations:
  - Default Prophet model
  - Prophet with yearly and weekly seasonality
  - Prophet with seasonality and US holidays
- Model evaluation using RMSE and MAE metrics
- Visualization of forecasts and model components
- Identification of best-performing model

## 🏗️ Project Structure

```
IBM-Machine-Learning/
├── btc_prophet.ipynb          # Main Jupyter notebook with analysis
├── Crypto_historical_data.csv # Historical cryptocurrency data
├── requirements.txt           # Python dependencies
├── setup_and_run.sh          # Setup script for automated execution
├── generate_outputs.py        # Script to execute notebook and save outputs
├── LICENSE                    # MIT License
├── README.md                  # This file
└── .gitignore                 # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository:
```bash
git clone https://github.com/khaosans/time-series-survival-analysis.git
cd IBM-Machine-Learning
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Run the notebook to generate visualizations:

   **Option A: Automated Script (Recommended)**
   ```bash
   python3 generate_outputs.py
   ```
   This script automatically executes all notebook cells and saves outputs.

   **Option B: Manual Execution**
   ```bash
   jupyter notebook btc_prophet.ipynb
   ```
   Then: `Kernel` → `Restart & Run All` → Save the notebook

   **Option C: Quick Setup Script**
   ```bash
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

   **Note**: To see visualizations on GitHub, you must run all cells and save the notebook. The notebook uses `%matplotlib inline` to embed plots directly in the output.

## 📦 Dependencies

Key libraries used in this project:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **prophet**: Time series forecasting
- **scikit-learn**: Machine learning metrics
- **statsmodels**: Statistical modeling

See `requirements.txt` for the complete list with versions.

## 📊 Data

The project uses historical cryptocurrency data from the Kaggle dataset:

**Dataset**: [Cryptocurrency Historical Prices - Top 100 (2025)](https://www.kaggle.com/datasets/isaaclopgu/cryptocurrency-historical-prices-top-100-2025)

The dataset is stored in `Crypto_historical_data.csv` and includes:

- **Date**: Timestamp for each data point
- **Open, High, Low, Close**: OHLC (Open-High-Low-Close) price data
- **Volume**: Trading volume
- **ticker**: Cryptocurrency identifier (BTC-USD)
- **name**: Cryptocurrency name

This dataset contains historical price data for the top 100 cryptocurrencies, from which we filter and analyze BTC-USD data.

## 🔬 Methodology

1. **Data Preparation**
   - Filter data for BTC-USD
   - Rename columns to Prophet format (ds, y)
   - Handle timezone information
   - Check and handle missing values

2. **Exploratory Data Analysis**
   - Plot time series data
   - Decompose time series into trend, seasonality, and residuals
   - Calculate rolling statistics

3. **Train-Test Split**
   - Split data into training (historical) and test (last 365 days) sets

4. **Model Training**
   - Train three Prophet model variations
   - Generate forecasts for the test period

5. **Model Evaluation**
   - Calculate RMSE and MAE for each model
   - Select best-performing model

6. **Visualization**
   - Plot forecast components (trend, seasonality, holidays)
   - Visualize forecasts with confidence intervals
   - Compare predictions with actual values

## 📈 Results

The best-performing model (Prophet with holidays) achieved:
- **RMSE**: ~20,712 USD
- **MAE**: ~18,695 USD

These metrics are evaluated on a 365-day holdout period.

## 🔍 Key Insights

1. **Trend**: Clear upward trajectory in BTC-USD prices over the historical period
2. **Seasonality**: Identified yearly and weekly patterns in price movements
3. **Holidays**: US holidays appear to influence cryptocurrency prices
4. **Uncertainty**: Confidence intervals widen over longer forecast horizons

## 🚧 Future Improvements

- Incorporate additional features (trading volume, sentiment data)
- Explore alternative models (ARIMA, LSTM)
- Fine-tune Prophet hyperparameters
- Implement time series cross-validation
- Model high-volatility periods separately

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Souriya Khaosanga

## 🙏 Acknowledgments

- **Facebook's Prophet team** for the excellent forecasting library
- **Isaac Lopgu** for providing the comprehensive cryptocurrency historical prices dataset on Kaggle
- **Kaggle** for hosting the dataset and providing the platform for data science collaboration

## 📚 References & Citations

### Software & Libraries

1. **Prophet**: Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45. [DOI: 10.1080/00031305.2017.1380080](https://doi.org/10.1080/00031305.2017.1380080)
   - [Prophet Documentation](https://facebook.github.io/prophet/)
   - [GitHub Repository](https://github.com/facebook/prophet)

2. **pandas**: McKinney, W. (2010). Data structures for statistical computing in python. In *Proceedings of the 9th Python in Science Conference*, 445, 51-56.
   - [pandas Documentation](https://pandas.pydata.org/)

3. **NumPy**: Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362. [DOI: 10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

4. **scikit-learn**: Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

5. **statsmodels**: Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with python. In *9th Python in Science Conference*.

### Methodological References

6. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts. [https://otexts.com/fpp3/](https://otexts.com/fpp3/)

7. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). John Wiley & Sons.

### Data Sources

8. **Cryptocurrency Historical Prices Dataset**: Lopgu, I. (2025). Cryptocurrency Historical Prices - Top 100 (2025). Kaggle. Retrieved from [https://www.kaggle.com/datasets/isaaclopgu/cryptocurrency-historical-prices-top-100-2025](https://www.kaggle.com/datasets/isaaclopgu/cryptocurrency-historical-prices-top-100-2025)

### Additional Resources

- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)
- [Facebook Research - Prophet Blog Post](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/)

