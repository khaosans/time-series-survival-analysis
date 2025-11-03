"""
ARIMA model implementation for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


def test_stationarity(timeseries: pd.Series, alpha: float = 0.05) -> dict:
    """
    Test time series stationarity using Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Test results
    """
    result = adfuller(timeseries.dropna())
    
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'is_stationary': result[1] <= alpha
    }


def make_stationary(timeseries: pd.Series, method: str = 'diff') -> tuple:
    """
    Make time series stationary.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Original time series
    method : str
        Method to make stationary ('diff' or 'log_diff')
    
    Returns:
    --------
    tuple : (stationary_series, original_series, method_used)
    """
    if method == 'diff':
        stationary = timeseries.diff().dropna()
        return stationary, timeseries, 'diff'
    
    elif method == 'log_diff':
        log_series = np.log(timeseries)
        stationary = log_series.diff().dropna()
        return stationary, timeseries, 'log_diff'
    
    else:
        raise ValueError(f"Unknown method: {method}")


def find_arima_params(timeseries: pd.Series, max_p: int = 5, max_d: int = 2, 
                      max_q: int = 5) -> tuple:
    """
    Find optimal ARIMA parameters using AIC.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Stationary time series
    max_p : int
        Maximum AR order
    max_d : int
        Maximum differencing order
    max_q : int
        Maximum MA order
    
    Returns:
    --------
    tuple : (p, d, q) optimal parameters
    """
    best_aic = np.inf
    best_params = (0, 0, 0)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                except:
                    continue
    
    return best_params


def fit_arima_model(timeseries: pd.Series, order: tuple = None, 
                   auto_select: bool = True) -> dict:
    """
    Fit ARIMA model to time series.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    order : tuple, optional
        ARIMA order (p, d, q). If None and auto_select=True, will find optimal
    auto_select : bool
        Whether to automatically select optimal parameters
    
    Returns:
    --------
    dict : Model results including fitted model, predictions, and metrics
    """
    # Make stationary if needed
    stationary_test = test_stationarity(timeseries)
    
    if not stationary_test['is_stationary']:
        stationary, original, method = make_stationary(timeseries, method='diff')
    else:
        stationary = timeseries
        original = timeseries
        method = None
    
    # Find optimal parameters if auto_select
    if auto_select or order is None:
        order = find_arima_params(stationary)
    
    # Fit model
    model = ARIMA(stationary, order=order)
    fitted_model = model.fit()
    
    # Get fitted values
    fitted_values = fitted_model.fittedvalues
    
    # Convert back to original scale if needed
    if method == 'diff':
        fitted_original = original.iloc[0] + fitted_values.cumsum()
        fitted_original = pd.Series(fitted_original, index=original.index[1:])
    elif method == 'log_diff':
        fitted_original = np.exp(np.log(original.iloc[0]) + fitted_values.cumsum())
        fitted_original = pd.Series(fitted_original, index=original.index[1:])
    else:
        fitted_original = fitted_values
    
    return {
        'model': fitted_model,
        'order': order,
        'fitted_values': fitted_original,
        'stationary': stationary,
        'method': method,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic
    }


def forecast_arima(fitted_model, steps: int, timeseries: pd.Series = None,
                  method: str = None) -> np.ndarray:
    """
    Generate forecasts using fitted ARIMA model.
    
    Parameters:
    -----------
    fitted_model : statsmodels ARIMA model
        Fitted ARIMA model
    steps : int
        Number of steps to forecast
    timeseries : pd.Series, optional
        Original time series (needed if inverse transform required)
    method : str, optional
        Transformation method used ('diff' or 'log_diff')
    
    Returns:
    --------
    np.ndarray : Forecast values
    """
    forecast = fitted_model.forecast(steps=steps)
    
    # Inverse transform if needed
    if method == 'diff' and timeseries is not None:
        last_value = timeseries.iloc[-1]
        forecast = last_value + np.cumsum(forecast)
    elif method == 'log_diff' and timeseries is not None:
        last_value = np.log(timeseries.iloc[-1])
        forecast = np.exp(last_value + np.cumsum(forecast))
    
    return forecast.values if hasattr(forecast, 'values') else forecast


def evaluate_arima(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate ARIMA model predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def plot_arima_results(timeseries: pd.Series, fitted_values: pd.Series,
                       forecasts: np.ndarray = None, 
                       save_path: str = None):
    """
    Plot ARIMA model results.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Original time series
    fitted_values : pd.Series
        Fitted values
    forecasts : np.ndarray, optional
        Forecast values
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(15, 6))
    plt.plot(timeseries.index, timeseries.values, label='Original', alpha=0.7)
    plt.plot(fitted_values.index, fitted_values.values, label='Fitted', alpha=0.8)
    
    if forecasts is not None:
        forecast_index = pd.date_range(
            start=timeseries.index[-1] + pd.Timedelta(days=1),
            periods=len(forecasts),
            freq='D'
        )
        plt.plot(forecast_index, forecasts, label='Forecast', linestyle='--')
    
    plt.title('ARIMA Model Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_acf_pacf(timeseries: pd.Series, lags: int = 40, save_path: str = None):
    """
    Plot ACF and PACF for time series.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    lags : int
        Number of lags to plot
    save_path : str, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    acf_values = acf(timeseries.dropna(), nlags=lags)
    pacf_values = pacf(timeseries.dropna(), nlags=lags)
    
    axes[0].plot(range(len(acf_values)), acf_values)
    axes[0].axhline(y=0, linestyle='--', color='black')
    axes[0].axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    axes[0].axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    axes[0].set_title('ACF (Autocorrelation Function)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(len(pacf_values)), pacf_values)
    axes[1].axhline(y=0, linestyle='--', color='black')
    axes[1].axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    axes[1].axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    axes[1].set_title('PACF (Partial Autocorrelation Function)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

