"""
Utility functions for the cryptocurrency ML project.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed value
    """
    import random
    import numpy as np
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For TensorFlow/Keras
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics for time series predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    dict : Dictionary with MAE, RMSE, MAPE metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE calculation with handling for zero values
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def split_time_series(data: pd.DataFrame, date_col: str, 
                     train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets based on date.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    date_col : str
        Name of date column
    train_ratio : float
        Ratio of data to use for training
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : Train and test dataframes
    """
    data = data.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    return train, test


def check_stationarity(timeseries: pd.Series, alpha: float = 0.05) -> dict:
    """
    Check stationarity of time series using Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Test results including p-value and stationarity status
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(timeseries.dropna())
    
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'is_stationary': result[1] <= alpha
    }

