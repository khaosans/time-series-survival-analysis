"""
Data preprocessing utilities for cryptocurrency dataset.
Handles data cleaning, transformation, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def load_and_clean_data(file_path: str, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Load cryptocurrency data and perform initial cleaning.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    ticker : str, optional
        Specific cryptocurrency ticker to filter (e.g., 'BTC-USD')
    
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by ticker if specified
    if ticker and 'ticker' in df.columns:
        df = df[df['ticker'] == ticker].copy()
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Method to handle missing values:
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with missing values
    
    Returns:
    --------
    pd.DataFrame : DataFrame with missing values handled
    """
    df = df.copy()
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill')
    elif method == 'backward_fill':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # If still have missing values, drop them
    df = df.dropna()
    
    return df


def detect_outliers(df: pd.DataFrame, columns: List[str], 
                   method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to check for outliers
    method : str
        Method for outlier detection:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
    
    Returns:
    --------
    pd.DataFrame : DataFrame with outlier information in new columns
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_outlier'] = np.abs((df[col] - mean) / std) > 3
    
    return df


def get_top_cryptocurrencies(df: pd.DataFrame, 
                            by: str = 'Volume', 
                            top_n: int = 10) -> pd.DataFrame:
    """
    Get top cryptocurrencies by specified metric.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    by : str
        Column to rank by (e.g., 'Volume', 'Close')
    top_n : int
        Number of top cryptocurrencies to return
    
    Returns:
    --------
    pd.DataFrame : Top cryptocurrencies with their metrics
    """
    if 'ticker' not in df.columns:
        raise ValueError("Dataframe must have 'ticker' column")
    
    if by not in df.columns:
        raise ValueError(f"Column '{by}' not found in dataframe")
    
    # Calculate average or latest value per ticker
    ticker_stats = df.groupby('ticker').agg({
        by: 'mean' if by in ['Volume'] else 'last',
        'Date': 'max'  # Latest date
    }).reset_index()
    
    # Sort and get top N
    top_crypto = ticker_stats.nlargest(top_n, by)
    
    return top_crypto


def prepare_time_series_data(df: pd.DataFrame, 
                            target_col: str = 'Close',
                            date_col: str = 'Date',
                            additional_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare data for time series modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column for prediction
    date_col : str
        Date column name
    additional_features : List[str], optional
        Additional features to include
    
    Returns:
    --------
    pd.DataFrame : Prepared dataframe for time series modeling
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Select relevant columns
    cols = [date_col, target_col]
    if additional_features:
        cols.extend([f for f in additional_features if f in df.columns])
    
    ts_data = df[cols].copy()
    
    # Ensure target column has no missing values
    ts_data = ts_data.dropna(subset=[target_col])
    
    return ts_data


def create_lag_features(df: pd.DataFrame, 
                       target_col: str,
                       lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column to create lags for
    lags : List[int]
        List of lag periods
    
    Returns:
    --------
    pd.DataFrame : DataFrame with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df

