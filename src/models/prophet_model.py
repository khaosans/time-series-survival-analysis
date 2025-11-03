"""
Prophet model implementation for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')


def prepare_prophet_data(df: pd.DataFrame, date_col: str, 
                         target_col: str) -> pd.DataFrame:
    """
    Prepare data for Prophet model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Date column name
    target_col : str
        Target column name
    
    Returns:
    --------
    pd.DataFrame : DataFrame with 'ds' and 'y' columns for Prophet
    """
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df[date_col]),
        'y': df[target_col].values
    })
    
    # Remove missing values
    prophet_df = prophet_df.dropna()
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    return prophet_df


def fit_prophet_model(df: pd.DataFrame, 
                     yearly_seasonality: bool = True,
                     weekly_seasonality: bool = True,
                     daily_seasonality: bool = False,
                     seasonality_mode: str = 'additive',
                     changepoint_prior_scale: float = 0.05,
                     seasonality_prior_scale: float = 10.0) -> dict:
    """
    Fit Prophet model to time series data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Prophet-formatted dataframe with 'ds' and 'y' columns
    yearly_seasonality : bool
        Whether to include yearly seasonality
    weekly_seasonality : bool
        Whether to include weekly seasonality
    daily_seasonality : bool
        Whether to include daily seasonality
    seasonality_mode : str
        'additive' or 'multiplicative'
    changepoint_prior_scale : float
        Flexibility of changepoints
    seasonality_prior_scale : float
        Strength of seasonality components
    
    Returns:
    --------
    dict : Model results including fitted model and predictions
    """
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    
    # Fit model
    model.fit(df)
    
    # Make predictions on training data
    forecast = model.predict(df)
    
    return {
        'model': model,
        'forecast': forecast,
        'fitted_values': forecast['yhat'].values,
        'parameters': {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale
        }
    }


def forecast_prophet(model, periods: int, freq: str = 'D') -> pd.DataFrame:
    """
    Generate future forecasts using Prophet model.
    
    Parameters:
    -----------
    model : Prophet
        Fitted Prophet model
    periods : int
        Number of periods to forecast
    freq : str
        Frequency of forecast ('D' for daily, 'H' for hourly, etc.)
    
    Returns:
    --------
    pd.DataFrame : Forecast dataframe with 'ds' and 'yhat' columns
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    return forecast


def evaluate_prophet(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate Prophet model predictions.
    
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


def plot_prophet_results(model, forecast: pd.DataFrame, 
                        original_data: pd.DataFrame = None,
                        save_path: str = None):
    """
    Plot Prophet model results.
    
    Parameters:
    -----------
    model : Prophet
        Fitted Prophet model
    forecast : pd.DataFrame
        Forecast dataframe from model.predict()
    original_data : pd.DataFrame, optional
        Original data for comparison
    save_path : str, optional
        Path to save plot
    """
    fig = model.plot(forecast)
    ax = fig.gca()
    
    if original_data is not None:
        ax.plot(original_data['ds'], original_data['y'], 
               'ko', markersize=3, label='Actual', alpha=0.5)
        ax.legend()
    
    ax.set_title('Prophet Model Results')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_prophet_components(model, forecast: pd.DataFrame,
                           save_path: str = None):
    """
    Plot Prophet model components (trend, seasonality).
    
    Parameters:
    -----------
    model : Prophet
        Fitted Prophet model
    forecast : pd.DataFrame
        Forecast dataframe from model.predict()
    save_path : str, optional
        Path to save plot
    """
    fig = model.plot_components(forecast)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_prophet_forecast_vs_actual(forecast: pd.DataFrame,
                                   actual: pd.Series,
                                   save_path: str = None):
    """
    Plot Prophet forecast vs actual values.
    
    Parameters:
    -----------
    forecast : pd.DataFrame
        Forecast dataframe
    actual : pd.Series
        Actual values
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(15, 6))
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['yhat'], 
            label='Forecast', alpha=0.8, linewidth=2)
    plt.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'],
                     alpha=0.3, label='Confidence Interval')
    
    # Plot actual if provided
    if actual is not None and len(actual) == len(forecast):
        plt.plot(forecast['ds'], actual.values, 
                'ko', markersize=2, label='Actual', alpha=0.6)
    
    plt.title('Prophet Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

