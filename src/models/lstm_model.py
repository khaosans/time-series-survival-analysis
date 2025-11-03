"""
LSTM model implementation for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')


def create_sequences(data: np.ndarray, lookback: int = 60) -> tuple:
    """
    Create sequences for LSTM model.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    lookback : int
        Number of time steps to look back
    
    Returns:
    --------
    tuple : (X, y) sequences
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)


def prepare_lstm_data(timeseries: pd.Series, lookback: int = 60,
                      train_ratio: float = 0.8, scale: bool = True) -> dict:
    """
    Prepare data for LSTM model.
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    lookback : int
        Number of time steps to look back
    train_ratio : float
        Ratio of data to use for training
    scale : bool
        Whether to scale the data
    
    Returns:
    --------
    dict : Dictionary with prepared data and scaler
    """
    values = timeseries.values.reshape(-1, 1)
    
    # Scale data
    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(values)
    
    # Create sequences
    X, y = create_sequences(values.flatten(), lookback=lookback)
    
    # Reshape X for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split into train and test
    split_idx = int(len(X) * train_ratio)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'lookback': lookback
    }


def build_lstm_model(input_shape: tuple, units: list = [50, 50],
                    dropout_rate: float = 0.2, learning_rate: float = 0.001) -> keras.Model:
    """
    Build LSTM model architecture.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    units : list
        Number of units in each LSTM layer
    dropout_rate : float
        Dropout rate
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    keras.Model : Compiled LSTM model
    """
    model = Sequential()
    
    # Add first LSTM layer
    model.add(LSTM(units=units[0], return_sequences=True, 
                   input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Add additional LSTM layers
    for i in range(1, len(units)):
        model.add(LSTM(units=units[i], return_sequences=(i < len(units) - 1)))
        model.add(Dropout(dropout_rate))
    
    # Add output layer
    model.add(Dense(1))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_lstm_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 50, batch_size: int = 32,
                    validation_split: float = 0.2,
                    patience: int = 10,
                    model_path: str = './results/models/lstm_model.h5') -> dict:
    """
    Train LSTM model.
    
    Parameters:
    -----------
    model : keras.Model
        LSTM model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray, optional
        Validation features
    y_val : np.ndarray, optional
        Validation targets
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    validation_split : float
        Fraction of training data to use for validation
    patience : int
        Early stopping patience
    model_path : str
        Path to save best model
    
    Returns:
    --------
    dict : Training history and best model
    """
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, 
                     restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', 
                       save_best_only=True, verbose=0)
    ]
    
    # Prepare validation data
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        validation_split = None
    else:
        validation_data = None
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    
    return {
        'model': model,
        'history': history,
        'model_path': model_path
    }


def predict_lstm(model: keras.Model, X: np.ndarray, 
                scaler: MinMaxScaler = None) -> np.ndarray:
    """
    Make predictions using LSTM model.
    
    Parameters:
    -----------
    model : keras.Model
        Trained LSTM model
    X : np.ndarray
        Input features
    scaler : MinMaxScaler, optional
        Scaler to inverse transform predictions
    
    Returns:
    --------
    np.ndarray : Predictions
    """
    predictions = model.predict(X, verbose=0)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()


def evaluate_lstm(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate LSTM model predictions.
    
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


def plot_lstm_training_history(history: keras.callbacks.History,
                              save_path: str = None):
    """
    Plot LSTM training history.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_lstm_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         dates: pd.DatetimeIndex = None,
                         save_path: str = None):
    """
    Plot LSTM predictions vs actual values.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : pd.DatetimeIndex, optional
        Date index for x-axis
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(15, 6))
    
    if dates is not None and len(dates) == len(y_true):
        plt.plot(dates, y_true, label='Actual', alpha=0.7, linewidth=2)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.8, linewidth=2)
        plt.xlabel('Date')
    else:
        plt.plot(y_true, label='Actual', alpha=0.7, linewidth=2)
        plt.plot(y_pred, label='Predicted', alpha=0.8, linewidth=2)
        plt.xlabel('Time Step')
    
    plt.title('LSTM Model: Predictions vs Actual')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

