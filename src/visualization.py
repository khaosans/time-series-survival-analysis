"""
Visualization utilities for cryptocurrency ML project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict


def set_plot_style(style: str = 'seaborn-v0_8', figsize: tuple = (12, 6)):
    """
    Set matplotlib plot style.
    
    Parameters:
    -----------
    style : str
        Matplotlib style
    figsize : tuple
        Default figure size
    """
    plt.style.use(style)
    sns.set_palette("husl")


def plot_time_series(data: pd.DataFrame, date_col: str, value_col: str,
                    title: str = 'Time Series Plot',
                    save_path: Optional[str] = None):
    """
    Plot time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to plot
    date_col : str
        Date column name
    value_col : str
        Value column name
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(15, 6))
    plt.plot(data[date_col], data[value_col], linewidth=1.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_model_comparison(predictions: Dict[str, np.ndarray],
                         actual: np.ndarray,
                         dates: Optional[pd.DatetimeIndex] = None,
                         title: str = 'Model Comparison',
                         save_path: Optional[str] = None):
    """
    Plot predictions from multiple models for comparison.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary with model names as keys and predictions as values
    actual : np.ndarray
        Actual values
    dates : pd.DatetimeIndex, optional
        Date index
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(15, 8))
    
    # Plot actual
    if dates is not None:
        plt.plot(dates, actual, label='Actual', linewidth=2.5, 
                color='black', alpha=0.8)
        x_values = dates
    else:
        plt.plot(actual, label='Actual', linewidth=2.5, 
                color='black', alpha=0.8)
        x_values = range(len(actual))
    
    # Plot predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for i, (model_name, pred) in enumerate(predictions.items()):
        plt.plot(x_values, pred, label=model_name, linewidth=2, 
                alpha=0.7, color=colors[i])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date' if dates is not None else 'Time Step', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = 'Model',
                   save_path: Optional[str] = None):
    """
    Plot residuals (errors) for a model.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Model name for title
    save_path : str, optional
        Path to save plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals over time
    axes[0].plot(residuals, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title(f'{model_name} - Residuals Over Time', fontweight='bold')
    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Residual', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title(f'{model_name} - Residuals Distribution', fontweight='bold')
    axes[1].set_xlabel('Residual', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None):
    """
    Plot comparison of metrics across models.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with model names as keys and metrics dicts as values
    save_path : str, optional
        Path to save plot
    """
    # Prepare data
    model_names = list(metrics.keys())
    metric_names = list(metrics[model_names[0]].keys())
    
    # Create subplots
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(len(model_names))
    width = 0.6
    
    for i, metric_name in enumerate(metric_names):
        values = [metrics[model][metric_name] for model in model_names]
        bars = axes[i].bar(x, values, width, alpha=0.8)
        axes[i].set_xlabel('Model', fontsize=11)
        axes[i].set_ylabel(metric_name, fontsize=11)
        axes[i].set_title(f'{metric_name} Comparison', fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_error_distribution(y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                           save_path: Optional[str] = None):
    """
    Plot error distribution for multiple models.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    predictions : dict
        Dictionary with model names and predictions
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, y_pred in predictions.items():
        errors = np.abs(y_true - y_pred)
        plt.hist(errors, alpha=0.6, label=model_name, bins=50, edgecolor='black')
    
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def create_metrics_table(metrics: Dict[str, Dict[str, float]],
                        save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a metrics comparison table.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with model names and metrics
    save_path : str, optional
        Path to save table as CSV
    
    Returns:
    --------
    pd.DataFrame : Metrics table
    """
    df = pd.DataFrame(metrics).T
    df = df.round(4)
    
    if save_path:
        df.to_csv(save_path)
    
    return df

