"""
Data loading utilities for Kaggle cryptocurrency dataset.
Handles dataset download and loading operations.
"""

import os
import kaggle
from pathlib import Path
import pandas as pd


def download_dataset(dataset_name, output_path='./data/crypto_data', unzip=True):
    """
    Download a dataset from Kaggle.
    
    Parameters:
    -----------
    dataset_name : str
        Kaggle dataset identifier (format: 'username/dataset-name')
    output_path : str
        Path where dataset should be downloaded
    unzip : bool
        Whether to unzip the downloaded files
    
    Returns:
    --------
    str : Path to downloaded dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Download dataset
    print(f"Downloading dataset: {dataset_name}")
    kaggle.api.dataset_download_files(
        dataset_name, 
        path=output_path, 
        unzip=unzip
    )
    
    print(f"Dataset downloaded to: {output_path}")
    return output_path


def list_dataset_files(data_path='./data/crypto_data'):
    """
    List all files in the downloaded dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to dataset directory
    
    Returns:
    --------
    list : List of file names
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    files = [f.name for f in data_dir.iterdir() if f.is_file()]
    return files


def load_dataset(file_path, **kwargs):
    """
    Load a CSV file from the dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    **kwargs : dict
        Additional arguments passed to pd.read_csv
    
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, **kwargs)


def get_dataset_info(data_path='./data/crypto_data'):
    """
    Get basic information about the dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to dataset directory
    
    Returns:
    --------
    dict : Dictionary with dataset information
    """
    files = list_dataset_files(data_path)
    info = {
        'path': data_path,
        'files': files,
        'num_files': len(files)
    }
    
    # Try to get info about CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        info['csv_files'] = csv_files
        # Load first CSV and get basic stats
        first_csv = os.path.join(data_path, csv_files[0])
        df_sample = load_dataset(first_csv, nrows=100)
        info['sample_columns'] = list(df_sample.columns)
        info['sample_shape'] = df_sample.shape
    
    return info


if __name__ == "__main__":
    # Example usage
    dataset_name = 'isaaclopgu/cryptocurrency-historical-prices-top-100-2025'
    
    # Download dataset
    download_dataset(dataset_name)
    
    # List files
    files = list_dataset_files()
    print(f"\nDataset files: {files}")
    
    # Get dataset info
    info = get_dataset_info()
    print(f"\nDataset info: {info}")

