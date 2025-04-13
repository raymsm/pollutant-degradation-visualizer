"""
Utility functions for data processing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

def parse_csv_data(file_content: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse CSV data into time and concentration arrays.
    
    Parameters:
    -----------
    file_content : str
        CSV file content as string
        
    Returns:
    --------
    tuple
        (time_points, concentration_points)
    """
    df = pd.read_csv(pd.StringIO(file_content))
    if 'time' not in df.columns or 'concentration' not in df.columns:
        raise ValueError("CSV must contain 'time' and 'concentration' columns")
    return df['time'].values, df['concentration'].values

def parse_text_data(time_text: str, conc_text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse text data into time and concentration arrays.
    
    Parameters:
    -----------
    time_text : str
        Time values as newline-separated string
    conc_text : str
        Concentration values as newline-separated string
        
    Returns:
    --------
    tuple
        (time_points, concentration_points)
    """
    time_points = np.array([float(x) for x in time_text.split()])
    conc_points = np.array([float(x) for x in conc_text.split()])
    return time_points, conc_points

def validate_data(time_points: np.ndarray, conc_points: np.ndarray) -> List[str]:
    """
    Validate time and concentration data.
    
    Parameters:
    -----------
    time_points : array_like
        Time points
    conc_points : array_like
        Concentration points
        
    Returns:
    --------
    list
        List of error messages, empty if validation passes
    """
    errors = []
    
    if len(time_points) != len(conc_points):
        errors.append("Number of time points must match number of concentration points")
    if len(time_points) < 3:
        errors.append("At least 3 data points are required for fitting")
    if any(t < 0 for t in time_points):
        errors.append("Time values must be non-negative")
    if any(c < 0 for c in conc_points):
        errors.append("Concentration values must be non-negative")
    if any(t1 >= t2 for t1, t2 in zip(time_points[:-1], time_points[1:])):
        errors.append("Time values must be strictly increasing")
        
    return errors

def calculate_statistics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """
    Calculate various statistics for model fit.
    
    Parameters:
    -----------
    predicted : array_like
        Predicted values
    actual : array_like
        Actual values
        
    Returns:
    --------
    dict
        Dictionary containing various statistics
    """
    residuals = actual - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((actual - np.mean(actual))**2)
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mse': mse
    } 