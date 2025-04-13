"""
Utility functions for data export.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import io
import base64
import numpy as np

def export_to_csv(time_points: np.ndarray, 
                 conc_points: np.ndarray, 
                 predicted: np.ndarray) -> str:
    """
    Export data to CSV format.
    
    Parameters:
    -----------
    time_points : array_like
        Time points
    conc_points : array_like
        Experimental concentration points
    predicted : array_like
        Predicted concentration points
        
    Returns:
    --------
    str
        CSV content as string
    """
    df = pd.DataFrame({
        'time': time_points,
        'experimental_concentration': conc_points,
        'predicted_concentration': predicted
    })
    return df.to_csv(index=False)

def export_plot_to_png(fig: plt.Figure) -> str:
    """
    Export plot to PNG format.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to export
        
    Returns:
    --------
    str
        Base64 encoded PNG image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def generate_report(results: Dict[str, Any]) -> str:
    """
    Generate a text report of the analysis results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
        
    Returns:
    --------
    str
        Formatted report text
    """
    report = []
    report.append("Pollutant Degradation Kinetics Analysis Report")
    report.append("=" * 50)
    report.append("\nModel Parameters:")
    for param, value in results['parameters'].items():
        report.append(f"{param}: {value:.4f}")
    
    report.append("\nFit Statistics:")
    for stat, value in results['statistics'].items():
        report.append(f"{stat}: {value:.4f}")
    
    return "\n".join(report) 