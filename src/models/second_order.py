"""
Second-order kinetic model implementation for pollutant degradation.
"""

import numpy as np
from scipy.optimize import curve_fit

def second_order_model(t, k, c0):
    """
    Calculate concentration using second-order kinetic model.
    
    Parameters:
    -----------
    t : array_like
        Time points
    k : float
        Rate constant
    c0 : float
        Initial concentration
        
    Returns:
    --------
    array_like
        Predicted concentrations at time points t
    """
    # Clip k to prevent overflow
    k_clipped = np.clip(k, 0, 1e3)
    denominator = 1 + k_clipped * c0 * t
    # Handle division by zero
    denominator = np.clip(denominator, 1e-10, np.inf)
    return c0 / denominator

def fit_second_order(time_points, conc_points, c0):
    """
    Fit second-order kinetic model to experimental data.
    
    Parameters:
    -----------
    time_points : array_like
        Experimental time points
    conc_points : array_like
        Experimental concentration points
    c0 : float
        Initial concentration
        
    Returns:
    --------
    tuple
        (k, r2, rmse) where:
        - k: fitted rate constant
        - r2: coefficient of determination
        - rmse: root mean square error
    """
    # Define bounds for k
    bounds = (0, 1e3)
    
    # Fit the model
    popt, pcov = curve_fit(lambda t, k: second_order_model(t, k, c0), 
                          time_points, conc_points,
                          bounds=bounds)
    k = popt[0]
    
    # Calculate statistics
    predicted = second_order_model(time_points, k, c0)
    residuals = conc_points - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((conc_points - np.mean(conc_points))**2)
    
    return k, r2, rmse 