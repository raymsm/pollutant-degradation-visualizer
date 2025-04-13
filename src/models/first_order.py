"""
First-order kinetic model implementation for pollutant degradation.
"""

import numpy as np
from scipy.optimize import curve_fit

def first_order_model(t, k, c0):
    """
    Calculate concentration using first-order kinetic model.
    
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
    # Clip k to prevent overflow in exp(-k*t)
    k_clipped = np.clip(k, 0, 1e3)  # Limit k to reasonable values
    # Calculate exponential term safely
    exp_term = np.exp(-k_clipped * t)
    # Handle any remaining overflow by clipping to reasonable range
    exp_term = np.clip(exp_term, 0, 1e100)
    return c0 * exp_term

def fit_first_order(time_points, conc_points, c0):
    """
    Fit first-order kinetic model to experimental data.
    
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
    # Define bounds for k to prevent overflow
    bounds = (0, 1e3)  # k must be between 0 and 1000
    
    # Fit the model with bounds
    popt, pcov = curve_fit(lambda t, k: first_order_model(t, k, c0), 
                          time_points, conc_points,
                          bounds=bounds)
    k = popt[0]
    
    # Calculate statistics
    predicted = first_order_model(time_points, k, c0)
    residuals = conc_points - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((conc_points - np.mean(conc_points))**2)
    
    return k, r2, rmse 