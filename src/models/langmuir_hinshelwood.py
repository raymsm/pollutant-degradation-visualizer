"""
Langmuir-Hinshelwood kinetic model implementation for pollutant degradation.
"""

import numpy as np
from scipy.optimize import curve_fit

def langmuir_hinshelwood_model(t, k, K, c0):
    """
    Calculate concentration using Langmuir-Hinshelwood kinetic model.
    
    Parameters:
    -----------
    t : array_like
        Time points
    k : float
        Rate constant
    K : float
        Adsorption equilibrium constant
    c0 : float
        Initial concentration
        
    Returns:
    --------
    array_like
        Predicted concentrations at time points t
    """
    # Clip parameters to prevent overflow
    k_clipped = np.clip(k, 0, 1e3)
    K_clipped = np.clip(K, 0, 1e3)
    
    # Calculate the model
    term1 = K_clipped * c0
    term2 = np.log(c0) - np.log(c0 * np.exp(-k_clipped * t))
    return (term1 * term2 + c0 - c0 * np.exp(-k_clipped * t)) / (1 + K_clipped * c0 * np.exp(-k_clipped * t))

def fit_langmuir_hinshelwood(time_points, conc_points, c0):
    """
    Fit Langmuir-Hinshelwood kinetic model to experimental data.
    
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
        (k, K, r2, rmse) where:
        - k: fitted rate constant
        - K: fitted adsorption equilibrium constant
        - r2: coefficient of determination
        - rmse: root mean square error
    """
    # Define bounds for parameters
    bounds = ([0, 0], [1e3, 1e3])
    
    # Initial guess for parameters
    p0 = [0.1, 0.1]
    
    # Fit the model
    popt, pcov = curve_fit(lambda t, k, K: langmuir_hinshelwood_model(t, k, K, c0), 
                          time_points, conc_points,
                          p0=p0, bounds=bounds)
    k, K = popt
    
    # Calculate statistics
    predicted = langmuir_hinshelwood_model(time_points, k, K, c0)
    residuals = conc_points - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((conc_points - np.mean(conc_points))**2)
    
    return k, K, r2, rmse 