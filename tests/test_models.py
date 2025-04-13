"""
Tests for kinetic models.
"""

import numpy as np
from src.models.first_order import first_order_model, fit_first_order

def test_first_order_model():
    """Test first-order model calculations."""
    # Test parameters
    t = np.array([0, 1, 2, 3])
    k = 0.1
    c0 = 10.0
    
    # Calculate expected values
    expected = c0 * np.exp(-k * t)
    
    # Calculate using model
    result = first_order_model(t, k, c0)
    
    # Assert results match
    np.testing.assert_array_almost_equal(result, expected)

def test_fit_first_order():
    """Test first-order model fitting."""
    # Generate synthetic data
    t = np.linspace(0, 10, 11)
    k_true = 0.2
    c0 = 10.0
    noise = 0.1 * np.random.randn(len(t))
    c = first_order_model(t, k_true, c0) + noise
    
    # Fit model
    k_fit, r2, rmse = fit_first_order(t, c, c0)
    
    # Assert reasonable results
    assert abs(k_fit - k_true) < 0.1  # Within 10% of true value
    assert r2 > 0.9  # Good fit
    assert rmse < 0.2  # Low error 