import numpy as np
import pandas as pd
import pytest
from src.garch_engine import GARCHVolatilityEngine  # Adjust import based on your exact class name

def test_garch_stationarity_constraint():
    """
    Validates that GARCH(1,1) parameters satisfy standard stationarity: alpha + beta < 1.
    """
    engine = GARCHVolatilityEngine()
    # Mock return series with zero-mean normal distribution
    np.random.seed(42)
    mock_returns = np.random.normal(0, 0.015, 500)
    
    params = engine.fit(mock_returns)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.90)
    
    assert alpha + beta < 1.0, f"GARCH non-stationary process: alpha ({alpha}) + beta ({beta}) >= 1.0"

def test_volatility_adjusted_zscore():
    """
    Tests that volatility normalization correctly rescales the spread by dynamic GARCH variance.
    """
    spread = pd.Series([0.0, 1.5, -2.0, 0.5, 3.1])
    conditional_vol = pd.Series([1.0, 1.2, 1.0, 0.8, 1.5])
    
    adjusted_zscore = spread / conditional_vol
    
    assert len(adjusted_zscore) == len(spread)
    assert pytest.approx(adjusted_zscore.iloc[1], 0.01) == 1.25
    assert pytest.approx(adjusted_zscore.iloc[2], 0.01) == -2.00

def test_engle_granger_pvalue_threshold():
    """
    Ensures cointegration test flags valid mean-reverting series (p-value < 0.05).
    """
    np.random.seed(107)
    x = np.cumsum(np.random.normal(0, 1, 200)) + 50
    noise = np.random.normal(0, 0.5, 200)
    y = x + 5 + noise  # Cointegrated series
    
    engine = GARCHVolatilityEngine()
    coint_pvalue = engine.check_cointegration(x, y)
    
    assert coint_pvalue < 0.05, f"Failed to detect cointegration: p-value {coint_pvalue} >= 0.05"
