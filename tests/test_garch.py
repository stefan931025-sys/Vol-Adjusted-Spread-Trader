import pytest
import pandas as pd
import numpy as np
from src.garch_engine import DynamicVolEngine


def test_dynamic_vol_engine_initialization():
    """Validates that DynamicVolEngine initializes correctly with a pandas Series."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    spread_data = pd.Series(np.random.normal(0, 1, 100), index=dates)
    
    engine = DynamicVolEngine(spread_data)
    assert isinstance(engine.spread, pd.Series)
    assert len(engine.spread) == 100


def test_forecast_conditional_volatility():
    """Validates that forecast_conditional_volatility returns a non-empty pandas Series."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    spread_data = pd.Series(np.random.normal(0, 1, 100), index=dates)
    
    engine = DynamicVolEngine(spread_data)
    vol_series = engine.forecast_conditional_volatility()
    
    assert isinstance(vol_series, pd.Series)
    assert not vol_series.empty


def test_calculate_vol_adjusted_zscore():
    """Validates that calculate_vol_adjusted_zscore returns the expected columns and structure."""
    dates = pd.date_range(start="2023-01-01", periods=150, freq="B")
    spread_data = pd.Series(np.linspace(10, 15, 150) + np.random.normal(0, 0.2, 150), index=dates)
    
    engine = DynamicVolEngine(spread_data)
    result_df = engine.calculate_vol_adjusted_zscore()
    
    assert isinstance(result_df, pd.DataFrame)
    assert "spread" in result_df.columns
    assert "garch_vol" in result_df.columns
    assert "dynamic_zscore" in result_df.columns
    assert not result_df.empty
