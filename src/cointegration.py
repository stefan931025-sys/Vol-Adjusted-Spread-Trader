import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class CointegrationEngine:
    def __init__(self, asset_y: pd.Series, asset_x: pd.Series):
        self.y = asset_y
        self.x = asset_x
        self.beta = None
        self.alpha = None
        self.spread = None

    def calculate_hedge_ratio(self) -> float:
        """Runs OLS regression: Y = alpha + beta * X"""
        X = sm.add_constant(self.x)
        model = sm.OLS(self.y, X).fit()
        self.alpha = model.params.iloc[0]
        self.beta = model.params.iloc[1]
        self.spread = self.y - (self.beta * self.x)
        return self.beta

    def check_stationarity(self, p_value_threshold: float = 0.05) -> dict:
        """Tests residual spread for stationarity via ADF test."""
        if self.spread is None:
            self.calculate_hedge_ratio()
        
        adf_result = adfuller(self.spread.dropna())
        p_value = adf_result[1]
        
        return {
            "adf_statistic": adf_result[0],
            "p_value": p_value,
            "is_stationary": p_value < p_value_threshold
        }
