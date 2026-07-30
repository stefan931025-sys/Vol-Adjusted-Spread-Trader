import pandas as pd
import numpy as np
from arch import arch_model

class DynamicVolEngine:
    def __init__(self, spread: pd.Series):
        self.spread = spread

    def forecast_conditional_volatility(self) -> pd.Series:
        """Fits GARCH(1,1) model on spread log-changes to forecast daily conditional vol."""
        spread_diff = self.spread.diff().dropna() * 100  # Rescaled for numerical stability
        
        # Fit GARCH(1,1)
        garch = arch_model(spread_diff, vol='Garch', p=1, q=1, dist='Normal')
        res = garch.fit(disp='off')
        
        # Extract conditional volatility and rescale back
        cond_vol = res.conditional_volatility / 100
        return cond_vol.reindex(self.spread.index).bfill()

    def calculate_vol_adjusted_zscore(self) -> pd.DataFrame:
        """Calculates dynamic Z-score scaled by dynamic GARCH volatility."""
        cond_vol = self.forecast_conditional_volatility()
        mean_spread = self.spread.rolling(window=60).mean()
        
        # Dynamic Z-Score
        dynamic_z = (self.spread - mean_spread) / cond_vol
        
        df = pd.DataFrame({
            'spread': self.spread,
            'garch_vol': cond_vol,
            'dynamic_zscore': dynamic_z
        })
        return df.dropna()
