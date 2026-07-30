import pandas as pd
import numpy as np

class DiscretionaryExecutionMatrix:
    def __init__(self, target_z_entry: float = 2.0, target_z_exit: float = 0.5):
        self.target_z_entry = target_z_entry
        self.target_z_exit = target_z_exit

    def generate_trade_ticket(self, symbol_y: str, symbol_x: str, current_z: float, 
                              hedge_ratio: float, garch_vol: float, capital: float = 100000) -> dict:
        """Generates a structured discretionary decision ticket."""
        
        action = "HOLD / NO TRADE"
        bias = "NEUTRAL"
        confidence_modifier = 1.0
        
        # Volatility Regime Filter
        # High volatility reduces position sizing to protect capital
        if garch_vol > 0.025:
            regime = "HIGH VOLATILITY (Reduce Exposure)"
            confidence_modifier = 0.6
        elif garch_vol < 0.008:
            regime = "LOW VOLATILITY (Mean Reversion Compression)"
            confidence_modifier = 1.2
        else:
            regime = "NORMAL VOLATILITY"
            confidence_modifier = 1.0

        # Entry logic
        if current_z >= self.target_z_entry:
            action = f"SHORT {symbol_y} / LONG {symbol_x}"
            bias = "SPREAD OVERBOUGHT -> Expect Reversion Down"
        elif current_z <= -self.target_z_entry:
            action = f"LONG {symbol_y} / SHORT {symbol_x}"
            bias = "SPREAD OVERSOLD -> Expect Reversion Up"
        elif abs(current_z) <= self.target_z_exit:
            action = "CLOSE POSITION / TAKE PROFIT"
            bias = "SPREAD AT EQUILIBRIUM"

        # Discretionary Position Sizing (Risk-weighted)
        base_allocation = capital * 0.10  # 10% base risk per trade
        adjusted_allocation = base_allocation * confidence_modifier

        return {
            "Pair": f"{symbol_y} / {symbol_x}",
            "Action": action,
            "Bias": bias,
            "Dynamic Z-Score": round(current_z, 2),
            "Hedge Ratio (Beta)": round(hedge_ratio, 4),
            "GARCH Volatility": f"{round(garch_vol * 100, 2)}%",
            "Volatility Regime": regime,
            "Recommended Capital Allocation": f"${adjusted_allocation:,.2f}"
        }
