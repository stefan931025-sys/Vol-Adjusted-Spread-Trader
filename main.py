import yfinance as yf
from tabulate import tabulate
from src.cointegration import CointegrationEngine
from src.garch_engine import DynamicVolEngine
from src.execution_matrix import DiscretionaryExecutionMatrix

def run_pipeline(ticker_y: str, ticker_x: str, start_date: str = "2024-01-01"):
    print(f"\n=======================================================")
    print(f"  VOL-ADJUSTED DISCRETIONARY PAIRS TRADING DESK")
    print(f"  Target Pair: {ticker_y} vs {ticker_x}")
    print(f"=======================================================\n")

    # 1. Fetch Data
    df = yf.download([ticker_y, ticker_x], start=start_date)['Close']
    df = df.dropna()

    # 2. Cointegration & Spread Construction
    coint_engine = CointegrationEngine(df[ticker_y], df[ticker_x])
    beta = coint_engine.calculate_hedge_ratio()
    coint_stats = coint_engine.check_stationarity()

    print(f"[*] Hedge Ratio (Beta): {beta:.4f}")
    print(f"[*] Cointegration ADF p-value: {coint_stats['p_value']:.4f} "
          f"({'PASS' if coint_stats['is_stationary'] else 'FAIL'})\n")

    # 3. Dynamic GARCH Volatility Engine
    vol_engine = DynamicVolEngine(coint_engine.spread)
    vol_df = vol_engine.calculate_vol_adjusted_zscore()

    latest_z = vol_df['dynamic_zscore'].iloc[-1]
    latest_vol = vol_df['garch_vol'].iloc[-1]

    # 4. Discretionary Decision Matrix
    exec_matrix = DiscretionaryExecutionMatrix(target_z_entry=2.0, target_z_exit=0.5)
    ticket = exec_matrix.generate_trade_ticket(ticker_y, ticker_x, latest_z, beta, latest_vol)

    # Output Decision Matrix
    table_data = [[k, v] for k, v in ticket.items()]
    print(tabulate(table_data, headers=["Metric / Parameter", "Discretionary Value"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    # Example: Energy Sector Majors
    run_pipeline("XOM", "CVX")
