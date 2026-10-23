# Vol-Adjusted-Spread-Trader

![CI/CD Pipeline](https://github.com/stefan931025-sys/Vol-Adjusted-Spread-Trader/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Hybrid statistical arbitrage pairs trading architecture that dynamically rescales cointegration spreads using time-series GARCH(1,1) volatility forecasting to systematically filter false mean-reversion signals under volatility regime shifts.

---

## Key Features

* **GARCH(1,1) Volatility Filtering:** Dynamic conditional variance estimation ($\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$) to normalize statistical spread z-scores against cluster volatility.
* **Cointegration Analytics:** Engle-Granger two-step cointegration testing for stationary spread detection with rolling hedge ratio derivation via OLS/TLS.
* **Risk & Mean-Reversion Rules:** Half-life of mean-reversion estimation via Ornstein-Uhlenbeck processes to eliminate structural regime breaks.
* **Automated CI/CD Test Suite:** Production unit testing via `pytest` covering stationarity constraints, likelihood maximization, and z-score scaling.

---

## Mathematical Architecture

1. **Spread Dynamics:**
   $$\text{Spread}_t = Y_t - \gamma X_t$$
   *Where $\gamma$ is the cointegration hedge ratio derived from total least squares.*

2. **GARCH(1,1) Conditional Variance Normalization:**
   $$\sigma_t^2 = \omega + \alpha e_{t-1}^2 + \beta \sigma_{t-1}^2$$
   $$Z_t^{\text{adj}} = \frac{\text{Spread}_t - \mu_{\text{spread}}}{\sigma_t}$$

3. **Execution Decision Rules:**
   * **Long Entry:** $Z_t^{\text{adj}} < -2.0$ (and $p$-value $< 0.05$)
   * **Short Entry:** $Z_t^{\text{adj}} > +2.0$ (and $p$-value $< 0.05$)
   * **Exit Target:** $Z_t^{\text{adj}} \rightarrow 0$ or maximum holding period threshold based on half-life $t_{1/2}$.

---

## Repository Structure

```text
Vol-Adjusted-Spread-Trader/
├── .devcontainer/        # Devcontainer configuration for Codespaces
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions automated test workflow
├── src/
│   ├── garch_engine.py   # GARCH(1,1) likelihood and volatility estimator
│   └── main.py           # Main strategy pipeline execution script
├── tests/
│   └── test_garch.py     # Automated pytest suite
├── requirements.txt      # Python dependencies
└── README.md             # System documentation
