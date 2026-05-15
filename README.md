# RegimeTrading

**Regime-aware automated trading system with seven Streamlit dashboards.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

RegimeTrading classifies the market into volatility regimes in real time — Low Vol, Medium Vol, High Vol, Extreme Vol — using a Gaussian HMM with a mathematically proven no-look-ahead guarantee, then scales portfolio exposure accordingly. Five independent circuit breakers protect capital regardless of what the model believes. The system ships as a monorepo: a `core/` package consumed by seven Streamlit dashboards, each with a deliberately distinct design language.

---

## How to Use This App to Buy Stocks

RegimeTrading combines regime-aware signals with live TradingView technical analysis to give you a structured, data-driven process before placing any order.

### Suggested buying workflow

| Step | Tool | What you learn |
|------|------|----------------|
| 1 | **Dashboard 1 — Regime Detection** | Confirms current market regime (Bull / Bear / Uncertain) — determines whether to buy at all |
| 2 | **TradingView screen / top_gainers** | Screens live stocks by momentum, volume, or technicals to build a shortlist |
| 3 | **TradingView combined_analysis** | Multi-timeframe technical analysis on each candidate ticker |
| 4 | **Dashboard 5 — Multi-Asset Backtest** | Walk-forward backtest of the HMM strategy vs buy-and-hold for your ticker |
| 5 | **Dashboard 4 — Portfolio Risk** | Stress-tests the position against 2008 GFC, 2020 COVID crash, 2022 rate shock |
| 6 | **Dashboard 6 — Sentiment** | VADER sentiment on Google News RSS — catches negative news before you commit |
| 7 | **Dashboard 7 — Correlation Breaks** | Detects divergence from correlated assets — early warning of regime shifts |
| 8 | **Alpaca paper trade** | Places order through `broker.py` with all five safety circuit breakers active |

### What each dashboard contributes

| Dashboard | Buying signal it provides |
|-----------|--------------------------|
| **1 — Regime Detection** | Only buy in Low/Medium Vol regimes; avoid Extreme Vol or Uncertain |
| **2 — Monte Carlo** | Probability distribution of future returns — right-sizes position |
| **3 — Sensitivity** | Finds the optimal SMA entry/exit parameters for the ticker |
| **4 — Portfolio Risk** | Max drawdown and concentration risk before adding the position |
| **5 — Multi-Asset Backtest** | Confirms HMM strategy beats buy-and-hold on this specific ticker |
| **6 — Sentiment** | Recency-weighted news sentiment with momentum arrows |
| **7 — Correlation Breaks** | Z-score break alerts across five asset pairs — spot divergence early |

### Safety guardrails
All orders route through five independent circuit breakers in `core/safety.py` — daily loss limit (2%), weekly loss limit (5%), max drawdown (15%), single-asset concentration (25%), and order rate (20 per 60 s). These fire regardless of what the HMM model believes. Paper trading is the default; live orders require `LIVE_TRADING=true` in `.env` plus an explicit in-code confirmation.

---

## What this demonstrates

This project was built to show a coherent set of production-readiness practices in a single codebase:

- **Causal time-series ML** — Forward filtering (not Viterbi) in the HMM layer, verified at every import against prefix-only refits. Look-ahead bias is tested in CI and blocks all further development if it fails.
- **Separation of safety from intelligence** — The five circuit breakers (`core/safety.py`) are pure functions that never touch the HMM. If the model is wrong, safety still fires.
- **Design system discipline** — `REGIME_COLORS` and every UI helper live in exactly one file (`core/design_system.py`). All seven dashboards import from it; none redefine it.
- **Walk-forward validation only** — `core/backtest.py` allows only rolling train/test splits. Full-period in-sample backtests are architecturally excluded.
- **Seven distinct design languages** — Bloomberg terminal, deep-space nebula, minimal Jupyter, premium fintech, asset-colored, newsroom serif, and SOC. CSS is scoped per page; no cross-dashboard token leakage is enforced as a regression.

---

## Architecture

```
                        .env (ALPACA_KEY_ID, ALPACA_SECRET)
                              |
                    +---------+----------+
                    |    core/ package   |
                    |                   |
  OHLCV (yfinance) --> data.py          |
                    |      |            |
                    |  hmm_utils.py ----+---> RegimeResult
                    |   (BIC sweep,     |     (posteriors,
                    |  forward filter,  |      labels,
                    |  stability filter)|      confidence)
                    |      |            |
                    |  verify.py <------+  (import-time self-check,
                    |  LOOKAHEAD_CHECK_PASSED flag)
                    |                   |
                    |  allocation.py    |  target_exposure(regime, conf)
                    |  safety.py        |  5 independent circuit breakers
                    |  broker.py        |  Alpaca paper/live wrapper
                    |  backtest.py      |  walk-forward only
                    |  design_system.py |  REGIME_COLORS, helpers
                    +-------------------+
                              |
              +---------------+----------------+
              |               |                |
     pages/1_Regime   pages/2_Monte    pages/3_Sensitivity
     _Detection.py    _Carlo.py        .py
     (Bloomberg        (Nebula bg,     (IBM Plex,
      terminal)         fan chart)      sweep heatmap)

     pages/4_Portfolio  pages/5_Multi    pages/6_Sentiment
     _Risk.py           _Asset_Backtest  .py
     (Fintech cards,    .py              (Newsreader font,
      stress tests)     (--accent CSS     VADER + RSS)
                        variable)

     pages/7_Correlation_Breaks.py
     (SOC dark, @keyframes pulse)
```

### Data flow (single query)

1. `data.py` fetches OHLCV from yfinance and caches for 3600 s.
2. `hmm_utils.fit_and_filter()` engineers three features (log-return, realized vol, HL range %), sweeps n_components 3–6 by BIC, then runs the forward algorithm to get per-bar posteriors — never Viterbi.
3. `verify.forward_filter()` is the only function that touches the HMM lattice. On import it runs a self-check on synthetic data at 20 sampled time points; `LOOKAHEAD_CHECK_PASSED` is set to `True` or `False` and displayed as a badge on every dashboard.
4. The stability filter holds a new regime label for 3 consecutive bars before promoting it; rolling chop > 4 transitions in 20 bars overrides to `"Uncertain"`.
5. `allocation.target_exposure(regime, confidence)` returns exposure linearly interpolated between the regime baseline and 0.50 neutral as confidence falls.
6. `broker.AlpacaBroker.submit_order()` gates on `safety.check_all()` before every order. Any triggered breaker rejects the order and writes an audit record to `logs/trades.json` atomically.

### Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| ML | hmmlearn 0.3.0 (pinned) | Exposes `_hmmc.forward_log` C extension for causal filtering |
| Numerics | numpy < 2.0 | hmmlearn C-extension ABI compatibility |
| Dashboards | Streamlit multi-page | Sidebar navigation, per-page `st.cache_data` |
| Charts | Plotly | Interactive, dark-theme-friendly |
| Broker | alpaca-py | Paper and live via single `paper=` flag |
| Sentiment | NLTK VADER + feedparser | Zero-dependency NLP; Google News RSS |
| Data | yfinance + pandas | OHLCV with 3600 s cache |
| Env | python-dotenv | `.env` file, `LIVE_TRADING` string-compare gate |

---

## Quickstart

### Prerequisites

- Python 3.11+
- An [Alpaca](https://alpaca.markets) account (paper trading is free)

### Local setup

```bash
# 1. Clone
git clone https://github.com/your-username/RegimeTrading.git
cd RegimeTrading

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .              # makes the core/ package importable as "core"

# 4. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
# Edit .env — fill in ALPACA_KEY_ID and ALPACA_SECRET

# 5. Download VADER lexicon (required for Dashboard 6)
python -m nltk.downloader vader_lexicon

# 6. Run
streamlit run app.py
```

Open `http://localhost:8501` — the seven dashboards appear in the sidebar.

---

## Configuration

All environment variables are loaded from `.env` via `python-dotenv`.

| Variable | Required | Description | Example |
|---|---|---|---|
| `ALPACA_KEY_ID` | Yes | Alpaca API key | `PKXXXXXXXXXXXXXXXX` |
| `ALPACA_SECRET` | Yes | Alpaca API secret | `xxxxxxxxxxxxxxxxxxxx` |
| `ALPACA_BASE_URL` | No | Trading endpoint | `https://paper-api.alpaca.markets` |
| `LIVE_TRADING` | No | Enable live orders (string comparison, must be exactly `"true"`) | `false` |

`LIVE_TRADING=false` is the hardcoded default. Setting it to `true` is necessary but not sufficient for a live order — `submit_order()` also requires `live_confirmed=True` in the call, preventing accidental live execution from automated code paths.

---

## Running tests

```bash
# Gate tests (must pass before anything else)
pytest tests/test_no_lookahead.py    # proves forward filter is causal
pytest tests/test_safety.py          # proves all 5 breakers fire independently

# Full suite (38 tests)
pytest

# Single test with output
pytest tests/test_no_lookahead.py::TestForwardFilterNoLookahead::test_forward_filter_matches_prefix_at_each_t -v -s
```

The no-look-ahead gate works by fitting one HMM, running `forward_filter` on the full sequence, then re-running it on 20 prefix subsequences `X[:t+1]`. The sorted posteriors at position `t` must agree within 1e-6 in both cases. If they do not, `LOOKAHEAD_CHECK_PASSED` is set to `False` and every dashboard surfaces an error badge instead of rendering results.

---

## Dashboards

### 1. Regime Detection — Bloomberg terminal
Dark background, `#00d4ff` cyan accent, monospace numbers. Live HMM regime overlay on a candlestick chart with color-banded regime zones. Look-ahead verification badge in the header.

### 2. Monte Carlo Simulation — Deep space nebula
Background `#060614`, Space Mono + DM Sans. 200 simulation paths rendered as low-alpha overlapping curves; their density creates the glow effect. Drawdown analysis panel. Overfitting warning fires when the observed backtest result lands above the 90th simulated percentile.

### 3. Sensitivity Analysis — Minimal Jupyter
Background `#0f1117`, IBM Plex Mono + IBM Plex Sans, muted green `#22c55e`, no glow effects. SMA crossover parameter sweep across fast_ma, slow_ma, stop_loss, take_profit. Robustness score 0–100 derived from the coefficient of variation across the sweep surface; classified as Robust / Moderate / Fragile.

### 4. Portfolio Risk — Premium fintech
Plus Jakarta Sans + JetBrains Mono, indigo `#6366f1`, gradient metric cards with hover lift. Current regime per position from the live HMM. 60-day rolling correlation heatmap. Historical stress-test drawer covering 2008 GFC, 2020 COVID crash, and 2022 rate shock.

### 5. Multi-Asset Backtester — Asset-colored
A single `--accent` CSS variable re-tints the entire page to the selected asset's color: SPY → cyan, BTC → orange, GLD → gold, TLT → purple. Walk-forward backtest per asset with regime timeline strips stacked for all four assets. Sharpe improvement ranking vs. buy-and-hold baseline.

### 6. Sentiment Analysis — Newsroom serif
Newsreader for headings, `#111318` background, 4 px card radius, editorial near-white accent. Google News RSS parsed via feedparser, scored with NLTK VADER. Semicircular SVG sentiment gauges per ticker. Recency-weighted aggregate score with momentum arrows.

### 7. Correlation Break Detector — SOC dark
Background `#08080c`, Share Tech Mono throughout. Z-score break detection across five asset pairs. Normal-state cards are intentionally bland. `@keyframes pulse` animation activates only on Significant and Extreme alert cards. Alert records append atomically to `logs/alerts.json`.

---

## Project structure

```
RegimeTrading/
├── core/                        # Shared package — imported by all dashboards
│   ├── design_system.py         # REGIME_COLORS, regime_badge(), metric_card(),
│   │                            #   section_header(), get_plotly_layout()
│   ├── hmm_utils.py             # BIC sweep, forward filter, stability filter
│   ├── verify.py                # Causal self-check; LOOKAHEAD_CHECK_PASSED flag
│   ├── allocation.py            # target_exposure(regime, confidence) -> float
│   ├── safety.py                # 5 independent circuit breakers + atomic state
│   ├── broker.py                # AlpacaBroker with safety gate + audit log
│   ├── data.py                  # yfinance OHLCV loader (cache TTL 3600 s)
│   └── backtest.py              # Walk-forward backtester (no in-sample permitted)
├── pages/                       # Streamlit multi-page app (auto-discovered)
│   ├── 1_Regime_Detection.py    # Bloomberg terminal
│   ├── 2_Monte_Carlo.py         # Deep space nebula
│   ├── 3_Sensitivity.py         # Minimal Jupyter
│   ├── 4_Portfolio_Risk.py      # Premium fintech
│   ├── 5_Multi_Asset_Backtest.py# Asset-colored
│   ├── 6_Sentiment.py           # Newsroom serif
│   └── 7_Correlation_Breaks.py  # SOC dark + alert pulses
├── tests/
│   ├── test_no_lookahead.py     # GATE: proves forward filter causality (5 tests)
│   ├── test_safety.py           # GATE: proves all 5 breakers fire (18 tests)
│   └── test_allocation.py       # Parametrized regime/confidence coverage (15 tests)
├── logs/
│   ├── alerts.json              # Correlation break alerts (atomic append)
│   ├── trades.json              # Order audit log (atomic append)
│   └── safety_state.json        # Persisted circuit breaker state
├── app.py                       # Streamlit hub (page_config, landing screen)
├── pyproject.toml               # core/ editable install
├── requirements.txt             # Pinned hmmlearn==0.3.0, numpy<2.0
├── .env.example                 # Environment variable template
└── .streamlit/config.toml       # headless=true, base="dark"
```

---

## Key invariants

These are load-bearing constraints, not style preferences. Breaking any of them introduces a correctness bug:

1. **No look-ahead bias.** `hmm_utils.py` calls `verify.forward_filter()` exclusively. `model.predict()` (Viterbi) is never used anywhere in the codebase. `tests/test_no_lookahead.py` enforces this in CI.
2. **Safety is independent of HMM.** `core/safety.py` imports nothing from `core/hmm_utils.py`. `tests/test_safety.py` exercises all five breakers with no model involved.
3. **`REGIME_COLORS` is defined in exactly one place.** `core/design_system.py`. Dashboards import it; they do not redefine it.
4. **Paper trading is the default.** Live orders require both `LIVE_TRADING=true` in the environment and `live_confirmed=True` in the `submit_order()` call.
5. **CSS is scoped per dashboard.** The seven pages use distinct hex codes, fonts, border radii, and animation rules. No design token from one page may leak to another.

---

## License

MIT. Use freely; trade responsibly.
