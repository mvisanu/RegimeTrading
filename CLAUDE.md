# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: COMPLETE

All 8 core modules, 7 dashboards, and 38 tests are implemented and passing.
`prompt.md` remains the authoritative spec for requirements and exact design tokens.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                         # makes core/ importable everywhere
copy .env.example .env                   # fill in Alpaca credentials
python -m nltk.downloader vader_lexicon  # required for Dashboard 6

# Run
streamlit run app.py

# Tests
pytest tests/test_no_lookahead.py        # gate: must pass
pytest tests/test_safety.py              # gate: must pass
pytest                                   # all 38 tests
pytest tests/test_no_lookahead.py::test_name -v  # single test
```

## Architecture

Monorepo: shared `core/` package consumed by 7 independent Streamlit dashboards in `pages/`.

```
RegimeTrading/
├── core/
│   ├── design_system.py   # REGIME_COLORS + regime_badge/metric_card/section_header/get_plotly_layout
│   ├── data.py            # yfinance loader — callers use @st.cache_data(ttl=3600)
│   ├── hmm_utils.py       # Gaussian HMM, forward filter, BIC sweep, stability filter
│   ├── verify.py          # look-ahead bias check — runs on every import of hmm_utils
│   ├── allocation.py      # target_exposure(regime, confidence) → float
│   ├── safety.py          # 5 circuit breakers, atomic state to logs/safety_state.json
│   ├── broker.py          # Alpaca wrapper, paper default, safety gate on every order
│   └── backtest.py        # walk-forward backtest (1y train / 6mo test)
├── pages/
│   ├── 1_Regime_Detection.py   # Bloomberg terminal — HMM regime timeline
│   ├── 2_Monte_Carlo.py        # Deep space nebula — 200-curve fan chart
│   ├── 3_Sensitivity.py        # Clean minimal — SMA crossover parameter sweep
│   ├── 4_Portfolio_Risk.py     # Premium fintech — gradient cards, stress tests
│   ├── 5_Multi_Asset_Backtest.py  # Asset-colored — CSS --accent per tab
│   ├── 6_Sentiment.py          # Newsroom serif — VADER + Google News RSS
│   └── 7_Correlation_Breaks.py # SOC dark — z-score breaks, pulse animations
├── tests/
│   ├── test_no_lookahead.py    # GATE: causal HMM guarantee
│   ├── test_safety.py          # GATE: breakers independent of HMM
│   └── test_allocation.py
├── logs/                       # alerts.json, trades.json, safety_state.json (atomic writes)
├── app.py                      # Streamlit entry point
├── pyproject.toml
└── requirements.txt
```

## Core module details

- **`hmm_utils.py`** — Uses `_hmmc.forward_log` (C extension), NOT `model.predict()` (Viterbi is look-ahead biased). BIC sweeps n_components 3–6. Regimes labeled by ascending realized vol. Stability filter: 3-bar minimum + rolling chop detection flags "Uncertain". Triggers `verify.py` on every import.
- **`verify.py`** — Compares `forward_filter(model, X)[t]` vs `forward_filter(model, X[:t+1])[-1]` at 20 random time points. Tolerance 1e-6. Sets `LOOKAHEAD_CHECK_PASSED: bool` at module level. Dashboards check this before rendering.
- **`safety.py`** — Daily 2%, weekly 5%, max DD 15%, concentration 25%, rate 20/60s. State persisted via `os.replace()` atomic writes. `reset_state()` available for tests.
- **`broker.py`** — `submit_order()` calls `safety.check_all()` first. `LIVE_TRADING=true` (exact string, case-sensitive) required for live orders. Appends to `logs/trades.json` atomically.

## Non-negotiable invariants

1. **No look-ahead bias.** Forward filtering only. Walk-forward backtests only.
2. **Safety is independent of HMM.** Circuit breakers must fire even if HMM is broken.
3. **`REGIME_COLORS` in exactly one place** — `core/design_system.py`. Import it; never redefine it.
4. **Paper trading is the default.** `LIVE_TRADING=true` env var + dashboard confirmation for live.
5. **Dashboard CSS is scoped to its page.** Seven distinct design languages — no cross-leakage.

## Dashboard design languages

| # | Dashboard | Aesthetic | Key tokens |
|---|-----------|-----------|-----------|
| 1 | Regime Detection | Bloomberg terminal | `#0e1117` bg, `#00d4ff` cyan, monospace |
| 2 | Monte Carlo | Deep space nebula | `#060614` bg, Space Mono / DM Sans, glow curves |
| 3 | Sensitivity | Clean minimal | `#0f1117` bg, IBM Plex, `#22c55e` green, no glow |
| 4 | Portfolio Risk | Premium fintech | `#0e1016` bg, `#6366f1` indigo, Plus Jakarta Sans |
| 5 | Multi-Asset | Asset-colored | `#0c0c14` bg, `--accent` CSS var, Outfit / Fira Code |
| 6 | Sentiment | Newsroom serif | `#111318` bg, Newsreader font, 4px radius |
| 7 | Correlation Break | SOC dark | `#08080c` bg, Share Tech Mono, pulse @keyframes |

## Environment variables

| Var | Default | Notes |
|-----|---------|-------|
| `ALPACA_KEY_ID` | — | Required for broker.py |
| `ALPACA_SECRET` | — | Required for broker.py |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Paper endpoint default |
| `LIVE_TRADING` | `false` | Must be exact string `"true"` to enable live orders |

## Python conventions

Python 3.11+. Type hints on all public functions. Module docstrings on every module. NaN rows dropped before HMM fitting (not mid-stream). `numpy < 2.0` required (hmmlearn C-extension incompatibility with numpy 2.x).
