# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repo is **spec-only**. The single file `prompt.md` is the complete build specification for a regime-aware automated trading system with seven Streamlit dashboards. No code, tests, dependencies, or `.env.example` exist yet — they are all to be created per the spec.

Treat `prompt.md` as the source of truth for requirements. When in doubt, re-read the relevant section rather than guessing — the spec pins exact hex codes, font names, regime thresholds, and acceptance criteria.

## Architecture (planned)

Monorepo with a single shared `core/` package consumed by seven independent Streamlit dashboards:

- `core/hmm_utils.py` — Gaussian HMM regime detector. **Must use forward filtering** (`_do_forward_log_pass`), never `model.predict()` (Viterbi smooths the full sequence and is look-ahead biased). Sweeps `n_components` 3–6 by BIC. Regimes are labeled by ascending realized vol (`"Low Vol"`, `"Medium Vol"`, ...) — never by returns. Stability filter: regime is "active" only after 3 consecutive bars; >4 label flips in any rolling 20-bar window flags `"Uncertain"`.
- `core/verify.py` — Self-check harness comparing streaming forward-filter output to a refit at each `t` using only `X[:t+1]`. Tolerance 1e-6. Surfaces a ✅ badge on dashboards; failure must block the dashboard from rendering results.
- `core/allocation.py` — Pure `target_exposure(regime, confidence) -> float`. Fixed mapping: Low 95%, Medium 80%, High 60%, Extreme 30%, Uncertain 50%.
- `core/safety.py` — Five circuit breakers (daily 2%, weekly 5%, max DD 15%, position concentration 25%, order rate 20/60s). **Must be independent of the HMM** — if the model is wrong, safety still works. State persists to `logs/safety_state.json`.
- `core/broker.py` — Alpaca wrapper. Defaults to paper; live trading requires both `LIVE_TRADING=true` env var **and** an in-dashboard confirmation. Every `submit_order` calls `safety.check()` first.
- `core/data.py` — yfinance loader. Every caller in a Streamlit page must wrap with `@st.cache_data(ttl=3600)`.
- `core/backtest.py` — Walk-forward only (e.g., 1y train / 6mo test rolling). No full-period in-sample backtests.
- `core/design_system.py` — Single source of `REGIME_COLORS` and the helpers `regime_badge`, `metric_card`, `section_header`, `get_plotly_layout`. **Dashboards import these — never redefine them locally.**

`logs/alerts.json` and `logs/trades.json` must be appended atomically (no truncation on concurrent writes).

## Non-negotiable invariants

These are load-bearing — get them right from day one:

1. **No look-ahead bias anywhere.** Forward filtering for HMM, walk-forward for backtests. `tests/test_no_lookahead.py` enforces this in CI and must pass before anything else is considered done.
2. **Safety is independent of HMM.** `tests/test_safety.py` must prove every breaker fires regardless of regime state.
3. **`REGIME_COLORS` lives in exactly one place** (`core/design_system.py`). Importing it anywhere else is required; redefining it anywhere else is a bug.
4. **Paper trading is the default.** Live orders require `LIVE_TRADING=true` env + dashboard confirmation.
5. **Each dashboard's CSS is scoped to its page.** The seven dashboards have intentionally different design languages (Bloomberg terminal, deep-space nebula, minimal Jupyter, premium fintech, asset-colored, newsroom serif, SOC) — leakage between them is a regression. Fonts, hex codes, radii, and spacing in `prompt.md` are exact, not suggestions.

## Dashboard design languages (quick reference)

The seven dashboards deliberately look different. When working on one, do not reach for tokens from another:

1. **Regime Detection** — Bloomberg terminal, dark + cyan accent (`#00d4ff`), monospace numbers.
2. **Monte Carlo** — Deep space nebula, `#060614` bg, Space Mono / DM Sans, low-alpha overlapping curves create the glow.
3. **Sensitivity** — Clean minimal Jupyter, `#0f1117` bg, IBM Plex Mono/Sans, muted green `#22c55e`, no glow.
4. **Portfolio Risk** — Premium fintech, gradient cards, Plus Jakarta Sans + JetBrains Mono, indigo `#6366f1`, hover lift.
5. **Multi-Asset Backtester** — Asset-colored: a single `--accent` CSS variable swaps to the selected asset's color (SPY cyan, BTC orange, GLD gold, TLT purple) and re-tints the whole page.
6. **Sentiment** — Newsroom serif (Newsreader for headers), `#111318` bg, 4px card radius, editorial near-white accent.
7. **Correlation Break** — SOC-style, `#08080c` bg, Share Tech Mono. Normal cards intentionally bland; only Significant/Extreme alerts pulse via `@keyframes`.

## Commands (to be created)

None of these exist yet — they're what the spec implies should work once built:

```bash
# Install
pip install -r requirements.txt

# Run a dashboard (Streamlit picks up pages from the dashboards/ folder)
streamlit run dashboards/1_regime_detection.py

# Tests — the first two are acceptance gates
pytest tests/test_no_lookahead.py
pytest tests/test_safety.py
pytest                                # all tests
pytest tests/test_no_lookahead.py::test_name -v   # single test
```

Required env vars (in `.env`, loaded via `python-dotenv`): `ALPACA_KEY_ID`, `ALPACA_SECRET`, `ALPACA_BASE_URL` (default paper), `LIVE_TRADING` (must be `true` to allow live orders).

## Build order

The spec prescribes this order — follow it. The first two items block everything else:

1. `core/design_system.py`, `core/data.py`
2. `core/hmm_utils.py` + `core/verify.py` + `tests/test_no_lookahead.py` — **do not proceed until this test passes**
3. `core/allocation.py`, `core/safety.py` + `tests/test_safety.py`
4. `core/broker.py` (paper-only first), `core/backtest.py`
5. Dashboards 1–7 in order. Ship each fully styled before starting the next — styling is not a polish pass.

## Python conventions

Python 3.11+. Type hints on all public functions, docstrings on every module. Drop NaNs before fitting the HMM, not during feature engineering downstream.
