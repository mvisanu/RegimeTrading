# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: COMPLETE + SELF-IMPROVEMENT LAYER

All 8 core modules, 8 dashboards, `swing/` self-improvement package, and 349 tests passing.
`prompt.md` remains the authoritative spec for original requirements and design tokens.

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
pytest                                   # all 349 tests
pytest tests/test_swing_stats.py -v      # swing package tests
pytest tests/test_no_lookahead.py::test_name -v  # single test
```

## Architecture

Monorepo: shared `core/` package + `swing/` self-improvement package consumed by 8 independent Streamlit dashboards in `pages/`.

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
├── swing/
│   ├── watchlist.json     # 141 swing setups (status/tp_steps_hit/regime_at_add fields)
│   ├── outcomes.json      # append-only closed-position log (one record per exit)
│   ├── stats.py           # rebuild() + load() — aggregate outcomes → pattern_stats.json
│   ├── warn.py            # check() — HMM regime + win-rate advisory (WarnResult)
│   ├── sync.py            # run() — Alpaca closed orders → outcomes.json (idempotent)
│   ├── trader.py          # check_stops/tps/regime + execute_auto_buy (via AlpacaBroker)
│   └── improvement/
│       ├── pattern_stats.json  # win rate by pattern × regime (rebuilt each sync)
│       ├── sync_state.json     # last_sync_ts + daily_buy_count
│       ├── sync_log.md         # append-only sync history
│       └── rules.md            # auto-appended when cell crosses win_rate < 40% (n ≥ 5)
├── pages/
│   ├── 1_Regime_Detection.py   # Bloomberg terminal — HMM regime timeline
│   ├── 2_Monte_Carlo.py        # Deep space nebula — 200-curve fan chart
│   ├── 3_Sensitivity.py        # Clean minimal — SMA crossover parameter sweep
│   ├── 4_Portfolio_Risk.py     # Premium fintech — gradient cards, stress tests
│   ├── 5_Multi_Asset_Backtest.py  # Asset-colored — CSS --accent per tab
│   ├── 6_Sentiment.py          # Newsroom serif — VADER + Google News RSS
│   ├── 7_Correlation_Breaks.py # SOC dark — z-score breaks, pulse animations
│   └── 8_Swing_Improvement.py  # Quant terminal — pattern×regime heatmap, outcomes
├── .claude/skills/
│   └── trading-improvement.md  # /swing:sync, /swing:warn, /swing:promote, watchlist auto-hook
├── tests/
│   ├── test_no_lookahead.py    # GATE: causal HMM guarantee
│   ├── test_safety.py          # GATE: breakers independent of HMM
│   ├── test_allocation.py
│   ├── test_swing_stats.py     # 8 tests — pattern×regime aggregation
│   ├── test_swing_warn.py      # 5 tests — regime-aware advisory
│   ├── test_swing_sync.py      # 6 tests — Alpaca poller (mocked)
│   └── test_swing_trader.py    # 7 tests — auto-execution (mocked)
├── logs/                       # alerts.json, trades.json, safety_state.json (atomic writes)
├── app.py                      # Streamlit entry point
├── pyproject.toml              # includes both core* and swing* editable packages
└── requirements.txt
```

## Core module details

- **`hmm_utils.py`** — Uses `_hmmc.forward_log` (C extension), NOT `model.predict()` (Viterbi is look-ahead biased). BIC sweeps n_components 3–6. Regimes labeled by ascending realized vol. Stability filter: 3-bar minimum + rolling chop detection flags "Uncertain". Triggers `verify.py` on every import.
- **`verify.py`** — Compares `forward_filter(model, X)[t]` vs `forward_filter(model, X[:t+1])[-1]` at 20 random time points. Tolerance 1e-6. Sets `LOOKAHEAD_CHECK_PASSED: bool` at module level. Dashboards check this before rendering.
- **`safety.py`** — Daily 2%, weekly 5%, max DD 15%, concentration 25%, rate 20/60s. State persisted via `os.replace()` atomic writes. `reset_state()` available for tests.
- **`broker.py`** — `submit_order()` calls `safety.check_all()` first. `LIVE_TRADING=true` (exact string, case-sensitive) required for live orders. Appends to `logs/trades.json` atomically.

## Swing self-improvement module details

- **`swing/stats.py`** — `rebuild(outcomes_path, stats_path)` groups outcomes by `(pattern, regime_at_add)`, computes `n / win_rate / avg_tp_pct / avg_pnl_pct`. Win rate counts only `full_win` + `partial_win`; `breakeven` and `loss` are non-wins. Appends to `rules.md` when a cell newly crosses below `win_rate < 0.40` with `n ≥ 5`. All writes atomic via `os.replace()`.
- **`swing/warn.py`** — `check(symbol, pattern, confidence) → WarnResult`. Runs HMM on 90 days of SPY via `core.data.load_ohlcv` + `core.hmm_utils.fit_and_filter`. Returns `should_warn=True` when `n ≥ 5` and `win_rate < 0.40`. When `n < 5` always returns `should_warn=False` with "Insufficient data" message — never warns on thin data.
- **`swing/sync.py`** — `run() → SyncResult`. Fetches Alpaca closed orders since `last_sync_ts`, matches against watchlist by symbol, builds outcome records with `tp_pct_complete = [0, 33, 66, 100][tp_steps_hit]`. Idempotent via record_id deduplication. Calls `stats.rebuild()` after writing. All paths are module-level variables (monkeypatchable).
- **`swing/trader.py`** — `check_stops/check_tps/check_regime/execute_auto_buy`. All orders via `AlpacaBroker().submit_order(..., live_confirmed=True)` — circuit breakers always fire. `_EXIT_REGIMES = {"Extreme Vol", "Uncertain"}`. `_MAX_DAILY_BUYS = 3` (tracked in `sync_state.json`). `execute_auto_buy` runs `swing_warn.check()` before every buy and skips if `should_warn=True`.

## Claude skill commands

- **`/swing:sync`** — fetch closed Alpaca orders, rebuild stats, report new outcomes + threshold crossings
- **`/swing:warn SYMBOL PATTERN`** — check historical edge in current HMM regime
- **`/swing:promote`** — user-approved promotion of `rules.md` entries to `CLAUDE.md`
- **Auto-hook** — ANY TIME Claude adds an entry to `swing/watchlist.json`, it MUST run `/swing:warn` first, show the result, and ask for confirmation if `should_warn=True`. New entries always include `status: "watching"`, `tp_steps_hit: 0`, `regime_at_add` from the warn result.

## Non-negotiable invariants

1. **No look-ahead bias.** Forward filtering only. Walk-forward backtests only.
2. **Safety is independent of HMM.** Circuit breakers must fire even if HMM is broken.
3. **`REGIME_COLORS` in exactly one place** — `core/design_system.py`. Import it; never redefine it.
4. **Paper trading is the default.** `LIVE_TRADING=true` env var + `live_confirmed=True` in `submit_order()` for live.
5. **Dashboard CSS is scoped to its page.** Eight distinct design languages — no cross-leakage.
6. **Swing warn hook is mandatory.** Every watchlist add must run `/swing:warn` first. Never suppress the warning silently. If `should_warn=True`, require explicit user confirmation.
7. **`rules.md` is append-only.** Never edit or delete entries. Entries are promoted to `CLAUDE.md` only via `/swing:promote` with explicit per-rule user approval.

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
| 8 | Swing Improvement | Quant terminal | `#0a0a0f` bg, `#f59e0b` amber, JetBrains Mono, 2px radius |

## Environment variables

| Var | Default | Notes |
|-----|---------|-------|
| `ALPACA_KEY_ID` | — | Required for broker.py |
| `ALPACA_SECRET` | — | Required for broker.py |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | Paper endpoint default |
| `LIVE_TRADING` | `false` | Must be exact string `"true"` to enable live orders |

## Python conventions

Python 3.11+. Type hints on all public functions. Module docstrings on every module. NaN rows dropped before HMM fitting (not mid-stream). `numpy < 2.0` required (hmmlearn C-extension incompatibility with numpy 2.x).
