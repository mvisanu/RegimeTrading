# Automated Trading Bot — Claude Code Build Prompt

Build a regime-aware, safety-gated, broker-connected automated trading system with seven Streamlit dashboards. This document is the complete specification. Implement it as a monorepo where shared logic lives in a `core/` package and each dashboard is its own page that imports from core.

---

## 0. Architecture & Project Layout

```
trading_bot/
├── core/
│   ├── __init__.py
│   ├── hmm_utils.py            # Forward-algorithm HMM, regime labeling, stability filter
│   ├── design_system.py        # Shared tokens: REGIME_COLORS, regime_badge(), metric_card(),
│   │                           # section_header(), get_plotly_layout()
│   ├── allocation.py           # Regime → portfolio exposure mapping
│   ├── safety.py               # Circuit breakers (independent of HMM)
│   ├── broker.py               # Alpaca API wrapper (paper + live, env-gated)
│   ├── data.py                 # yfinance loader with caching
│   ├── backtest.py             # Walk-forward backtester, metrics (Sharpe, MDD, etc.)
│   └── verify.py               # Look-ahead bias verification harness
├── dashboards/
│   ├── 1_regime_detection.py
│   ├── 2_monte_carlo.py
│   ├── 3_sensitivity.py
│   ├── 4_portfolio_risk.py
│   ├── 5_multi_asset_backtester.py
│   ├── 6_sentiment.py
│   └── 7_correlation_break.py
├── logs/                       # alerts.json, trades.json
├── tests/
│   ├── test_no_lookahead.py    # MUST PASS — see Section 1
│   ├── test_safety.py
│   └── test_allocation.py
├── requirements.txt
├── .env.example                # ALPACA_KEY_ID, ALPACA_SECRET, ALPACA_BASE_URL
└── README.md
```

**Dependencies:** `streamlit`, `yfinance`, `hmmlearn`, `pandas`, `numpy`, `plotly`, `scipy`, `nltk` (VADER), `feedparser`, `alpaca-py`, `python-dotenv`.

**Constraints that apply everywhere:**
- Python 3.11+, type hints on all public functions, docstrings on every module.
- No look-ahead bias anywhere. Forward filtering only on HMM. Walk-forward only on backtests.
- Every dashboard imports `REGIME_COLORS` and the design helpers from `core.design_system` — never redefine them locally.
- Cache `yfinance` calls with `@st.cache_data(ttl=3600)`.
- Every dashboard has the same sidebar pattern: inputs at top, "Run" button, status indicator.

---

## 1. Core: `hmm_utils.py` — The Brain (CRITICAL)

This module is the foundation. Multiple dashboards depend on it. Get it right.

### Features to engineer from OHLCV

- `log_return = log(close / close.shift(1))`
- `realized_vol = log_return.rolling(20).std()`
- `hl_range_pct = (high - low) / close`
- Drop all NaNs **before** fitting.

### Model selection
- `hmmlearn.GaussianHMM`, `covariance_type="full"`, `n_iter=200`.
- Sweep `n_components` from 3 to 6, pick the lowest **BIC**.
- Expose chosen `n` to the caller so dashboards can display it.

### Forward algorithm (NON-NEGOTIABLE)

Do **not** use `model.predict()` — Viterbi smooths over the entire sequence and is look-ahead biased. Implement forward filtering:

```python
def forward_filter(model, X):
    """
    Returns posterior P(state_t | obs_1..t) using only data up to time t.
    Shape: (T, n_components).
    """
    log_prob, fwd_lattice = model._do_forward_log_pass(X)
    posteriors = np.exp(fwd_lattice - fwd_lattice.max(axis=1, keepdims=True))
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    return posteriors
```

Regime label at time `t` = `argmax` of the row-`t` posterior. Confidence = the max posterior value.

### Regime labeling
- After fitting, compute mean realized vol per latent state.
- Sort ascending by vol. Label `["Low Vol", "Medium Vol", "High Vol", "Extreme Vol", ...]` — never by returns, always by vol.
- Return a label-mapping dict so all downstream code uses the same ordering.

### Stability filter
- A regime is only "active" after **3 consecutive bars** showing it.
- If the regime label changes more than **4 times in any rolling 20-bar window**, flag the period as `"Uncertain"` and surface this state to the dashboard.

### Verification harness (`core/verify.py`)

On import of `hmm_utils`, run a self-check:

```python
def verify_no_lookahead(model, X):
    """
    Refit/refilter at time t using only X[:t+1] for t in a sample of indices,
    compare to the streaming forward_filter output. They must match within 1e-6.
    Raise AssertionError if not.
    """
```

Surface a green ✅ "Look-ahead bias check passed" badge in every dashboard that uses HMM. If the check fails, the dashboard refuses to render results and shows a red error banner.

`tests/test_no_lookahead.py` must enforce this in CI.

---

## 2. Core: `allocation.py` — Position Sizing

Map regime label → portfolio exposure:

| Regime       | Exposure |
|--------------|----------|
| Low Vol      | 95%      |
| Medium Vol   | 80%      |
| High Vol     | 60%      |
| Extreme Vol  | 30%      |
| Uncertain    | 50%      |

Expose as a pure function `target_exposure(regime: str, confidence: float) -> float`. Optionally scale by confidence (e.g., interpolate toward 50% as confidence → 0). Keep this dead simple and unit-tested.

---

## 3. Core: `safety.py` — Circuit Breakers (Independent of HMM)

Safety **must not depend on the HMM**. If the model is wrong, safety still works.

Implement these breakers, each as an independent check returning `(triggered: bool, reason: str)`:

- **Daily loss limit:** equity drops > 2% in one trading day → halt new entries for the rest of the day.
- **Weekly loss limit:** equity drops > 5% over rolling 5 trading days → halt for 2 days.
- **Max drawdown limit:** equity drops > 15% from peak → halt and require manual reset.
- **Position concentration:** any single position > 25% of portfolio → block adds, force trim.
- **Order rate limit:** > 20 orders in 60 seconds → cool-down for 5 minutes.

Persist breaker state to `logs/safety_state.json` so it survives restarts. Provide `safety.status()` returning a dict the dashboards can render.

---

## 4. Core: `broker.py` — Alpaca Wrapper

- Use `alpaca-py`. Read keys from `.env` via `python-dotenv`.
- Default to **paper trading** (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`). Require an explicit `LIVE_TRADING=true` env var to allow live orders, and even then prompt-confirm in the dashboard.
- Methods: `get_account()`, `get_positions()`, `submit_order()`, `cancel_all()`, `get_clock()`.
- Every `submit_order` must call `safety.check()` first and refuse if any breaker is triggered.
- Log every order attempt (accepted or rejected) to `logs/trades.json`.

---

## 5. Core: `design_system.py` — Shared Tokens

Provide:

```python
REGIME_COLORS = {
    "Low Vol":     "#10b981",  # green
    "Medium Vol":  "#3b82f6",  # blue
    "High Vol":    "#f59e0b",  # amber
    "Extreme Vol": "#ef4444",  # red
    "Uncertain":   "#64748b",  # slate
}
ACCENT_CYAN = "#00d4ff"
```

Plus helper functions:
- `regime_badge(regime, confidence, glow=True)` — HTML pill with regime color and optional glow shadow.
- `metric_card(label, value, color=None, border_side="left")` — standardized stat card.
- `section_header(text)` — consistent section title.
- `get_plotly_layout(theme="dark")` — base Plotly layout dict; dashboards override per their design language.

Each dashboard injects its own page-level CSS on top of this base. Do not let one dashboard's CSS leak into another — scope by Streamlit page.

---

## Dashboard 1 — Regime Detection

**Purpose:** Visualize the HMM brain in action on a single ticker.

**Sidebar inputs:** ticker (default SPY), date range (default last 3y), optional `n_regimes` override (default auto-BIC), "Run Analysis" button.

**Layout:**

1. **Top bar (full width):** ticker (large bold) · regime badge with glow · confidence as large cyan number · stability status ("Stable" / "Uncertain") · number of regimes detected · ✅ no-look-ahead badge.

2. **Hero chart (full width, tall):** Plotly using `get_plotly_layout()`. Price as clean white line (toggle to candlesticks). Vertical background bands colored by regime at 12–15% opacity using `REGIME_COLORS`. Regime transitions appear as background color changes. This is the visual centerpiece — make it ~600px tall.

3. **Regime Statistics:** `section_header("Regime Statistics")` then a grid of `metric_card()`s, one per regime, each showing: regime name with colored left border, mean return, mean volatility, mean volume ratio, % of time in regime.

4. **Confidence timeline (shorter, below):** area chart of max-posterior confidence over time. Fill `ACCENT_CYAN` at 30% opacity.

**Theme:** Bloomberg-terminal feel — dark background, monospace numbers, cyan accent.

---

## Dashboard 2 — Monte Carlo Simulation

**Purpose:** Stress-test a backtest's path dependency.

**Functionality:**
1. CSV upload (`date, trade_return`) **or** "Generate Demo Data" button (~200 trades with slight positive edge).
2. Run **1,000 simulations**: shuffle trade order, add per-iteration noise (±0.3%). Starting capital $100,000. Track equity curve and max drawdown per iteration.
3. Compute: median final value, 5th/95th percentiles, P(loss), P(drawdown > 20%), P(drawdown > 30%), max-DD distribution, percentile rank of original backtest.
4. **Overfitting flag:** if original > 90th percentile of simulations, raise a prominent warning.

**Visual Design — Deep Space / Nebula:**
- Background `#060614`. Cards `#0d0d24`, border `rgba(100,120,255,0.08)`.
- Primary `#4d8eff`, success `#00e676`, danger `#ff5252`.
- Fonts: **Space Mono** for numbers, **DM Sans** for labels.
- Metric numbers 40px bold `#4d8eff`. Labels 12px `#7a7a9a`, **not** uppercase.
- Card radius 16px.
- Page background uses a CSS `radial-gradient` — subtle dark blue glow centered, fading to deep navy at edges.
- The fan chart with 1,000 overlapping curves naturally creates a nebula glow — lean into it. Use low per-line alpha (~0.03) so density emerges from overlap.

---

## Dashboard 3 — Sensitivity Analysis

**Purpose:** Show which parameters are robust vs fragile.

**Functionality:**
1. Built-in demo: SMA crossover on SPY, 5y daily. Parameters:
   - `fast_ma`: base 10, range 5–30, step 1
   - `slow_ma`: base 50, range 20–100, step 5
   - `stop_loss_pct`: base 2.0, range 0.5–5.0, step 0.5
   - `take_profit_pct`: base 4.0, range 1.0–10.0, step 0.5
2. For each parameter, sweep its range holding others at base, run backtest, record total return, Sharpe, max DD, win rate.
3. **Robustness score 0–100** per parameter from coefficient of variation. Overall = mean. Above 70 = Robust, 40–70 = Moderate, below 40 = Fragile.

**Visual Design — Clean Minimal (Jupyter but beautiful):**
- Background `#0f1117`. Cards `#1a1c25`, border `rgba(255,255,255,0.06)`.
- Primary `#22c55e` (muted analytical green). Warning `#f59e0b`. Danger `#ef4444`.
- Fonts: **IBM Plex Mono** for numbers, **IBM Plex Sans** for text.
- Lots of whitespace. No glow. Clean line charts of metric-vs-parameter for each sweep.

---

## Dashboard 4 — Portfolio Risk

**Purpose:** Premium fintech-grade portfolio view with regime overlay, correlations, and historical stress tests.

**Functionality:**
1. **Demo positions** (editable via sidebar table or CSV upload):
   - SPY: 100 sh @ $540 → $558
   - QQQ: 50 sh @ $480 → $495
   - AAPL: 75 sh @ $210 → $218
   - GLD: 40 sh @ $235 → $242
   - TLT: 60 sh @ $88 → $85
2. **Regime overlay:** run HMM per position via `hmm_utils`, show current regime + confidence + days in regime.
3. **Correlation:** 60-day rolling correlation matrix. Flag any pair above 0.85.
4. **Stress test** with hardcoded historical drawdowns:
   - **2008:** SPY −56%, QQQ −54%, AAPL −61%, GLD +21%, TLT +33%
   - **2020 COVID:** SPY −34%, QQQ −28%, AAPL −31%, GLD −3%, TLT +21%
   - **2022:** SPY −25%, QQQ −33%, AAPL −30%, GLD −4%, TLT −31%
5. **Watchlist:** extra tickers monitored with regime + confidence.

**Visual Design — Premium Fintech:**
- Background `#0e1016`. Cards `linear-gradient(135deg, #161923, #1a1e2e)`. Border `rgba(255,255,255,0.04)`.
- Primary `#6366f1` (indigo). Success `#10b981`. Danger `#f43f5e`. Warning `#f59e0b`.
- Fonts: **Plus Jakarta Sans** for text, **JetBrains Mono** for numbers.
- Portfolio value: 48–56px, **font-weight 500** (not 800 — elegant not heavy), subtle indigo text-shadow.
- Labels 12px `#94a3b8`. Card radius 16px. Box shadow `0 4px 24px rgba(0,0,0,0.3)`. Hover lift `translateY(-2px)` with border brightening to `rgba(99,102,241,0.2)`.

**Layout:**
- **Top bar:** portfolio value (huge elegant) · P&L (teal/rose with arrow) · "5 positions · 3 in favorable regime · market open/closed".
- **Left column (60%) Positions:** each as gradient card. Ticker 20px bold. Regime badge as rounded pill with glow. Entry → current with thin arrow between. **P&L bar:** 4px-tall horizontal bar extending green-right or red-left proportional to P&L %.
- **Right column (40%):**
  - **Correlation heatmap:** deep navy → indigo → white scale. Cells > 0.85 get rose border + warning dot. Small monospace numbers in each cell.
  - **Stress test:** three scenario rows, each with name, estimated loss as large colored number, percentage, thin damage bar. Background tints: green if < 10% loss, amber 10–20%, rose > 20%.
  - **Watchlist:** compact rows — ticker, price, regime badge, confidence as thin progress bar in regime color.

---

## Dashboard 5 — Multi-Asset Regime Backtester

**Purpose:** Walk-forward backtest the regime-allocation strategy across multiple assets.

**Functionality:**
1. Default assets: **SPY, BTC-USD, GLD, TLT** (configurable in sidebar).
2. Per asset: download 5y from yfinance → HMM via `hmm_utils` (forward algo) → apply allocation (95% Low Vol, 60% High Vol, etc.) → **walk-forward backtest** (1y train, 6mo test, rolling).
3. **Benchmarks:** buy-and-hold + 200-day SMA trend-following per asset.
4. **Stress test:** isolate performance during 2008 (Sep '08–Mar '09), 2020 (Feb–Apr '20), 2022 (Jan–Oct '22).
5. Rank assets by **Sharpe improvement over buy-and-hold**.

**Visual Design — Asset-Colored (the dashboard's identity shifts per asset):**
- Background `#0c0c14`. Cards `#141420`, border `rgba(255,255,255,0.05)`.
- Fonts: **Outfit** for headings, **Fira Code** for numbers. Card radius 12px.
- **Asset color constants:**
  - SPY `#00d4ff` (cyan)
  - BTC-USD `#f7921a` (bitcoin orange)
  - GLD `#ffd700` (gold)
  - TLT `#a78bfa` (soft purple)
  - Extra tickers cycle a preset palette of distinct colors.
- **The signature trick:** when an asset tab is selected, the entire dashboard's accent shifts to that asset's color — card borders tint, chart lines, glows. Selecting BTC makes everything warm orange; GLD makes it golden. Implement via a single `--accent` CSS variable swapped on tab change.

**Layout:**
- **Top:** asset tabs as rounded pills in each asset's color at 20% opacity. Active tab full color with glow. Under each tab: small summary (return %, Sharpe Δ with arrow).
- **Hero chart:** equity curves — strategy line in selected asset's color (bold), buy-and-hold as muted gray dashed line. Grid lines at 5% opacity in asset color.
- **Regime Timeline Strips (unique to this dashboard):** all assets stacked as horizontal bars, one above the other, each segmented with `REGIME_COLORS`. Asset name labeled left in its assigned color. Bars 20–40px tall. When two assets are in the same regime simultaneously it's immediately visible.
- **Comparison table:** dark styled, asset names in their colors. Sharpe Improvement column green+↑ if positive, red+↓ if negative. Best asset row gets a subtle glow border in its color.
- **Stress test:** three columns (2008, 2020, 2022). Per crisis, mini horizontal bar chart per asset — strategy drawdown in asset color, buy-and-hold as gray bar beside it.

---

## Dashboard 6 — Sentiment Analysis

**Purpose:** Morning intelligence briefing on news sentiment for watched tickers.

**Functionality:**
1. **Data source:** Google News RSS — `https://news.google.com/rss/search?q={ticker}+stock`. No API key needed. Capture title, source, published date, snippet via `feedparser`.
2. **Sentiment:** VADER from `nltk`. Score each article −1 to +1. Aggregate per ticker weighted by recency (last 24h weighted 2×). Compute momentum = recent score − articles older than 3 days.
3. **Key drivers:** top 5 articles per ticker by absolute sentiment score.
4. **Default tickers:** SPY, AAPL, NVDA, TSLA, BTC-USD (configurable).
5. Disclaimer at bottom about automated-analysis limitations.

**Visual Design — Newsroom Briefing:**
- Background `#111318` (slightly lighter dark). Cards `#1a1d25`, border `rgba(255,255,255,0.06)`.
- Primary accent `#e2e8f0` (near-white — editorial, not colored).
- Bullish `#22c55e`. Bearish `#ef4444`. Neutral `#64748b`.
- Fonts: **Newsreader** (serif) for title/header — this is what makes the dashboard immediately feel different. **Inter** for body. **JetBrains Mono** for numbers.
- Card radius **4px** (barely rounded — editorial). Thin 1px `#2a2d35` horizontal rules between sections like newspaper column dividers.

**Layout:**
- **Header:** "Market Sentiment Briefing" in Newsreader, 28px, letter-spacing 4px, near-white. Editorial date format: "Wednesday, May 13, 2026". "X articles analyzed" in small muted text. Thin rule below.
- **Ticker cards (horizontal row):** each shows ticker (24px bold sans), **semicircular sentiment gauge** (red→gray on left, gray→green on right, needle at current score, score number below in matching color), momentum arrow (↑ green / ↓ red), article count. 3px left border in sentiment color.
- **Detail panel** (when ticker selected): section divider with ticker name in serif. Articles as **compact rows, not cards** — thin bottom border between, left dot (4px) in sentiment color, source in muted small caps + date, headline in white 15px regular. Should look like a news feed.
- **Aggregate sentiment bar (bottom):** single horizontal bar centered at 0. Each ticker a colored segment extending left (bearish) or right (bullish). Ticker labels on segments.

---

## Dashboard 7 — Correlation Break Detector

**Purpose:** Quiet monitoring station that screams when correlations rupture.

**Functionality:**
1. **Default pairs:** SPY/QQQ, GLD/TLT, SPY/IWM, BTC-USD/ETH-USD, SPY/EEM (configurable).
2. **Data:** yfinance daily, last 3y. Compute 20-day and 60-day rolling correlations. Compute historical mean and std of the 60-day correlation per pair.
3. **Break detection via z-score** of current 60-day correlation vs its history:
   - `z > -1.5` → **Normal**
   - `z < -1.5` → **Notable** (amber)
   - `z < -2.0` → **Significant** (orange)
   - `z < -2.5` → **Extreme** (red)
4. **Historical context:** when a break is detected, find previous instances with similar z-scores; show asset returns 5/10/20 days after. Include caveat that past does not guarantee future.
5. **Persist alerts** to `logs/alerts.json` (timestamp, pair, z-score, severity).

**Visual Design — Security Operations Center:**
- Background `#08080c` (nearly black — darkest of all dashboards). Cards `#0e0e14`, border `rgba(255,255,255,0.03)` — barely visible.
- Normal accent `#334155` (muted slate — deliberately quiet).
- Notable `#f59e0b`. Significant `#f97316`. Extreme `#ef4444`.
- Font: **Share Tech Mono**. 14px for status, 12px for details.
- **The point:** quiet normal cards contrast with screaming alert cards. Don't over-design normal state — its blandness is the identity.

**Layout:**
- **Main chart (selected pair):** 60-day rolling correlation line in `#94a3b8`. Historical mean as dashed `#334155`. ±2σ threshold as dashed `#ef4444`. Area fill: nothing above threshold (clean), `#ef4444` at 8% below. Historical break periods as background columns of red at 4% opacity. Grid `rgba(255,255,255,0.02)` — barely there.
- **Below:** two small side-by-side charts — 20-day vs 60-day correlation. Helps distinguish short-term breaks (20d only) from sustained (both).
- **Conditional panel** (only renders when a pair is Notable or worse — its mere appearance is the alert): historical context table with previous break dates, z-scores, forward returns colored green/red, average outcomes bold, muted caveat. Card has a left border in the alert color.
- **Alert log (expandable, bottom):** recent alerts from `logs/alerts.json` — timestamp, z-score, severity badge. Compact monospace rows.
- **Alert cards should pulse** with a CSS `@keyframes` animation when severity is Significant or Extreme. Normal cards must not animate at all.

**Sidebar:** pair management, correlation window, z-score threshold, date range, "Check" button.

---

## Acceptance Criteria (must pass before declaring done)

1. `pytest tests/test_no_lookahead.py` passes — proves regime labels at time `t` only depend on `X[:t+1]`.
2. `pytest tests/test_safety.py` passes — proves all five circuit breakers trigger independently of HMM state.
3. Every dashboard renders with no console errors on `streamlit run`.
4. The no-look-ahead ✅ badge appears on every dashboard that uses HMM.
5. Alpaca calls default to paper; live trading requires both `LIVE_TRADING=true` env var **and** an in-dashboard confirmation.
6. `REGIME_COLORS` is defined in **exactly one place** (`core/design_system.py`) and imported everywhere else.
7. Each dashboard's CSS is scoped to its page and matches its specified design language verbatim — fonts, hex codes, radii, spacing.
8. `logs/alerts.json` and `logs/trades.json` are appended atomically (no truncation on concurrent writes).
9. README documents: setup, env vars, how to run each dashboard, the no-look-ahead guarantee, the safety-breaker behavior.

---

## Build Order (suggested)

1. `core/design_system.py`, `core/data.py`
2. `core/hmm_utils.py` + `core/verify.py` + `tests/test_no_lookahead.py` — **block on tests passing**
3. `core/allocation.py`, `core/safety.py` + `tests/test_safety.py`
4. `core/broker.py` (paper-only first), `core/backtest.py`
5. Dashboards in numerical order. Ship each one fully styled before starting the next — don't leave styling as a "polish pass."

Treat the no-look-ahead verification and the safety breakers as the load-bearing parts of the system. Everything else can be iterated on; those two must be correct from day one.