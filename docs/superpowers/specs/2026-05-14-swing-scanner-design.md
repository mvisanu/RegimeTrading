# Swing Scanner — Design Spec

**Date:** 2026-05-14  
**Status:** Approved  

---

## Overview

A regime-aware swing-trade scanner that reads `swing/watchlist.json`, runs HMM regime detection on each symbol, scores and ranks candidates, and surfaces the top 10 on the `app.py` landing page with a per-symbol approve button that places a paper order via Alpaca.

---

## Files

```
swing/
├── watchlist.json          # existing input — list of candidate symbols
└── scanner.py              # scoring + sizing logic (new, pure functions)

app.py                      # existing — add scanner UI section (modified)
```

No new dashboard page. The scanner lives on the landing page so it is the first thing seen on launch.

---

## Data input

`swing/watchlist.json` — array of objects with these fields used by the scanner:

| Field | Type | Used for |
|-------|------|----------|
| `symbol` | str | ticker lookup |
| `entry` | float | R/R calculation, order price reference |
| `stop` | float | R/R denominator, risk per share |
| `tp_ladder` | list[float] | tp1 = tp_ladder[0] for R/R numerator |
| `confidence` | float 0–1 | watchlist confidence weight |
| `atr20` | float | ATR-based position sizing |
| `added_ts` | ISO datetime str | recency score |

Symbols with missing `tp_ladder` (empty list) receive `rr_score = 0.0` and will not appear in the top 10 unless all other candidates also have no tp ladder.

---

## Scoring formula

```
regime_score = {
    "Low Vol":    1.00,
    "Medium Vol": 0.75,
    "Uncertain":  0.50,
    "High Vol":   SKIP,       # hard gate — excluded from ranking
    "Extreme Vol":SKIP,       # hard gate — excluded from ranking
}

rr_score      = min((tp1 - entry) / (entry - stop), 3.0) / 3.0   # cap at 3:1 R/R
recency_score = 1.0 / (1.0 + days_since_added)                    # today = 1.0, 1 day ago ≈ 0.5

final_score = regime_score × watchlist_confidence × rr_score × recency_score
```

Regime and confidence come from `hmm_utils.fit_and_filter()` on 1-year daily OHLCV fetched via `core/data.py`. The `current_regime` and `current_confidence` are the last bar's values from `RegimeResult.stable_labels[-1]` and `RegimeResult.confidence[-1]`.

---

## Position sizing (ATR-based, 1% risk per trade)

```
risk_dollars = account_equity × 0.01
shares       = max(1, floor(risk_dollars / atr20))
estimated_cost = shares × entry
```

`account_equity` is fetched live from `AlpacaBroker.get_account()["equity"]` at scan time. If the broker is unavailable, sizing falls back to a $10,000 placeholder and displays a warning.

---

## `swing/scanner.py` — public functions

```python
def load_watchlist(path: str | Path) -> list[dict]:
    """Load and validate watchlist JSON. Returns list of symbol dicts."""

def scan(watchlist: list[dict], top_n: int = 10) -> list[ScanResult]:
    """
    For each symbol: fetch OHLCV, run HMM, apply hard gate, score.
    Returns top_n ScanResult objects sorted by final_score descending.
    Skips symbols that fail data fetch or HMM fit (logs warning, continues).
    """

@dataclass
class ScanResult:
    symbol: str
    regime: str
    regime_confidence: float
    watchlist_confidence: float
    rr_score: float
    recency_score: float
    final_score: float
    shares: int
    estimated_cost: float
    entry: float
    stop: float
    tp1: float | None
```

All functions are pure / side-effect-free. No Streamlit imports. No broker calls. Testable in isolation.

---

## `app.py` — scanner UI section

The landing page gains a **"Swing Scanner"** section below the existing intro text.

### Layout

```
[ Run Scan ]   (button, triggers scan on click)

─── Top 10 Buy Candidates ──────────────────────────────────────
 # | Symbol | Regime     | Score | Shares | Est. Cost | Stop  | TP1   | Action
 1 | F      | Low Vol    | 0.61  |   72   |  $979     | $11.80| $13.97| [Buy F]
 2 | PLTR   | Medium Vol | 0.44  |    3   |  $397     | $128.76|$134  | [Buy PLTR]
...
10 | SNAP   | Uncertain  | 0.09  |   31   |  $166     | $5.19 | —     | [Buy SNAP]
────────────────────────────────────────────────────────────────
```

- **"Run Scan"** button triggers `scanner.scan()` and stores results in `st.session_state["scan_results"]`. A spinner is shown during the scan.
- Results table is rendered with `st.dataframe` for the read-only columns, plus one `st.button(f"Buy {symbol}")` per row rendered alongside.
- Clicking **"Buy {symbol}"** calls `AlpacaBroker.submit_order(symbol, qty=shares, side="buy")` for that symbol only. Shows a success toast (`st.success`) or error message (`st.error`).
- After a buy is placed, the row button changes to a disabled "Ordered ✓" state via `st.session_state`.
- Symbols skipped by the hard gate are shown in a collapsed expander ("Skipped — bearish regime") so the user can see what was excluded.

### Session state keys

| Key | Type | Purpose |
|-----|------|---------|
| `scan_results` | list[ScanResult] | last scan output |
| `scan_running` | bool | spinner guard |
| `ordered` | set[str] | symbols already bought this session |
| `account_equity` | float | cached from broker at scan time |

---

## Error handling

| Failure | Behaviour |
|---------|-----------|
| Symbol data fetch fails | Skip symbol, add to warnings list shown after scan |
| HMM fit fails (all covariance types) | Skip symbol, same warning list |
| Broker unavailable at scan | Use $10,000 placeholder equity, show orange warning |
| Safety breaker fires on buy | `st.error` with breaker name, order not placed |
| `tp_ladder` empty | `rr_score = 0.0`, symbol stays in pool but scores low |

---

## Safety

All orders route through `broker.AlpacaBroker.submit_order()` which calls `safety.check_all()` internally. No additional safety logic required in the scanner. Circuit breakers remain independent of HMM as per the non-negotiable invariants.

Paper trading is the default (`LIVE_TRADING` env var not set to `"true"`).

---

## Testing

New tests in `tests/test_scanner.py`:

- `test_load_watchlist_valid` — loads fixture JSON, returns correct count
- `test_load_watchlist_missing_file` — raises FileNotFoundError
- `test_score_low_vol_beats_medium_vol` — same confidence/rr/recency, Low Vol scores higher
- `test_hard_gate_excludes_high_vol` — High Vol symbol absent from results
- `test_rr_score_capped_at_3` — R/R > 3.0 treated as 3.0
- `test_position_sizing_atr` — shares = floor(equity * 0.01 / atr20)
- `test_scan_skips_on_data_error` — symbol with no data is skipped, others still returned

---

## Non-negotiable invariants (unchanged)

1. `scanner.py` never calls `model.predict()` — only `hmm_utils.fit_and_filter()` which uses forward filtering.
2. `scanner.py` has zero imports from Streamlit — pure logic only.
3. `REGIME_COLORS` is not redefined — scanner uses `core.design_system.REGIME_COLORS` for any colour lookups.
4. Paper trading default — no `live_confirmed=True` passed from scanner.
