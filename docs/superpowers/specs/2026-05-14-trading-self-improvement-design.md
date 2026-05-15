# Trading Self-Improvement Skill — Design Spec

**Date:** 2026-05-14  
**Status:** Approved  
**Scope:** `swing/` package, `pages/8_Swing_Improvement.py`, `.claude/skills/trading-improvement.md`

---

## 1. Problem Statement

The existing system generates watchlist entries (`swing/watchlist.json`) with pattern labels, confidence scores, entry/stop/TP levels, and technical indicators — but records no outcome data. There is no feedback loop: a `gap-up` setup that fails 80% of the time in `High Vol` regimes looks identical to one that succeeds 80% of the time in `Low Vol`. This skill closes that loop by:

1. Auto-detecting closed positions from Alpaca and logging structured outcomes
2. Aggregating win rates by pattern × regime
3. Warning before adding a new watchlist entry if historical edge is poor in the current regime
4. Auto-executing buys, partial sells, stop-outs, and regime-driven exits

---

## 2. Architecture Overview

```
swing/
├── __init__.py              # package exports
├── watchlist.json           # existing — gains `status` field
├── outcomes.json            # NEW — one record per closed position
└── improvement/
    ├── pattern_stats.json   # NEW — aggregated win rates by pattern × regime
    ├── sync_state.json      # NEW — last_sync_ts for incremental Alpaca fetch
    ├── sync_log.md          # NEW — append-only human-readable sync history
    └── rules.md             # NEW — audit trail of promoted pattern warnings

swing/
├── sync.py                  # Alpaca poller → outcomes.json
├── stats.py                 # outcomes.json → pattern_stats.json
├── trader.py                # auto-buy / auto-sell executor
└── warn.py                  # regime-aware advisory check

pages/
└── 8_Swing_Improvement.py  # Dashboard 8 — quant terminal

.claude/skills/
└── trading-improvement.md  # Claude skill: /swing:sync, /swing:warn
```

All file writes use the atomic `os.replace()` pattern already established in `core/safety.py` and `core/broker.py`.

---

## 3. Data Schema

### 3.1 `swing/outcomes.json`

A JSON array. One record appended per closed position. Never mutated — append only.

```json
{
  "id": "OUT-20260514-001",
  "symbol": "PLTR",
  "pattern": "gap-up",
  "confidence": 0.6,
  "entry": 132.18,
  "stop": 128.76,
  "tp_ladder": [135.0, 140.0, 145.0],
  "added_ts": "2026-05-14T11:06:02-0400",
  "regime_at_add": "Low Vol",
  "close_ts": "2026-05-15T16:00:00Z",
  "close_price": 137.5,
  "tp_steps_hit": 1,
  "tp_steps_total": 3,
  "tp_pct_complete": 33,
  "pnl_pct": 4.02,
  "outcome": "partial_win",
  "triggered_by": "tp1_auto"
}
```

**`outcome` values:**
| Value | Condition |
|---|---|
| `full_win` | `tp_pct_complete == 100` |
| `partial_win` | `tp_pct_complete` in {33, 66} |
| `breakeven` | Closed above entry, no TP hit |
| `loss` | Stop hit |

**`triggered_by` values:** `tp1_auto`, `tp2_auto`, `stop_auto`, `regime_exit`, `manual`

### 3.2 `swing/improvement/pattern_stats.json`

Rebuilt from scratch on every sync. Never manually edited.

```json
{
  "gap-up": {
    "Low Vol":     {"n": 12, "avg_tp_pct": 61, "avg_pnl_pct": 4.1,  "win_rate": 0.75},
    "Medium Vol":  {"n":  8, "avg_tp_pct": 44, "avg_pnl_pct": 2.3,  "win_rate": 0.63},
    "High Vol":    {"n":  5, "avg_tp_pct": 18, "avg_pnl_pct": -1.2, "win_rate": 0.20},
    "Extreme Vol": {"n":  2, "avg_tp_pct":  0, "avg_pnl_pct": -3.8, "win_rate": 0.00},
    "Uncertain":   {"n":  3, "avg_tp_pct": 22, "avg_pnl_pct": 0.4,  "win_rate": 0.33}
  },
  "downtrend-break": { ... },
  "oversold-bounce": { ... },
  "_meta": {
    "last_rebuilt": "2026-05-14T16:05:00Z",
    "total_outcomes": 47
  }
}
```

**Minimum sample threshold:** `n >= 5` required before a cell drives warnings or auto-buy suppression. Cells with `n < 5` are treated as "insufficient data."

### 3.3 `swing/watchlist.json` — new `status` field

Existing entries gain one new field:

```json
{
  "symbol": "PLTR",
  "status": "watching",
  ...
}
```

**`status` lifecycle:** `watching` → `active` (position opened) → `closed` (fully exited)

Entries also gain two tracking fields written by `trader.py`:
- `regime_at_add: str` — current HMM regime recorded at the moment Claude adds the entry (written by the skill's auto-hook, not sync.py)
- `tp_steps_hit: int` — number of TP ladder levels reached so far (0–3); updated by `trader.check_tps()`

Only `sync.py` and `trader.py` write these fields. Humans can write any value. All writes are atomic.

---

## 4. Module Specifications

### 4.1 `swing/sync.py`

**Public API:** `run() -> SyncResult`

**Behaviour:**
1. Load `sync_state.json` to get `last_sync_ts` (defaults to 30 days ago on first run)
2. Call Alpaca `get_orders(status="closed", after=last_sync_ts)`
3. For each closed order, find matching `watchlist.json` entry by symbol
4. Compute `tp_steps_hit` by counting how many `tp_ladder` levels `close_price` has reached or exceeded
5. Compute `tp_pct_complete = [0, 33, 66, 100][tp_steps_hit]` (index into fixed ladder; avoids floating-point rounding errors)
6. Compute `pnl_pct = (close_price - entry) / entry * 100`
7. Determine `regime_at_add` from watchlist entry (stored at add time — not re-computed)
8. Append outcome record to `outcomes.json` atomically
9. Update `watchlist.json` entry `status` to `closed` if fully exited
10. Call `stats.rebuild()`
11. Update `last_sync_ts` in `sync_state.json`
12. Append one-line summary to `sync_log.md`

**No HMM calls** — sync uses the `regime_at_add` that was recorded when the watchlist entry was created.

### 4.2 `swing/stats.py`

**Public API:**
- `rebuild(outcomes_path, stats_path) -> None`
- `load(stats_path) -> dict`

**Behaviour of `rebuild`:**
1. Read `outcomes.json`
2. Group records by `(pattern, regime_at_add)`
3. For each group compute: `n`, `avg_tp_pct`, `avg_pnl_pct`, `win_rate` (fraction where `outcome` is `full_win` or `partial_win` — `breakeven` and `loss` both count as non-wins)
4. Write `pattern_stats.json` atomically
5. If any cell newly crosses warning threshold (`win_rate < 0.40` with `n >= 5`), append entry to `rules.md`

### 4.3 `swing/warn.py`

**Public API:** `check(symbol: str, pattern: str, confidence: float) -> WarnResult`

```python
@dataclass
class WarnResult:
    symbol: str
    pattern: str
    regime: str
    regime_confidence: float
    win_rate: float | None     # None if n < 5
    avg_tp_pct: float | None
    n: int
    should_warn: bool
    message: str
```

**Behaviour:**
1. Load SPY OHLCV via `core.data` (90 days, cached)
2. Run `core.hmm_utils.fit_and_filter()` on SPY to get current regime + confidence
3. Load `pattern_stats.json` via `stats.load()`
4. Look up `pattern_stats[pattern][regime]`
5. Set `should_warn = True` when `n >= 5` and `win_rate < 0.40`
6. Return `WarnResult` with human-readable `message`

**Never warns** when `n < 5` — returns informational message noting insufficient data.

### 4.4 `swing/trader.py`

**Public API:**
- `check_stops(positions: list[dict]) -> list[TradeAction]`
- `check_tps(positions: list[dict]) -> list[TradeAction]`
- `check_regime() -> list[TradeAction]`
- `execute_auto_buy(entry: dict) -> TradeAction`

All actions route through `core.broker.AlpacaBroker.submit_order()`. Circuit breakers in `core/safety.py` fire on every call regardless of trigger source.

**Stop trigger (`check_stops`):**
- Fetch live prices from Alpaca for all `active` watchlist entries
- If `live_price <= stop`: submit market-sell for full position, set `status = "closed"`, log outcome as `loss`

**TP1 trigger (`check_tps`):**
- If `live_price >= tp_ladder[0]` and `tp_steps_hit == 0`: submit market-sell for 33% of position, update entry `tp_steps_hit = 1`
- If `live_price >= tp_ladder[1]` and `tp_steps_hit == 1`: sell another 33%, update `tp_steps_hit = 2`
- If `live_price >= tp_ladder[2]` and `tp_steps_hit == 2`: sell remainder, set `status = "closed"`

**Regime exit (`check_regime`):**
- Run HMM on SPY (same as `warn.py`)
- If regime is `Extreme Vol` or `Uncertain`: market-sell all `active` positions, set all to `closed`, log outcome as `regime_exit`

**Auto-buy (`execute_auto_buy`):**
- Called at 9:35 AM ET for all `watching` entries added since last market open
- Runs `warn.check()` first — if `should_warn = True`, skips buy and logs skip reason
- If clear: submits limit buy at `entry` price, sets `status = "active"`
- Maximum 3 auto-buys per calendar day (hard cap, tracked in `sync_state.json`)

**Hard constraints:**
- `LIVE_TRADING=true` in `.env` required for real orders — paper mode silently if absent
- Never adds new entries to `watchlist.json` — only humans and Claude add entries
- Never overrides circuit breakers

---

## 5. Dashboard 8 — Swing Improvement

**File:** `pages/8_Swing_Improvement.py`  
**Design language:** Quant terminal — `#0a0a0f` background, `#f59e0b` amber accent, JetBrains Mono throughout, 2px card radius, no glow effects.

**Three sections:**

### 5.1 Pattern × Regime Heatmap
- 3 rows (patterns) × 5 columns (regimes)
- Cell color: red (`win_rate < 0.40`) → amber (0.40–0.60) → green (> 0.60)
- Cell text: `win_rate %` large, `n={n}` and `avg_tp={avg_tp_pct}%` small
- Cells with `n < 5`: hatched grey, text "n/a (n={n})"

### 5.2 Outcome Timeline
- Plotly scatter: x = `close_ts`, y = `tp_pct_complete`, color = pattern
- Hover: symbol, regime, entry, close price, triggered_by
- Lets user spot when a pattern stopped working after a regime shift

### 5.3 Recent Outcomes Table
- Last 30 closed positions, sorted by `close_ts` descending
- Columns: symbol, pattern, regime, entry, close, TP%, P&L%, outcome badge, triggered_by
- Outcome badge colors match `REGIME_COLORS` convention: green=win, amber=partial, red=loss

**Header controls:**
- "Last synced: {timestamp}" 
- "Sync Now" button → calls `swing.sync.run()` inline
- Data quality badge: green ≥ 20 outcomes, amber 5–19, red < 5

**No HMM** — this dashboard reads pre-computed files only. No `verify.py` check needed.

---

## 6. Claude Skill

**File:** `.claude/skills/trading-improvement.md`

### Commands

**`/swing:sync`**
Invokes `swing.sync.run()`. Reports:
- How many new outcomes were logged and which symbols
- Whether any cells newly crossed the warning threshold
- Current `pattern_stats.json` summary table

**`/swing:warn SYMBOL PATTERN`**
Invokes `swing.warn.check()`. Presents:
```
⚠ PLTR / gap-up — CAUTION
Current regime:  High Vol (confidence 0.81)
Historical edge: 20% win rate, avg 18% TP ladder (n=5)
Recommendation:  Poor edge in this regime. Consider
                 waiting for Low/Medium Vol or sizing ≤ 0.5×.
```
When `n < 5`: `"Insufficient data for gap-up in High Vol (n=2)."`

**Auto-hook on watchlist adds:**
Any time Claude adds a symbol to `watchlist.json`, it runs `/swing:warn` first and presents the result. If `should_warn = True`, Claude asks for explicit confirmation before writing the entry. Claude never suppresses the warning silently.

**`/swing:promote`**
Reviews `swing/improvement/rules.md` and asks the user which rules to promote to `CLAUDE.md` as permanent project constraints. Never auto-promotes — requires explicit user approval for each rule. Includes the source stats (n, win_rate, date) in the promotion entry so the origin is never lost.

---

## 7. Scheduling

**Daily at 4:05 PM ET (market close):**
1. `sync.run()` — fetch all closed orders, rebuild stats
2. `trader.check_stops()` — catch any stops that fired at/after close
3. `trader.check_tps()` — catch TP hits at/after close
4. `trader.check_regime()` — regime-driven exits if needed
5. Append to `sync_log.md`

**Intraday every 30 minutes (9:35 AM – 3:55 PM ET):**
Steps 2–4 only (stop/TP/regime checks). No Alpaca order history fetch — price checks only.

**Daily at 9:35 AM ET (market open + 5 min):**
`trader.execute_auto_buy()` for all `watching` entries added since last market open.

**Manual anytime:** `/swing:sync` via Claude skill.

---

## 8. Security Constraints

- All auto-orders route through `core.broker.submit_order()` — 5 circuit breakers always active
- `LIVE_TRADING=true` required for real money — schedule defaults to paper silently
- Schedule never adds watchlist entries — only humans and Claude (via explicit add request) do
- Auto-buy hard cap: 3 per calendar day
- `rules.md` is append-only and includes full source stats — no silent rule injection
- No external data is written to `CLAUDE.md` without explicit `/swing:promote` + user approval per rule
- `swing/improvement/` files are never read by the HMM or `verify.py` — no prompt injection surface into the causal ML layer
