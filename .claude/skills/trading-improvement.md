---
name: trading-improvement
description: Swing trade self-improvement — sync outcomes from Alpaca, warn before adding watchlist entries, promote pattern rules. Trigger on /swing:sync, /swing:warn, /swing:promote, or any watchlist add.
metadata:
  type: skill
---

# Trading Self-Improvement Skill

## Commands

### /swing:sync
Run `swing.sync.run()` to fetch closed positions from Alpaca and update outcomes.

```python
from swing import sync
result = sync.run()
```

Report back:
- How many new outcomes were logged and which symbols
- Whether any pattern×regime cell newly crossed the warning threshold
- A summary table of current `pattern_stats.json`

If Alpaca credentials are missing (ALPACA_KEY_ID or ALPACA_SECRET not in .env), say so clearly — do not silently return 0 outcomes.

---

### /swing:warn SYMBOL PATTERN
Check if a symbol/pattern has poor historical edge in the current regime.

```python
from swing import warn
result = warn.check(symbol="SYMBOL", pattern="PATTERN", confidence=0.6)
print(result.message)
```

Format the result as:
```
[CLEAR|⚠ CAUTION]: SYMBOL / PATTERN
Current regime:  {regime} (confidence {regime_confidence:.0%})
Historical edge: {win_rate:.0%} win rate | avg TP {avg_tp_pct}% | n={n}
Recommendation:  {message}
```

If n < 5: report "Insufficient data (n=X)" — never invent confidence from thin data.

---

### Auto-hook: Watchlist Add
ANY TIME you add a symbol to `swing/watchlist.json`, you MUST:
1. Run `/swing:warn SYMBOL PATTERN` first
2. Show the result to the user
3. If `should_warn = True`, ask for explicit confirmation before writing the entry
4. When writing the entry, include `regime_at_add` from the warn result, `status: "watching"`, `tp_steps_hit: 0`

Never suppress the warning silently. Never add an entry without running the warn check.

**Entry template:**
```json
{
  "symbol": "SYMBOL",
  "pattern": "PATTERN",
  "setup": "dip",
  "added_ts": "YYYY-MM-DDTHH:MM:SS-0400",
  "entry_estimate": 0.0,
  "entry": 0.0,
  "stop": 0.0,
  "tp_ladder": [],
  "confidence": 0.6,
  "source": "swing-cycle",
  "ema8": 0.0,
  "atr20": 0.0,
  "sma200": 0.0,
  "status": "watching",
  "tp_steps_hit": 0,
  "regime_at_add": "{regime from warn result}"
}
```

---

### /swing:promote
Review `swing/improvement/rules.md` and ask the user which rules to promote to `CLAUDE.md`.

For each rule, show:
- The pattern and regime
- The win rate and n
- The date it was flagged

Ask: "Promote this rule to CLAUDE.md? (yes/no)"

Only promote rules the user explicitly approves. Include the source stats in the CLAUDE.md entry so the origin is never lost. Format:

```
## Trading Rule (promoted {date})
- Pattern `{pattern}` in `{regime}` has {win_rate:.0%} win rate (n={n} as of {flagged_date})
- Avoid adding new watchlist entries with this pattern×regime combination
```

Remove promoted entries from `rules.md` after writing to `CLAUDE.md`.

---

## Safety Reminders
- All orders route through `core.broker.AlpacaBroker.submit_order()` — 5 circuit breakers always active
- `LIVE_TRADING=true` in `.env` is required for real money — paper is the default
- Never auto-promote rules to `CLAUDE.md` without user approval per rule
- Auto-buy cap: 3 per calendar day
