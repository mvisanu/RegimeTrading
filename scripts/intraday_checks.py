"""Intraday Checks — called by Task Scheduler every 30 min 9:35–15:55 ET Mon-Fri."""
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

_ET = ZoneInfo("America/New_York")
now_et = datetime.now(_ET)
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

if now_et.weekday() >= 5:
    print(f"[{ts}] Weekend — skipping.")
    sys.exit(0)

open_t = now_et.replace(hour=9, minute=35, second=0, microsecond=0)
close_t = now_et.replace(hour=15, minute=55, second=0, microsecond=0)

if not (open_t <= now_et <= close_t):
    print(f"[{ts}] Outside market hours ({now_et.strftime('%H:%M ET')}) — skipping.")
    sys.exit(0)

from swing import trader

stop_actions = trader.check_stops()
tp_actions = trader.check_tps()
regime_actions = trader.check_regime()

all_actions = stop_actions + tp_actions + regime_actions
if not all_actions:
    print(f"[{ts}] No triggers.")
for a in all_actions:
    print(f"[{ts}] {a.symbol} → {a.action}: {a.reason}")
