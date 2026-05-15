"""Daily Sync — called by Task Scheduler at 4:05 PM ET Mon-Fri."""
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from swing import sync

ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
print(f"[{ts}] Starting daily sync…")

result = sync.run()
print(f"[{ts}] Done — new_outcomes={result.new_outcomes} errors={result.errors}")
