"""
pages/9_Cron_Jobs.py
====================
Dashboard 9 — Scheduled Jobs.

Design language: Ops terminal.
Background: #0b0f13  Accent: #22d3ee (cyan)  Font: IBM Plex Mono
Card radius: 4px  Status dots: green / amber / red.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

_REPO_ROOT = Path(__file__).parent.parent
_SYNC_STATE_PATH = _REPO_ROOT / "swing" / "improvement" / "sync_state.json"
_SYNC_LOG_PATH = _REPO_ROOT / "swing" / "improvement" / "sync_log.md"

_ET = ZoneInfo("America/New_York")

_BG = "#0b0f13"
_CARD_BG = "#131920"
_BORDER = "#1e2a38"
_ACCENT = "#22d3ee"
_GREEN = "#22c55e"
_AMBER = "#f59e0b"
_RED = "#ef4444"
_TEXT = "#e2e8f0"
_MUTED = "#64748b"
_FONT = "'IBM Plex Mono', 'Fira Code', monospace"

st.set_page_config(
    page_title="Cron Jobs",
    page_icon="⏱",
    layout="wide",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {{
    font-family: {_FONT};
    background-color: {_BG};
    color: {_TEXT};
}}
.job-card {{
    background: {_CARD_BG};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.job-title {{
    font-size: 15px;
    font-weight: 600;
    color: {_ACCENT};
    margin-bottom: 4px;
}}
.job-desc {{
    font-size: 12px;
    color: {_MUTED};
    margin-bottom: 10px;
}}
.dot-green  {{ color: {_GREEN}; font-size: 18px; line-height: 1; }}
.dot-amber  {{ color: {_AMBER}; font-size: 18px; line-height: 1; }}
.dot-red    {{ color: {_RED};   font-size: 18px; line-height: 1; }}
.kv-label   {{ color: {_MUTED}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
.kv-value   {{ color: {_TEXT};  font-size: 13px; }}
.schedule-chip {{
    display: inline-block;
    background: #0f1e2e;
    border: 1px solid {_ACCENT}44;
    border-radius: 4px;
    color: {_ACCENT};
    font-size: 11px;
    padding: 2px 8px;
    margin-right: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_sync_state() -> dict:
    if not _SYNC_STATE_PATH.exists():
        return {}
    try:
        return json.loads(_SYNC_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _fmt_ts(dt: datetime | None) -> str:
    if dt is None:
        return "Never"
    et = dt.astimezone(_ET)
    return et.strftime("%Y-%m-%d %H:%M ET")


def _staleness_dot(last_run: datetime | None, warn_hours: float, crit_hours: float) -> str:
    if last_run is None:
        return '<span class="dot-red">●</span>'
    age_h = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
    if age_h < warn_hours:
        return '<span class="dot-green">●</span>'
    if age_h < crit_hours:
        return '<span class="dot-amber">●</span>'
    return '<span class="dot-red">●</span>'


def _next_daily(hour: int, minute: int) -> str:
    """Next occurrence of HH:MM ET."""
    now_et = datetime.now(_ET)
    target = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_et >= target:
        target += timedelta(days=1)
    return target.strftime("%Y-%m-%d %H:%M ET")


def _next_intraday_30min() -> str:
    """Next 30-min slot between 09:35 and 15:55 ET."""
    now_et = datetime.now(_ET)
    market_open = now_et.replace(hour=9, minute=35, second=0, microsecond=0)
    market_close = now_et.replace(hour=15, minute=55, second=0, microsecond=0)

    if now_et < market_open:
        return market_open.strftime("%Y-%m-%d %H:%M ET")
    if now_et > market_close:
        next_day = (now_et + timedelta(days=1)).replace(hour=9, minute=35, second=0, microsecond=0)
        return next_day.strftime("%Y-%m-%d %H:%M ET")

    # find next 30-min mark
    elapsed = (now_et - market_open).total_seconds() // 1800
    next_slot = market_open + timedelta(minutes=30 * (elapsed + 1))
    if next_slot > market_close:
        next_day = (now_et + timedelta(days=1)).replace(hour=9, minute=35, second=0, microsecond=0)
        return next_day.strftime("%Y-%m-%d %H:%M ET")
    return next_slot.strftime("%Y-%m-%d %H:%M ET")


def _recent_sync_log_lines(n: int = 5) -> list[str]:
    if not _SYNC_LOG_PATH.exists():
        return []
    try:
        lines = [l.strip() for l in _SYNC_LOG_PATH.read_text().splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _kv(label: str, value: str) -> str:
    return (
        f'<div class="kv-label">{label}</div>'
        f'<div class="kv-value">{value}</div>'
    )


def _chip(text: str) -> str:
    return f'<span class="schedule-chip">{text}</span>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    state = _load_sync_state()

    last_sync_dt = _parse_ts(state.get("last_sync_ts"))
    daily_count = state.get("daily_buy_count", 0) if state.get("buy_date") == datetime.now(_ET).date().isoformat() else 0
    buy_date = state.get("buy_date", "—")

    st.markdown(f"## ⏱ Scheduled Jobs")
    st.markdown(
        f"<small style='color:{_MUTED}'>All times in US/Eastern. "
        f"Market hours: 09:30–16:00 ET Mon–Fri.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # -----------------------------------------------------------------------
    # Job 1: Daily Sync
    # -----------------------------------------------------------------------
    dot = _staleness_dot(last_sync_dt, warn_hours=26, crit_hours=50)
    st.markdown(f"""
<div class="job-card">
  <div class="job-title">{dot} &nbsp;Daily Sync</div>
  <div class="job-desc">Fetches closed Alpaca orders → outcomes.json → rebuilds pattern stats → Discord summary</div>
  <div style="margin-bottom:10px">
    {_chip("4:05 PM ET")} {_chip("Daily")} {_chip("swing.sync.run()")}
  </div>
  <div style="display:flex;gap:40px">
    <div>{_kv("Last run", _fmt_ts(last_sync_dt))}</div>
    <div>{_kv("Next scheduled", _next_daily(16, 5))}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("▶ Run Daily Sync now"):
        if st.button("Run sync.run()", key="run_sync"):
            from swing import sync as swing_sync
            with st.spinner("Syncing Alpaca orders…"):
                result = swing_sync.run()
            if result.errors:
                st.error(f"Errors: {result.errors}")
            else:
                st.success(f"{result.new_outcomes} new outcomes — {result.symbols_matched or 'no new symbols'}")
            st.rerun()

    # -----------------------------------------------------------------------
    # Job 2: Intraday Checks
    # -----------------------------------------------------------------------
    st.markdown(f"""
<div class="job-card">
  <div class="job-title"><span class="dot-green">●</span> &nbsp;Intraday Checks</div>
  <div class="job-desc">Checks live prices against stop / TP levels and current HMM regime for all active positions</div>
  <div style="margin-bottom:10px">
    {_chip("Every 30 min")} {_chip("09:35–15:55 ET")} {_chip("check_stops · check_tps · check_regime")}
  </div>
  <div style="display:flex;gap:40px">
    <div>{_kv("Next slot", _next_intraday_30min())}</div>
    <div>{_kv("Functions", "trader.check_stops / check_tps / check_regime")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("▶ Run Intraday Checks now"):
        if st.button("Run stops + TPs + regime check", key="run_checks"):
            from swing import trader
            with st.spinner("Running checks…"):
                stop_actions = trader.check_stops()
                tp_actions = trader.check_tps()
                regime_actions = trader.check_regime()
            all_actions = stop_actions + tp_actions + regime_actions
            if not all_actions:
                st.info("No actions taken — no active positions triggered.")
            else:
                for a in all_actions:
                    st.write(f"`{a.symbol}` — **{a.action}** — {a.reason}")

    # -----------------------------------------------------------------------
    # Job 3: Auto-Buy
    # -----------------------------------------------------------------------
    dot3 = '<span class="dot-green">●</span>' if daily_count < 3 else '<span class="dot-amber">●</span>'
    st.markdown(f"""
<div class="job-card">
  <div class="job-title">{dot3} &nbsp;Auto-Buy</div>
  <div class="job-desc">Buys all 'watching' entries that clear the win-rate threshold in the current regime (max 3/day)</div>
  <div style="margin-bottom:10px">
    {_chip("09:35 AM ET")} {_chip("Daily")} {_chip("trader.execute_auto_buy()")}
  </div>
  <div style="display:flex;gap:40px">
    <div>{_kv("Today's buys", f"{daily_count} / 3 (as of {buy_date})")}</div>
    <div>{_kv("Next scheduled", _next_daily(9, 35))}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("▶ Run Auto-Buy now"):
        st.warning("This will submit real orders if LIVE_TRADING=true.", icon="⚠️")
        if st.button("Run execute_auto_buy()", key="run_buy"):
            from swing import trader
            with st.spinner("Running auto-buy…"):
                actions = trader.execute_auto_buy()
            if not actions:
                st.info("No watching entries found.")
            else:
                for a in actions:
                    if a.action == "buy":
                        st.success(f"`{a.symbol}` — {a.reason}")
                    else:
                        st.info(f"`{a.symbol}` — skipped: {a.reason}")

    # -----------------------------------------------------------------------
    # Sync log
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### Recent Sync Log")
    log_lines = _recent_sync_log_lines(10)
    if log_lines:
        st.code("\n".join(log_lines), language=None)
    else:
        st.markdown(
            f"<small style='color:{_MUTED}'>No sync history yet — run Daily Sync to populate.</small>",
            unsafe_allow_html=True,
        )


main()
