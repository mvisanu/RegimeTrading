"""
pages/8_Swing_Improvement.py
============================
Dashboard 8 — Swing Trade Self-Improvement.

Design language: Quant terminal.
Background: #0a0a0f  Accent: #f59e0b (amber)  Font: JetBrains Mono
Card radius: 2px  No glow effects.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from swing import stats as swing_stats

_REPO_ROOT = Path(__file__).parent.parent
_OUTCOMES_PATH = _REPO_ROOT / "swing" / "outcomes.json"
_STATS_PATH = _REPO_ROOT / "swing" / "improvement" / "pattern_stats.json"
_SYNC_STATE_PATH = _REPO_ROOT / "swing" / "improvement" / "sync_state.json"

_BG = "#0a0a0f"
_CARD_BG = "#12120f"
_ACCENT = "#f59e0b"
_RED = "#ef4444"
_GREEN = "#22c55e"
_TEXT = "#e2e8f0"
_MUTED = "#64748b"
_FONT = "'JetBrains Mono', 'Fira Code', monospace"

_PATTERNS = ["gap-up", "downtrend-break", "oversold-bounce"]
_REGIMES = ["Low Vol", "Medium Vol", "High Vol", "Extreme Vol", "Uncertain"]

st.set_page_config(
    page_title="Swing Improvement",
    page_icon="📈",
    layout="wide",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {{
    font-family: {_FONT};
    background-color: {_BG};
    color: {_TEXT};
}}
.metric-card {{
    background: {_CARD_BG};
    border: 1px solid #1e1e1a;
    border-radius: 2px;
    padding: 12px 16px;
    margin-bottom: 8px;
}}
.badge-win    {{ background:#14532d; color:{_GREEN}; padding:2px 8px; border-radius:2px; font-size:11px; }}
.badge-partial{{ background:#78350f; color:{_ACCENT}; padding:2px 8px; border-radius:2px; font-size:11px; }}
.badge-break  {{ background:#1e293b; color:{_MUTED};  padding:2px 8px; border-radius:2px; font-size:11px; }}
.badge-loss   {{ background:#7f1d1d; color:{_RED};    padding:2px 8px; border-radius:2px; font-size:11px; }}
</style>
""", unsafe_allow_html=True)


def _load_outcomes() -> list[dict]:
    if not _OUTCOMES_PATH.exists():
        return []
    try:
        return json.loads(_OUTCOMES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _last_sync_ts() -> str:
    if not _SYNC_STATE_PATH.exists():
        return "Never"
    try:
        state = json.loads(_SYNC_STATE_PATH.read_text())
        return state.get("last_sync_ts", "Never")
    except (json.JSONDecodeError, OSError):
        return "Never"


def _data_quality_badge(n: int) -> str:
    if n >= 20:
        return f'<span style="color:{_GREEN}">● Data quality: GOOD ({n} outcomes)</span>'
    if n >= 5:
        return f'<span style="color:{_ACCENT}">● Data quality: LOW ({n} outcomes, need ≥ 20)</span>'
    return f'<span style="color:{_RED}">● Data quality: INSUFFICIENT ({n} outcomes, need ≥ 5)</span>'


def _render_header(outcomes: list[dict]) -> None:
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown(f"## 📈 Swing Self-Improvement")
        st.markdown(
            _data_quality_badge(len(outcomes)),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"<small style='color:{_MUTED}'>Last synced: {_last_sync_ts()}</small>", unsafe_allow_html=True)
    with col3:
        if st.button("⟳ Sync Now", use_container_width=True):
            from swing import sync as swing_sync
            with st.spinner("Syncing Alpaca orders…"):
                result = swing_sync.run()
            if result.errors:
                st.error(f"Sync errors: {result.errors}")
            else:
                st.success(f"{result.new_outcomes} new outcomes logged.")
            st.rerun()


def _render_heatmap(stats: dict) -> None:
    st.markdown(f"### Pattern × Regime Win Rate")
    st.markdown(
        f"<small style='color:{_MUTED}'>Red &lt;40% | Amber 40–60% | Green &gt;60% | Grey = insufficient data (n&lt;5)</small>",
        unsafe_allow_html=True,
    )

    z_vals, text_vals = [], []
    for pattern in _PATTERNS:
        row_z, row_text = [], []
        for regime in _REGIMES:
            cell = stats.get(pattern, {}).get(regime, {})
            n = cell.get("n", 0)
            if n < 5:
                row_z.append(None)
                row_text.append(f"n/a<br>(n={n})")
            else:
                wr = cell["win_rate"]
                row_z.append(wr * 100)
                row_text.append(
                    f"{wr:.0%}<br>n={n} | tp={cell['avg_tp_pct']:.0f}%"
                )
        z_vals.append(row_z)
        text_vals.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=_REGIMES,
        y=_PATTERNS,
        text=text_vals,
        texttemplate="%{text}",
        colorscale=[[0, _RED], [0.4, _ACCENT], [0.6, _ACCENT], [1.0, _GREEN]],
        zmin=0, zmax=100,
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font={"color": _TEXT, "family": _FONT, "size": 12},
        margin={"l": 140, "r": 20, "t": 20, "b": 40},
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_timeline(outcomes: list[dict]) -> None:
    if not outcomes:
        st.info("No outcome data yet. Run a sync once positions close.")
        return

    st.markdown("### Outcome Timeline")
    df = pd.DataFrame(outcomes)
    df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce", utc=True)

    color_map = {"gap-up": _ACCENT, "downtrend-break": "#818cf8", "oversold-bounce": "#34d399"}
    fig = go.Figure()
    for pattern in _PATTERNS:
        sub = df[df["pattern"] == pattern]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["close_ts"],
            y=sub["tp_pct_complete"],
            mode="markers",
            name=pattern,
            marker=dict(color=color_map.get(pattern, _TEXT), size=8, opacity=0.8),
            customdata=sub[["symbol", "regime_at_add", "entry", "close_price", "triggered_by"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Regime: %{customdata[1]}<br>"
                "Entry: $%{customdata[2]:.2f} → $%{customdata[3]:.2f}<br>"
                "TP%%: %{y}%%<br>"
                "Trigger: %{customdata[4]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font={"color": _TEXT, "family": _FONT, "size": 11},
        height=300,
        margin={"l": 48, "r": 24, "t": 20, "b": 40},
        yaxis={"title": {"text": "TP% Complete", "font": {"color": _MUTED}},
               "range": [-5, 105], "tickfont": {"color": _MUTED}},
        xaxis={"tickfont": {"color": _MUTED}},
        legend={"bgcolor": _CARD_BG, "bordercolor": "#1e1e1a",
                "font": {"color": _TEXT}},
    )
    st.plotly_chart(fig, use_container_width=True)


_BADGE_MAP = {
    "full_win": "badge-win",
    "partial_win": "badge-partial",
    "breakeven": "badge-break",
    "loss": "badge-loss",
}


def _render_table(outcomes: list[dict]) -> None:
    if not outcomes:
        return

    st.markdown("### Recent Outcomes (last 30)")
    recent = sorted(outcomes, key=lambda r: r.get("close_ts", ""), reverse=True)[:30]
    rows = []
    for r in recent:
        badge_cls = _BADGE_MAP.get(r.get("outcome", ""), "badge-break")
        badge = f'<span class="{badge_cls}">{r.get("outcome","")}</span>'
        rows.append({
            "Symbol": r.get("symbol", ""),
            "Pattern": r.get("pattern", ""),
            "Regime": r.get("regime_at_add", ""),
            "Entry": f"${r.get('entry', 0):.2f}",
            "Close": f"${r.get('close_price', 0):.2f}",
            "TP%": f"{r.get('tp_pct_complete', 0)}%",
            "P&L%": f"{r.get('pnl_pct', 0):+.2f}%",
            "Outcome": badge,
            "Trigger": r.get("triggered_by", ""),
        })

    df = pd.DataFrame(rows)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


def main() -> None:
    outcomes = _load_outcomes()
    stats = swing_stats.load()

    _render_header(outcomes)
    st.markdown("---")
    _render_heatmap(stats)
    st.markdown("---")
    _render_timeline(outcomes)
    st.markdown("---")
    _render_table(outcomes)


main()
