"""
pages/2_Monte_Carlo.py — Deep Space / Nebula Monte Carlo Simulation Dashboard.

Design language: deep space — nearly-black background, 1,000 overlapping equity curves
form a nebula-like glow, Space Mono numbers, DM Sans labels, 16px-radius cards.

Displays:
  - Sidebar: CSV upload OR demo data generation, simulation count slider, run button
  - Metric row (5 cards): Median Final Value, 5th Pct, 95th Pct, P(Loss), P(DD>20%)
  - Optional overfitting warning banner (if original backtest > 90th percentile)
  - Fan chart hero (nebula): 200 rendered curves at opacity 0.03 + median + percentile band
  - Max Drawdown distribution histogram
  - Stats table: 5th / 25th / 50th / 75th / 95th percentile summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Must be the FIRST Streamlit call ---
st.set_page_config(page_title="Monte Carlo", page_icon="🌌", layout="wide")

# ---------------------------------------------------------------------------
# Deep Space / Nebula CSS injection
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

.stApp {
    background: radial-gradient(ellipse at 50% 0%, rgba(77,142,255,0.08) 0%, #060614 60%);
    font-family: 'DM Sans', sans-serif;
}

.stSidebar { background: #0a0a1a !important; }

[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

.metric-card {
    background: #0d0d24;
    border: 1px solid rgba(100,120,255,0.08);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card .number {
    font-family: 'Space Mono', monospace;
    font-size: 40px;
    font-weight: 700;
    color: #4d8eff;
    line-height: 1.1;
}

.metric-card .number.danger {
    color: #ff5252;
}

.metric-card .number.warning {
    color: #f59e0b;
}

.metric-card .number.success {
    color: #00e676;
}

.metric-card .label {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #7a7a9a;
    margin-top: 4px;
}

.overfit-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.4);
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}

.overfit-warning .title {
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    color: #f59e0b;
    margin-bottom: 6px;
}

.overfit-warning .body {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #c9a84c;
    line-height: 1.5;
}

.section-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #7a7a9a;
    letter-spacing: 0.04em;
    margin: 1.2rem 0 0.5rem 0;
    padding-left: 10px;
    border-left: 2px solid rgba(77,142,255,0.4);
}

/* Sidebar elements */
.stButton > button {
    background-color: rgba(77,142,255,0.08);
    border: 1px solid rgba(77,142,255,0.3);
    color: #4d8eff;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    transition: background 0.15s;
}

.stButton > button:hover {
    background-color: rgba(77,142,255,0.18);
    border-color: #4d8eff;
    color: #ffffff;
}

.stSlider [data-baseweb="slider"] {
    color: #4d8eff;
}

/* Table styling */
.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    background: #0d0d24;
    border-radius: 12px;
    overflow: hidden;
}

.stats-table th {
    background: rgba(77,142,255,0.08);
    color: #7a7a9a;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    padding: 10px 16px;
    text-align: right;
    border-bottom: 1px solid rgba(100,120,255,0.08);
}

.stats-table th:first-child { text-align: left; }

.stats-table td {
    color: #c0c0e0;
    padding: 10px 16px;
    text-align: right;
    border-bottom: 1px solid rgba(100,120,255,0.05);
}

.stats-table td:first-child {
    text-align: left;
    color: #4d8eff;
    font-weight: 700;
}

.stats-table tr:last-child td {
    border-bottom: none;
}

.stats-table tr:hover td {
    background: rgba(77,142,255,0.04);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------


def _max_drawdown(equity: np.ndarray) -> float:
    """Compute the maximum drawdown as a positive fraction."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(-dd.min())


def run_monte_carlo(
    trade_returns: list[float],
    n_sims: int = 1000,
    starting_capital: float = 100_000.0,
) -> dict:
    """Run Monte Carlo simulation by shuffling trade order with small noise."""
    rng = np.random.default_rng(seed=None)  # fresh seed each run
    n_trades = len(trade_returns)
    trade_arr = np.array(trade_returns)

    # Original unshuffled equity curve
    orig_equity = starting_capital * np.cumprod(1 + trade_arr)
    orig_max_dd = _max_drawdown(orig_equity)

    all_final: list[float] = []
    all_max_dd: list[float] = []
    all_curves: list[np.ndarray] = []

    for _ in range(n_sims):
        noise = rng.normal(0, 0.003, n_trades)  # ±0.3% noise
        shuffled = rng.permutation(trade_arr) + noise
        equity = starting_capital * np.cumprod(1 + shuffled)
        all_final.append(float(equity[-1]))
        all_max_dd.append(_max_drawdown(equity))
        all_curves.append(equity)

    finals = np.array(all_final)
    max_dds = np.array(all_max_dd)

    # Percentile rank of original backtest vs simulations
    percentile_rank = float(np.mean(finals < orig_equity[-1]))

    return {
        "all_curves": all_curves,
        "all_final": finals,
        "all_max_dd": max_dds,
        "orig_equity": orig_equity,
        "orig_max_dd": orig_max_dd,
        "median_final": float(np.median(finals)),
        "p5_final": float(np.percentile(finals, 5)),
        "p25_final": float(np.percentile(finals, 25)),
        "p75_final": float(np.percentile(finals, 75)),
        "p95_final": float(np.percentile(finals, 95)),
        "p_loss": float(np.mean(finals < starting_capital)),
        "p_dd_20": float(np.mean(max_dds > 0.20)),
        "p_dd_30": float(np.mean(max_dds > 0.30)),
        "percentile_rank": percentile_rank,
        "starting_capital": starting_capital,
        "n_sims": n_sims,
        "n_trades": n_trades,
    }


def generate_demo_trades(n: int = 200, seed: int = 42) -> list[float]:
    """Generate demo trades with slight positive edge."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.004, 0.025, n)  # slight positive edge
    return base.tolist()


def _compute_sharpe(equity: np.ndarray, starting_capital: float) -> float | None:
    """Compute annualised Sharpe from an equity curve (assuming 252 trades/year)."""
    if len(equity) < 2:
        return None
    rets = np.diff(equity) / equity[:-1]
    if np.std(rets) == 0:
        return None
    return float(np.mean(rets) / np.std(rets) * np.sqrt(252))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[list[float] | None, int, bool]:
    """Render sidebar and return (trade_returns_or_None, n_sims, run_clicked)."""
    st.sidebar.markdown(
        '<p style="color:#4d8eff;font-family:\'DM Sans\',sans-serif;'
        'font-size:0.7rem;letter-spacing:0.08em;font-weight:500;'
        'margin-bottom:0.8rem;">MONTE CARLO / INPUTS</p>',
        unsafe_allow_html=True,
    )

    # Data source
    uploaded = st.sidebar.file_uploader(
        "Upload trades CSV",
        type=["csv"],
        help="Expects columns: date, trade_return",
    )

    trade_returns: list[float] | None = None

    if uploaded is not None:
        try:
            df_csv = pd.read_csv(uploaded)
            if "trade_return" not in df_csv.columns:
                st.sidebar.error("CSV must have a 'trade_return' column.")
            else:
                trade_returns = df_csv["trade_return"].dropna().tolist()
                st.sidebar.success(f"{len(trade_returns)} trades loaded from CSV.")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"CSV parse error: {exc}")

    if trade_returns is None:
        use_demo = st.sidebar.button(
            "Generate Demo Data",
            use_container_width=True,
            key="gen_demo",
        )
        if use_demo:
            st.session_state["demo_trades"] = generate_demo_trades(200)
            st.session_state.pop("mc_result", None)

        if "demo_trades" in st.session_state:
            trade_returns = st.session_state["demo_trades"]
            n_loaded = len(trade_returns)
            st.sidebar.markdown(
                f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.75rem;'
                f'color:#00e676;margin-top:0.3rem;">Demo: {n_loaded} synthetic trades loaded</p>',
                unsafe_allow_html=True,
            )

    st.sidebar.markdown(
        '<div style="height:1px;background:rgba(100,120,255,0.12);margin:1rem 0;"></div>',
        unsafe_allow_html=True,
    )

    n_sims = int(
        st.sidebar.slider(
            "Number of simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="n_sims_slider",
        )
    )

    st.sidebar.markdown(
        '<div style="height:1px;background:rgba(100,120,255,0.12);margin:1rem 0;"></div>',
        unsafe_allow_html=True,
    )

    run_clicked = st.sidebar.button(
        "▶ Run Simulation",
        use_container_width=True,
        key="run_sim_btn",
    )

    return trade_returns, n_sims, run_clicked


# ---------------------------------------------------------------------------
# HTML card builders
# ---------------------------------------------------------------------------


def _metric_card_html(
    label: str,
    number: str,
    number_class: str = "",
) -> str:
    """Return a deep-space metric card HTML string."""
    cls = f"number {number_class}".strip()
    return (
        f'<div class="metric-card">'
        f'<div class="{cls}">{number}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )


def _section_title(text: str) -> None:
    st.markdown(f'<p class="section-title">{text}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Metric row
# ---------------------------------------------------------------------------


def render_metric_row(result: dict) -> None:
    """Render the 5-card metric row."""
    starting = result["starting_capital"]
    median = result["median_final"]
    p5 = result["p5_final"]
    p95 = result["p95_final"]
    p_loss = result["p_loss"]
    p_dd_20 = result["p_dd_20"]

    # Color class for P(Loss) — red if > 30%, warning if > 15%
    loss_cls = "danger" if p_loss > 0.30 else ("warning" if p_loss > 0.15 else "success")
    dd_cls = "danger" if p_dd_20 > 0.40 else ("warning" if p_dd_20 > 0.20 else "")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            _metric_card_html("Median Final Value", f"${median:,.0f}"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card_html("5th Percentile", f"${p5:,.0f}", "warning"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card_html("95th Percentile", f"${p95:,.0f}", "success"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _metric_card_html("P(Loss)", f"{p_loss * 100:.1f}%", loss_cls),
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            _metric_card_html("P(DD > 20%)", f"{p_dd_20 * 100:.1f}%", dd_cls),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Overfitting warning banner
# ---------------------------------------------------------------------------


def render_overfit_warning(result: dict) -> None:
    """Show prominent warning if original backtest > 90th percentile of simulations."""
    rank = result["percentile_rank"]
    if rank <= 0.90:
        return

    st.markdown(
        f'<div class="overfit-warning">'
        f'<div class="title">Potential Overfitting Detected</div>'
        f'<div class="body">'
        f"The original backtest outperformed <strong>{rank * 100:.1f}%</strong> of random "
        f"trade-order permutations. Results this extreme (above the 90th percentile) often "
        f"indicate curve-fitting to historical trade sequence rather than genuine edge. "
        f"Treat forward-test results with caution."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Fan chart (the nebula hero)
# ---------------------------------------------------------------------------


def render_fan_chart(result: dict) -> None:
    """Render the nebula fan chart with up to 200 rendered simulation curves."""
    _section_title("Equity Curve Fan Chart — Monte Carlo Nebula")

    all_curves = result["all_curves"]
    orig_equity = result["orig_equity"]
    starting_capital = result["starting_capital"]
    n_trades = result["n_trades"]
    n_sims = result["n_sims"]

    # X-axis: trade number (1-indexed)
    x_trades = list(range(1, n_trades + 1))

    # Compute percentile bands from all simulations
    all_curves_arr = np.array(all_curves)  # shape: (n_sims, n_trades)
    p5_curve = np.percentile(all_curves_arr, 5, axis=0)
    p95_curve = np.percentile(all_curves_arr, 95, axis=0)
    median_curve = np.median(all_curves_arr, axis=0)

    fig = go.Figure()

    # --- Fan: render up to 200 random curves for performance ---
    n_render = min(200, n_sims)
    rng_idx = np.random.default_rng(seed=0)
    indices = rng_idx.choice(n_sims, size=n_render, replace=False)

    for idx in indices:
        fig.add_trace(
            go.Scatter(
                x=x_trades,
                y=all_curves[int(idx)].tolist(),
                mode="lines",
                line={"color": "rgba(77,142,255,0.03)", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --- 5th/95th percentile shaded band ---
    fig.add_trace(
        go.Scatter(
            x=x_trades + x_trades[::-1],
            y=p95_curve.tolist() + p5_curve.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(77,142,255,0.08)",
            line={"color": "rgba(0,0,0,0)"},
            name="5th–95th Pct Band",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # --- Median curve ---
    fig.add_trace(
        go.Scatter(
            x=x_trades,
            y=median_curve.tolist(),
            mode="lines",
            line={"color": "#4d8eff", "width": 2.5},
            name="Median",
            showlegend=True,
            hovertemplate="Trade %{x}<br>Median: $%{y:,.0f}<extra></extra>",
        )
    )

    # --- Original backtest curve ---
    orig_x = list(range(1, len(orig_equity) + 1))
    fig.add_trace(
        go.Scatter(
            x=orig_x,
            y=orig_equity.tolist(),
            mode="lines",
            line={"color": "#00e676", "width": 2, "dash": "dash"},
            name="Original Backtest",
            showlegend=True,
            hovertemplate="Trade %{x}<br>Backtest: $%{y:,.0f}<extra></extra>",
        )
    )

    # --- Starting capital horizontal reference line ---
    fig.add_hline(
        y=starting_capital,
        line_color="rgba(255,255,255,0.2)",
        line_dash="dot",
        line_width=1,
        annotation_text=f"Start ${starting_capital:,.0f}",
        annotation_font_color="rgba(255,255,255,0.4)",
        annotation_font_size=10,
        annotation_position="bottom right",
    )

    fig.update_layout(
        paper_bgcolor="#060614",
        plot_bgcolor="#060614",
        height=500,
        margin={"l": 60, "r": 24, "t": 40, "b": 40},
        font={"color": "#c0c0e0", "family": "'DM Sans', sans-serif", "size": 12},
        xaxis={
            "title": "Trade Number",
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": "#7a7a9a"},
            "titlefont": {"color": "#7a7a9a"},
        },
        yaxis={
            "title": "Portfolio Value ($)",
            "showgrid": False,
            "zeroline": False,
            "tickprefix": "$",
            "tickfont": {"color": "#7a7a9a"},
            "titlefont": {"color": "#7a7a9a"},
        },
        legend={
            "bgcolor": "rgba(13,13,36,0.8)",
            "bordercolor": "rgba(100,120,255,0.15)",
            "borderwidth": 1,
            "font": {"color": "#c0c0e0", "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        hoverlabel={
            "bgcolor": "#0d0d24",
            "bordercolor": "rgba(77,142,255,0.3)",
            "font": {"color": "#c0c0e0", "size": 12},
        },
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Max Drawdown Distribution histogram
# ---------------------------------------------------------------------------


def render_dd_histogram(result: dict) -> None:
    """Render the max drawdown distribution histogram."""
    _section_title("Max Drawdown Distribution Across Simulations")

    all_max_dd = result["all_max_dd"]
    orig_max_dd = result["orig_max_dd"]

    # Build color gradient from blue (#4d8eff) to red (#ff5252)
    # Plotly histogram doesn't support per-bar color natively — we use a single
    # color scale via marker colorscale on a histogram with marker.color set to
    # the bin midpoints by using a Scatter-based workaround.
    # For simplicity: use a single color with the gradient effect via colorscale.

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=(all_max_dd * 100).tolist(),  # convert to %
            nbinsx=40,
            marker=dict(
                color=(all_max_dd * 100).tolist(),
                colorscale=[
                    [0.0, "#4d8eff"],
                    [0.5, "#f59e0b"],
                    [1.0, "#ff5252"],
                ],
                showscale=False,
                line={"width": 0},
            ),
            opacity=0.85,
            name="Max DD Distribution",
            hovertemplate="DD: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        )
    )

    # Original backtest DD vertical line
    fig.add_vline(
        x=orig_max_dd * 100,
        line_color="#00e676",
        line_dash="dash",
        line_width=2,
        annotation_text=f"Backtest DD {orig_max_dd * 100:.1f}%",
        annotation_font_color="#00e676",
        annotation_font_size=11,
        annotation_position="top right",
    )

    fig.update_layout(
        paper_bgcolor="#060614",
        plot_bgcolor="#060614",
        height=250,
        margin={"l": 60, "r": 24, "t": 32, "b": 40},
        font={"color": "#c0c0e0", "family": "'DM Sans', sans-serif", "size": 12},
        xaxis={
            "title": "Max Drawdown (%)",
            "showgrid": False,
            "zeroline": False,
            "ticksuffix": "%",
            "tickfont": {"color": "#7a7a9a"},
            "titlefont": {"color": "#7a7a9a"},
        },
        yaxis={
            "title": "Frequency",
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": "#7a7a9a"},
            "titlefont": {"color": "#7a7a9a"},
        },
        bargap=0.02,
        showlegend=False,
        hoverlabel={
            "bgcolor": "#0d0d24",
            "bordercolor": "rgba(77,142,255,0.3)",
            "font": {"color": "#c0c0e0", "size": 12},
        },
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------


def render_stats_table(result: dict) -> None:
    """Render the percentile stats table."""
    _section_title("Percentile Summary Table")

    all_curves_arr = np.array(result["all_curves"])  # (n_sims, n_trades)
    all_max_dd = result["all_max_dd"]
    starting_capital = result["starting_capital"]
    final_values = result["all_final"]

    percentiles = [5, 25, 50, 75, 95]
    rows = []

    for pct in percentiles:
        final_val = float(np.percentile(final_values, pct))
        max_dd_val = float(np.percentile(all_max_dd, pct))

        # Sharpe: use the curve at the closest percentile index
        idx = int(np.argsort(final_values)[int(len(final_values) * pct / 100)])
        curve = all_curves_arr[idx]
        sharpe = _compute_sharpe(curve, starting_capital)
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "—"

        rows.append(
            {
                "percentile": f"{pct}th",
                "final_value": f"${final_val:,.0f}",
                "sharpe": sharpe_str,
                "max_dd": f"{max_dd_val * 100:.1f}%",
            }
        )

    header = (
        "<thead><tr>"
        "<th>Percentile</th>"
        "<th>Final Value</th>"
        "<th>Sharpe</th>"
        "<th>Max Drawdown</th>"
        "</tr></thead>"
    )

    body_rows = ""
    for r in rows:
        body_rows += (
            f"<tr>"
            f"<td>{r['percentile']}</td>"
            f"<td>{r['final_value']}</td>"
            f"<td>{r['sharpe']}</td>"
            f"<td>{r['max_dd']}</td>"
            f"</tr>"
        )

    table_html = (
        f'<table class="stats-table">{header}<tbody>{body_rows}</tbody></table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------


def render_welcome() -> None:
    """Show instructions when no simulation has been run yet."""
    st.markdown(
        """
<div style="
    border: 1px solid rgba(100,120,255,0.12);
    border-left: 3px solid #4d8eff;
    background: #0d0d24;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-top: 2rem;
    max-width: 720px;
">
  <p style="
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    color: #7a7a9a;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
  ">MONTE CARLO / READY</p>
  <h2 style="
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 1rem 0;
  ">Deep Space Monte Carlo Simulator</h2>
  <p style="font-family:'DM Sans',sans-serif;font-size:0.85rem;color:#94a3b8;line-height:1.7;margin-bottom:0.8rem;">
    Upload a trades CSV or generate demo data, then press
    <span style="color:#4d8eff;font-weight:600;">▶ Run Simulation</span>
    to stress-test your trade sequence across 1,000+ random permutations.
  </p>
  <p style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#7a7a9a;line-height:1.6;margin:0;">
    CSV format:
    <span style="font-family:'Space Mono',monospace;color:#4d8eff;">date, trade_return</span>
    — one row per trade, return as decimal (e.g. 0.015 = 1.5%).
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Page title
    st.markdown(
        '<h1 style="font-family:\'Space Mono\',monospace;font-size:1.6rem;'
        'font-weight:700;color:#e2e8f0;margin-bottom:0.2rem;'
        'letter-spacing:0.02em;">Monte Carlo Simulation</h1>'
        '<p style="font-family:\'DM Sans\',sans-serif;font-size:0.82rem;'
        'color:#7a7a9a;margin-top:0;margin-bottom:1.4rem;">'
        "Trade-sequence stress test — 1,000 shuffled equity paths forming a nebula"
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    trade_returns, n_sims, run_clicked = render_sidebar()

    # Run simulation
    if run_clicked:
        if trade_returns is None or len(trade_returns) < 5:
            st.error(
                "No trade data available. Upload a CSV or generate demo data first."
            )
            st.stop()

        with st.spinner(f"Running {n_sims:,} simulations across {len(trade_returns)} trades…"):
            result = run_monte_carlo(trade_returns, n_sims=n_sims)
            st.session_state["mc_result"] = result

        st.rerun()

    # Render results if available
    if "mc_result" not in st.session_state:
        render_welcome()
        return

    result: dict = st.session_state["mc_result"]

    # Overfitting warning (before metrics, so it's seen first)
    render_overfit_warning(result)

    # Metric row
    render_metric_row(result)

    # Spacer
    st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)

    # Fan chart hero
    render_fan_chart(result)

    # Two-column layout for DD histogram + stats table
    col_hist, col_table = st.columns([1.4, 1])

    with col_hist:
        render_dd_histogram(result)

    with col_table:
        st.markdown('<div style="height:1.6rem;"></div>', unsafe_allow_html=True)
        render_stats_table(result)

    # Simulation metadata footer
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;'
        f'color:#7a7a9a;margin-top:1.2rem;">'
        f"Simulation: {result['n_sims']:,} paths · {result['n_trades']} trades · "
        f"starting capital ${result['starting_capital']:,.0f}"
        f"</p>",
        unsafe_allow_html=True,
    )


main()
