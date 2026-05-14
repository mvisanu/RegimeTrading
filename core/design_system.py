"""
design_system.py — Single source of truth for all visual tokens and UI helpers
used across the 7 Streamlit dashboards in the RegimeTrading application.

Rules enforced by this module:
- This file never imports from other ``core/`` modules.
- ``REGIME_COLORS`` is defined exactly once here; no dashboard may redefine it.
- ``regime_badge`` and ``metric_card`` return raw HTML strings; callers are
  responsible for passing them to ``st.markdown(html, unsafe_allow_html=True)``.
- ``section_header`` is the only function that imports / calls Streamlit directly.
"""

from __future__ import annotations

import copy
import html as _html
from typing import Any

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------

REGIME_COLORS: dict[str, str] = {
    "Low Vol":     "#10b981",  # emerald-500  — calm market
    "Medium Vol":  "#3b82f6",  # blue-500     — normal activity
    "High Vol":    "#f59e0b",  # amber-500    — elevated risk
    "Extreme Vol": "#ef4444",  # red-500      — crisis conditions
    "Uncertain":   "#64748b",  # slate-500    — insufficient data
}

ACCENT_CYAN: str = "#00d4ff"

# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------


def regime_badge(regime: str, confidence: float, glow: bool = True) -> str:
    """Return an HTML ``<span>`` pill for the given volatility regime.

    Args:
        regime:     One of the keys in ``REGIME_COLORS`` (e.g. ``"Low Vol"``).
                    Falls back to the ``"Uncertain"`` color for unknown keys.
        confidence: Confidence score in the range [0, 1].  Displayed as a
                    rounded integer percentage inside the badge.
        glow:       When ``True`` a CSS ``box-shadow`` in the regime color is
                    applied, producing a subtle glow effect.

    Returns:
        HTML string — a single ``<span>`` element.  Pass it to
        ``st.markdown(html, unsafe_allow_html=True)``.

    Example::

        html = regime_badge("High Vol", confidence=0.82)
        st.markdown(html, unsafe_allow_html=True)
    """
    color = REGIME_COLORS.get(regime, REGIME_COLORS["Uncertain"])
    pct = round(max(0.0, min(1.0, confidence)) * 100)
    label = f"{_html.escape(regime)} {pct}%"

    shadow_css = (
        f"box-shadow: 0 0 8px 2px {color}88, 0 0 2px 1px {color};"
        if glow
        else ""
    )

    style = (
        f"background-color:{color};"
        "color:#ffffff;"
        "font-weight:700;"
        "font-size:0.78rem;"
        "padding:3px 10px;"
        "border-radius:999px;"
        "display:inline-block;"
        "letter-spacing:0.02em;"
        f"{shadow_css}"
    )

    return f'<span style="{style}">{label}</span>'


def metric_card(
    label: str,
    value: str,
    color: str | None = None,
    border_side: str = "left",
) -> str:
    """Return an HTML stat card with a colored accent border.

    Args:
        label:       Short descriptor shown in small muted text above the value.
        value:       Primary metric displayed in larger bold text.
        color:       CSS color for the accent border.  Defaults to
                     ``ACCENT_CYAN`` when ``None``.
        border_side: Which side carries the accent border.  One of
                     ``"left"`` (default), ``"right"``, ``"top"``,
                     ``"bottom"``.

    Returns:
        HTML string — a ``<div>`` card element.  Pass it to
        ``st.markdown(html, unsafe_allow_html=True)``.

    Example::

        html = metric_card("Sharpe Ratio", "1.84", color="#10b981")
        st.markdown(html, unsafe_allow_html=True)
    """
    accent = color if color is not None else ACCENT_CYAN

    valid_sides = {"left", "right", "top", "bottom"}
    side = border_side if border_side in valid_sides else "left"
    border_css = f"border-{side}:3px solid {accent};"

    container_style = (
        "background-color:#1e2130;"
        "border-radius:8px;"
        "padding:14px 18px;"
        "margin:4px 0;"
        f"{border_css}"
        "display:inline-block;"
        "min-width:120px;"
    )

    label_style = (
        "color:#94a3b8;"   # slate-400
        "font-size:0.72rem;"
        "font-weight:500;"
        "text-transform:uppercase;"
        "letter-spacing:0.05em;"
        "margin-bottom:4px;"
        "display:block;"
    )

    value_style = (
        f"color:{accent};"
        "font-size:1.45rem;"
        "font-weight:700;"
        "line-height:1.2;"
        "display:block;"
    )

    return (
        f'<div style="{container_style}">'
        f'<span style="{label_style}">{_html.escape(label)}</span>'
        f'<span style="{value_style}">{_html.escape(value)}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------


def section_header(text: str) -> None:
    """Render a consistently styled section heading via ``st.markdown``.

    This is the *only* function in this module that imports or calls Streamlit.

    Args:
        text: Heading text to display.

    Example::

        section_header("Portfolio Overview")
    """
    import streamlit as st  # local import — keeps module importable without st

    style = (
        "color:#f8fafc;"        # near-white
        "font-size:1.15rem;"
        "font-weight:600;"
        "letter-spacing:0.01em;"
        "margin:1.2rem 0 0.4rem 0;"
        f"border-left:3px solid {ACCENT_CYAN};"
        "padding-left:10px;"
    )

    st.markdown(f'<h3 style="{style}">{_html.escape(text)}</h3>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly layout factory
# ---------------------------------------------------------------------------

_DARK_BG = "#0e1117"
_GRID_COLOR = "#1e2535"
_TEXT_COLOR = "#e2e8f0"
_FONT_FAMILY = "Inter, 'Segoe UI', sans-serif"


def get_plotly_layout(theme: str = "dark") -> dict[str, Any]:
    """Return a base Plotly layout dict suitable for dark-theme dashboards.

    Args:
        theme: Layout theme identifier.  Currently only ``"dark"`` is
               supported; passing any other value returns the same dark
               layout (reserved for future light-theme support).

    Returns:
        A ``dict`` that can be spread into a ``go.Figure`` layout or passed
        directly to ``fig.update_layout(**get_plotly_layout())``.

    Example::

        import plotly.graph_objects as go
        from core.design_system import get_plotly_layout

        fig = go.Figure()
        fig.update_layout(**get_plotly_layout())
    """
    axis_defaults: dict = {
        "gridcolor": _GRID_COLOR,
        "gridwidth": 1,
        "zerolinecolor": _GRID_COLOR,
        "zerolinewidth": 1,
        "tickfont": {"color": _TEXT_COLOR, "family": _FONT_FAMILY, "size": 11},
        "title": {"font": {"color": _TEXT_COLOR, "family": _FONT_FAMILY, "size": 12}},
        "linecolor": _GRID_COLOR,
        "showgrid": True,
    }

    return {
        "paper_bgcolor": _DARK_BG,
        "plot_bgcolor": _DARK_BG,
        "font": {
            "color": _TEXT_COLOR,
            "family": _FONT_FAMILY,
            "size": 12,
        },
        "xaxis": copy.deepcopy(axis_defaults),
        "yaxis": copy.deepcopy(axis_defaults),
        "legend": {
            "bgcolor": "#1e2130",
            "bordercolor": _GRID_COLOR,
            "borderwidth": 1,
            "font": {"color": _TEXT_COLOR, "family": _FONT_FAMILY, "size": 11},
        },
        "margin": {"l": 48, "r": 24, "t": 40, "b": 40},
        "hoverlabel": {
            "bgcolor": "#1e2130",
            "bordercolor": _GRID_COLOR,
            "font": {"color": _TEXT_COLOR, "family": _FONT_FAMILY, "size": 12},
        },
    }
