"""
Shared helpers for dashboard tests.

Provides:
- make_st_mock()      — build a reusable streamlit mock
- load_dashboard(n)   — load pages/N_*.py as a Python module
- make_ohlcv_df(n)    — deterministic OHLCV DataFrame for testing
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent


def make_st_mock() -> types.ModuleType:
    """Return a minimal streamlit mock suitable for importing dashboard modules."""
    st = types.ModuleType("streamlit")

    def _noop(*a, **kw):
        return None

    def _cache_data(ttl=None, **kwargs):
        def _deco(fn):
            return fn
        return _deco

    st.set_page_config = _noop
    st.markdown = _noop
    st.write = _noop
    st.cache_data = _cache_data
    st.cache_resource = lambda fn: fn
    st.session_state = {}
    st.spinner = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
    st.stop = _noop
    st.error = _noop
    st.warning = _noop
    st.info = _noop
    st.success = _noop
    st.rerun = _noop
    st.caption = _noop
    st.expander = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
    st.container = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
    def _make_col_mock():
        m = MagicMock()
        m.__enter__ = lambda s, *a: s
        m.__exit__ = lambda s, *a: None
        m.markdown = _noop
        m.write = _noop
        m.metric = _noop
        m.plotly_chart = _noop
        m.error = _noop
        m.warning = _noop
        m.info = _noop
        m.caption = _noop
        m.success = _noop
        m.button = MagicMock(return_value=False)
        m.checkbox = MagicMock(return_value=True)
        m.selectbox = MagicMock(return_value=None)
        m.number_input = MagicMock(return_value=4)
        m.slider = MagicMock(return_value=1000)
        m.text_input = MagicMock(return_value="")
        m.text_area = MagicMock(return_value="")
        return m

    def _columns(spec):
        """Return exactly as many column mocks as requested."""
        if isinstance(spec, int):
            n = spec
        else:
            n = len(spec)
        return [_make_col_mock() for _ in range(n)]

    def _tabs(labels):
        return [_make_col_mock() for _ in labels]

    st.columns = _columns
    st.tabs = _tabs
    st.radio = MagicMock(return_value="Line")
    st.selectbox = MagicMock(return_value="SPY/QQQ")
    st.plotly_chart = _noop
    st.text_area = MagicMock(return_value="ticker,shares,entry,current\nSPY,100,540,558")
    st.text_input = MagicMock(return_value="SPY")
    st.file_uploader = MagicMock(return_value=None)
    st.button = MagicMock(return_value=False)
    st.number_input = MagicMock(return_value=4)
    st.checkbox = MagicMock(return_value=True)
    st.slider = MagicMock(return_value=1000)
    st.date_input = MagicMock(return_value=None)

    class _Sidebar:
        """Sidebar mock that supports `with st.sidebar:` context manager."""
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    sidebar = _Sidebar()
    sidebar.markdown = _noop
    sidebar.text_input = MagicMock(return_value="SPY")
    sidebar.text_area = MagicMock(return_value="SPY/QQQ\nGLD/TLT")
    sidebar.date_input = MagicMock(return_value=None)
    sidebar.number_input = MagicMock(return_value=4)
    sidebar.checkbox = MagicMock(return_value=True)
    sidebar.button = MagicMock(return_value=False)
    sidebar.slider = MagicMock(return_value=1000)
    sidebar.file_uploader = MagicMock(return_value=None)
    sidebar.error = _noop
    sidebar.success = _noop
    sidebar.warning = _noop
    sidebar.caption = _noop
    sidebar.info = _noop
    sidebar.selectbox = MagicMock(return_value="SPY/QQQ")
    sidebar.radio = MagicMock(return_value="Line")
    sidebar.write = _noop
    st.sidebar = sidebar

    return st


_DASHBOARD_PATTERNS = {
    1: "1_Regime_Detection.py",
    2: "2_Monte_Carlo.py",
    3: "3_Sensitivity.py",
    4: "4_Portfolio_Risk.py",
    5: "5_Multi_Asset_Backtest.py",
    6: "6_Sentiment.py",
    7: "7_Correlation_Breaks.py",
}


def load_dashboard(n: int, st_mock: types.ModuleType | None = None) -> types.ModuleType:
    """Load a dashboard module from pages/ by its numeric index.

    Parameters
    ----------
    n:
        Dashboard number 1-7.
    st_mock:
        A pre-built streamlit mock. Defaults to make_st_mock().

    Returns
    -------
    The loaded module object with all top-level names accessible.
    """
    if st_mock is None:
        st_mock = make_st_mock()

    filename = _DASHBOARD_PATTERNS[n]
    path = _REPO_ROOT / "pages" / filename

    # Ensure streamlit mock is in sys.modules before loading
    sys.modules["streamlit"] = st_mock

    module_name = f"dashboard_{n}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Register in sys.modules BEFORE exec so dataclass and other
    # metaclasses can resolve cls.__module__ via sys.modules.get(...)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def make_ohlcv_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    high = close * (1.0 + rng.uniform(0.001, 0.01, n))
    low = close * (1.0 - rng.uniform(0.001, 0.01, n))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.002, n)),
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=pd.date_range("2020-01-01", periods=n),
    )
