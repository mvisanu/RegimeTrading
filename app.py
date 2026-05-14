"""
RegimeTrading — Streamlit multi-page hub.

Run with:
    streamlit run app.py

Pages in the pages/ subdirectory are auto-discovered by Streamlit for navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="RegimeTrading",
    page_icon="📈",
    layout="wide",
)

st.title("RegimeTrading")
st.subheader("Adaptive algorithmic trading driven by market-regime detection")

st.markdown(
    """
    This dashboard provides tools for monitoring market regimes, reviewing trade signals,
    and managing the automated trading bot.

    Use the sidebar to navigate between pages once they are available.
    """
)

st.info("No pages loaded yet. Dashboard pages will appear in the sidebar as they are added.")
