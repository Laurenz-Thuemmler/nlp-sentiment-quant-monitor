"""
AI-Driven Event Sentiment Monitor - Main Application.

This Streamlit application provides an institutional-grade dashboard for
real-time financial news sentiment analysis using the FinBERT model.

Run with:
    streamlit run app.py
"""

import os
# Fix for Protobuf / Anaconda environment conflict
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

from config import (
    DEFAULT_TICKERS,
    PAGE_TITLE,
    PAGE_ICON,
    SIGNAL_THRESHOLDS,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR
)
from src import FinBertAnalyzer, NewsIngestor, MarketDataLoader, DashboardCharts


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Custom Styling
# =============================================================================
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stMetric:hover {
        border-color: #58a6ff;
        transition: border-color 0.3s ease;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .news-positive { color: #00cc96; }
    .news-negative { color: #ef553b; }
    .news-neutral { color: #ffa15a; }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8b949e;
        margin-top: 0;
    }

    /* Signal badge styling */
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .signal-strong-buy { background-color: #00cc96; color: #000; }
    .signal-buy { background-color: #3fb950; color: #000; }
    .signal-neutral { background-color: #ffa15a; color: #000; }
    .signal-sell { background-color: #f85149; color: #fff; }
    .signal-strong-sell { background-color: #da3633; color: #fff; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached Resource: FinBERT Analyzer
# =============================================================================
@st.cache_resource(show_spinner="Initializing FinBERT Sentiment Engine...")
def get_analyzer() -> FinBertAnalyzer:
    """Initialize and cache the FinBERT analyzer."""
    return FinBertAnalyzer()


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_market_data(ticker: str) -> Tuple[pd.DataFrame, float]:
    """
    Load market data for the given ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Tuple of (price_history_df, current_price).
    """
    loader = MarketDataLoader(ticker)
    price_df = loader.get_price_history(period="7d", interval="1h")
    current_price = loader.get_current_price()
    return price_df, current_price


def load_news_with_sentiment(ticker: str, analyzer: FinBertAnalyzer) -> pd.DataFrame:
    """
    Load news and perform sentiment analysis.

    Args:
        ticker: Stock ticker symbol.
        analyzer: FinBERT analyzer instance.

    Returns:
        DataFrame with news and sentiment columns.
    """
    ingestor = NewsIngestor(ticker)
    news_df = ingestor.fetch_news()

    if news_df.empty:
        return pd.DataFrame()

    # Run sentiment analysis
    sentiment_results = analyzer.analyze_headlines(news_df["title"].tolist())

    # Merge results
    news_df = pd.concat([
        news_df.reset_index(drop=True),
        sentiment_results.reset_index(drop=True)
    ], axis=1)

    return news_df


def calculate_metrics(news_df: pd.DataFrame, lookback_hours: int = 24) -> Tuple[int, float, str]:
    """
    Calculate dashboard metrics from news data.

    Args:
        news_df: DataFrame with sentiment analysis results.
        lookback_hours: Hours to look back for metrics.

    Returns:
        Tuple of (news_volume, avg_sentiment, signal_label).
    """
    if news_df.empty:
        return 0, 0.0, "NEUTRAL"

    # Filter to lookback period
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    recent_news = news_df[news_df["published"] > cutoff]

    volume = len(recent_news)

    if volume == 0:
        return 0, 0.0, "NEUTRAL"

    avg_sentiment = recent_news["sentiment_numeric"].mean()

    # Determine signal
    if avg_sentiment > SIGNAL_THRESHOLDS["strong_buy"]:
        signal = "STRONG BUY"
    elif avg_sentiment > SIGNAL_THRESHOLDS["buy"]:
        signal = "BUY"
    elif avg_sentiment < SIGNAL_THRESHOLDS["strong_sell"]:
        signal = "STRONG SELL"
    elif avg_sentiment < SIGNAL_THRESHOLDS["sell"]:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return volume, avg_sentiment, signal


def get_signal_color(signal: str) -> str:
    """Get the color for a signal badge."""
    colors = {
        "STRONG BUY": POSITIVE_COLOR,
        "BUY": "#3fb950",
        "NEUTRAL": NEUTRAL_COLOR,
        "SELL": "#f85149",
        "STRONG SELL": NEGATIVE_COLOR
    }
    return colors.get(signal, NEUTRAL_COLOR)


def style_sentiment_label(val: str) -> str:
    """Apply color styling to sentiment labels."""
    colors = {
        "positive": POSITIVE_COLOR,
        "negative": NEGATIVE_COLOR,
        "neutral": NEUTRAL_COLOR
    }
    color = colors.get(val.lower() if isinstance(val, str) else val, "white")
    return f"color: {color}; font-weight: bold;"


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("Control Panel")

selected_ticker = st.sidebar.selectbox(
    "Select Asset Ticker",
    options=DEFAULT_TICKERS,
    index=0
)

if st.sidebar.button("Refresh Analysis", use_container_width=True):
    st.cache_data.clear()

st.sidebar.markdown("---")

# Engine Stats
analyzer = get_analyzer()
st.sidebar.markdown("### Sentiment Engine")
st.sidebar.info(f"""
**Model:** FinBERT (BERT-base)
**Device:** {analyzer.device_name}
**Task:** Financial Sentiment
**Provider:** ProsusAI / Hugging Face
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Signal Interpretation")
st.sidebar.markdown(f"""
<div style="line-height: 1.4;">
<span style="color: {POSITIVE_COLOR}; font-weight: bold;">STRONG BUY</span><br>
<span style="margin-left: 10px; font-size: 0.9em;">Sentiment > 0.4</span><br>
<span style="color: #3fb950; font-weight: bold;">BUY</span><br>
<span style="margin-left: 10px; font-size: 0.9em;">Sentiment > 0.1</span><br>
<span style="color: {NEUTRAL_COLOR}; font-weight: bold;">NEUTRAL</span><br>
<span style="margin-left: 10px; font-size: 0.9em;">-0.1 to 0.1</span><br>
<span style="color: #f85149; font-weight: bold;">SELL</span><br>
<span style="margin-left: 10px; font-size: 0.9em;">Sentiment < -0.1</span><br>
<span style="color: {NEGATIVE_COLOR}; font-weight: bold;">STRONG SELL</span><br>
<span style="margin-left: 10px; font-size: 0.9em;">Sentiment < -0.4</span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Main Dashboard
# =============================================================================
st.markdown('<p class="main-header">AI-Driven Event Sentiment Monitor</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Institutional-Grade Sentiment Analysis for <b>{selected_ticker}</b></p>', unsafe_allow_html=True)

st.markdown("---")

# Load Data
with st.spinner(f"Analyzing {selected_ticker} market data and news sentiment..."):
    price_df, current_price = load_market_data(selected_ticker)
    news_df = load_news_with_sentiment(selected_ticker, analyzer)

# Handle data loading errors
if price_df.empty:
    st.error(f"Failed to load market data for {selected_ticker}. Please verify the ticker symbol.")
    st.stop()

# Calculate Metrics
volume_24h, avg_sentiment, signal = calculate_metrics(news_df)


# =============================================================================
# Metrics Row
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current Price",
        value=f"${current_price:,.2f}"
    )

with col2:
    st.metric(
        label="News Volume (24h)",
        value=volume_24h
    )

with col3:
    st.metric(
        label="Avg Sentiment (24h)",
        value=f"{avg_sentiment:+.3f}"
    )

with col4:
    signal_color = get_signal_color(signal)
    st.metric(
        label="AI Sentiment Signal",
        value=signal
    )


# =============================================================================
# Charts Section
# =============================================================================
st.markdown("---")

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    fig_overlay = DashboardCharts.plot_price_sentiment_overlay(
        price_df,
        news_df,
        title=f"{selected_ticker} - Price & News Sentiment Correlation (7 Days)"
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

with chart_col2:
    fig_gauge = DashboardCharts.plot_sentiment_gauge(avg_sentiment)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Add sentiment distribution
    fig_dist = DashboardCharts.plot_sentiment_distribution(news_df)
    st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# News Table Section
# =============================================================================
st.markdown("---")
st.markdown("### Recent News & Sentiment Analysis")

if not news_df.empty:
    # Prepare display dataframe
    display_df = news_df[["published", "title", "label", "score"]].copy()
    display_df = display_df.sort_values("published", ascending=False)
    display_df.columns = ["Published", "Headline", "Sentiment", "Confidence"]

    # Format columns
    display_df["Published"] = display_df["Published"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
    display_df["Sentiment"] = display_df["Sentiment"].str.capitalize()

    # Apply styling
    styled_df = display_df.style.applymap(
        style_sentiment_label,
        subset=["Sentiment"]
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )
else:
    st.info(f"No recent news found for {selected_ticker}. This may be due to limited RSS feed availability.")


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b949e; padding: 20px 0;'>
    <p style='font-size: 0.9em; margin-bottom: 5px;'>
        <b>Real-Time AI Sentiment Monitor</b> | Built with FinBERT & Streamlit
    </p>
    <p style='font-size: 0.75em;'>
        Model: ProsusAI/finbert | Data: Yahoo Finance RSS & yfinance |
        Sentiment labels map to: Positive (+1), Neutral (0), Negative (-1)
    </p>
</div>
""", unsafe_allow_html=True)
