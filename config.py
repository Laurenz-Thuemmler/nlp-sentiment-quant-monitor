"""
Central Configuration for Institutional Sentiment Monitor.

This module contains all configurable parameters including model settings,
default tickers, API endpoints, chart themes, and cache TTL values.
"""

from typing import List, Dict

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_ID: str = "ProsusAI/finbert"

# Sentiment label to numeric mapping
SENTIMENT_MAP: Dict[str, int] = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# Signal thresholds for trading signals
SIGNAL_THRESHOLDS: Dict[str, float] = {
    "strong_buy": 0.4,
    "buy": 0.1,
    "sell": -0.1,
    "strong_sell": -0.4
}

# =============================================================================
# Default Asset Universe
# =============================================================================
DEFAULT_TICKERS: List[str] = [
    "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META"
]

# =============================================================================
# Data Sources
# =============================================================================
RSS_URL_PATTERN: str = "https://finance.yahoo.com/rss/headline?s={ticker}"

# =============================================================================
# Chart Configuration
# =============================================================================
CHART_THEME: str = "plotly_dark"
POSITIVE_COLOR: str = "#00cc96"  # Green
NEGATIVE_COLOR: str = "#ef553b"  # Red
NEUTRAL_COLOR: str = "#ffa15a"   # Yellow/Orange

# =============================================================================
# Cache TTL Configuration (seconds)
# =============================================================================
NEWS_CACHE_TTL: int = 300         # 5 minutes
MARKET_DATA_CACHE_TTL: int = 60   # 1 minute for intraday

# =============================================================================
# Application Settings
# =============================================================================
PAGE_TITLE: str = "AI-Driven Event Sentiment Monitor"
PAGE_ICON: str = ":chart_with_upwards_trend:"
DEFAULT_PERIOD: str = "7d"
DEFAULT_INTERVAL: str = "1h"
