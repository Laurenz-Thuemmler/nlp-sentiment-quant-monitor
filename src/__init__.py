"""
Source package for Real-Time AI Sentiment Monitor.

This package contains the core modules:
- sentiment_engine: FinBERT-based sentiment analysis
- news_ingestor: Yahoo Finance RSS feed ingestion
- market_data: yfinance market data provider
- visualizer: Plotly chart generation

Example:
    from src import FinBertAnalyzer, NewsIngestor, MarketDataLoader, DashboardCharts
"""

from src.sentiment_engine import FinBertAnalyzer
from src.news_ingestor import NewsIngestor
from src.market_data import MarketDataLoader
from src.visualizer import DashboardCharts

__all__ = [
    "FinBertAnalyzer",
    "NewsIngestor",
    "MarketDataLoader",
    "DashboardCharts"
]
