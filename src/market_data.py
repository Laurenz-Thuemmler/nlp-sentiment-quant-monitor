"""
Market Data Provider using yfinance.

This module provides the MarketDataLoader class for fetching historical
OHLCV (Open, High, Low, Close, Volume) data and real-time price quotes
from Yahoo Finance via the yfinance library.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKET_DATA_CACHE_TTL, DEFAULT_PERIOD, DEFAULT_INTERVAL

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """
    Market data provider using Yahoo Finance.

    This class handles fetching and caching of historical price data
    and current quotes for a given ticker symbol.

    Attributes:
        ticker (str): Stock ticker symbol.
        _yf_ticker (yf.Ticker): yfinance Ticker object.

    Example:
        >>> loader = MarketDataLoader("NVDA")
        >>> price_df = loader.get_price_history(period="7d")
        >>> print(loader.get_current_price())
        123.45
    """

    def __init__(self, ticker: str):
        """
        Initialize the market data loader.

        Args:
            ticker: Stock ticker symbol.
        """
        self.ticker = ticker.upper()
        self._yf_ticker: Optional[yf.Ticker] = None

    @property
    def yf_ticker(self) -> yf.Ticker:
        """Lazy-load the yfinance Ticker object."""
        if self._yf_ticker is None:
            self._yf_ticker = yf.Ticker(self.ticker)
        return self._yf_ticker

    @staticmethod
    def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the timestamp column name.

        Args:
            df: DataFrame with Date or Datetime index.

        Returns:
            pd.DataFrame: DataFrame with 'timestamp' column.
        """
        df = df.reset_index()

        # Handle various column names from yfinance
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "timestamp"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})

        return df

    @staticmethod
    def _remove_timezone(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove timezone information from timestamp column.

        This is necessary for compatibility with merge_asof operations
        that require timezone-naive datetimes.

        Args:
            df: DataFrame with timestamp column.

        Returns:
            pd.DataFrame: DataFrame with timezone-naive timestamps.
        """
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df

    def get_price_history(
        self,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            period: Time period for historical data.
                Valid values: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
            interval: Data granularity.
                Valid values: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp (datetime): Bar timestamp
                - open (float): Opening price
                - high (float): High price
                - low (float): Low price
                - close (float): Closing price
                - volume (int): Trading volume
        """
        return self._fetch_history_cached(self.ticker, period, interval)

    @st.cache_data(ttl=MARKET_DATA_CACHE_TTL, show_spinner=False)
    def _fetch_history_cached(
        _self,
        ticker: str,
        period: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Fetch historical price data with caching.

        Args:
            ticker: Ticker symbol (for cache key).
            period: Data period (e.g., "7d", "1mo", "1y").
            interval: Data interval (e.g., "1h", "1d").

        Returns:
            pd.DataFrame: OHLCV data with timestamp column.
        """
        try:
            logger.info(f"Fetching {period} of {interval} data for {ticker}")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No historical data returned for {ticker}")
                return pd.DataFrame()

            df = MarketDataLoader._normalize_timestamp_column(df)
            df = MarketDataLoader._remove_timezone(df)

            # Select and rename columns for consistency
            columns_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }

            result_columns = ["timestamp"]
            for old_col, new_col in columns_map.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
                    result_columns.append(new_col)

            # Filter to only the columns we need
            available_columns = [c for c in result_columns if c in df.columns]
            df = df[available_columns]

            logger.info(f"Fetched {len(df)} price bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """
        Get the most recent closing price.

        Returns:
            float: Current/latest price, or 0.0 if unavailable.
        """
        try:
            # Try fast_info first (faster, uses cached data)
            if hasattr(self.yf_ticker, "fast_info"):
                info = self.yf_ticker.fast_info
                if hasattr(info, "last_price") and info.last_price:
                    return float(info.last_price)

            # Fallback to history
            data = self.yf_ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])

            return 0.0

        except Exception as e:
            logger.error(f"Error fetching current price for {self.ticker}: {e}")
            return 0.0

    def get_price_change(self, lookback_days: int = 1) -> Tuple[float, float]:
        """
        Calculate price change over a lookback period.

        Args:
            lookback_days: Number of days to look back.

        Returns:
            Tuple[float, float]: (absolute_change, percent_change)
        """
        try:
            data = self.yf_ticker.history(period=f"{lookback_days + 1}d")
            if len(data) < 2:
                return 0.0, 0.0

            current = data["Close"].iloc[-1]
            previous = data["Close"].iloc[0]

            absolute_change = current - previous
            percent_change = (absolute_change / previous) * 100 if previous != 0 else 0.0

            return float(absolute_change), float(percent_change)

        except Exception:
            return 0.0, 0.0


if __name__ == "__main__":
    # Test the loader
    loader = MarketDataLoader("NVDA")
    history = loader.get_price_history()
    print(f"Loaded {len(history)} price bars for NVDA")
    if not history.empty:
        print("\nColumns:", history.columns.tolist())
        print("\nFirst 5 rows:")
        print(history.head())
        print(f"\nCurrent Price: ${loader.get_current_price():,.2f}")
        abs_change, pct_change = loader.get_price_change(7)
        print(f"7-day Change: ${abs_change:,.2f} ({pct_change:+.2f}%)")
