"""
News Ingestion Module for Yahoo Finance RSS Feeds.

This module provides the NewsIngestor class for fetching, parsing, and
cleaning financial news headlines from Yahoo Finance RSS feeds.

Feed URL Pattern: https://finance.yahoo.com/rss/headline?s={TICKER}
"""

from __future__ import annotations
import os
import sys
import logging
import time
from datetime import datetime
from typing import List

import feedparser
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RSS_URL_PATTERN, NEWS_CACHE_TTL

logger = logging.getLogger(__name__)


class NewsIngestor:
    """
    RSS feed ingestor for Yahoo Finance news.

    This class handles fetching, parsing, and cleaning of financial news
    from Yahoo Finance RSS feeds. Results are cached using Streamlit's
    caching mechanism.

    Attributes:
        ticker (str): Stock ticker symbol.
        url (str): Constructed RSS feed URL.

    Example:
        >>> ingestor = NewsIngestor("AAPL")
        >>> news_df = ingestor.fetch_news()
        >>> print(news_df.columns.tolist())
        ['title', 'link', 'published']
    """

    def __init__(self, ticker: str):
        """
        Initialize the news ingestor for a specific ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "NVDA").
        """
        self.ticker = ticker.upper()
        self.url = RSS_URL_PATTERN.format(ticker=self.ticker)

    @staticmethod
    def _parse_published_date(entry) -> datetime:
        """
        Parse the published date from a feed entry.

        Args:
            entry: Feedparser entry object.

        Returns:
            datetime: Parsed datetime or current time as fallback.
        """
        # Try published_parsed first
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime.fromtimestamp(time.mktime(entry.published_parsed))
            except (ValueError, TypeError, OverflowError):
                pass

        # Fallback to updated_parsed
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                return datetime.fromtimestamp(time.mktime(entry.updated_parsed))
            except (ValueError, TypeError, OverflowError):
                pass

        # Last resort: current time
        return datetime.now()

    @staticmethod
    def _clean_title(title: str) -> str:
        """
        Clean and normalize a news title.

        Args:
            title: Raw title string.

        Returns:
            str: Cleaned title string.
        """
        if not title:
            return ""
        # Remove excessive whitespace
        return " ".join(title.split())

    def _parse_feed(self, feed) -> List[dict]:
        """
        Parse feed entries into dictionaries.

        Args:
            feed: Feedparser feed object.

        Returns:
            List[dict]: List of parsed news items.
        """
        items = []

        for entry in feed.entries:
            try:
                title = self._clean_title(getattr(entry, "title", ""))
                link = getattr(entry, "link", "")
                published = self._parse_published_date(entry)

                if title:  # Only include entries with titles
                    items.append({
                        "title": title,
                        "link": link,
                        "published": published
                    })
            except Exception as e:
                logger.warning(f"Error parsing feed entry: {e}")
                continue

        return items

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        return pd.DataFrame(columns=["title", "link", "published"])

    def fetch_news(self) -> pd.DataFrame:
        """
        Fetch news headlines for the configured ticker.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - title (str): News headline
                - link (str): URL to full article
                - published (datetime): Publication timestamp
        """
        return self._fetch_cached(self.ticker, self.url)

    @st.cache_data(ttl=NEWS_CACHE_TTL, show_spinner=False)
    def _fetch_cached(_self, ticker: str, url: str) -> pd.DataFrame:
        """
        Fetch news with caching support.

        Note: The _self parameter name with underscore tells Streamlit
        not to hash this parameter (since self is unhashable).

        Args:
            ticker: Ticker symbol (for cache key).
            url: RSS feed URL.

        Returns:
            pd.DataFrame: DataFrame with news data.
        """
        try:
            logger.info(f"Fetching news for {ticker} from {url}")
            feed = feedparser.parse(url)

            # Check for HTTP errors
            if hasattr(feed, "status") and feed.status >= 400:
                logger.error(f"HTTP error {feed.status} for {ticker}")
                return NewsIngestor._empty_dataframe()

            # Check for bozo errors (malformed XML)
            if feed.bozo and not feed.entries:
                logger.error(f"Feed parse error for {ticker}: {feed.bozo_exception}")
                return NewsIngestor._empty_dataframe()

            if not feed.entries:
                logger.info(f"No news entries found for {ticker}")
                return NewsIngestor._empty_dataframe()

            items = _self._parse_feed(feed)

            if not items:
                return NewsIngestor._empty_dataframe()

            df = pd.DataFrame(items)
            df["published"] = pd.to_datetime(df["published"])
            df = df.sort_values("published", ascending=False).reset_index(drop=True)

            logger.info(f"Fetched {len(df)} news items for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return NewsIngestor._empty_dataframe()

    def get_news_count(self) -> int:
        """
        Get the total count of available news items.

        Returns:
            int: Number of news items.
        """
        return len(self.fetch_news())


if __name__ == "__main__":
    # Test the ingestor
    ingestor = NewsIngestor("AAPL")
    news = ingestor.fetch_news()
    print(f"Fetched {len(news)} news items for AAPL")
    if not news.empty:
        print("\nFirst 5 headlines:")
        for _, row in news.head().iterrows():
            print(f"  [{row['published']}] {row['title'][:60]}...")
