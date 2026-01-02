"""
Visualization Module for Dashboard Charts.

This module provides the DashboardCharts class with static methods for
generating Plotly figures used in the Streamlit dashboard, including:
- Price & Sentiment Overlay Chart
- Sentiment Gauge Indicator
- Sentiment Distribution Bar Chart
"""

from __future__ import annotations
import os
import sys
from typing import Dict

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CHART_THEME,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    SIGNAL_THRESHOLDS
)


class DashboardCharts:
    """
    Static chart factory for dashboard visualizations.

    All methods are static and return Plotly Figure objects that can
    be rendered directly in Streamlit using st.plotly_chart().
    """

    # Color mapping for sentiment labels
    COLOR_MAP: Dict[str, str] = {
        "positive": POSITIVE_COLOR,
        "negative": NEGATIVE_COLOR,
        "neutral": NEUTRAL_COLOR
    }

    @staticmethod
    def _prepare_news_with_prices(
        price_df: pd.DataFrame,
        news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge news data with nearest price points.

        This performs an as-of merge to find the closest price bar
        for each news item, enabling overlay visualization.

        Args:
            price_df: Price data with 'timestamp' and 'close' columns.
            news_df: News data with 'published' and 'label' columns.

        Returns:
            pd.DataFrame: Merged data with price at each news timestamp.
        """
        if news_df.empty or price_df.empty:
            return pd.DataFrame()

        # Ensure timezone-naive datetimes for merge compatibility
        news_copy = news_df.copy()
        price_copy = price_df.copy()

        news_copy["published"] = pd.to_datetime(news_copy["published"]).dt.tz_localize(None)
        price_copy["timestamp"] = pd.to_datetime(price_copy["timestamp"]).dt.tz_localize(None)

        # Sort for merge_asof requirement
        news_sorted = news_copy.sort_values("published")
        price_sorted = price_copy.sort_values("timestamp")

        merged = pd.merge_asof(
            news_sorted,
            price_sorted[["timestamp", "close"]],
            left_on="published",
            right_on="timestamp",
            direction="nearest"
        )

        return merged

    @staticmethod
    def plot_price_sentiment_overlay(
        price_df: pd.DataFrame,
        news_df: pd.DataFrame,
        title: str = "Price & News Sentiment Correlation"
    ) -> go.Figure:
        """
        Create an overlay chart with price line and sentiment scatter points.

        The chart displays:
        - A line chart of closing prices over time
        - Scatter points at news event times, colored by sentiment
        - Hover tooltips showing headline text and confidence score

        Args:
            price_df: DataFrame with 'timestamp' and 'close' columns.
            news_df: DataFrame with 'published', 'title', 'label', 'score' columns.
            title: Chart title.

        Returns:
            go.Figure: Plotly figure object.
        """
        fig = go.Figure()

        # 1. Add Price Line
        if not price_df.empty:
            fig.add_trace(go.Scatter(
                x=price_df["timestamp"],
                y=price_df["close"],
                mode="lines",
                name="Price (Close)",
                line=dict(color="rgba(200, 200, 200, 0.8)", width=2),
                hovertemplate="<b>Price:</b> $%{y:,.2f}<br><b>Time:</b> %{x}<extra></extra>"
            ))

        # 2. Add Sentiment Scatter Points
        if not news_df.empty and "label" in news_df.columns:
            merged = DashboardCharts._prepare_news_with_prices(price_df, news_df)

            if not merged.empty:
                for label, color in DashboardCharts.COLOR_MAP.items():
                    mask = merged["label"] == label
                    subset = merged[mask]

                    if subset.empty:
                        continue

                    # Build hover text with headline and confidence
                    hover_text = subset.apply(
                        lambda row: (
                            f"<b>{row['title'][:80]}{'...' if len(str(row['title'])) > 80 else ''}</b><br>"
                            f"<b>Sentiment:</b> {row['label'].capitalize()}<br>"
                            f"<b>Confidence:</b> {row['score']:.1%}<br>"
                            f"<b>Price:</b> ${row['close']:,.2f}"
                        ),
                        axis=1
                    )

                    fig.add_trace(go.Scatter(
                        x=subset["published"],
                        y=subset["close"],
                        mode="markers",
                        name=label.capitalize(),
                        marker=dict(
                            color=color,
                            size=12,
                            line=dict(width=1.5, color="white"),
                            symbol="circle"
                        ),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>"
                    ))

        # 3. Configure Layout
        fig.update_layout(
            template=CHART_THEME,
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="Time",
            yaxis_title="Price ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            hovermode="closest",
            height=450
        )

        return fig

    @staticmethod
    def plot_sentiment_gauge(
        avg_sentiment: float,
        title: str = "24h Sentiment Signal"
    ) -> go.Figure:
        """
        Create a gauge chart displaying average sentiment.

        The gauge shows sentiment on a scale from -1 (bearish) to +1 (bullish)
        with color bands indicating negative, neutral, and positive zones.

        Args:
            avg_sentiment: Average sentiment score (-1 to 1).
            title: Chart title.

        Returns:
            go.Figure: Plotly gauge figure.
        """
        # Clamp value to valid range
        clamped_value = max(-1.0, min(1.0, avg_sentiment))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=clamped_value,
            number=dict(
                font=dict(size=36),
                valueformat=".2f"
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=title, font=dict(size=16)),
            gauge=dict(
                axis=dict(
                    range=[-1, 1],
                    tickwidth=1,
                    tickcolor="white",
                    tickvals=[-1, 0, 1],
                    ticktext=["-1", "0", "+1"],
                    tickfont=dict(size=12)
                ),
                bar=dict(color="white", thickness=0.2),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[-1, SIGNAL_THRESHOLDS["sell"]], color=NEGATIVE_COLOR),
                    dict(range=[SIGNAL_THRESHOLDS["sell"], SIGNAL_THRESHOLDS["buy"]], color=NEUTRAL_COLOR),
                    dict(range=[SIGNAL_THRESHOLDS["buy"], 1], color=POSITIVE_COLOR)
                ],
                threshold=dict(
                    line=dict(color="yellow", width=4),
                    thickness=0.8,
                    value=clamped_value
                )
            )
        ))

        # Add annotations for Bearish/Bullish labels at the bottom corners
        fig.add_annotation(
            x=0.02, y=-0.12,
            text="Bearish",
            showarrow=False,
            font=dict(size=11, color="#ef553b"),
            xref="paper", yref="paper"
        )
        fig.add_annotation(
            x=0.98, y=-0.12,
            text="Bullish",
            showarrow=False,
            font=dict(size=11, color="#00cc96"),
            xref="paper", yref="paper"
        )

        fig.update_layout(
            template=CHART_THEME,
            height=300,
            margin=dict(l=40, r=40, t=60, b=60)
        )

        return fig

    @staticmethod
    def plot_sentiment_distribution(news_df: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart showing sentiment distribution.

        Args:
            news_df: DataFrame with 'label' column.

        Returns:
            go.Figure: Plotly bar chart figure.
        """
        if news_df.empty or "label" not in news_df.columns:
            # Return empty figure with message
            fig = go.Figure()
            fig.update_layout(
                template=CHART_THEME,
                title="Sentiment Distribution",
                height=280,
                annotations=[dict(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )]
            )
            return fig

        # Count sentiments
        counts = news_df["label"].value_counts()

        # Ensure all labels are present
        for label in ["positive", "neutral", "negative"]:
            if label not in counts:
                counts[label] = 0

        fig = go.Figure(data=[
            go.Bar(
                x=["Positive", "Neutral", "Negative"],
                y=[counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)],
                marker_color=[POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR],
                text=[counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)],
                textposition="auto"
            )
        ])

        fig.update_layout(
            template=CHART_THEME,
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=280,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig


if __name__ == "__main__":
    # Test with mock data
    import numpy as np
    from datetime import datetime, timedelta

    # Create mock price data
    dates = pd.date_range(end=datetime.now(), periods=168, freq="H")
    prices = 100 + np.cumsum(np.random.randn(168) * 0.5)
    price_df = pd.DataFrame({"timestamp": dates, "close": prices})

    # Create mock news data
    news_dates = [datetime.now() - timedelta(hours=i*12) for i in range(10)]
    news_df = pd.DataFrame({
        "published": news_dates,
        "title": [f"Sample headline {i}" for i in range(10)],
        "label": np.random.choice(["positive", "neutral", "negative"], 10),
        "score": np.random.uniform(0.6, 0.99, 10)
    })

    # Test charts
    fig1 = DashboardCharts.plot_price_sentiment_overlay(price_df, news_df)
    fig2 = DashboardCharts.plot_sentiment_gauge(0.35)
    fig3 = DashboardCharts.plot_sentiment_distribution(news_df)

    print("Charts created successfully!")
    print(f"Overlay chart has {len(fig1.data)} traces")
    print(f"Gauge value: {fig2.data[0].value}")
    print(f"Distribution: {[trace.y for trace in fig3.data]}")
