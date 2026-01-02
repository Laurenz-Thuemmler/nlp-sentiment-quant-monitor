"""
Sentiment Analysis Engine using FinBERT.

This module provides the FinBertAnalyzer class which wraps the Hugging Face
transformers pipeline for financial sentiment classification. The model is
cached using Streamlit's @st.cache_resource decorator for optimal performance.

Model: ProsusAI/finbert
- Pre-trained on Financial PhraseBank
- Fine-tuned for sentiment classification
- Labels: positive, negative, neutral
"""

from __future__ import annotations
import os
import logging
from typing import List, Optional

# Force PyTorch backend for transformers (avoid Keras/TensorFlow issues)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import pandas as pd
import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, Pipeline

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_ID, SENTIMENT_MAP

logger = logging.getLogger(__name__)


class FinBertAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.

    This class manages the loading and inference of the FinBERT model.
    The model is cached at the Streamlit resource level to avoid
    reloading on each request.

    Attributes:
        model_id (str): Hugging Face model identifier.
        device (int): CUDA device index (0) or CPU (-1).

    Example:
        >>> analyzer = FinBertAnalyzer()
        >>> results = analyzer.analyze_headlines(["Stock surges on earnings"])
        >>> print(results['label'].iloc[0])
        'positive'
    """

    def __init__(self, model_id: str = MODEL_ID):
        """
        Initialize the FinBERT analyzer.

        Args:
            model_id: Hugging Face model identifier. Defaults to ProsusAI/finbert.
        """
        self.model_id = model_id
        self.device = self._detect_device()
        self._pipeline: Optional[Pipeline] = None

    @staticmethod
    def _detect_device() -> int:
        """
        Detect available compute device.

        Returns:
            int: 0 for CUDA GPU, -1 for CPU.
        """
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected, using GPU for inference")
            return 0
        logger.info("No CUDA GPU detected, using CPU for inference")
        return -1

    @property
    def device_name(self) -> str:
        """Human-readable device name."""
        return "CUDA GPU" if self.device == 0 else "CPU"

    @staticmethod
    @st.cache_resource(show_spinner="Loading FinBERT model (~420MB)...")
    def _load_pipeline(model_id: str, device: int) -> Pipeline:
        """
        Load the FinBERT sentiment analysis pipeline.

        This method is decorated with @st.cache_resource to ensure the model
        is loaded only once per session and shared across all users.

        Args:
            model_id: Hugging Face model identifier.
            device: Target device (-1 for CPU, 0+ for GPU).

        Returns:
            Pipeline: Configured sentiment analysis pipeline.
        """
        # Set environment variable for Protobuf compatibility
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        # Explicitly load PyTorch model to avoid Keras/TensorFlow issues
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # Determine device string
        device_str = f"cuda:{device}" if device >= 0 else "cpu"

        return pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device_str,
            truncation=True,
            max_length=512
        )

    def get_pipeline(self) -> Pipeline:
        """
        Get or load the sentiment analysis pipeline.

        Returns:
            Pipeline: Ready-to-use sentiment analysis pipeline.
        """
        if self._pipeline is None:
            self._pipeline = self._load_pipeline(self.model_id, self.device)
        return self._pipeline

    def analyze_headlines(self, headlines: List[str]) -> pd.DataFrame:
        """
        Perform batch sentiment analysis on a list of headlines.

        Args:
            headlines: List of news headline strings.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - label: Sentiment label (positive/negative/neutral)
                - score: Confidence score (0-1)
                - sentiment_numeric: Numeric mapping (-1, 0, 1)
        """
        if not headlines:
            return pd.DataFrame(columns=["label", "score", "sentiment_numeric"])

        try:
            classifier = self.get_pipeline()
            results = classifier(headlines)

            df = pd.DataFrame(results)
            df["sentiment_numeric"] = df["label"].map(SENTIMENT_MAP)

            return df

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            st.error(f"Sentiment analysis error: {e}")
            return pd.DataFrame(columns=["label", "score", "sentiment_numeric"])

    def analyze_single(self, text: str) -> Optional[dict]:
        """
        Analyze sentiment for a single text.

        Args:
            text: Input text for sentiment analysis.

        Returns:
            dict with label, score, sentiment_numeric or None if analysis fails.
        """
        df = self.analyze_headlines([text])
        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "label": row["label"],
            "score": row["score"],
            "sentiment_numeric": row["sentiment_numeric"]
        }


if __name__ == "__main__":
    # Test the analyzer
    analyzer = FinBertAnalyzer()
    print(f"Device: {analyzer.device_name}")

    test_headlines = [
        "NVIDIA reports record quarterly revenue, beating estimates",
        "Company announces massive layoffs amid restructuring",
        "Stock trades sideways as investors await earnings"
    ]

    results = analyzer.analyze_headlines(test_headlines)
    print("\nSentiment Analysis Results:")
    print(results)
