# Real-Time AI Sentiment Analysis Engine (FinBERT)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](HIER_DEIN_STREAMLIT_LINK_EINFÜGEN)
[![Model: FinBERT](https://img.shields.io/badge/Model-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Live Dashboard:** [https://nlp-sentiment-quant-monitor.streamlit.app]

## Abstract

This project implements an institutional-grade event monitoring system that leverages Transfer Learning (BERT) to quantify market sentiment from unstructured financial news. By integrating real-time RSS ingestion with the FinBERT Large Language Model, the engine extracts alpha signals and correlates them with high-frequency price movements.

The system processes financial headlines through a pre-trained transformer architecture, mapping linguistic features to sentiment classifications that can inform systematic trading strategies.

---

## Dashboard Interface

![AI Sentiment Monitor Dashboard](assets/dashboard_overview.png)
*Figure 1: Real-time analysis showing price-sentiment correlation, 24h aggregate signal gauge, and volume distribution.*

---

## System Architecture

The application is built on a modular, object-oriented framework designed for scalability and low-latency inference:

```mermaid
graph TD
    A[RSS Feeds] -->|Raw XML| B(NewsIngestor)
    C[Yahoo Finance API] -->|OHLCV Data| D(MarketDataLoader)
    B -->|Cleaned Text| E{FinBertAnalyzer}
    E -->|Tokens| F[HuggingFace Transformer]
    F -->|Logits| G[Softmax Layer]
    G -->|Sentiment Score| H[Signal Generator]
    H --> I[Streamlit Dashboard]
    D --> I