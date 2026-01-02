#!/usr/bin/env python3
"""
Launcher script for the AI Sentiment Monitor.

This script sets required environment variables before launching Streamlit
to avoid protobuf and Keras compatibility issues with Anaconda.

Usage:
    python run.py
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE importing anything else
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow in transformers
os.environ["USE_TORCH"] = "1"           # Force PyTorch backend

# Now we can safely import and run streamlit
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")

    # Run streamlit with the app
    sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
    sys.exit(stcli.main())
