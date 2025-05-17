# src/__init__.py
from dotenv import load_dotenv

# Load environment variables at package import time
load_dotenv()

# Define default configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 500

# Import public classes
from .agent import SimpleQAAgent

__all__ = ['SimpleQAAgent']