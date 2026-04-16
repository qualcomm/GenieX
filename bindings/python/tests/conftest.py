"""
Pytest configuration and fixtures for geniex Python bindings tests.
"""

import sys
from pathlib import Path

# Add parent directory to Python path to import geniex without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from geniex import setup_logging

# Get project root directory (bindings/python/../../)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Setup logging for tests
setup_logging(level=10)  # DEBUG level
