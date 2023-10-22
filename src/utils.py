"""
Utility functions
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return absolute path to project root. Modify if file is moved from root."""
    return Path(__file__).parent.parent