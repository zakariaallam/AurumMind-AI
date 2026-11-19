"""Data ingestion module for AurumMind AI."""

from .data_loader import DataLoader
from .schema import OHLCVSchema

__all__ = ["DataLoader", "OHLCVSchema"]
