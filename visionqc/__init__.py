"""VisionQC - a classical machine-vision pipeline for industrial part inspection."""

from .config import load_config
from .pipeline import Inspector, InspectionReport, PartResult

__all__ = ["load_config", "Inspector", "InspectionReport", "PartResult"]
__version__ = "0.1.0"
