"""
AI Sleep Constructs: Production-ready framework for engineered sleep cycles in language models.

This package provides classes and utilities for implementing light and deep sleep modes
that enable offline optimization, performance monitoring, and adaptive learning in LLMs.

License: CC-BY-NC-SA 4.0
"""

__version__ = "0.1.0"
__author__ = "AI Sleep Constructs Team"
__license__ = "CC-BY-NC-SA-4.0"

from .model_state import LMModelState
from .performance_monitor import LMPerformanceMonitor
from .sleep_controller import AISleepController

__all__ = [
    "LMModelState",
    "LMPerformanceMonitor",
    "AISleepController",
]
