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

# C2C (Cache-to-Cache) components
try:
    from .c2c import C2CFuser, CacheProjector, LayerGate, KVCache, KVSpec
    from .hf_adapters import (
        HFCacheExtractor,
        get_pkv_from_outputs,
        set_pkv_in_inputs,
        convert_pkv_to_kvcache,
        convert_kvcache_to_pkv,
        get_model_config
    )
    C2C_AVAILABLE = True
except ImportError:
    C2C_AVAILABLE = False

__all__ = [
    "LMModelState",
    "LMPerformanceMonitor",
    "AISleepController",
]

# Add C2C exports if torch is available
if C2C_AVAILABLE:
    __all__.extend([
        "C2CFuser",
        "CacheProjector",
        "LayerGate",
        "KVCache",
        "KVSpec",
        "HFCacheExtractor",
        "get_pkv_from_outputs",
        "set_pkv_in_inputs",
        "convert_pkv_to_kvcache",
        "convert_kvcache_to_pkv",
        "get_model_config",
    ])
