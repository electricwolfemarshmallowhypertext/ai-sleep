"""
HuggingFace Integration: Utilities for integrating AI Sleep with HuggingFace models.

This module provides helper functions and classes for applying AI Sleep constructs
to HuggingFace Transformers models.
"""

from typing import Any, Dict, List, Optional, Union
import warnings

from .sleep_controller import AISleepController, OptimizationStrategy, SleepTrigger
from .model_state import SleepMode


class HuggingFaceModelAdapter:
    """
    Adapter for integrating AI Sleep with HuggingFace models.

    This class provides utilities to wrap HuggingFace models with AI Sleep
    capabilities, enabling sleep cycles and optimizations on pre-trained models.

    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> adapter = HuggingFaceModelAdapter(model, model_id="gpt2")
        >>> adapter.enable_sleep_cycles()
        >>> adapter.initiate_light_sleep()
    """

    def __init__(
        self,
        model: Any,
        model_id: str,
        enable_monitoring: bool = True,
        auto_adapt: bool = True,
    ):
        """
        Initialize HuggingFace model adapter.

        Args:
            model: HuggingFace model instance (or None for mock)
            model_id: Model identifier
            enable_monitoring: Whether to enable performance monitoring
            auto_adapt: Whether to enable adaptive learning
        """
        self.model = model
        self.model_id = model_id
        self.controller = AISleepController(
            model_id=model_id,
            enable_monitoring=enable_monitoring,
            auto_adapt=auto_adapt,
        )
        self.sleep_enabled = False

        # Extract model configuration if available
        if model is not None and hasattr(model, "config"):
            self._extract_model_config()

    def _extract_model_config(self) -> None:
        """Extract relevant configuration from HuggingFace model."""
        config = self.model.config

        # Store model architecture info
        self.num_layers = getattr(config, "num_hidden_layers", 12)
        self.num_heads = getattr(config, "num_attention_heads", 12)
        self.hidden_size = getattr(config, "hidden_size", 768)

    def enable_sleep_cycles(self) -> None:
        """Enable AI Sleep cycles for the model."""
        self.sleep_enabled = True

    def disable_sleep_cycles(self) -> None:
        """Disable AI Sleep cycles for the model."""
        self.sleep_enabled = False

    def initiate_light_sleep(self, duration: int = 300) -> bool:
        """
        Initiate light sleep optimization cycle.

        Args:
            duration: Sleep duration in seconds

        Returns:
            True if sleep was initiated
        """
        if not self.sleep_enabled:
            warnings.warn("Sleep cycles not enabled. Call enable_sleep_cycles() first.")
            return False

        self.controller.configure_light_sleep(duration=duration)
        return self.controller.initiate_sleep(
            trigger=SleepTrigger.MANUAL, mode=SleepMode.LIGHT_SLEEP
        )

    def initiate_deep_sleep(self, duration: int = 1800) -> bool:
        """
        Initiate deep sleep optimization cycle.

        Args:
            duration: Sleep duration in seconds

        Returns:
            True if sleep was initiated
        """
        if not self.sleep_enabled:
            warnings.warn("Sleep cycles not enabled. Call enable_sleep_cycles() first.")
            return False

        self.controller.configure_deep_sleep(duration=duration)
        return self.controller.initiate_sleep(
            trigger=SleepTrigger.MANUAL, mode=SleepMode.DEEP_SLEEP
        )

    def wake_model(self) -> bool:
        """
        Wake the model from sleep.

        Returns:
            True if wake was successful
        """
        return self.controller.wake_up()

    def track_inference_metrics(
        self,
        perplexity: Optional[float] = None,
        loss: Optional[float] = None,
        inference_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
    ) -> None:
        """
        Track inference metrics for performance monitoring.

        Args:
            perplexity: Model perplexity score
            loss: Loss value
            inference_time: Inference time in seconds
            memory_usage: Memory usage in MB
        """
        if not self.controller.performance_monitor:
            return

        monitor = self.controller.performance_monitor

        if perplexity is not None:
            monitor.track_metric("perplexity", perplexity)
        if loss is not None:
            monitor.track_metric("loss", loss)
        if inference_time is not None:
            monitor.track_metric("inference_time", inference_time)
        if memory_usage is not None:
            monitor.track_metric("memory_usage", memory_usage)

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get comprehensive model status.

        Returns:
            Dictionary with model status information
        """
        status = {
            "model_id": self.model_id,
            "sleep_enabled": self.sleep_enabled,
            "current_mode": self.controller.model_state.current_mode.value,
            "sleep_statistics": self.controller.get_sleep_statistics(),
        }

        if self.controller.performance_monitor:
            status["monitoring"] = (
                self.controller.performance_monitor.get_monitoring_summary()
            )

        return status


def create_sleep_enabled_model(
    model_name_or_path: str,
    enable_monitoring: bool = True,
    auto_adapt: bool = True,
    load_model: bool = False,
) -> HuggingFaceModelAdapter:
    """
    Create a sleep-enabled HuggingFace model.

    This is a convenience function for quickly wrapping HuggingFace models
    with AI Sleep capabilities.

    Args:
        model_name_or_path: HuggingFace model identifier or path
        enable_monitoring: Whether to enable performance monitoring
        auto_adapt: Whether to enable adaptive learning
        load_model: Whether to actually load the model (requires transformers library)

    Returns:
        HuggingFaceModelAdapter instance

    Example:
        >>> adapter = create_sleep_enabled_model("gpt2")
        >>> adapter.enable_sleep_cycles()
        >>> adapter.initiate_light_sleep()
    """
    model = None

    if load_model:
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_name_or_path)
        except ImportError:
            warnings.warn(
                "transformers library not available. Running in mock mode. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            warnings.warn(f"Could not load model: {e}. Running in mock mode.")

    adapter = HuggingFaceModelAdapter(
        model=model,
        model_id=model_name_or_path,
        enable_monitoring=enable_monitoring,
        auto_adapt=auto_adapt,
    )

    return adapter


def configure_optimal_sleep_schedule(
    adapter: HuggingFaceModelAdapter, workload_type: str = "continuous"
) -> None:
    """
    Configure optimal sleep schedule based on workload type.

    Args:
        adapter: HuggingFaceModelAdapter instance
        workload_type: Type of workload ('continuous', 'batch', 'interactive')
    """
    if workload_type == "continuous":
        # Frequent light sleep for continuous operation
        adapter.controller.configure_light_sleep(
            duration=300,
            strategies=[
                OptimizationStrategy.GRADIENT_CLIPPING,
                OptimizationStrategy.KV_CACHE_MANAGEMENT,
            ],
        )
        adapter.controller.configure_deep_sleep(
            duration=1800,
            strategies=[
                OptimizationStrategy.ATTENTION_HEAD_PRUNING,
                OptimizationStrategy.SEMANTIC_CONSOLIDATION,
            ],
        )

    elif workload_type == "batch":
        # Less frequent but longer sleep cycles
        adapter.controller.configure_light_sleep(
            duration=600,
            strategies=[
                OptimizationStrategy.GRADIENT_CLIPPING,
                OptimizationStrategy.KV_CACHE_MANAGEMENT,
                OptimizationStrategy.ADAPTIVE_LEARNING_RATE,
            ],
        )
        adapter.controller.configure_deep_sleep(
            duration=3600,
            strategies=[
                OptimizationStrategy.ATTENTION_HEAD_PRUNING,
                OptimizationStrategy.SEMANTIC_CONSOLIDATION,
                OptimizationStrategy.LAYER_NORM_RECALIBRATION,
                OptimizationStrategy.SECURITY_PATCHING,
            ],
        )

    elif workload_type == "interactive":
        # Quick, minimal impact sleep cycles
        adapter.controller.configure_light_sleep(
            duration=60, strategies=[OptimizationStrategy.KV_CACHE_MANAGEMENT]
        )
        adapter.controller.configure_deep_sleep(
            duration=600,
            strategies=[
                OptimizationStrategy.SEMANTIC_CONSOLIDATION,
                OptimizationStrategy.LAYER_NORM_RECALIBRATION,
            ],
        )
