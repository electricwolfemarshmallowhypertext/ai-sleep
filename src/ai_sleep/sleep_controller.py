"""
AISleepController: Orchestrator for language model sleep cycles.

This module provides the main controller for managing sleep cycles, coordinating
light and deep sleep modes, and applying optimization strategies to language models.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .model_state import LMModelState, SleepMode
from .performance_monitor import LMPerformanceMonitor, PerformanceAlert


class SleepTrigger(Enum):
    """Enumeration of contextual triggers for sleep initiation."""

    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_DETECTED = "anomaly_detected"
    DRIFT_DETECTED = "drift_detected"
    MEMORY_PRESSURE = "memory_pressure"
    MANUAL = "manual"
    IDLE_TIMEOUT = "idle_timeout"


class OptimizationStrategy(Enum):
    """Optimization strategies available during sleep modes."""

    GRADIENT_CLIPPING = "gradient_clipping"
    ATTENTION_HEAD_PRUNING = "attention_head_pruning"
    KV_CACHE_MANAGEMENT = "kv_cache_management"
    SEMANTIC_CONSOLIDATION = "semantic_consolidation"
    LAYER_NORM_RECALIBRATION = "layer_norm_recalibration"
    SECURITY_PATCHING = "security_patching"
    ADAPTIVE_LEARNING_RATE = "adaptive_learning_rate"


class AISleepController:
    """
    Main controller for orchestrating AI sleep cycles.

    This class coordinates light and deep sleep modes, applies various optimization
    strategies, manages contextual triggers, and integrates with HuggingFace models.

    The controller implements a state machine for sleep cycles with the following modes:
    - AWAKE: Normal operation mode
    - LIGHT_SLEEP: Quick optimizations (gradient clipping, KV cache management)
    - DEEP_SLEEP: Intensive optimizations (pruning, consolidation, recalibration)

    Attributes:
        model_id (str): Identifier for the controlled model
        model_state (LMModelState): State manager for the model
        performance_monitor (LMPerformanceMonitor): Performance tracking

    Example:
        >>> controller = AISleepController("gpt-neo-125M")
        >>> controller.configure_light_sleep(duration=300, strategies=[...])
        >>> controller.initiate_sleep(SleepTrigger.SCHEDULED)
    """

    def __init__(
        self, model_id: str, enable_monitoring: bool = True, auto_adapt: bool = True
    ):
        """
        Initialize AI Sleep Controller.

        Args:
            model_id: Unique identifier for the model
            enable_monitoring: Whether to enable performance monitoring
            auto_adapt: Whether to enable adaptive learning
        """
        self.model_id = model_id
        self.model_state = LMModelState(model_id=model_id)
        self.enable_monitoring = enable_monitoring
        self.auto_adapt = auto_adapt

        # Performance monitoring
        if enable_monitoring:
            self.performance_monitor = LMPerformanceMonitor(model_id=model_id)
            self.performance_monitor.register_alert_callback(
                self._handle_performance_alert
            )
        else:
            self.performance_monitor = None

        # Sleep configuration
        self.light_sleep_config: Dict[str, Any] = {
            "duration": 300,  # seconds
            "strategies": [
                OptimizationStrategy.GRADIENT_CLIPPING,
                OptimizationStrategy.KV_CACHE_MANAGEMENT,
            ],
            "gradient_clip_norm": 1.0,
            "kv_cache_retention": 0.8,  # Retain 80% of cache
        }

        self.deep_sleep_config: Dict[str, Any] = {
            "duration": 1800,  # seconds
            "strategies": [
                OptimizationStrategy.ATTENTION_HEAD_PRUNING,
                OptimizationStrategy.SEMANTIC_CONSOLIDATION,
                OptimizationStrategy.LAYER_NORM_RECALIBRATION,
            ],
            "pruning_threshold": 0.1,  # Prune heads with importance < 10%
            "consolidation_batch_size": 1000,
            "recalibration_epsilon": 1e-5,
        }

        # Trigger thresholds
        self.trigger_thresholds: Dict[SleepTrigger, Any] = {
            SleepTrigger.PERFORMANCE_DEGRADATION: 0.15,  # 15% degradation
            SleepTrigger.ANOMALY_DETECTED: 1,  # Any anomaly
            SleepTrigger.DRIFT_DETECTED: 0.05,  # p-value < 0.05
            SleepTrigger.MEMORY_PRESSURE: 0.9,  # 90% memory usage
            SleepTrigger.IDLE_TIMEOUT: 3600,  # 1 hour
        }

        # Adaptive learning
        self.learning_rate = 0.001
        self.learning_rate_schedule: List[Tuple[int, float]] = []
        self.adaptive_history: List[Dict[str, Any]] = []

        # Security
        self.security_patches_applied: List[Dict[str, Any]] = []
        self.vulnerability_checks: List[str] = [
            "gradient_explosion",
            "attention_collapse",
            "cache_overflow",
        ]

        # Callbacks
        self.sleep_callbacks: Dict[str, List[Callable]] = {
            "on_sleep_start": [],
            "on_sleep_end": [],
            "on_optimization_complete": [],
        }

        # Statistics
        self.total_sleep_cycles = 0
        self.total_light_sleep_time = 0.0
        self.total_deep_sleep_time = 0.0
        self.last_sleep_time: Optional[datetime] = None

    def configure_light_sleep(
        self,
        duration: Optional[int] = None,
        strategies: Optional[List[OptimizationStrategy]] = None,
        **kwargs,
    ) -> None:
        """
        Configure light sleep mode parameters.

        Args:
            duration: Sleep duration in seconds
            strategies: List of optimization strategies to apply
            **kwargs: Additional strategy-specific parameters
        """
        if duration is not None:
            self.light_sleep_config["duration"] = duration
        if strategies is not None:
            self.light_sleep_config["strategies"] = strategies
        self.light_sleep_config.update(kwargs)

    def configure_deep_sleep(
        self,
        duration: Optional[int] = None,
        strategies: Optional[List[OptimizationStrategy]] = None,
        **kwargs,
    ) -> None:
        """
        Configure deep sleep mode parameters.

        Args:
            duration: Sleep duration in seconds
            strategies: List of optimization strategies to apply
            **kwargs: Additional strategy-specific parameters
        """
        if duration is not None:
            self.deep_sleep_config["duration"] = duration
        if strategies is not None:
            self.deep_sleep_config["strategies"] = strategies
        self.deep_sleep_config.update(kwargs)

    def set_trigger_threshold(self, trigger: SleepTrigger, threshold: Any) -> None:
        """
        Set threshold for a specific sleep trigger.

        Args:
            trigger: Sleep trigger type
            threshold: Threshold value
        """
        self.trigger_thresholds[trigger] = threshold

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for sleep cycle events.

        Args:
            event: Event name (on_sleep_start, on_sleep_end, on_optimization_complete)
            callback: Callback function
        """
        if event in self.sleep_callbacks:
            self.sleep_callbacks[event].append(callback)

    def initiate_sleep(
        self, trigger: SleepTrigger, mode: Optional[SleepMode] = None
    ) -> bool:
        """
        Initiate a sleep cycle.

        Args:
            trigger: Reason for initiating sleep
            mode: Specific sleep mode (auto-determined if None)

        Returns:
            True if sleep was initiated successfully
        """
        # Determine sleep mode if not specified
        if mode is None:
            mode = self._determine_sleep_mode(trigger)

        # Transition to transitioning state
        try:
            self.model_state.transition_to(SleepMode.TRANSITIONING)
        except ValueError as e:
            return False

        # Execute callbacks
        for callback in self.sleep_callbacks["on_sleep_start"]:
            callback(mode, trigger)

        # Transition to target sleep mode
        self.model_state.transition_to(mode)
        self.last_sleep_time = datetime.now()

        # Apply optimizations based on mode
        if mode == SleepMode.LIGHT_SLEEP:
            self._execute_light_sleep()
        elif mode == SleepMode.DEEP_SLEEP:
            self._execute_deep_sleep()

        self.total_sleep_cycles += 1

        return True

    def wake_up(self) -> bool:
        """
        Wake the model from sleep mode.

        Returns:
            True if wake was successful
        """
        if self.model_state.current_mode == SleepMode.AWAKE:
            return True

        # Calculate sleep duration
        if self.last_sleep_time is not None:
            sleep_duration = (datetime.now() - self.last_sleep_time).total_seconds()
            if self.model_state.current_mode == SleepMode.LIGHT_SLEEP:
                self.total_light_sleep_time += sleep_duration
            elif self.model_state.current_mode == SleepMode.DEEP_SLEEP:
                self.total_deep_sleep_time += sleep_duration

        # Transition through transitioning state
        self.model_state.transition_to(SleepMode.TRANSITIONING)
        self.model_state.transition_to(SleepMode.AWAKE)

        # Execute callbacks
        for callback in self.sleep_callbacks["on_sleep_end"]:
            callback()

        return True

    def _determine_sleep_mode(self, trigger: SleepTrigger) -> SleepMode:
        """
        Determine appropriate sleep mode based on trigger.

        Args:
            trigger: Sleep trigger

        Returns:
            Recommended sleep mode
        """
        # Deep sleep for serious issues or scheduled maintenance
        deep_sleep_triggers = {SleepTrigger.DRIFT_DETECTED, SleepTrigger.SCHEDULED}

        if trigger in deep_sleep_triggers:
            return SleepMode.DEEP_SLEEP

        # Light sleep for minor issues
        return SleepMode.LIGHT_SLEEP

    def _execute_light_sleep(self) -> None:
        """Execute light sleep optimizations."""
        config = self.light_sleep_config

        for strategy in config["strategies"]:
            if strategy == OptimizationStrategy.GRADIENT_CLIPPING:
                self._apply_gradient_clipping(config.get("gradient_clip_norm", 1.0))
            elif strategy == OptimizationStrategy.KV_CACHE_MANAGEMENT:
                self._manage_kv_cache(config.get("kv_cache_retention", 0.8))
            elif strategy == OptimizationStrategy.ADAPTIVE_LEARNING_RATE:
                self._adjust_learning_rate()

            # Notify optimization complete
            for callback in self.sleep_callbacks["on_optimization_complete"]:
                callback(strategy)

    def _execute_deep_sleep(self) -> None:
        """Execute deep sleep optimizations."""
        config = self.deep_sleep_config

        for strategy in config["strategies"]:
            if strategy == OptimizationStrategy.ATTENTION_HEAD_PRUNING:
                self._prune_attention_heads(config.get("pruning_threshold", 0.1))
            elif strategy == OptimizationStrategy.SEMANTIC_CONSOLIDATION:
                self._consolidate_semantic_memory(
                    config.get("consolidation_batch_size", 1000)
                )
            elif strategy == OptimizationStrategy.LAYER_NORM_RECALIBRATION:
                self._recalibrate_layer_norms(config.get("recalibration_epsilon", 1e-5))
            elif strategy == OptimizationStrategy.SECURITY_PATCHING:
                self._apply_security_patches()

            # Notify optimization complete
            for callback in self.sleep_callbacks["on_optimization_complete"]:
                callback(strategy)

    def _apply_gradient_clipping(self, max_norm: float) -> None:
        """
        Apply gradient clipping optimization.

        Args:
            max_norm: Maximum gradient norm
        """
        norms = self.model_state.clip_gradients(max_norm=max_norm)

        # Record metrics
        if self.performance_monitor:
            avg_norm = np.mean(list(norms.values())) if norms else 0.0
            self.performance_monitor.track_metric("gradient_norm", avg_norm)

    def _manage_kv_cache(self, retention_ratio: float) -> None:
        """
        Manage KV cache by clearing old entries.

        Args:
            retention_ratio: Ratio of cache to retain (0.0 to 1.0)
        """
        cache_size = len(self.model_state.kv_cache)

        if cache_size == 0:
            return

        # Clear oldest entries
        entries_to_clear = int(cache_size * (1 - retention_ratio))
        if entries_to_clear > 0:
            layer_ids = sorted(
                self.model_state.kv_cache.keys(),
                key=lambda k: self.model_state.kv_cache[k].get(
                    "timestamp", datetime.min
                ),
            )[:entries_to_clear]

            self.model_state.clear_kv_cache(layer_ids)

        # Record metrics
        if self.performance_monitor:
            new_size = len(self.model_state.kv_cache)
            self.performance_monitor.track_metric("kv_cache_size", new_size)

    def _prune_attention_heads(self, threshold: float) -> None:
        """
        Prune low-importance attention heads.

        Args:
            threshold: Importance threshold for pruning
        """
        # Simulate attention head importance calculation
        # In production, this would analyze actual attention patterns
        num_layers = 12  # Example: 12 layers
        num_heads = 12  # Example: 12 heads per layer

        pruned_count = 0

        for layer_id in range(num_layers):
            layer_name = f"layer_{layer_id}"

            # Simulate importance scores (would be calculated from actual model)
            importance_scores = np.random.random(num_heads)

            # Prune heads below threshold
            heads_to_prune = [
                i for i, score in enumerate(importance_scores) if score < threshold
            ]

            if heads_to_prune:
                self.model_state.prune_attention_heads(layer_name, heads_to_prune)
                pruned_count += len(heads_to_prune)

        # Record metrics
        if self.performance_monitor:
            self.performance_monitor.track_metric("pruned_heads", pruned_count)

    def _consolidate_semantic_memory(self, batch_size: int) -> None:
        """
        Consolidate semantic information during deep sleep.

        Args:
            batch_size: Number of items to process in consolidation
        """
        # Simulate semantic consolidation
        # In production, this would process and compress semantic representations
        consolidation_data = {
            "processed_items": batch_size,
            "compression_ratio": 0.7,
            "timestamp": datetime.now(),
        }

        self.model_state.consolidate_semantic_memory(
            f"consolidation_{self.total_sleep_cycles}", consolidation_data
        )

        # Record metrics
        if self.performance_monitor:
            self.performance_monitor.track_metric(
                "semantic_memory_size", len(self.model_state.semantic_memory)
            )

    def _recalibrate_layer_norms(self, epsilon: float) -> None:
        """
        Recalibrate layer normalization parameters.

        Args:
            epsilon: Small constant for numerical stability
        """
        # Simulate layer norm recalibration
        # In production, this would analyze and adjust actual layer norm parameters
        num_layers = 12

        for layer_id in range(num_layers):
            layer_name = f"layer_{layer_id}"

            # Simulate recalibrated parameters
            gamma = 1.0 + np.random.normal(0, 0.01)  # Small adjustment to scale
            beta = np.random.normal(0, 0.01)  # Small adjustment to shift

            self.model_state.update_layer_norm(layer_name, gamma, beta)

        # Record metrics
        if self.performance_monitor:
            self.performance_monitor.track_metric(
                "layer_norms_recalibrated", num_layers
            )

    def _apply_security_patches(self) -> None:
        """Apply security patches and vulnerability fixes."""
        patches_applied = []

        for vulnerability in self.vulnerability_checks:
            # Simulate vulnerability checks and patching
            # In production, this would perform actual security checks
            if np.random.random() > 0.8:  # 20% chance of vulnerability
                patch = {
                    "vulnerability": vulnerability,
                    "severity": np.random.choice(["low", "medium", "high"]),
                    "timestamp": datetime.now(),
                    "status": "patched",
                }
                patches_applied.append(patch)

        self.security_patches_applied.extend(patches_applied)

        # Record metrics
        if self.performance_monitor:
            self.performance_monitor.track_metric(
                "security_patches", len(patches_applied)
            )

    def _adjust_learning_rate(self) -> None:
        """Adjust learning rate adaptively based on performance."""
        if not self.auto_adapt or not self.performance_monitor:
            return

        # Get recent loss trend
        loss_trend = self.performance_monitor.get_metric_trend("loss", window=10)

        if len(loss_trend) < 5:
            return

        # Simple adaptive strategy: reduce if loss plateaued, increase if improving
        recent_improvement = loss_trend[-1] < np.mean(loss_trend[-5:])

        if recent_improvement:
            self.learning_rate *= 1.05  # Increase by 5%
        else:
            self.learning_rate *= 0.95  # Decrease by 5%

        # Clamp learning rate
        self.learning_rate = np.clip(self.learning_rate, 1e-6, 1e-1)

        self.learning_rate_schedule.append(
            (self.total_sleep_cycles, self.learning_rate)
        )

        # Record metrics
        self.performance_monitor.track_metric("learning_rate", self.learning_rate)

    def _handle_performance_alert(self, alert: PerformanceAlert) -> None:
        """
        Handle performance alerts from monitoring system.

        Args:
            alert: Performance alert object
        """
        # Auto-initiate sleep based on alert type
        if alert.alert_type == "anomaly":
            trigger = SleepTrigger.ANOMALY_DETECTED
        elif alert.alert_type == "drift":
            trigger = SleepTrigger.DRIFT_DETECTED
        elif alert.severity == "high":
            trigger = SleepTrigger.PERFORMANCE_DEGRADATION
        else:
            return

        # Only initiate if currently awake
        if self.model_state.current_mode == SleepMode.AWAKE:
            self.initiate_sleep(trigger)

    def get_sleep_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive sleep cycle statistics.

        Returns:
            Dictionary containing sleep statistics
        """
        total_time = (datetime.now() - self.model_state.creation_time).total_seconds()
        awake_time = (
            total_time - self.total_light_sleep_time - self.total_deep_sleep_time
        )

        return {
            "model_id": self.model_id,
            "total_sleep_cycles": self.total_sleep_cycles,
            "current_mode": self.model_state.current_mode.value,
            "total_light_sleep_time": self.total_light_sleep_time,
            "total_deep_sleep_time": self.total_deep_sleep_time,
            "total_awake_time": awake_time,
            "sleep_ratio": (
                (self.total_light_sleep_time + self.total_deep_sleep_time) / total_time
                if total_time > 0
                else 0
            ),
            "last_sleep": (
                self.last_sleep_time.isoformat() if self.last_sleep_time else None
            ),
            "learning_rate": self.learning_rate,
            "security_patches_applied": len(self.security_patches_applied),
        }

    def export_state(self, filepath: str) -> None:
        """
        Export complete controller state.

        Args:
            filepath: Path to save state
        """
        self.model_state.save_state(filepath)

    @classmethod
    def from_state(cls, filepath: str) -> "AISleepController":
        """
        Create controller from saved state.

        Args:
            filepath: Path to saved state

        Returns:
            Restored AISleepController instance
        """
        model_state = LMModelState.load_state(filepath)
        controller = cls(model_id=model_state.model_id)
        controller.model_state = model_state
        return controller
