"""
LMModelState: State management for language model sleep cycles.

This module provides comprehensive state tracking, serialization, and restoration
capabilities for language models undergoing sleep optimization cycles.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class SleepMode(Enum):
    """Enumeration of available sleep modes."""

    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    TRANSITIONING = "transitioning"


@dataclass
class StateSnapshot:
    """Represents a point-in-time snapshot of model state."""

    timestamp: datetime
    sleep_mode: SleepMode
    metrics: Dict[str, float]
    parameters_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LMModelState:
    """
    Manages the state of a language model during sleep cycles.

    This class tracks model parameters, attention states, KV cache, layer norms,
    and other critical components that need to be managed during light and deep
    sleep optimization cycles.

    Attributes:
        model_id (str): Unique identifier for the model instance
        current_mode (SleepMode): Current sleep mode of the model
        state_history (List[StateSnapshot]): Historical snapshots of model state
        kv_cache (Dict): Key-Value cache for attention mechanisms
        attention_heads (Dict): Active attention head configurations
        layer_norms (Dict): Layer normalization parameters
        gradients (Dict): Gradient information for optimization

    Example:
        >>> state = LMModelState(model_id="gpt-neo-125M")
        >>> state.transition_to(SleepMode.LIGHT_SLEEP)
        >>> state.save_snapshot("checkpoint.pkl")
    """

    def __init__(
        self,
        model_id: str,
        initial_mode: SleepMode = SleepMode.AWAKE,
        track_history: bool = True,
    ):
        """
        Initialize model state tracker.

        Args:
            model_id: Unique identifier for this model instance
            initial_mode: Starting sleep mode (default: AWAKE)
            track_history: Whether to maintain state history (default: True)
        """
        self.model_id = model_id
        self.current_mode = initial_mode
        self.track_history = track_history
        self.state_history: List[StateSnapshot] = []

        # Core state components
        self.kv_cache: Dict[str, Any] = {}
        self.attention_heads: Dict[str, Dict[str, Any]] = {}
        self.layer_norms: Dict[str, Dict[str, float]] = {}
        self.gradients: Dict[str, np.ndarray] = {}
        self.semantic_memory: Dict[str, Any] = {}

        # Metadata
        self.creation_time = datetime.now()
        self.last_transition_time = datetime.now()
        self.sleep_cycle_count = 0
        self.total_awake_time = 0.0
        self.total_sleep_time = 0.0

        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            "perplexity": [],
            "loss": [],
            "inference_time": [],
            "memory_usage": [],
        }

    def transition_to(self, new_mode: SleepMode) -> None:
        """
        Transition the model to a new sleep mode.

        Args:
            new_mode: Target sleep mode to transition to

        Raises:
            ValueError: If transition is not valid
        """
        if new_mode == self.current_mode:
            return

        # Validate transition
        valid_transitions = {
            SleepMode.AWAKE: [SleepMode.LIGHT_SLEEP, SleepMode.TRANSITIONING],
            SleepMode.LIGHT_SLEEP: [
                SleepMode.AWAKE,
                SleepMode.DEEP_SLEEP,
                SleepMode.TRANSITIONING,
            ],
            SleepMode.DEEP_SLEEP: [SleepMode.LIGHT_SLEEP, SleepMode.TRANSITIONING],
            SleepMode.TRANSITIONING: [
                SleepMode.AWAKE,
                SleepMode.LIGHT_SLEEP,
                SleepMode.DEEP_SLEEP,
            ],
        }

        if new_mode not in valid_transitions.get(self.current_mode, []):
            raise ValueError(
                f"Invalid transition from {self.current_mode.value} to {new_mode.value}"
            )

        old_mode = self.current_mode
        self.current_mode = new_mode
        self.last_transition_time = datetime.now()

        if new_mode in [SleepMode.LIGHT_SLEEP, SleepMode.DEEP_SLEEP]:
            self.sleep_cycle_count += 1

    def update_kv_cache(self, layer_id: str, keys: Any, values: Any) -> None:
        """
        Update Key-Value cache for a specific layer.

        Args:
            layer_id: Identifier for the model layer
            keys: Key tensors for attention
            values: Value tensors for attention
        """
        self.kv_cache[layer_id] = {
            "keys": keys,
            "values": values,
            "timestamp": datetime.now(),
        }

    def clear_kv_cache(self, layer_ids: Optional[List[str]] = None) -> None:
        """
        Clear KV cache for specified layers or all layers.

        Args:
            layer_ids: Specific layers to clear, or None for all
        """
        if layer_ids is None:
            self.kv_cache.clear()
        else:
            for layer_id in layer_ids:
                self.kv_cache.pop(layer_id, None)

    def prune_attention_heads(self, layer_id: str, head_ids: List[int]) -> None:
        """
        Mark attention heads as pruned for a specific layer.

        Args:
            layer_id: Identifier for the model layer
            head_ids: List of attention head indices to prune
        """
        if layer_id not in self.attention_heads:
            self.attention_heads[layer_id] = {"active": [], "pruned": []}

        self.attention_heads[layer_id]["pruned"].extend(head_ids)
        self.attention_heads[layer_id]["active"] = [
            h
            for h in self.attention_heads[layer_id].get("active", [])
            if h not in head_ids
        ]

    def update_layer_norm(self, layer_id: str, gamma: float, beta: float) -> None:
        """
        Update layer normalization parameters.

        Args:
            layer_id: Identifier for the model layer
            gamma: Scale parameter
            beta: Shift parameter
        """
        self.layer_norms[layer_id] = {
            "gamma": gamma,
            "beta": beta,
            "timestamp": datetime.now().isoformat(),
        }

    def store_gradient(self, param_name: str, gradient: np.ndarray) -> None:
        """
        Store gradient information for a parameter.

        Args:
            param_name: Name of the parameter
            gradient: Gradient array
        """
        self.gradients[param_name] = gradient

    def clip_gradients(self, max_norm: float = 1.0) -> Dict[str, float]:
        """
        Apply gradient clipping to stored gradients.

        Args:
            max_norm: Maximum gradient norm (default: 1.0)

        Returns:
            Dictionary of original gradient norms before clipping
        """
        original_norms = {}

        for param_name, gradient in self.gradients.items():
            grad_norm = np.linalg.norm(gradient)
            original_norms[param_name] = float(grad_norm)

            if grad_norm > max_norm:
                self.gradients[param_name] = gradient * (max_norm / grad_norm)

        return original_norms

    def consolidate_semantic_memory(self, key: str, value: Any) -> None:
        """
        Store consolidated semantic information.

        Args:
            key: Memory key/identifier
            value: Semantic information to store
        """
        self.semantic_memory[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "consolidation_cycle": self.sleep_cycle_count,
        }

    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Record a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value to record
        """
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append(value)

    def create_snapshot(self) -> StateSnapshot:
        """
        Create a snapshot of current model state.

        Returns:
            StateSnapshot object containing current state
        """
        metrics = {
            name: values[-1] if values else 0.0
            for name, values in self.performance_metrics.items()
        }

        # Create a simple hash of the current state
        state_str = (
            f"{self.model_id}_{self.sleep_cycle_count}_{self.current_mode.value}"
        )
        params_hash = str(hash(state_str))

        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            sleep_mode=self.current_mode,
            metrics=metrics,
            parameters_hash=params_hash,
            metadata={
                "sleep_cycle_count": self.sleep_cycle_count,
                "kv_cache_size": len(self.kv_cache),
                "attention_heads_count": len(self.attention_heads),
                "semantic_memory_size": len(self.semantic_memory),
            },
        )

        if self.track_history:
            self.state_history.append(snapshot)

        return snapshot

    def save_state(self, filepath: Union[str, Path]) -> None:
        """
        Save complete model state to disk.

        Args:
            filepath: Path where state should be saved
        """
        filepath = Path(filepath)

        state_dict = {
            "model_id": self.model_id,
            "current_mode": self.current_mode.value,
            "kv_cache": self.kv_cache,
            "attention_heads": self.attention_heads,
            "layer_norms": self.layer_norms,
            "gradients": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.gradients.items()
            },
            "semantic_memory": self.semantic_memory,
            "performance_metrics": self.performance_metrics,
            "sleep_cycle_count": self.sleep_cycle_count,
            "creation_time": self.creation_time.isoformat(),
            "last_transition_time": self.last_transition_time.isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load_state(cls, filepath: Union[str, Path]) -> "LMModelState":
        """
        Load model state from disk.

        Args:
            filepath: Path to saved state file

        Returns:
            Restored LMModelState instance
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)

        instance = cls(
            model_id=state_dict["model_id"],
            initial_mode=SleepMode(state_dict["current_mode"]),
        )

        instance.kv_cache = state_dict["kv_cache"]
        instance.attention_heads = state_dict["attention_heads"]
        instance.layer_norms = state_dict["layer_norms"]
        instance.gradients = {
            k: np.array(v) for k, v in state_dict["gradients"].items()
        }
        instance.semantic_memory = state_dict["semantic_memory"]
        instance.performance_metrics = state_dict["performance_metrics"]
        instance.sleep_cycle_count = state_dict["sleep_cycle_count"]
        instance.creation_time = datetime.fromisoformat(state_dict["creation_time"])
        instance.last_transition_time = datetime.fromisoformat(
            state_dict["last_transition_time"]
        )

        return instance

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state.

        Returns:
            Dictionary containing state summary information
        """
        return {
            "model_id": self.model_id,
            "current_mode": self.current_mode.value,
            "sleep_cycle_count": self.sleep_cycle_count,
            "kv_cache_entries": len(self.kv_cache),
            "attention_heads_tracked": len(self.attention_heads),
            "layer_norms_tracked": len(self.layer_norms),
            "gradient_params": len(self.gradients),
            "semantic_memory_entries": len(self.semantic_memory),
            "performance_metrics": list(self.performance_metrics.keys()),
            "uptime_seconds": (datetime.now() - self.creation_time).total_seconds(),
        }
