# API Reference

Complete API reference for AI Sleep Constructs.

## Table of Contents

- [ai_sleep](#ai_sleep)
  - [LMModelState](#lmmodelstate)
  - [LMPerformanceMonitor](#lmperformancemonitor)
  - [AISleepController](#aisleepcontroller)
  - [HuggingFaceModelAdapter](#huggingfacemodeladapter)
- [Enumerations](#enumerations)
- [Data Classes](#data-classes)

---

## ai_sleep

Main package containing all core components.

```python
from ai_sleep import (
    LMModelState,
    LMPerformanceMonitor,
    AISleepController
)
```

---

## LMModelState

Manages the state of a language model during sleep cycles.

### Class Definition

```python
class LMModelState:
    def __init__(
        self,
        model_id: str,
        initial_mode: SleepMode = SleepMode.AWAKE,
        track_history: bool = True
    )
```

### Parameters

- **model_id** (str): Unique identifier for this model instance
- **initial_mode** (SleepMode, optional): Starting sleep mode. Default: `SleepMode.AWAKE`
- **track_history** (bool, optional): Whether to maintain state history. Default: `True`

### Attributes

- **model_id** (str): Model identifier
- **current_mode** (SleepMode): Current sleep mode
- **state_history** (List[StateSnapshot]): Historical snapshots
- **kv_cache** (Dict): Key-Value cache for attention
- **attention_heads** (Dict): Attention head configurations
- **layer_norms** (Dict): Layer normalization parameters
- **gradients** (Dict): Gradient information
- **semantic_memory** (Dict): Consolidated semantic information
- **performance_metrics** (Dict): Performance metric history

### Methods

#### transition_to

```python
def transition_to(self, new_mode: SleepMode) -> None
```

Transition the model to a new sleep mode.

**Parameters:**
- **new_mode** (SleepMode): Target sleep mode

**Raises:**
- **ValueError**: If transition is invalid

**Example:**
```python
state = LMModelState("gpt-neo")
state.transition_to(SleepMode.LIGHT_SLEEP)
```

#### update_kv_cache

```python
def update_kv_cache(self, layer_id: str, keys: Any, values: Any) -> None
```

Update Key-Value cache for a specific layer.

**Parameters:**
- **layer_id** (str): Layer identifier
- **keys** (Any): Key tensors
- **values** (Any): Value tensors

#### clear_kv_cache

```python
def clear_kv_cache(self, layer_ids: Optional[List[str]] = None) -> None
```

Clear KV cache for specified layers or all layers.

**Parameters:**
- **layer_ids** (Optional[List[str]]): Specific layers to clear, or None for all

#### prune_attention_heads

```python
def prune_attention_heads(self, layer_id: str, head_ids: List[int]) -> None
```

Mark attention heads as pruned for a specific layer.

**Parameters:**
- **layer_id** (str): Layer identifier
- **head_ids** (List[int]): Indices of heads to prune

#### update_layer_norm

```python
def update_layer_norm(self, layer_id: str, gamma: float, beta: float) -> None
```

Update layer normalization parameters.

**Parameters:**
- **layer_id** (str): Layer identifier
- **gamma** (float): Scale parameter
- **beta** (float): Shift parameter

#### clip_gradients

```python
def clip_gradients(self, max_norm: float = 1.0) -> Dict[str, float]
```

Apply gradient clipping to stored gradients.

**Parameters:**
- **max_norm** (float, optional): Maximum gradient norm. Default: 1.0

**Returns:**
- **Dict[str, float]**: Original gradient norms before clipping

#### consolidate_semantic_memory

```python
def consolidate_semantic_memory(self, key: str, value: Any) -> None
```

Store consolidated semantic information.

**Parameters:**
- **key** (str): Memory key/identifier
- **value** (Any): Semantic information to store

#### save_state

```python
def save_state(self, filepath: Union[str, Path]) -> None
```

Save complete model state to disk.

**Parameters:**
- **filepath** (Union[str, Path]): Path where state should be saved

#### load_state

```python
@classmethod
def load_state(cls, filepath: Union[str, Path]) -> "LMModelState"
```

Load model state from disk.

**Parameters:**
- **filepath** (Union[str, Path]): Path to saved state file

**Returns:**
- **LMModelState**: Restored state instance

---

## LMPerformanceMonitor

Comprehensive performance monitoring for language models.

### Class Definition

```python
class LMPerformanceMonitor:
    def __init__(
        self,
        model_id: str,
        drift_window: int = 100,
        anomaly_method: str = "zscore",
        anomaly_threshold: float = 3.0
    )
```

### Parameters

- **model_id** (str): Unique identifier for the model
- **drift_window** (int, optional): Window size for drift detection. Default: 100
- **anomaly_method** (str, optional): Anomaly detection method. Default: "zscore"
  - Options: "zscore", "iqr", "moving_avg"
- **anomaly_threshold** (float, optional): Threshold for anomaly detection. Default: 3.0

### Methods

#### register_metric

```python
def register_metric(
    self,
    metric_name: str,
    track_drift: bool = True,
    track_anomalies: bool = True,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None
) -> None
```

Register a new metric for monitoring.

**Parameters:**
- **metric_name** (str): Name of the metric
- **track_drift** (bool): Enable drift detection
- **track_anomalies** (bool): Enable anomaly detection
- **min_threshold** (Optional[float]): Minimum acceptable value
- **max_threshold** (Optional[float]): Maximum acceptable value

#### track_metric

```python
def track_metric(self, metric_name: str, value: float) -> None
```

Record a metric value and perform checks.

**Parameters:**
- **metric_name** (str): Name of the metric
- **value** (float): Metric value to record

#### establish_baseline

```python
def establish_baseline(self, metric_name: str) -> None
```

Mark current samples as baseline for drift detection.

**Parameters:**
- **metric_name** (str): Name of the metric

#### check_drift

```python
def check_drift(self, metric_name: str) -> Tuple[bool, float]
```

Check if drift has occurred for a metric.

**Parameters:**
- **metric_name** (str): Name of the metric

**Returns:**
- **Tuple[bool, float]**: (drift_detected, p_value)

#### get_metric_statistics

```python
def get_metric_statistics(self, metric_name: str) -> Dict[str, float]
```

Get statistical summary for a metric.

**Parameters:**
- **metric_name** (str): Name of the metric

**Returns:**
- **Dict[str, float]**: Statistical measures (mean, median, std, min, max, count, latest)

#### register_alert_callback

```python
def register_alert_callback(
    self,
    callback: Callable[[PerformanceAlert], None]
) -> None
```

Register a callback function for alerts.

**Parameters:**
- **callback** (Callable): Function to call with PerformanceAlert objects

---

## AISleepController

Main controller for orchestrating AI sleep cycles.

### Class Definition

```python
class AISleepController:
    def __init__(
        self,
        model_id: str,
        enable_monitoring: bool = True,
        auto_adapt: bool = True
    )
```

### Parameters

- **model_id** (str): Unique identifier for the model
- **enable_monitoring** (bool, optional): Enable performance monitoring. Default: True
- **auto_adapt** (bool, optional): Enable adaptive learning. Default: True

### Methods

#### configure_light_sleep

```python
def configure_light_sleep(
    self,
    duration: Optional[int] = None,
    strategies: Optional[List[OptimizationStrategy]] = None,
    **kwargs
) -> None
```

Configure light sleep mode parameters.

**Parameters:**
- **duration** (Optional[int]): Sleep duration in seconds
- **strategies** (Optional[List[OptimizationStrategy]]): Optimization strategies
- ****kwargs**: Additional strategy-specific parameters

**Example:**
```python
controller.configure_light_sleep(
    duration=300,
    strategies=[
        OptimizationStrategy.GRADIENT_CLIPPING,
        OptimizationStrategy.KV_CACHE_MANAGEMENT
    ],
    gradient_clip_norm=1.0,
    kv_cache_retention=0.8
)
```

#### configure_deep_sleep

```python
def configure_deep_sleep(
    self,
    duration: Optional[int] = None,
    strategies: Optional[List[OptimizationStrategy]] = None,
    **kwargs
) -> None
```

Configure deep sleep mode parameters.

**Parameters:**
- **duration** (Optional[int]): Sleep duration in seconds
- **strategies** (Optional[List[OptimizationStrategy]]): Optimization strategies
- ****kwargs**: Additional strategy-specific parameters

#### set_trigger_threshold

```python
def set_trigger_threshold(self, trigger: SleepTrigger, threshold: Any) -> None
```

Set threshold for a specific sleep trigger.

**Parameters:**
- **trigger** (SleepTrigger): Sleep trigger type
- **threshold** (Any): Threshold value

#### register_callback

```python
def register_callback(self, event: str, callback: Callable) -> None
```

Register a callback for sleep cycle events.

**Parameters:**
- **event** (str): Event name
  - Options: "on_sleep_start", "on_sleep_end", "on_optimization_complete"
- **callback** (Callable): Callback function

#### initiate_sleep

```python
def initiate_sleep(
    self,
    trigger: SleepTrigger,
    mode: Optional[SleepMode] = None
) -> bool
```

Initiate a sleep cycle.

**Parameters:**
- **trigger** (SleepTrigger): Reason for initiating sleep
- **mode** (Optional[SleepMode]): Specific sleep mode (auto-determined if None)

**Returns:**
- **bool**: True if sleep was initiated successfully

#### wake_up

```python
def wake_up(self) -> bool
```

Wake the model from sleep mode.

**Returns:**
- **bool**: True if wake was successful

#### get_sleep_statistics

```python
def get_sleep_statistics(self) -> Dict[str, Any]
```

Get comprehensive sleep cycle statistics.

**Returns:**
- **Dict[str, Any]**: Sleep statistics

---

## HuggingFaceModelAdapter

Adapter for integrating AI Sleep with HuggingFace models.

### Class Definition

```python
class HuggingFaceModelAdapter:
    def __init__(
        self,
        model: Any,
        model_id: str,
        enable_monitoring: bool = True,
        auto_adapt: bool = True
    )
```

### Parameters

- **model** (Any): HuggingFace model instance (or None for mock)
- **model_id** (str): Model identifier
- **enable_monitoring** (bool): Enable performance monitoring
- **auto_adapt** (bool): Enable adaptive learning

### Methods

#### enable_sleep_cycles

```python
def enable_sleep_cycles(self) -> None
```

Enable AI Sleep cycles for the model.

#### initiate_light_sleep

```python
def initiate_light_sleep(self, duration: int = 300) -> bool
```

Initiate light sleep optimization cycle.

**Parameters:**
- **duration** (int): Sleep duration in seconds

**Returns:**
- **bool**: True if sleep was initiated

#### track_inference_metrics

```python
def track_inference_metrics(
    self,
    perplexity: Optional[float] = None,
    loss: Optional[float] = None,
    inference_time: Optional[float] = None,
    memory_usage: Optional[float] = None
) -> None
```

Track inference metrics for performance monitoring.

**Parameters:**
- **perplexity** (Optional[float]): Model perplexity score
- **loss** (Optional[float]): Loss value
- **inference_time** (Optional[float]): Inference time in seconds
- **memory_usage** (Optional[float]): Memory usage in MB

---

## Enumerations

### SleepMode

```python
class SleepMode(Enum):
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    TRANSITIONING = "transitioning"
```

### SleepTrigger

```python
class SleepTrigger(Enum):
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_DETECTED = "anomaly_detected"
    DRIFT_DETECTED = "drift_detected"
    MEMORY_PRESSURE = "memory_pressure"
    MANUAL = "manual"
    IDLE_TIMEOUT = "idle_timeout"
```

### OptimizationStrategy

```python
class OptimizationStrategy(Enum):
    GRADIENT_CLIPPING = "gradient_clipping"
    ATTENTION_HEAD_PRUNING = "attention_head_pruning"
    KV_CACHE_MANAGEMENT = "kv_cache_management"
    SEMANTIC_CONSOLIDATION = "semantic_consolidation"
    LAYER_NORM_RECALIBRATION = "layer_norm_recalibration"
    SECURITY_PATCHING = "security_patching"
    ADAPTIVE_LEARNING_RATE = "adaptive_learning_rate"
```

---

## Data Classes

### StateSnapshot

```python
@dataclass
class StateSnapshot:
    timestamp: datetime
    sleep_mode: SleepMode
    metrics: Dict[str, float]
    parameters_hash: str
    metadata: Dict[str, Any]
```

### PerformanceAlert

```python
@dataclass
class PerformanceAlert:
    timestamp: datetime
    alert_type: str
    severity: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    metadata: Dict[str, Any]
```

---

## Utility Functions

### create_sleep_enabled_model

```python
def create_sleep_enabled_model(
    model_name_or_path: str,
    enable_monitoring: bool = True,
    auto_adapt: bool = True,
    load_model: bool = False
) -> HuggingFaceModelAdapter
```

Create a sleep-enabled HuggingFace model.

**Parameters:**
- **model_name_or_path** (str): HuggingFace model identifier
- **enable_monitoring** (bool): Enable performance monitoring
- **auto_adapt** (bool): Enable adaptive learning
- **load_model** (bool): Whether to load actual model

**Returns:**
- **HuggingFaceModelAdapter**: Adapter instance

### configure_optimal_sleep_schedule

```python
def configure_optimal_sleep_schedule(
    adapter: HuggingFaceModelAdapter,
    workload_type: str = "continuous"
) -> None
```

Configure optimal sleep schedule based on workload type.

**Parameters:**
- **adapter** (HuggingFaceModelAdapter): Model adapter
- **workload_type** (str): Workload type
  - Options: "continuous", "batch", "interactive"

---

## Type Hints

The library extensively uses type hints for better IDE support and type checking:

```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
```

## Error Handling

### Common Exceptions

- **ValueError**: Invalid parameter values or state transitions
- **IOError**: File operations (save/load state)
- **KeyError**: Missing metric or configuration

### Best Practices

```python
try:
    controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
except ValueError as e:
    print(f"Invalid transition: {e}")
```

---

For architecture details, see [architecture.md](architecture.md).
For usage examples, see [../examples/](../examples/).
