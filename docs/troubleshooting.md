# Troubleshooting Guide

Common issues and solutions when using AI Sleep Constructs.

## Table of Contents

- [Installation Issues](#installation-issues)
- [State Transition Errors](#state-transition-errors)
- [Performance Monitoring Issues](#performance-monitoring-issues)
- [Integration Problems](#integration-problems)
- [Memory and Performance](#memory-and-performance)
- [Testing Issues](#testing-issues)
- [Best Practices](#best-practices)

---

## Installation Issues

### Issue: Module not found

**Symptom:**
```python
ModuleNotFoundError: No module named 'ai_sleep'
```

**Solution:**
```bash
# Ensure package is installed
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/ai-sleep/src"
```

### Issue: Dependency conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv_fresh
source venv_fresh/bin/activate
pip install -e .

# Or use specific versions
pip install numpy==1.21.0 scipy==1.7.0
```

### Issue: HuggingFace transformers not found

**Symptom:**
```
UserWarning: transformers library not available. Running in mock mode.
```

**Solution:**
```bash
# Install HuggingFace extras
pip install -e ".[huggingface]"

# Or install directly
pip install transformers>=4.20.0
```

---

## State Transition Errors

### Issue: Invalid transition error

**Symptom:**
```python
ValueError: Invalid transition from awake to deep_sleep
```

**Cause:** Direct transition from AWAKE to DEEP_SLEEP is not allowed.

**Solution:**
```python
# Must transition through LIGHT_SLEEP
controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.DEEP_SLEEP)
```

### Issue: Cannot transition while in transitioning state

**Symptom:**
```python
ValueError: Invalid transition from transitioning to light_sleep
```

**Cause:** Attempting to initiate sleep while already transitioning.

**Solution:**
```python
# Wait for transition to complete
import time
time.sleep(0.1)  # Brief wait for transition

# Or check current mode first
if controller.model_state.current_mode != SleepMode.TRANSITIONING:
    controller.initiate_sleep(trigger, mode)
```

### Issue: Model stuck in transitioning state

**Symptom:** Model remains in TRANSITIONING mode indefinitely.

**Cause:** Exception during transition interrupted the process.

**Solution:**
```python
# Manually reset to AWAKE (use with caution)
controller.model_state.current_mode = SleepMode.AWAKE

# Or create new controller
controller = AISleepController(
    model_id=old_controller.model_id
)
```

---

## Performance Monitoring Issues

### Issue: Drift not detected when expected

**Symptom:** Performance has clearly degraded, but drift detection doesn't trigger.

**Cause:** 
1. Insufficient samples for statistical significance
2. Baseline not established
3. Gradual drift below detection threshold

**Solution:**
```python
# Ensure sufficient samples (at least 30)
for i in range(50):
    monitor.track_metric("loss", value)

# Establish baseline explicitly
monitor.establish_baseline("loss")

# Lower significance threshold
monitor = LMPerformanceMonitor(
    model_id="model",
    drift_window=50,  # Smaller window
    anomaly_threshold=2.0  # More sensitive
)
```

### Issue: Too many false positive alerts

**Symptom:** Alerts triggered for normal variation.

**Cause:** Thresholds too sensitive.

**Solution:**
```python
# Increase anomaly threshold
monitor = LMPerformanceMonitor(
    model_id="model",
    anomaly_threshold=4.0  # Less sensitive
)

# Or use different method
monitor = LMPerformanceMonitor(
    model_id="model",
    anomaly_method="iqr"  # More robust to outliers
)

# Widen acceptable ranges
monitor.register_metric(
    "perplexity",
    min_threshold=1.0,  # Wider range
    max_threshold=10.0
)
```

### Issue: Metrics not being tracked

**Symptom:** `get_metric_statistics()` returns empty dict.

**Cause:** Metric not registered or tracking not started.

**Solution:**
```python
# Register metric before tracking
monitor.register_metric("custom_metric")

# Then track values
monitor.track_metric("custom_metric", value)

# Verify registration
print(monitor.metrics.keys())
```

---

## Integration Problems

### Issue: HuggingFace model not loading

**Symptom:**
```
Warning: Could not load model: ... Running in mock mode.
```

**Cause:** Model not available or transformers not installed.

**Solution:**
```python
# Install transformers
pip install transformers

# For mock testing, this is expected behavior
adapter = create_sleep_enabled_model(
    "gpt2",
    load_model=False  # Explicitly use mock mode
)

# For actual model
adapter = create_sleep_enabled_model(
    "gpt2",
    load_model=True  # Load real model
)
```

### Issue: Callbacks not executing

**Symptom:** Registered callbacks don't fire during sleep cycles.

**Cause:** Callback registered for wrong event or has errors.

**Solution:**
```python
# Verify event name is correct
valid_events = ["on_sleep_start", "on_sleep_end", "on_optimization_complete"]

# Add error handling to callback
def safe_callback(*args):
    try:
        # Your callback logic
        print(f"Callback executed with args: {args}")
    except Exception as e:
        print(f"Callback error: {e}")

controller.register_callback("on_sleep_start", safe_callback)
```

### Issue: State not persisting after save/load

**Symptom:** Loaded state doesn't match saved state.

**Cause:** State components not serializable or path issues.

**Solution:**
```python
import pickle
from pathlib import Path

# Use absolute paths
save_path = Path("/absolute/path/to/state.pkl").resolve()

# Verify save succeeded
controller.export_state(save_path)
assert save_path.exists()

# Verify load
loaded = AISleepController.from_state(save_path)
assert loaded.model_id == controller.model_id

# For non-serializable components, implement custom serialization
```

---

## Memory and Performance

### Issue: High memory usage

**Symptom:** Memory consumption grows over time.

**Cause:** 
1. State history accumulating
2. KV cache not being cleared
3. Performance metrics accumulating

**Solution:**
```python
# Disable history tracking
state = LMModelState(model_id="model", track_history=False)

# Clear KV cache more aggressively
controller.configure_light_sleep(
    kv_cache_retention=0.5  # Clear 50% of cache
)

# Limit metric history
monitor.metrics["loss"] = monitor.metrics["loss"][-1000:]  # Keep last 1000

# Clear old alerts
monitor.clear_alerts()
```

### Issue: Sleep cycles taking too long

**Symptom:** Model unavailable for extended periods.

**Cause:** Sleep duration too long or optimizations too intensive.

**Solution:**
```python
# Reduce sleep duration
controller.configure_light_sleep(duration=60)  # 1 minute
controller.configure_deep_sleep(duration=300)  # 5 minutes

# Reduce optimization scope
controller.configure_light_sleep(
    strategies=[OptimizationStrategy.KV_CACHE_MANAGEMENT]  # Only cache management
)

# Schedule during low-traffic periods
import schedule
schedule.every().day.at("03:00").do(
    lambda: controller.initiate_sleep(SleepTrigger.SCHEDULED, SleepMode.DEEP_SLEEP)
)
```

### Issue: Slow drift detection

**Symptom:** Drift checks take too long.

**Cause:** Large window sizes or many samples.

**Solution:**
```python
# Reduce drift window
monitor = LMPerformanceMonitor(
    model_id="model",
    drift_window=50  # Smaller window
)

# Check drift less frequently
# Instead of every iteration:
if iteration % 100 == 0:  # Check every 100 iterations
    drift_detected, p_value = monitor.check_drift("loss")
```

---

## Testing Issues

### Issue: Tests failing with numpy boolean error

**Symptom:**
```
AssertionError: np.False_ is not an instance of <class 'bool'>
```

**Cause:** NumPy boolean returned instead of Python boolean.

**Solution:**
```python
# Already fixed in codebase, but if extending:
# Convert numpy booleans to Python booleans
drift_detected = bool(p_value < self.alpha)
return drift_detected, float(p_value)
```

### Issue: Tests timeout

**Symptom:** Tests hang and eventually timeout.

**Cause:** Waiting for sleep cycles to complete.

**Solution:**
```python
# Use shorter durations in tests
controller.configure_light_sleep(duration=1)  # 1 second for tests
controller.configure_deep_sleep(duration=2)

# Or mock time.sleep
from unittest.mock import patch

with patch('time.sleep'):
    controller.initiate_sleep(trigger, mode)
```

---

## Best Practices

### Proper Initialization

```python
# Always use context-appropriate settings
# For development
controller = AISleepController(
    model_id="dev-model",
    enable_monitoring=True,  # Full monitoring
    auto_adapt=True
)

# For production
controller = AISleepController(
    model_id="prod-model",
    enable_monitoring=True,
    auto_adapt=True
)
controller.configure_light_sleep(duration=300)
controller.configure_deep_sleep(duration=1800)
```

### Error Handling

```python
# Always wrap sleep operations
try:
    success = controller.initiate_sleep(trigger, mode)
    if not success:
        logger.warning("Sleep initiation failed")
except ValueError as e:
    logger.error(f"Invalid sleep transition: {e}")
except Exception as e:
    logger.error(f"Unexpected error during sleep: {e}")
```

### Resource Management

```python
# Clean up resources
try:
    # Your code
    controller.initiate_sleep(...)
finally:
    # Save state before exit
    controller.export_state("final_state.pkl")
    
    # Clear large caches
    controller.model_state.clear_kv_cache()
    
    # Clear monitoring history
    if controller.performance_monitor:
        controller.performance_monitor.clear_alerts()
```

### Monitoring Configuration

```python
# Set up comprehensive monitoring
monitor = controller.performance_monitor

# Register all important metrics
for metric in ["perplexity", "loss", "latency", "memory"]:
    monitor.register_metric(
        metric,
        track_drift=True,
        track_anomalies=True
    )

# Set up alert callbacks
def handle_alert(alert):
    if alert.severity == "high":
        # Log to monitoring system
        logger.critical(f"High severity alert: {alert.message}")
        # Potentially initiate sleep
        if alert.alert_type == "drift":
            controller.initiate_sleep(SleepTrigger.DRIFT_DETECTED)

monitor.register_alert_callback(handle_alert)
```

### Testing Integration

```python
# Test in stages
# 1. Test without actual model
adapter = create_sleep_enabled_model("gpt2", load_model=False)
adapter.enable_sleep_cycles()
adapter.initiate_light_sleep(duration=1)
adapter.wake_model()

# 2. Test with small model
adapter = create_sleep_enabled_model("distilgpt2", load_model=True)
# ... test operations

# 3. Test with production model
# Only after stages 1-2 pass
```

---

## Getting Help

If you're still experiencing issues:

1. **Check the Examples**: Review [examples/](../examples/) for working code
2. **Read the Docs**: See [architecture.md](architecture.md) and [api_reference.md](api_reference.md)
3. **Enable Debug Logging**: 
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **Create a Minimal Reproduction**: Isolate the issue
5. **Open an Issue**: Provide details on [GitHub](https://github.com/electricwolfemarshmallowhypertext/ai-sleep/issues)

### Issue Template

When reporting issues, include:

```
**Environment:**
- Python version: 
- OS: 
- ai-sleep version: 
- numpy version: 
- scipy version: 

**Expected behavior:**
[What you expected to happen]

**Actual behavior:**
[What actually happened]

**Minimal reproduction:**
```python
# Code to reproduce the issue
```

**Error message:**
```
[Full error traceback]
```
```

---

## FAQ

**Q: Can I use AI Sleep with models not from HuggingFace?**
A: Yes! Extend `LMModelState` and `AISleepController` for your architecture. See [custom_architecture_template.py](../examples/custom_architecture_template.py).

**Q: How often should I run sleep cycles?**
A: Depends on workload. Light sleep: every 4-8 hours. Deep sleep: daily or weekly.

**Q: Will sleep cycles improve accuracy?**
A: They primarily maintain performance and efficiency. Accuracy improvements depend on the specific optimizations applied.

**Q: Can I run sleep cycles in parallel?**
A: No, each controller manages one model. For multiple models, create separate controllers.

**Q: Is it safe to interrupt a sleep cycle?**
A: Not recommended. Always let cycles complete, or call `wake_up()` to safely exit.

---

For more information, see the [documentation index](../README.md).
