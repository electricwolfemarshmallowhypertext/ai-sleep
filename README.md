## HEAD
# AI Sleep Constructs

Production-ready Python framework for engineered sleep cycles in language models. Light and deep sleep modes enable offline optimization, performance monitoring, gradient clipping, semantic consolidation, and adaptive learning.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17547016.svg)](https://doi.org/10.5281/zenodo.17547016)

## Overview

AI Sleep Constructs provides a comprehensive framework for implementing chronological intelligence in stateful language model systems. The library enables:

- **Light Sleep Mode**: Quick optimizations including gradient clipping, KV cache management, and adaptive learning
- **Deep Sleep Mode**: Intensive optimizations including attention head pruning, semantic consolidation, layer norm recalibration, and security patching
- **Performance Monitoring**: Drift detection, anomaly detection, and comprehensive metrics tracking
- **Contextual Triggers**: Automated sleep initiation based on performance degradation, anomalies, drift, or scheduled intervals
- **HuggingFace Integration**: Seamless integration with HuggingFace Transformers models

## Installation

```bash
pip install -e .
```

For HuggingFace integration:
```bash
pip install -e ".[huggingface]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from ai_sleep import AISleepController, SleepTrigger
from ai_sleep.model_state import SleepMode

# Create a controller
controller = AISleepController(
    model_id="gpt-neo-125M",
    enable_monitoring=True,
    auto_adapt=True
)

# Configure sleep modes
controller.configure_light_sleep(duration=300)
controller.configure_deep_sleep(duration=1800)

# Initiate sleep cycle
controller.initiate_sleep(
    trigger=SleepTrigger.MANUAL,
    mode=SleepMode.LIGHT_SLEEP
)

# Wake up
controller.wake_up()

# Get statistics
stats = controller.get_sleep_statistics()
print(stats)
```

### HuggingFace Integration

```python
from ai_sleep.huggingface_integration import (
    create_sleep_enabled_model,
    configure_optimal_sleep_schedule
)

# Create sleep-enabled model
adapter = create_sleep_enabled_model(
    "gpt2",
    enable_monitoring=True,
    auto_adapt=True,
    load_model=False  # Set to True to load actual model
)

# Configure for your workload
configure_optimal_sleep_schedule(adapter, workload_type="continuous")

# Enable and use sleep cycles
adapter.enable_sleep_cycles()
adapter.initiate_light_sleep(duration=300)

# Track metrics
adapter.track_inference_metrics(
    perplexity=3.5,
    loss=0.42,
    inference_time=0.1
)

# Wake model
adapter.wake_model()

# Get status
status = adapter.get_model_status()
```

### Performance Monitoring

```python
from ai_sleep import LMPerformanceMonitor

# Create monitor
monitor = LMPerformanceMonitor(model_id="gpt-neo-125M")

# Register metrics with thresholds
monitor.register_metric(
    "perplexity",
    track_drift=True,
    track_anomalies=True,
    min_threshold=1.0,
    max_threshold=10.0
)

# Track metrics
monitor.track_metric("perplexity", 3.5)
monitor.track_metric("loss", 0.42)

# Check for drift
drift_detected, p_value = monitor.check_drift("perplexity")

# Get statistics
stats = monitor.get_metric_statistics("perplexity")
print(f"Mean: {stats['mean']}, Std: {stats['std']}")

# Get alerts
alerts = monitor.get_recent_alerts(severity="high")
```

### C2C (Cache-to-Cache) Deep Sleep

C2C enables direct KV cache transfer between models for warm-start generation:

```python
from ai_sleep import C2CFuser, HFCacheExtractor, convert_kvcache_to_pkv
from transformers import AutoModelForCausalLM

# Load models
model_a = AutoModelForCausalLM.from_pretrained("gpt2")
model_b = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Extract cache from Model A (priming context)
extractor = HFCacheExtractor(model_a)
cache_a = extractor.extract_cache(priming_input_ids)

# Extract initial cache from Model B
cache_b = extractor_b.extract_cache(initial_input_ids)

# Create C2C fuser
fuser = C2CFuser(n_layers=24, head_dim=64, rank=32)

# Fuse A's knowledge into B
fused_cache = fuser.fuse(cache_a, cache_b)

# Continue generation with Model B using fused cache
pkv = convert_kvcache_to_pkv(fused_cache)
outputs = model_b(next_tokens, past_key_values=pkv)
```

**Benefits:**
- 5-10x faster than text-based relay
- Preserves latent knowledge (85-99% quality)
- Enables cross-model knowledge transfer

See [`docs/c2c.md`](docs/c2c.md) for comprehensive C2C documentation.

## Architecture

The framework consists of four main components:

### 1. LMModelState
Manages model state during sleep cycles:
- KV cache tracking and management
- Attention head configuration
- Layer normalization parameters
- Gradient storage and clipping
- Semantic memory consolidation
- State serialization and restoration

### 2. LMPerformanceMonitor
Comprehensive performance monitoring:
- Drift detection using statistical methods (Kolmogorov-Smirnov test)
- Anomaly detection (Z-score, IQR, moving average methods)
- Metric tracking and trending
- Alert generation and callbacks
- Performance statistics

### 3. AISleepController
Orchestrates sleep cycles:
- Light and deep sleep mode management
- Optimization strategy execution
- Contextual trigger handling
- Adaptive learning rate adjustment
- Security patching
- HuggingFace model integration

### 4. C2C (Cache-to-Cache)
Cross-model KV cache fusion:
- Direct cache transfer between models
- Low-rank projection adapters
- Per-layer fusion gating
- Training via distillation
- Integration with sleep cycles
- Contextual trigger handling
- Adaptive learning rate adjustment
- Security patching
- HuggingFace model integration

## Features

### Light Sleep Mode
- **Gradient Clipping**: Prevents gradient explosion
- **KV Cache Management**: Optimizes memory usage
- **Adaptive Learning Rate**: Adjusts learning rate based on performance

### Deep Sleep Mode
- **Attention Head Pruning**: Removes low-importance attention heads
- **Semantic Consolidation**: Compresses and consolidates semantic representations
- **Layer Norm Recalibration**: Re-adjusts layer normalization parameters
- **Security Patching**: Applies security fixes and vulnerability checks

### Contextual Triggers
- Scheduled maintenance
- Performance degradation detection
- Anomaly detection
- Drift detection
- Memory pressure
- Idle timeout
- Manual initiation

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

You are free to:
- Share and adapt this work for non-commercial purposes
- Require attribution to the original author

Under the following conditions:
- **Attribution**: Credit must be given to Tionne Smith, Antiparty, Inc.
- **NonCommercial**: Use is restricted to non-commercial applications
- **ShareAlike**: Any adaptations must be licensed under the same CC-BY-NC-SA 4.0 terms

For full legal details, see the [Creative Commons Legal Code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Citation

If you use this project in research or publication, please cite:

```bibtex
@techreport{smith2025aisleep,
  author = {Smith, Tionne},
  title = {AI Sleep Constructs: Implementing Chronological Intelligence in Stateful Systems},
  institution = {Antiparty, Inc.},
  year = {2025},
  type = {Technical Note},
  doi = {10.5281/zenodo.17547016},
  url = {https://doi.org/10.5281/zenodo.17547016}
}
```

Plain text citation:
```
Smith, T. (2025). AI Sleep Constructs: Implementing Chronological Intelligence in Stateful Systems. Technical Note, Zenodo. https://doi.org/10.5281/zenodo.17547016
```

## Contributing

This project is licensed under CC-BY-NC-SA 4.0. Contributions must comply with the license terms. Please ensure:
- All contributions are for non-commercial use
- Proper attribution is maintained
- Derivative works use the same license

## Acknowledgments

Created by Tionne Smith, Antiparty, Inc.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/electricwolfemarshmallowhypertext/ai-sleep).

# AI Sleep Constructs: Implementing Chronological Intelligence in Stateful Systems

A production-ready Python framework for implementing engineered sleep cycles in language models and stateful AI systems. Enables continuous offline optimization, memory consolidation, and adaptive learning through light and deep sleep modes.

**DOI:** [10.5281/zenodo.17547016](https://doi.org/10.5281/zenodo.17547016)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Testing](#testing)
- [License](#license)
- [Citation](#citation)

## Overview

Current AI systems operate in a continuous present without offline reflection, resulting in stateless architectures that cannot maintain identity or learn from experience across sessions. AI Sleep Constructs addresses this fundamental limitation by implementing engineered sleep cycles—inspired by biological sleep—that enable AI systems to consolidate memory, detect and heal from anomalies, and maintain persistent identity.

This framework provides concrete implementations of:
- **Light Sleep Mode**: Rapid maintenance including gradient clipping, attention head pruning, and KV cache management
- **Deep Sleep Mode**: Comprehensive self-healing via diagnostics, semantic consolidation, layer norm recalibration, security patching, and adaptive learning
- **Contextual Triggering**: Intelligent decision logic based on performance drift, anomalies, system load, and user activity

## Features

- ✅ Production-ready sleep cycle controller
- ✅ Performance drift detection and anomaly flagging
- ✅ HuggingFace transformer integration
- ✅ Semantic consolidation via VAE bottlenecking
- ✅ Comprehensive unit tests
- ✅ Extensible architecture for custom AI systems
- ✅ Non-commercial open-source licensing (CC-BY-NC-SA 4.0)

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+
- NumPy

### From Antiparty, Inc.

(README for AI Sleep Constructs framework)
