# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **C2C (Cache-to-Cache)**: Cross-model KV cache fusion
  - `C2CFuser` module for cache-to-cache transfer between models
  - `CacheProjector` with low-rank adapters for efficient projection
  - `LayerGate` for learnable per-layer fusion control
  - `HFCacheExtractor` for HuggingFace model integration
  - Conversion utilities between HF and C2C formats
  - 19 comprehensive tests (all passing)
  - Complete documentation with architecture diagrams
  - Working demo (mock and real models)
  - Integration points for sleep cycles

### Planned
- WebSocket integration for real-time monitoring
- Distributed sleep coordination for multi-model systems
- Advanced visualization dashboard for sleep metrics
- Integration with MLflow for experiment tracking
- Support for quantized models
- C2C projector training during deep sleep cycles
- Automated C2C optimization via distillation

## [0.1.0] - 2025-01-07

### Added - Initial Release

#### Core Components
- **LMModelState**: Complete state management for language models during sleep cycles
  - KV cache tracking and management
  - Attention head configuration and pruning support
  - Layer normalization parameter tracking
  - Gradient storage and clipping
  - Semantic memory consolidation
  - State serialization and restoration
  - State history tracking

- **LMPerformanceMonitor**: Comprehensive performance monitoring system
  - Drift detection using Kolmogorov-Smirnov statistical test
  - Multi-method anomaly detection (Z-score, IQR, moving average)
  - Flexible metric tracking and trending
  - Alert generation with configurable callbacks
  - Performance statistics and summaries
  - Threshold-based monitoring

- **AISleepController**: Main orchestration controller for sleep cycles
  - Light sleep mode with quick optimizations
  - Deep sleep mode with intensive optimizations
  - Configurable sleep durations and strategies
  - Contextual trigger system
  - Sleep cycle callbacks and event handling
  - Adaptive learning rate adjustment
  - Security patching mechanism
  - Comprehensive sleep statistics

#### Optimization Strategies

##### Light Sleep Mode
- Gradient clipping to prevent gradient explosion
- KV cache management for memory optimization
- Adaptive learning rate adjustment based on performance

##### Deep Sleep Mode
- Attention head pruning for model compression
- Semantic memory consolidation for knowledge distillation
- Layer normalization recalibration for stability
- Security patching and vulnerability checks

#### Contextual Triggers
- Scheduled maintenance cycles
- Performance degradation detection
- Anomaly detection
- Drift detection
- Memory pressure monitoring
- Idle timeout
- Manual initiation

#### HuggingFace Integration
- `HuggingFaceModelAdapter` for wrapping HuggingFace models
- `create_sleep_enabled_model()` convenience function
- `configure_optimal_sleep_schedule()` for workload-specific configuration
- Support for continuous, batch, and interactive workloads
- Inference metrics tracking integration

#### Testing
- 66 comprehensive unit tests
- Test coverage for all core components
- Tests for state management, monitoring, and control
- HuggingFace integration tests
- All tests passing

#### Documentation
- Comprehensive README with usage examples
- Detailed docstrings for all public APIs (Google-style)
- Quick start guide
- Architecture overview
- HuggingFace integration examples
- Performance monitoring examples

#### Project Infrastructure
- Python package structure with `setup.py`
- Requirements management
- CC-BY-NC-SA 4.0 licensing
- DOI citation information (10.5281/zenodo.17547016)
- Git repository initialization

### Technical Specifications

- **Python Version**: 3.8+
- **Core Dependencies**: 
  - numpy >= 1.21.0
  - scipy >= 1.7.0
- **Optional Dependencies**:
  - transformers >= 4.20.0 (for HuggingFace integration)

### Breaking Changes

None (initial release)

### Known Issues

None at this time

### Contributors

- Tionne Smith, Antiparty, Inc. (Initial development)

## Release Notes

### Version 0.1.0 - "Foundation"

This initial release establishes the foundation for AI Sleep Constructs, providing a production-ready framework for implementing engineered sleep cycles in language models. The framework enables:

1. **Offline Optimization**: Models can undergo optimization without impacting online performance
2. **Performance Monitoring**: Continuous tracking with drift and anomaly detection
3. **Adaptive Learning**: Dynamic adjustment based on performance trends
4. **Memory Management**: Efficient KV cache and attention head management
5. **Security**: Built-in vulnerability checking and patching

The framework is designed with extensibility in mind, allowing adaptation to various AI architectures beyond language models.

---

## Version Numbering

- **Major version** (X.0.0): Incompatible API changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

## Changelog Guidelines

When contributing, please update this file following these guidelines:

1. Add new entries under "Unreleased" section
2. Use subsections: Added, Changed, Deprecated, Removed, Fixed, Security
3. Reference issue numbers where applicable
4. Be specific and concise
5. Move entries to appropriate version section upon release

[Unreleased]: https://github.com/electricwolfemarshmallowhypertext/ai-sleep/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/electricwolfemarshmallowhypertext/ai-sleep/releases/tag/v0.1.0
