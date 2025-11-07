# Rebase Plan: AI Sleep Repository Clean Commit History

This document outlines a structured clean commit sequence for the AI Sleep Constructs repository. The commits are organized logically to build the project from infrastructure to documentation, ensuring a clear and maintainable git history.

**Important Note:** This plan is for documentation purposes only. No git commands should be executed based on this plan without proper review and authorization.

---

## Commit Sequence

### Commit 1: Configure custom funding and repository automation (.github)

**Description:**
Establishes GitHub-specific configurations including funding information for project support, CI/CD workflows for automated testing and documentation validation, and configuration files for repository automation.

**Affected Files:**
- `.github/FUNDING.yml` - Custom funding configuration pointing to Antiparty contact page
- `.github/workflows/docs.yml` - Documentation build and validation workflow
- `.github/workflows/tests.yml` - Automated test execution workflow
- `.github/markdown-link-check-config.json` - Configuration for markdown link validation

**Purpose:**
Sets up the foundational GitHub infrastructure for automated quality checks and project sustainability.

---

### Commit 2: Add structured documentation for CI/CD and workflow integrity (docs)

**Description:**
Provides comprehensive documentation for understanding GitHub Actions workflow statuses, conclusions, and the CI/CD process. Includes architecture documentation, API reference, troubleshooting guides, and Cache-to-Cache (C2C) implementation details.

**Affected Files:**
- `docs/cicd.md` - CI/CD workflow statuses and conclusions reference
- `docs/architecture.md` - System architecture documentation
- `docs/api_reference.md` - Complete API documentation for all modules
- `docs/troubleshooting.md` - Common issues and solutions guide
- `docs/c2c.md` - Cache-to-Cache fusion documentation and safety guidelines

**Purpose:**
Establishes comprehensive technical documentation to support development, integration, and troubleshooting.

---

### Commit 3: Update examples and ensure CI/CD workflow stability (examples)

**Description:**
Adds practical example scripts demonstrating key features of the AI Sleep framework, including simple language model integration, custom architecture templates, and Cache-to-Cache fusion demonstrations.

**Affected Files:**
- `examples/simple_lm_example.py` - Basic usage example with sleep cycles
- `examples/custom_architecture_template.py` - Template for custom model architectures
- `examples/c2c_two_model_demo.py` - Demonstration of Cache-to-Cache cross-model fusion

**Purpose:**
Provides working code examples to help users quickly understand and implement AI Sleep Constructs features.

---

### Commit 4: Implement AI Sleep Constructs core framework (src/ai_sleep)

**Description:**
Implements the complete core framework for AI Sleep Constructs, including the main sleep controller, model state management, performance monitoring, HuggingFace integration, and Cache-to-Cache fusion capabilities.

**Affected Files:**
- `src/ai_sleep/__init__.py` - Package initialization and public API exports
- `src/ai_sleep/sleep_controller.py` - Main AISleepController orchestration
- `src/ai_sleep/model_state.py` - LMModelState for state management during sleep cycles
- `src/ai_sleep/performance_monitor.py` - LMPerformanceMonitor for drift and anomaly detection
- `src/ai_sleep/huggingface_integration.py` - HuggingFace model adapter and utilities
- `src/ai_sleep/hf_adapters.py` - Additional HuggingFace adapter implementations
- `src/ai_sleep/c2c.py` - Cache-to-Cache fusion implementation (C2CFuser, CacheProjector, LayerGate, HFCacheExtractor)

**Purpose:**
Delivers the complete production-ready framework for implementing engineered sleep cycles in language models with light and deep sleep modes, performance monitoring, and cross-model cache transfer.

---

### Commit 5: Refine and verify test suite for AI Sleep Constructs (tests)

**Description:**
Implements comprehensive unit tests covering all core components including sleep controller functionality, model state management, performance monitoring, HuggingFace integration, and Cache-to-Cache fusion capabilities. Ensures 100% test pass rate.

**Affected Files:**
- `tests/__init__.py` - Test package initialization
- `tests/test_sleep_controller.py` - Tests for AISleepController orchestration
- `tests/test_model_state.py` - Tests for LMModelState management
- `tests/test_performance_monitor.py` - Tests for monitoring and anomaly detection
- `tests/test_huggingface_integration.py` - Tests for HuggingFace adapter integration
- `tests/test_c2c.py` - Comprehensive tests for Cache-to-Cache fusion (19 tests)

**Purpose:**
Ensures code quality and correctness through comprehensive test coverage (66+ tests total) validating all framework functionality.

---

### Commit 6: Update ignore rules for Python, build, and CI artifacts (.gitignore)

**Description:**
Configures git to ignore Python bytecode, build artifacts, virtual environments, IDE configurations, and other generated files that should not be tracked in version control.

**Affected Files:**
- `.gitignore` - Comprehensive ignore patterns for Python development

**Purpose:**
Maintains a clean repository by excluding generated files and preventing accidental commits of build artifacts or local configurations.

---

### Commit 7: Update changelog to include Cache-to-Cache and workflow changes (CHANGELOG.md)

**Description:**
Documents all notable changes including the initial 0.1.0 release with core components, optimization strategies, and HuggingFace integration, plus unreleased Cache-to-Cache (C2C) fusion capabilities and planned future enhancements.

**Affected Files:**
- `CHANGELOG.md` - Complete project changelog with version 0.1.0 details and C2C additions

**Purpose:**
Provides a comprehensive, chronological record of project changes following Keep a Changelog format for transparency and version tracking.

---

### Commit 8: Add detailed contribution and development guidelines (CONTRIBUTING.md)

**Description:**
Establishes guidelines for contributing to the project including code standards, testing requirements, documentation expectations, pull request process, and development workflow best practices.

**Affected Files:**
- `CONTRIBUTING.md` - Comprehensive contribution guidelines and development standards

**Purpose:**
Creates clear expectations and processes for contributors to maintain code quality and project consistency.

---

### Commit 9: Add license file for open distribution under CC BY-NC-SA 4.0 (LICENSE)

**Description:**
Applies Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license to the project. This requires attribution to Tionne Smith and Antiparty, Inc., restricts commercial use, and requires derivative works to use the same license.

**Affected Files:**
- `LICENSE` - CC BY-NC-SA 4.0 license text

**Purpose:**
Legally protects the work while allowing open distribution and modification for non-commercial purposes with proper attribution.

---

### Commit 10: Revise README for clarity, structure, and proper formatting (README.md)

**Description:**
Creates a clear, well-structured project overview including installation instructions, quick start guide, usage examples for core features and Cache-to-Cache fusion, testing instructions, license information, and academic citation.

**Affected Files:**
- `README.md` - Primary project documentation and quick start guide

**Purpose:**
Serves as the entry point for new users and developers, providing essential information about the project, its features, and how to get started.

---

### Commit 11: Add security policy for reporting and handling vulnerabilities (SECURITY.md)

**Description:**
Defines security policy including supported versions, vulnerability reporting procedures, responsible disclosure guidelines, response timeline expectations, and security best practices for users of the framework.

**Affected Files:**
- `SECURITY.md` - Security policy and vulnerability reporting guidelines

**Purpose:**
Establishes clear procedures for responsible disclosure of security vulnerabilities and demonstrates commitment to security.

---

### Commit 12: Define core dependencies and enable Cache-to-Cache support (requirements.txt)

**Description:**
Specifies required Python packages including numpy and scipy for core functionality, transformers for HuggingFace integration, and PyTorch for Cache-to-Cache functionality with appropriate version constraints.

**Affected Files:**
- `requirements.txt` - Project dependencies with version requirements

**Purpose:**
Ensures reproducible installations by explicitly declaring all required dependencies and their version constraints.

---

### Commit 13: Implement setup configuration for packaging and HuggingFace integration (setup.py)

**Description:**
Configures Python package setup with metadata, dependencies, classifiers, and extras for HuggingFace integration and development tools. Enables pip installation and PyPI distribution.

**Affected Files:**
- `setup.py` - Package configuration and installation setup

**Purpose:**
Enables standard Python package installation and distribution, making the framework easily installable via pip and compatible with Python packaging ecosystem.

---

## Summary

This rebase plan organizes the AI Sleep Constructs repository into 13 logical commits that build the project incrementally:

1. **Infrastructure** (Commits 1-3): GitHub automation, documentation, examples
2. **Core Implementation** (Commits 4-5): Framework code and tests
3. **Project Configuration** (Commits 6-13): Ignore rules, changelog, contributing guidelines, license, README, security policy, dependencies, and setup

The sequence ensures:
- Foundation is established before building upon it
- Core code is implemented and then verified by comprehensive tests
- Configuration files are added after the code they configure
- Documentation is comprehensive and well-organized
- The project follows best practices for open source development

**Reminder:** This is a documentation plan only. Any actual git rebase operations should be performed carefully with proper backups and team coordination.
