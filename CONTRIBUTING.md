# Contributing to AI Sleep Constructs

Thank you for your interest in contributing to AI Sleep Constructs! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Extending for Other AI Architectures](#extending-for-other-ai-architectures)
- [Pull Request Process](#pull-request-process)
- [License Compliance](#license-compliance)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai-sleep.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/electricwolfemarshmallowhypertext/ai-sleep.git
cd ai-sleep

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install optional dependencies
pip install -e ".[huggingface]"
```

## Contribution Guidelines

### Areas for Contribution

We welcome contributions in the following areas:

- **Bug fixes**: Address reported issues
- **New features**: Implement new optimization strategies or triggers
- **Documentation**: Improve existing docs or add new examples
- **Architecture extensions**: Adapt framework for new AI architectures
- **Performance improvements**: Optimize existing implementations
- **Test coverage**: Add tests for uncovered code paths

### Before You Start

- Check existing issues and pull requests to avoid duplicate work
- For major changes, open an issue first to discuss your proposal
- Ensure your contribution aligns with the project's non-commercial license

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Docstrings**: Google-style docstrings for all public classes and methods

### Code Quality Tools

Run these tools before submitting:

```bash
# Format code with black
black src/ tests/ examples/

# Check style with flake8
flake8 src/ tests/ examples/ --max-line-length=100

# Type checking with mypy
mypy src/ai_sleep/
```

### Documentation Standards

- All public classes, methods, and functions must have docstrings
- Use Google-style docstrings
- Include examples in docstrings for complex functionality
- Update relevant documentation files when adding features

Example docstring:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description with more details about what the function does,
    its purpose, and any important implementation notes.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return len(param1) == param2
```

## Testing Requirements

### Writing Tests

- All new code must include unit tests
- Aim for at least 80% code coverage
- Tests should be independent and repeatable
- Use descriptive test names that explain what is being tested

### Test Structure

```python
class TestYourFeature(unittest.TestCase):
    """Tests for YourFeature class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature = YourFeature()
        
    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        result = self.feature.do_something()
        self.assertEqual(result, expected_value)
        
    def test_error_handling(self):
        """Test that errors are handled properly."""
        with self.assertRaises(ValueError):
            self.feature.do_invalid_operation()
```

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_model_state -v

# Run with coverage
pip install pytest pytest-cov
pytest tests/ --cov=src/ai_sleep --cov-report=html
```

## Extending for Other AI Architectures

AI Sleep Constructs is designed to be extended beyond language models. Here's how to adapt it for other architectures:

### 1. Vision Models (CNNs, Vision Transformers)

```python
from ai_sleep import AISleepController
from ai_sleep.model_state import LMModelState

class VisionModelState(LMModelState):
    """Extended state for vision models."""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.feature_maps = {}
        self.spatial_attention_masks = {}
        
    def store_feature_map(self, layer_id: str, features):
        """Store intermediate feature maps."""
        self.feature_maps[layer_id] = features
        
    def prune_spatial_attention(self, layer_id: str, threshold: float):
        """Prune spatial attention regions below threshold."""
        # Implementation for spatial pruning
        pass

# Usage
vision_controller = AISleepController(
    model_id="vision-transformer-base",
    enable_monitoring=True
)
```

### 2. Reinforcement Learning Agents

```python
class RLAgentState(LMModelState):
    """Extended state for RL agents."""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.replay_buffer = {}
        self.policy_checkpoints = []
        
    def consolidate_experience(self, episodes: list):
        """Consolidate episodic memory during sleep."""
        # Implementation for experience replay consolidation
        pass
        
    def optimize_policy(self):
        """Optimize policy during deep sleep."""
        # Implementation for policy optimization
        pass
```

### 3. Graph Neural Networks

```python
class GNNState(LMModelState):
    """Extended state for graph neural networks."""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.graph_embeddings = {}
        self.message_passing_cache = {}
        
    def prune_graph_edges(self, threshold: float):
        """Prune low-importance edges."""
        # Implementation for graph pruning
        pass
```

### Architecture-Specific Optimization Strategies

When extending for new architectures, consider:

- **State Components**: What architecture-specific state needs tracking?
- **Optimization Strategies**: What optimizations make sense for this architecture?
- **Metrics**: What performance metrics should be monitored?
- **Triggers**: What conditions should initiate sleep cycles?

### Contributing New Architecture Support

When contributing support for a new architecture:

1. Create a new module in `src/ai_sleep/architectures/`
2. Extend `LMModelState` with architecture-specific state
3. Implement architecture-specific optimization strategies
4. Add comprehensive tests in `tests/architectures/`
5. Document the new architecture in `docs/architectures/`
6. Provide examples in `examples/architectures/`

## Pull Request Process

1. **Update Documentation**: Ensure all relevant documentation is updated
2. **Add Tests**: Include tests for new functionality
3. **Run Tests**: Verify all tests pass locally
4. **Update CHANGELOG**: Add entry to CHANGELOG.md
5. **Commit Message**: Use clear, descriptive commit messages
6. **PR Description**: Clearly describe what your PR does and why

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

Example:
```
feat: Add support for vision transformer architectures

- Extend LMModelState for vision models
- Implement spatial attention pruning
- Add feature map caching during sleep cycles

Closes #123
```

## License Compliance

All contributions must comply with the CC-BY-NC-SA 4.0 license:

- Contributions must be for non-commercial use
- You retain copyright to your contributions
- Contributions will be licensed under CC-BY-NC-SA 4.0
- Attribution to original authors must be maintained

By contributing, you agree that your contributions will be licensed under the CC-BY-NC-SA 4.0 license.

## Questions?

- Open an issue for questions about contributing
- Tag maintainers for urgent matters
- Join discussions in existing issues and PRs

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their specific contributions
- GitHub contributors page
- Project documentation when appropriate

Thank you for contributing to AI Sleep Constructs!
