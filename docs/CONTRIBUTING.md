# Contributing to Decision Transformer for Hospital AI

We welcome contributions to improve the Decision Transformer implementation for healthcare applications. This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- CUDA-capable GPU (recommended for development)
- Basic understanding of PyTorch and reinforcement learning

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/offline-rl-dt-jovy.git
cd offline-rl-dt-jovy
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/original-repo/offline-rl-dt-jovy.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dt-dev python=3.8
conda activate dt-dev
```

### 2. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy jupyter
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests
pytest tests/

# Run demo
python demo.py --quick

# Check code style
black --check src/
flake8 src/
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **Feature Additions**: Add new functionality
3. **Performance Improvements**: Optimize existing code
4. **Documentation**: Improve or add documentation
5. **Tests**: Add or improve test coverage
6. **Examples**: Add usage examples or tutorials

### Areas for Contribution

#### High Priority

- **CQL Integration**: Add Conservative Q-Learning critic for safety
- **Multi-task Learning**: Support for multiple hospital departments
- **Real-time Deployment**: API endpoints and web interface
- **Additional Datasets**: Support for more medical datasets
- **Advanced Evaluation**: More comprehensive evaluation metrics

#### Medium Priority

- **Model Variants**: Implement other transformer architectures
- **Data Augmentation**: Techniques for medical data
- **Hyperparameter Optimization**: Automated tuning
- **Visualization Tools**: Better analysis and plotting
- **Benchmarking**: Comprehensive performance comparisons

#### Low Priority

- **Documentation**: Tutorials and examples
- **Code Quality**: Refactoring and optimization
- **Testing**: Unit and integration tests
- **CI/CD**: Automated testing and deployment

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use black for formatting
black src/ tests/

# Use flake8 for linting
flake8 src/ tests/
```

### Code Organization

```
src/
├── model.py              # Core model implementation
├── dataset.py            # Dataset classes
├── train.py              # Training pipeline
├── eval.py               # Evaluation metrics
└── api_finetune.py       # API integration

tests/
├── test_model.py         # Model tests
├── test_dataset.py       # Dataset tests
├── test_train.py         # Training tests
└── test_eval.py          # Evaluation tests

docs/
├── API_REFERENCE.md      # API documentation
├── ARCHITECTURE.md       # System architecture
├── TUTORIAL.md           # Usage tutorial
└── CONTRIBUTING.md       # This file
```

### Naming Conventions

- **Classes**: PascalCase (`DecisionTransformer`)
- **Functions/Variables**: snake_case (`compute_loss`)
- **Constants**: UPPER_CASE (`MAX_LENGTH`)
- **Files**: snake_case (`decision_transformer.py`)

### Documentation Standards

All public functions and classes must have docstrings:

```python
def compute_loss(self, states, actions, rtgs, target_actions, mask=None):
    """
    Compute training loss for Decision Transformer.
    
    Args:
        states: State sequences [batch_size, seq_len, state_dim]
        actions: Action sequences [batch_size, seq_len, action_dim]
        rtgs: RTG sequences [batch_size, seq_len, 1]
        target_actions: Target actions for loss computation
        mask: Optional mask for valid timesteps
        
    Returns:
        Mean squared error loss
        
    Raises:
        ValueError: If input dimensions are incompatible
    """
    # Implementation here
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/

# Run in parallel
pytest -n auto
```

### Writing Tests

Create comprehensive tests for new functionality:

```python
import pytest
import torch
from src.model import DecisionTransformer, DecisionTransformerConfig

class TestDecisionTransformer:
    def test_model_creation(self):
        """Test model can be created with valid config."""
        config = DecisionTransformerConfig()
        model = DecisionTransformer(**config.to_dict())
        assert model is not None
        
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = DecisionTransformerConfig(max_length=10)
        model = DecisionTransformer(**config.to_dict())
        
        batch_size = 2
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        
        output = model(states, actions, rtgs)
        expected_shape = (batch_size, config.max_length, config.action_dim)
        assert output.shape == expected_shape
        
    def test_loss_computation(self):
        """Test loss computation works correctly."""
        config = DecisionTransformerConfig()
        model = DecisionTransformer(**config.to_dict())
        
        # Create test data
        batch_size, seq_len = 2, 10
        states = torch.randn(batch_size, seq_len, config.state_dim)
        actions = torch.randn(batch_size, seq_len, config.action_dim)
        rtgs = torch.randn(batch_size, seq_len, 1)
        targets = torch.randn(batch_size, seq_len, config.action_dim)
        
        loss = model.compute_loss(states, actions, rtgs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test model works with different batch sizes."""
        config = DecisionTransformerConfig()
        model = DecisionTransformer(**config.to_dict())
        
        states = torch.randn(batch_size, config.max_length, config.state_dim)
        actions = torch.randn(batch_size, config.max_length, config.action_dim)
        rtgs = torch.randn(batch_size, config.max_length, 1)
        
        output = model(states, actions, rtgs)
        assert output.shape[0] == batch_size
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Test training and inference speed
4. **Regression Tests**: Ensure existing functionality works

## Documentation

### Documentation Standards

- **README.md**: Project overview and quick start
- **API Reference**: Complete API documentation
- **Tutorials**: Step-by-step usage guides
- **Architecture**: System design and implementation details
- **Code Comments**: Inline documentation for complex logic

### Writing Documentation

```markdown
## Function Name

Brief description of what the function does.

### Parameters

- `param1` (type): Description of parameter
- `param2` (type): Description of parameter

### Returns

Description of return value.

### Example

```python
# Example usage
result = function_name(param1, param2)
```

### Notes

Additional information or warnings.
```

### Building Documentation

```bash
# Generate API documentation
python -m pydoc -w src.model

# Check documentation coverage
python -m pydoc -w src/
```

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write code following the style guide
   - Add tests for new functionality
   - Update documentation
   - Ensure all tests pass

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

### Submitting a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**:
   - Use the provided PR template
   - Provide clear description of changes
   - Reference any related issues
   - Add screenshots for UI changes

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Tests pass locally
   - [ ] Added tests for new functionality
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No merge conflicts
   ```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Code Review**: Maintainers review code quality and functionality
3. **Testing**: Reviewers test the changes locally
4. **Approval**: At least one maintainer approval required
5. **Merge**: Changes merged to main branch

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for solutions
3. **Try latest version** to see if issue is fixed

### Creating a Good Issue

Use the issue template:

```markdown
## Bug Report / Feature Request

### Description
Clear description of the issue or feature request.

### Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 1.12.0]
- CUDA version: [e.g., 11.6]

### Steps to Reproduce (for bugs)
1. Step one
2. Step two
3. Step three

### Expected Behavior
What you expected to happen.

### Actual Behavior
What actually happened.

### Additional Context
Any other relevant information.
```

### Issue Labels

- **bug**: Something isn't working
- **enhancement**: New feature or request
- **documentation**: Improvements to documentation
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention needed
- **question**: Further information requested

## Development Workflow

### Daily Development

1. **Start work**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/new-feature
   ```

2. **Make changes**:
   - Write code
   - Add tests
   - Update documentation

3. **Test changes**:
   ```bash
   pytest tests/
   python demo.py --quick
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add: feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/new-feature
   ```

### Release Process

1. **Update version** in `__init__.py` or `setup.py`
2. **Update changelog** with new features and fixes
3. **Create release branch** from main
4. **Run full test suite** and integration tests
5. **Create release** on GitHub with release notes
6. **Tag version** and create GitHub release

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: [Contact information]

### Mentorship

- **Good First Issues**: Labeled for newcomers
- **Code Reviews**: Detailed feedback on pull requests
- **Documentation**: Comprehensive guides and tutorials

### Code of Conduct

We follow a code of conduct to ensure a welcoming environment:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow community guidelines

## Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **Release Notes**: Credit for contributions
- **Documentation**: Attribution for significant contributions

Thank you for contributing to Decision Transformer for Hospital AI! Your contributions help advance AI for healthcare decision-making.
