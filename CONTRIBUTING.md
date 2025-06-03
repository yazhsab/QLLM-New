# Contributing to Quantum LLM

Thank you for your interest in contributing to Quantum LLM! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment Setup](#development-environment-setup)
  - [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Pull Requests](#pull-requests)
- [Coding Standards](#coding-standards)
  - [Python Style Guide](#python-style-guide)
  - [Documentation](#documentation)
  - [Testing](#testing)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Review Process](#review-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when contributing to the project.

## Getting Started

### Development Environment Setup

1. **Fork the repository**:
   - Fork the repository to your GitHub account.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/quantum-llm.git
   cd quantum-llm
   ```

3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/quantum-llm.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the package in development mode
   ```

### Project Structure

The project is organized as follows:

```
quantum_llm/
├── __init__.py                     # Package initialization
├── qllm_base.py                    # Base implementation
├── qllm_advanced.py                # Advanced implementation
├── tokenization.py                 # Tokenization utilities
├── basic_quantum_circuits/         # Basic quantum circuits
│   ├── __init__.py
│   └── data_encoding.py            # Data encoding circuits
├── variational_circuits/           # Variational quantum circuits
│   ├── __init__.py
│   └── variational_circuits.py     # Variational circuit implementations
├── qnn_models/                     # Quantum neural network models
│   ├── __init__.py
│   └── advanced_qnn.py             # Advanced QNN implementations
└── multimodal/                     # Multimodal support
    ├── __init__.py
    ├── encoders.py                 # Modality-specific encoders
    ├── data_encoding.py            # Multimodal quantum encoding
    └── model.py                    # Multimodal QLLM implementation
```

## Development Workflow

### Branching Strategy

- `main`: The main branch contains the stable version of the code.
- `develop`: Development branch where features are integrated.
- Feature branches: Create a new branch for each feature or bug fix.

To create a new feature branch:

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

### Commit Messages

Follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add quantum attention mechanism for multimodal processing

- Implement quantum attention circuit with 8 qubits
- Add tests for the new attention mechanism
- Update documentation

Fixes #123
```

### Pull Requests

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a pull request from your fork to the main repository's `develop` branch.

3. In your pull request description:
   - Describe the changes you've made
   - Reference any related issues
   - Include any necessary context for reviewers

4. Wait for the CI checks to pass and address any feedback from reviewers.

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
- Use 4 spaces for indentation (no tabs).
- Maximum line length is 88 characters (following Black's default).
- Use docstrings for all public modules, functions, classes, and methods.

We recommend using the following tools:
- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting

You can run these tools with:
```bash
black quantum_llm tests
isort quantum_llm tests
flake8 quantum_llm tests
```

### Documentation

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Document all public APIs, including functions, classes, and methods.
- Include examples in docstrings where appropriate.
- Update the documentation when you change the code.

Example:
```python
def create_data_embedding_circuit(embedding_type, n_qubits, n_features):
    """
    Creates a quantum circuit for embedding classical data.
    
    Args:
        embedding_type (str): Type of embedding ('angle', 'amplitude', 'hybrid').
        n_qubits (int): Number of qubits in the circuit.
        n_features (int): Number of classical features to embed.
        
    Returns:
        callable: A function that takes classical data and returns a quantum circuit.
        
    Examples:
        >>> circuit_fn = create_data_embedding_circuit("angle", 4, 8)
        >>> circuit = circuit_fn(data)
    """
```

### Testing

- Write unit tests for all new code.
- Ensure all tests pass before submitting a pull request.
- Aim for high test coverage, especially for critical components.

To run tests:
```bash
pytest tests/
```

## Reporting Issues

When reporting issues, please include:

1. A clear and descriptive title.
2. A detailed description of the issue.
3. Steps to reproduce the issue.
4. Expected behavior and actual behavior.
5. Your environment (OS, Python version, package versions).
6. Any relevant logs or error messages.

## Feature Requests

For feature requests, please include:

1. A clear and descriptive title.
2. A detailed description of the proposed feature.
3. Any relevant background or context.
4. If possible, a sketch or mockup of how the feature would work.

## Review Process

All contributions go through a review process:

1. Automated checks (CI/CD) verify that tests pass and code meets style guidelines.
2. At least one maintainer reviews the code for:
   - Correctness
   - Code quality
   - Test coverage
   - Documentation
3. Feedback is provided and changes may be requested.
4. Once approved, the contribution is merged.

Thank you for contributing to Quantum LLM!