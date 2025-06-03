# Quantum LLM

A production-grade enterprise open-source implementation of a quantum-enhanced large language model using PennyLane.

## Overview

Quantum LLM is a general-purpose quantum language model that leverages quantum computing through PennyLane to potentially achieve quantum advantage over classical models. The implementation is designed to be hardware-agnostic, allowing it to run on quantum simulators now and real quantum hardware in the future.

## Features

- **Quantum-Enhanced Language Model**: Core language model architecture with quantum circuit integration
- **Multimodal Support**: Process text, images, and audio inputs with quantum-enhanced processing
- **Hardware-Agnostic Design**: Works with quantum simulators and can be adapted for real quantum hardware
- **Modular Architecture**: Easily extensible for domain-specific applications
- **High-Performance**: Optimized for throughput and low latency
- **Production-Ready**: Enterprise-grade implementation with security and scalability in mind

## Architecture

The Quantum LLM architecture consists of several key components:

### Core Quantum LLM

- **Quantum Circuit Architecture**: Hardware-agnostic quantum circuits for data encoding and processing
- **Tokenization & Embedding**: Advanced tokenization with quantum-enhanced embeddings
- **Model Architecture**: Quantum transformer and RNN implementations
- **Training Pipeline**: Efficient training with distributed capabilities
- **Inference Engine**: Optimized for high throughput and low latency

### Quantum Components

- **Data Encoding Circuits**: Various methods for embedding classical data into quantum states
- **Multimodal Encoding**: Specialized quantum circuits for encoding text, images, and audio
- **Variational Circuits**: Parameterized quantum circuits for different model architectures
- **Quantum Neural Networks**: Quantum implementations of transformer and RNN architectures
- **Quantum Memory**: Mechanisms for preserving quantum state information

## Installation

```bash
# Clone the repository
git clone https://github.com/organization/quantum-llm.git
cd quantum-llm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

For development setup, please refer to the [Contributing](#contributing) section and the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Quick Start

```python
import torch
from quantum_llm import QLLMBase, QLLMTokenizer

# Create tokenizer
tokenizer = QLLMTokenizer(vocab_size=1000, max_length=64)
tokenizer.train_from_texts(["Your training text here"])

# Create model
model = QLLMBase(
    vocab_size=len(tokenizer.token_to_id),
    embedding_dim=16,
    n_qubits=4,
    n_layers=1,
    max_seq_length=32,
    device="default.qubit",
    embedding_type="angle",
    circuit_type="quantum_transformer"
)

# Generate text
input_ids = torch.tensor([tokenizer.encode("Your prompt here")])
generated_ids = model.generate(input_ids, max_length=20)
generated_text = tokenizer.decode(generated_ids[0].tolist())
print(generated_text)
```

## Examples

The `examples` directory contains several examples demonstrating how to use the Quantum LLM:

- `basic_qllm_example.py`: Basic usage of the Quantum LLM components
- `quantum_advantage_demo.py`: Demonstration of potential quantum advantage
- `multimodal_qllm_example.py`: Example of using the multimodal capabilities with text, images, and audio

To run an example:

```bash
python examples/basic_qllm_example.py
```

## Advanced Usage

### Using Different Quantum Circuits

```python
from quantum_llm import QLLMAdvanced

# Create advanced model with specific quantum components
model = QLLMAdvanced(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    n_qubits=n_qubits,
    n_layers=n_layers,
    n_heads=n_heads,
    device="default.qubit",
    embedding_type="fourier",
    circuit_type="quantum_transformer",
    use_quantum_attention=True,
    use_rotary_embeddings=True
)
```

### Using Multimodal Capabilities

```python
from quantum_llm import MultimodalQLLM, QLLMTokenizer
import torch

# Create tokenizer
tokenizer = QLLMTokenizer(vocab_size=1000, max_length=64)
tokenizer.train_from_texts(["Your training text here"])

# Create multimodal model
model = MultimodalQLLM(
    vocab_size=len(tokenizer.token_to_id),
    embedding_dim=16,
    n_qubits=8,
    n_layers=2,
    n_heads=2,
    device="default.qubit",
    supported_modalities=["text", "image", "audio"],
    qubit_allocations={
        "text": 4,
        "image": 2,
        "audio": 2
    }
)

# Prepare multimodal inputs
inputs = {
    "text": torch.tensor([tokenizer.encode("Your text here")]),
    "image": torch.randn(1, 3, 32, 32),  # Example image tensor
    "audio": torch.randn(1, 1, 16000)    # Example audio tensor
}

# Generate text from multimodal inputs
generated_ids = model.generate(inputs, max_length=20)
generated_text = tokenizer.decode(generated_ids[0].tolist())
print(generated_text)
```

### Custom Quantum Circuits

```python
from quantum_llm.basic_quantum_circuits.data_encoding import create_data_embedding_circuit
from quantum_llm.variational_circuits.variational_circuits import create_variational_circuit

# Create custom embedding circuit
embedding_circuit = create_data_embedding_circuit("hybrid", n_qubits, n_features)

# Create custom variational circuit
variational_circuit = create_variational_circuit("custom", n_qubits, n_layers)
```

## Project Structure

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

## Requirements

- Python 3.8+
- PennyLane
- PyTorch
- NumPy
- tqdm
- Pillow (for image processing)
- librosa (for audio processing)

## Contributing

Contributions are welcome and greatly appreciated! We're excited to welcome contributors to the Quantum LLM project.

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please read this document before participating in our community.

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed information on:
- Setting up your development environment
- Our coding standards and guidelines
- How to submit pull requests
- How to report issues
- The review process

### Getting Started with Contributing

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/quantum-llm.git`
3. Set up your development environment as described in [CONTRIBUTING.md](CONTRIBUTING.md)
4. Create a branch for your changes
5. Make your changes and commit them
6. Push to your fork and submit a pull request

## Security

We take the security of Quantum LLM seriously. If you believe you've found a security vulnerability, please follow the guidelines in our [Security Policy](SECURITY.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The MIT License is a permissive license that allows for reuse with few restrictions. You are free to use, modify, distribute, and sell this software, provided you include the original copyright notice and license terms.

## Citation

If you use Quantum LLM in your research, please cite:

```
@software{quantum_llm,
  author = {Quantum LLM Contributors},
  title = {Quantum LLM: A Quantum-Enhanced Large Language Model},
  year = {2025},
  url = {https://github.com/organization/quantum-llm}
}
```

## Acknowledgements

- PennyLane for the quantum computing framework
- PyTorch for the classical deep learning components
