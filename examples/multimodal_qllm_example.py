"""
Example demonstrating the use of the Multimodal Quantum LLM.

This script shows how to create and use a multimodal quantum language model
that can process text, images, and audio inputs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import os
import sys

# Add parent directory to path to import quantum_llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_llm.multimodal.model import MultimodalQLLM
from quantum_llm.tokenization import QLLMTokenizer


def load_sample_image(size=(32, 32)):
    """
    Load a sample image for demonstration.
    
    Args:
        size: Size to resize the image to
        
    Returns:
        torch.Tensor: Image tensor [1, 3, height, width]
    """
    # Create a simple gradient image
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xv, yv = np.meshgrid(x, y)
    
    # Create RGB channels
    r = xv
    g = yv
    b = (xv + yv) / 2
    
    # Stack channels
    img = np.stack([r, g, b], axis=0)
    
    # Convert to tensor
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def load_sample_audio(duration=1.0, sr=16000):
    """
    Generate a sample audio signal for demonstration.
    
    Args:
        duration: Duration of the audio in seconds
        sr: Sample rate
        
    Returns:
        torch.Tensor: Audio tensor [1, 1, time]
    """
    # Create a simple sine wave
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Generate a chord (multiple frequencies)
    freqs = [440, 550, 660]  # A4, C#5, E5 (A major chord)
    signal = np.zeros_like(t)
    
    for f in freqs:
        signal += 0.2 * np.sin(2 * np.pi * f * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Convert to tensor
    audio_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return audio_tensor


def main():
    """Main function demonstrating multimodal QLLM usage."""
    print("Initializing Multimodal Quantum LLM...")
    
    # Create tokenizer
    vocab_size = 1000
    tokenizer = QLLMTokenizer(vocab_size=vocab_size, max_length=64)
    
    # Train tokenizer on a small sample text
    sample_text = """
    Quantum computing leverages quantum mechanical phenomena to perform computations.
    Large language models process and generate human-like text based on patterns learned during training.
    Combining quantum computing with language models could potentially enhance their capabilities.
    """
    tokenizer.train_from_texts([sample_text])
    
    # Create multimodal QLLM model
    model = MultimodalQLLM(
        vocab_size=len(tokenizer.token_to_id),
        embedding_dim=16,
        n_qubits=8,
        n_layers=1,
        n_heads=2,
        max_seq_length=32,
        device="default.qubit",
        embedding_type="angle",
        circuit_type="quantum_transformer",
        supported_modalities=["text", "image", "audio"],
        modality_embedding_types={
            "text": "angle",
            "image": "amplitude",
            "audio": "fourier"
        },
        qubit_allocations={
            "text": 4,
            "image": 2,
            "audio": 2
        }
    )
    
    print("Model initialized successfully!")
    
    # Prepare inputs
    print("Preparing multimodal inputs...")
    
    # Text input
    text = "Quantum computing with language models"
    text_tokens = torch.tensor([tokenizer.encode(text)])
    print(f"Text input shape: {text_tokens.shape}")
    
    # Image input
    image = load_sample_image()
    print(f"Image input shape: {image.shape}")
    
    # Audio input
    audio = load_sample_audio()
    print(f"Audio input shape: {audio.shape}")
    
    # Combine inputs
    inputs = {
        "text": text_tokens,
        "image": image,
        "audio": audio
    }
    
    # Forward pass
    print("Running forward pass through the model...")
    with torch.no_grad():
        logits = model.forward(inputs)
    
    print(f"Output logits shape: {logits.shape}")
    
    # Generate text
    print("Generating text from multimodal inputs...")
    generated_ids = model.generate(inputs, max_length=20)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    print("\nGenerated text:")
    print(generated_text)
    
    print("\nMultimodal Quantum LLM demonstration completed successfully!")


if __name__ == "__main__":
    main()