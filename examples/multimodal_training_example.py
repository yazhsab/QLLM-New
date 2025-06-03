"""
Example demonstrating training and evaluation of the Multimodal Quantum LLM.

This script shows how to create a multimodal dataset, train a multimodal quantum
language model, and evaluate its performance.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import os
import sys
import json
from torch.utils.data import DataLoader

# Add parent directory to path to import quantum_llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_llm import (
    QLLMTokenizer,
    MultimodalQLLM,
    MultimodalDataset,
    MultimodalTrainer,
    MultimodalEvaluator,
    MultimodalPreprocessor,
    TextPreprocessor,
    ImagePreprocessor,
    AudioPreprocessor
)


def generate_sample_data(num_samples=100, max_seq_length=32):
    """
    Generate sample multimodal data for demonstration.
    
    Args:
        num_samples: Number of samples to generate
        max_seq_length: Maximum sequence length for text
        
    Returns:
        list: List of data samples
    """
    # Sample texts
    texts = [
        "Quantum computing leverages quantum mechanical phenomena",
        "Large language models process and generate human-like text",
        "Combining quantum computing with language models enhances capabilities",
        "Multimodal learning integrates different types of data",
        "Quantum circuits can encode information from various modalities"
    ]
    
    # Generate data samples
    data = []
    for i in range(num_samples):
        # Select a random text
        text = texts[i % len(texts)]
        
        # Create a simple gradient image
        size = (32, 32)
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xv, yv = np.meshgrid(x, y)
        
        # Create RGB channels with some randomness
        r = xv + np.random.normal(0, 0.1, size)
        g = yv + np.random.normal(0, 0.1, size)
        b = (xv + yv) / 2 + np.random.normal(0, 0.1, size)
        
        # Stack channels
        img = np.clip(np.stack([r, g, b], axis=0), 0, 1)
        
        # Create a simple audio signal
        duration = 1.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Generate a chord with some randomness
        freqs = [440, 550, 660]  # A4, C#5, E5 (A major chord)
        signal = np.zeros_like(t)
        
        for f in freqs:
            signal += 0.2 * np.sin(2 * np.pi * f * t)
        
        # Add some noise
        signal += np.random.normal(0, 0.05, signal.shape)
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Create sample
        sample = {
            'text': text,
            'image': img,
            'audio': signal
        }
        
        data.append(sample)
    
    return data


def main():
    """Main function demonstrating multimodal training and evaluation."""
    print("Initializing Multimodal Quantum LLM training example...")
    
    # Create tokenizer
    vocab_size = 1000
    tokenizer = QLLMTokenizer(vocab_size=vocab_size, max_length=64)
    
    # Train tokenizer on a small sample text
    sample_text = """
    Quantum computing leverages quantum mechanical phenomena to perform computations.
    Large language models process and generate human-like text based on patterns learned during training.
    Combining quantum computing with language models could potentially enhance their capabilities.
    Multimodal learning integrates different types of data such as text, images, and audio.
    Quantum circuits can encode information from various modalities and process them coherently.
    """
    tokenizer.train_from_texts([sample_text])
    
    # Generate sample data
    print("Generating sample multimodal data...")
    data = generate_sample_data(num_samples=100, max_seq_length=32)
    
    # Split data into train and validation sets
    train_data = data[:80]
    val_data = data[80:]
    
    # Create preprocessor
    preprocessor = MultimodalPreprocessor(
        tokenizer=tokenizer,
        text_max_length=32,
        image_size=(32, 32),
        audio_sample_rate=16000,
        audio_duration=1.0
    )
    
    # Create datasets
    train_dataset = MultimodalDataset(
        data=train_data,
        preprocessor=preprocessor,
        modalities=["text", "image", "audio"]
    )
    
    val_dataset = MultimodalDataset(
        data=val_data,
        preprocessor=preprocessor,
        modalities=["text", "image", "audio"]
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # Create multimodal QLLM model
    print("Creating multimodal quantum LLM model...")
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
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu"  # Use GPU if available
    )
    
    # Train model
    print("Training model...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=3,
        gradient_accumulation_steps=1,
        log_interval=5,
        save_path="models/multimodal_qllm"
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_perplexity'], label='Train')
    plt.plot(history['val_perplexity'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training Perplexity')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multimodal_training_history.png')
    
    # Create evaluator
    evaluator = MultimodalEvaluator(model=model, device="cpu")
    
    # Prepare sample inputs for generation
    sample_inputs = preprocessor({
        'text': "Quantum computing with",
        'image': train_data[0]['image'],
        'audio': train_data[0]['audio']
    })
    
    # Generate text
    print("\nGenerating text from multimodal inputs...")
    generated_texts = evaluator.evaluate_text_generation(
        inputs=sample_inputs,
        tokenizer=tokenizer,
        max_length=20,
        temperature=0.8,
        top_k=5,
        num_samples=3
    )
    
    print("\nGenerated texts:")
    for i, text in enumerate(generated_texts):
        print(f"Sample {i+1}: {text}")
    
    print("\nMultimodal Quantum LLM training and evaluation completed successfully!")


if __name__ == "__main__":
    main()