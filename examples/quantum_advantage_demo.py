"""
Quantum Advantage Demonstration for Quantum LLM.

This script demonstrates how to compare classical and quantum LLM components
to showcase potential quantum advantages.
"""

import torch
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from quantum_llm import (
    QLLMBase,
    QLLMAdvanced,
    QLLMTokenizer,
    QLLMDataset,
    QTransformer,
    QRNN
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Sample text data with long-range dependencies for demonstration
texts = [
    "The cat, which was sitting on the mat that had been placed in the corner of the room, suddenly jumped up.",
    "Although the algorithm was complex, the programmer who designed it explained that it could solve the problem efficiently.",
    "The quantum computer, unlike classical computers that use bits, leverages quantum bits or qubits to perform calculations.",
    "In the novel, the protagonist, who had been introduced in the first chapter as a mysterious figure, finally revealed his true identity.",
    "The neural network architecture, which consisted of multiple layers with various activation functions, achieved state-of-the-art performance.",
    "Quantum entanglement, a phenomenon where particles become correlated in such a way that the quantum state of each particle cannot be described independently, is a key resource.",
    "The language model, trained on a diverse corpus of text from books, articles, and websites, could generate coherent paragraphs on various topics.",
    "The experiment, which had been carefully designed to test the hypothesis proposed by the research team last year, yielded surprising results.",
    "Quantum superposition allows qubits to exist in multiple states simultaneously, which enables quantum computers to explore multiple solutions in parallel.",
    "The transformer architecture, with its self-attention mechanism that can capture long-range dependencies in sequences, revolutionized natural language processing."
]


class ClassicalTransformer(torch.nn.Module):
    """Simple classical transformer for comparison."""
    
    def __init__(self, vocab_size, embedding_dim, n_heads, n_layers):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim*4,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.output_projection = torch.nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        transformer_output = self.transformer(embeddings)
        logits = self.output_projection(transformer_output)
        return logits


def measure_contextual_coherence(model, tokenizer, texts, window_size=10, stride=5):
    """
    Measure contextual coherence across long-range dependencies.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        texts: List of text documents
        window_size: Size of context window
        stride: Stride for sliding window
        
    Returns:
        float: Average coherence score
    """
    model.eval()
    scores = []
    
    for text in texts:
        # Tokenize text
        tokens = tokenizer.encode(text, add_special_tokens=False, padding=False, truncation=False)
        
        # Process in windows
        for i in range(0, len(tokens) - window_size * 2, stride):
            # Get context and target
            context = tokens[i:i + window_size]
            target = tokens[i + window_size:i + window_size * 2]
            
            # Prepare input
            input_ids = torch.tensor([context])
            
            # Get model prediction
            with torch.no_grad():
                if isinstance(model, (QLLMBase, QLLMAdvanced)):
                    logits = model.forward(input_ids)
                    # Get predictions for the next tokens
                    pred_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
                else:
                    logits = model(input_ids)
                    pred_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
                
                # Calculate accuracy (exact match)
                accuracy = np.mean(pred_ids[:len(target)] == target[:len(pred_ids)])
                scores.append(accuracy)
    
    # Calculate average score
    return np.mean(scores) if scores else 0.0


def measure_inference_time(model, input_ids, n_runs=10):
    """
    Measure inference time for a model.
    
    Args:
        model: Model to evaluate
        input_ids: Input token IDs
        n_runs: Number of runs to average
        
    Returns:
        float: Average inference time in seconds
    """
    model.eval()
    times = []
    
    # Warm-up run
    with torch.no_grad():
        _ = model.forward(input_ids)
    
    # Timed runs
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model.forward(input_ids)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Return average time
    return np.mean(times)


def compare_models(quantum_model, classical_model, tokenizer, texts):
    """
    Compare quantum and classical models on various metrics.
    
    Args:
        quantum_model: Quantum model to evaluate
        classical_model: Classical model to evaluate
        tokenizer: Tokenizer to use
        texts: List of text documents
        
    Returns:
        dict: Comparison results
    """
    # Prepare sample input
    sample_text = "The quantum computer performs calculations using"
    input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
    
    # Measure inference time
    print("Measuring inference time...")
    quantum_time = measure_inference_time(quantum_model, input_ids)
    classical_time = measure_inference_time(classical_model, input_ids)
    
    # Measure contextual coherence
    print("Measuring contextual coherence...")
    quantum_coherence = measure_contextual_coherence(quantum_model, tokenizer, texts)
    classical_coherence = measure_contextual_coherence(classical_model, tokenizer, texts)
    
    # Calculate advantages
    time_advantage = classical_time / quantum_time if quantum_time > 0 else 0
    coherence_advantage = quantum_coherence / classical_coherence if classical_coherence > 0 else 0
    
    return {
        'quantum_time': quantum_time,
        'classical_time': classical_time,
        'time_advantage': time_advantage,
        'quantum_coherence': quantum_coherence,
        'classical_coherence': classical_coherence,
        'coherence_advantage': coherence_advantage
    }


def plot_comparison(results):
    """
    Plot comparison results.
    
    Args:
        results: Comparison results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot inference time comparison
    models = ['Quantum', 'Classical']
    times = [results['quantum_time'], results['classical_time']]
    ax1.bar(models, times, color=['blue', 'orange'])
    ax1.set_title('Inference Time (seconds)')
    ax1.set_ylabel('Time (s)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot coherence comparison
    coherence = [results['quantum_coherence'], results['classical_coherence']]
    ax2.bar(models, coherence, color=['blue', 'orange'])
    ax2.set_title('Contextual Coherence Score')
    ax2.set_ylabel('Score')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add advantage text
    fig.suptitle(f"Quantum vs Classical Comparison\n" +
                f"Time Advantage: {results['time_advantage']:.2f}x, " +
                f"Coherence Advantage: {results['coherence_advantage']:.2f}x")
    
    plt.tight_layout()
    plt.savefig('quantum_advantage_comparison.png')
    print("Comparison plot saved as 'quantum_advantage_comparison.png'")


def main():
    print("Quantum Advantage Demonstration")
    print("-" * 50)
    
    # Create tokenizer
    print("Creating and training tokenizer...")
    tokenizer = QLLMTokenizer(vocab_size=1000, max_length=64)
    tokenizer.train_from_texts(texts)
    
    # Create dataset
    print("Creating dataset...")
    dataset = QLLMDataset(texts, tokenizer, block_size=32)
    
    # Create data loaders
    batch_size = 2
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # Model parameters
    vocab_size = len(tokenizer.token_to_id)
    embedding_dim = 16
    n_qubits = 4
    n_layers = 1
    n_heads = 2
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using {n_qubits} qubits and {n_layers} layers")
    
    # Create quantum model
    print("Creating quantum LLM model...")
    quantum_model = QLLMAdvanced(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_length=32,
        device="default.qubit",
        embedding_type="fourier",
        circuit_type="quantum_transformer",
        use_quantum_attention=True,
        use_rotary_embeddings=True
    )
    
    # Create classical model for comparison
    print("Creating classical model for comparison...")
    classical_model = ClassicalTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        n_layers=n_layers
    )
    
    # Train models (just a few steps for demonstration)
    print("Training models (demonstration only)...")
    
    # Optimizers
    quantum_optimizer = torch.optim.Adam(quantum_model.parameters(), lr=0.01)
    classical_optimizer = torch.optim.Adam(classical_model.parameters(), lr=0.01)
    
    # Loss function for classical model
    loss_fn = torch.nn.CrossEntropyLoss()
    
    n_epochs = 1  # Just for demonstration
    
    for epoch in range(n_epochs):
        # Train quantum model
        quantum_model.train()
        for batch in tqdm(train_loader, desc=f"Training quantum model (Epoch {epoch+1})"):
            input_ids, labels = batch
            
            # Forward pass
            logits = quantum_model.forward(input_ids)
            loss = quantum_model.loss_fn(logits, labels)
            
            # Backward pass
            quantum_optimizer.zero_grad()
            loss.backward()
            quantum_optimizer.step()
        
        # Train classical model
        classical_model.train()
        for batch in tqdm(train_loader, desc=f"Training classical model (Epoch {epoch+1})"):
            input_ids, labels = batch
            
            # Forward pass
            logits = classical_model(input_ids)
            # Reshape for loss function
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            loss = loss_fn(logits_flat, labels_flat)
            
            # Backward pass
            classical_optimizer.zero_grad()
            loss.backward()
            classical_optimizer.step()
    
    # Compare models
    print("\nComparing quantum and classical models...")
    results = compare_models(quantum_model, classical_model, tokenizer, texts)
    
    # Print results
    print("\nResults:")
    print(f"Quantum model inference time: {results['quantum_time']:.6f} seconds")
    print(f"Classical model inference time: {results['classical_time']:.6f} seconds")
    print(f"Time advantage: {results['time_advantage']:.2f}x")
    print(f"Quantum model coherence score: {results['quantum_coherence']:.4f}")
    print(f"Classical model coherence score: {results['classical_coherence']:.4f}")
    print(f"Coherence advantage: {results['coherence_advantage']:.2f}x")
    
    # Plot comparison
    plot_comparison(results)
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()