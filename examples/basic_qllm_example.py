"""
Basic example of using the Qynthra components.

This script demonstrates how to create, train, and use a basic Qynthra model.
"""

import torch
import numpy as np
import pennylane as qml
from tqdm import tqdm

from Qynthra import (
    QLLMBase,
    QLLMAdvanced,
    QLLMTokenizer,
    QLLMDataset
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Sample text data for demonstration
texts = [
    "Quantum computing leverages quantum mechanical phenomena to perform computations.",
    "Large language models have revolutionized natural language processing.",
    "Combining quantum computing with language models may lead to new capabilities.",
    "Quantum algorithms can potentially solve certain problems more efficiently.",
    "Neural networks learn representations from data through training.",
    "Quantum neural networks use quantum circuits as building blocks.",
    "Entanglement is a key resource in quantum information processing.",
    "Language models predict the next token based on context.",
    "Quantum language models may offer advantages in specific domains.",
    "Hybrid quantum-classical systems combine the strengths of both paradigms."
]

def main():
    print("Qynthra Basic Example")
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
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using {n_qubits} qubits and {n_layers} layers")
    
    # Create model
    print("Creating Qynthra model...")
    model = QLLMBase(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        max_seq_length=32,
        device="default.qubit",
        embedding_type="angle",
        circuit_type="quantum_transformer"
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train model (just a few steps for demonstration)
    print("Training model (demonstration only)...")
    n_epochs = 2
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            input_ids, labels = batch
            
            # Forward pass
            logits = model.forward(input_ids)
            loss = model.loss_fn(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.4f}")
    
    # Generate text
    print("\nGenerating text...")
    
    # Encode a prompt
    prompt = "Quantum computing"
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)])
    
    # Generate continuation
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=20,
        temperature=1.0,
        top_k=5
    )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    print("\nCreating advanced model...")
    # Create advanced model
    advanced_model = QLLMAdvanced(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_heads=2,
        max_seq_length=32,
        device="default.qubit",
        embedding_type="fourier",
        circuit_type="quantum_transformer",
        use_quantum_attention=True,
        use_rotary_embeddings=True
    )
    
    # Show model information
    print(f"Advanced model parameters: {advanced_model.n_params}")
    print("Model created successfully!")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()