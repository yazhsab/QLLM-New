"""
Base implementation of Quantum Large Language Model (QLLM).
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
from tqdm import tqdm

from qnn_models.advanced_qnn import QTransformer, QRNN
from basic_quantum_circuits.data_encoding import create_data_embedding_circuit
from variational_circuits.variational_circuits import create_variational_circuit


class QLLMBase:
    """
    Base class for Quantum Large Language Model.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 n_qubits: int,
                 n_layers: int = 2,
                 n_heads: int = 2,
                 max_seq_length: int = 128,
                 device: str = "default.qubit",
                 embedding_type: str = "angle",
                 circuit_type: str = "quantum_transformer"):
        """
        Initialize the QLLM base class.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the classical embedding
            n_qubits: Number of qubits
            n_layers: Number of quantum transformer layers
            n_heads: Number of attention heads
            max_seq_length: Maximum sequence length
            device: Quantum device to use
            embedding_type: Type of data embedding
            circuit_type: Type of quantum circuit
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.device_name = device
        self.embedding_type = embedding_type
        self.circuit_type = circuit_type
        
        # Initialize quantum devices
        self.devices = []
        for _ in range(max_seq_length):
            self.devices.append(qml.device(device, wires=n_qubits))
        
        # Create classical embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Create quantum embedding circuit
        self.quantum_embedding = create_data_embedding_circuit(
            embedding_type, n_qubits, n_qubits
        )
        
        # Create quantum circuit based on type
        if circuit_type == "quantum_transformer":
            self.quantum_circuit_fn = self._create_quantum_transformer
        elif circuit_type == "quantum_rnn":
            self.quantum_circuit_fn = self._create_quantum_rnn
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Initialize parameters
        self.init_params()
        
        # Training state
        self.trained = False
        self.training_history = {
            'loss': [],
            'perplexity': [],
            'val_loss': [],
            'val_perplexity': []
        }
        self.creation_time = time.strftime("%Y%m%d-%H%M%S")
    
    def _create_quantum_transformer(self, seq_idx: int):
        """
        Create a quantum transformer circuit for a specific sequence position.
        
        Args:
            seq_idx: Index in the sequence
            
        Returns:
            function: Quantum circuit function
        """
        @qml.qnode(self.devices[seq_idx])
        def circuit(inputs, params):
            # Embed classical data into quantum state
            self.quantum_embedding(inputs)
            
            # Apply quantum transformer circuit
            n_params_per_layer = self.n_qubits * 6  # Based on QTransformer implementation
            
            param_idx = 0
            
            # Transformer layers
            for layer in range(self.n_layers):
                # Multi-head self-attention
                qubits_per_head = self.n_qubits // self.n_heads
                
                for head in range(self.n_heads):
                    head_wires = list(range(head * qubits_per_head, (head + 1) * qubits_per_head))
                    
                    # Self-attention within each head
                    for wire in head_wires:
                        qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
                        param_idx += 3
                    
                    # Entangle qubits within head (attention mechanism)
                    for i in range(len(head_wires)):
                        for j in range(i+1, len(head_wires)):
                            qml.CNOT(wires=[head_wires[i], head_wires[j]])
                            qml.RZ(params[param_idx % len(params)], wires=head_wires[j])
                            qml.CNOT(wires=[head_wires[i], head_wires[j]])
                
                # Feed-forward network
                for wire in range(self.n_qubits):
                    qml.Rot(params[param_idx % len(params)], 
                           params[(param_idx+1) % len(params)], 
                           params[(param_idx+2) % len(params)], wires=wire)
                    param_idx += 3
            
            # Return expectation values for all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def _create_quantum_rnn(self, seq_idx: int):
        """
        Create a quantum RNN circuit for a specific sequence position.
        
        Args:
            seq_idx: Index in the sequence
            
        Returns:
            function: Quantum circuit function
        """
        @qml.qnode(self.devices[seq_idx])
        def circuit(inputs, params):
            # Embed classical data into quantum state
            self.quantum_embedding(inputs)
            
            # Split qubits into hidden and input/output
            n_hidden = self.n_qubits // 2
            hidden_wires = list(range(n_hidden))
            io_wires = list(range(n_hidden, self.n_qubits))
            
            # Initialize hidden state
            for wire in hidden_wires:
                qml.Hadamard(wire)
            
            param_idx = 0
            
            # Recurrent steps
            for step in range(self.n_layers):
                # Input-hidden interaction
                for h_wire in hidden_wires:
                    for io_wire in io_wires:
                        qml.CNOT(wires=[io_wire, h_wire])
                
                # Hidden state update
                for wire in hidden_wires:
                    qml.Rot(params[param_idx % len(params)], 
                           params[(param_idx+1) % len(params)], 
                           params[(param_idx+2) % len(params)], wires=wire)
                    param_idx += 3
                
                # Hidden-hidden interaction
                for i in range(len(hidden_wires) - 1):
                    qml.CNOT(wires=[hidden_wires[i], hidden_wires[i+1]])
                
                # Hidden-output interaction
                for h_wire in hidden_wires:
                    for io_wire in io_wires:
                        qml.CNOT(wires=[h_wire, io_wire])
                
                # Output update
                for wire in io_wires:
                    qml.Rot(params[param_idx % len(params)], 
                           params[(param_idx+1) % len(params)], 
                           params[(param_idx+2) % len(params)], wires=wire)
                    param_idx += 3
            
            # Return expectation values for output wires
            return [qml.expval(qml.PauliZ(i)) for i in io_wires]
        
        return circuit
    
    def init_params(self):
        """Initialize circuit parameters."""
        if self.circuit_type == "quantum_transformer":
            # Parameters for quantum transformer
            # Each layer has n_qubits * 6 parameters (based on QTransformer implementation)
            self.n_params = self.n_layers * self.n_qubits * 6
        elif self.circuit_type == "quantum_rnn":
            # Parameters for quantum RNN
            # Each layer has (n_hidden + n_io) * 3 parameters
            n_hidden = self.n_qubits // 2
            n_io = self.n_qubits - n_hidden
            self.n_params = self.n_layers * (n_hidden + n_io) * 3
        
        self.params = np.random.uniform(
            low=0, high=2*np.pi, size=self.n_params
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QLLM.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Classical embedding
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Process each sequence position with quantum circuits
        outputs = []
        
        for b in range(batch_size):
            seq_outputs = []
            
            for s in range(seq_len):
                # Get embedding for this position
                emb = embeddings[b, s].detach().numpy()
                
                # Reduce dimension if needed
                if len(emb) > self.n_qubits:
                    # Simple dimension reduction by averaging
                    emb = np.array([np.mean(emb[i:i+len(emb)//self.n_qubits]) 
                                   for i in range(0, len(emb), len(emb)//self.n_qubits)])[:self.n_qubits]
                
                # Normalize to [0, 2π]
                emb = 2 * np.pi * (emb - np.min(emb)) / (np.max(emb) - np.min(emb) + 1e-8)
                
                # Create and run quantum circuit for this position
                circuit = self.quantum_circuit_fn(s)
                result = circuit(emb, self.params)
                
                # Convert to tensor
                result_tensor = torch.tensor(result, dtype=torch.float32)
                
                # Project to vocabulary size
                if len(result_tensor) < self.vocab_size:
                    # Repeat and truncate
                    repeats = (self.vocab_size + len(result_tensor) - 1) // len(result_tensor)
                    result_tensor = result_tensor.repeat(repeats)[:self.vocab_size]
                else:
                    # Truncate
                    result_tensor = result_tensor[:self.vocab_size]
                
                seq_outputs.append(result_tensor)
            
            outputs.append(torch.stack(seq_outputs))
        
        return torch.stack(outputs)
    
    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross-entropy loss.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            labels: True labels [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Calculate cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_flat, labels_flat)
        
        return loss
    
    def fit(self, 
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: Optional[torch.utils.data.DataLoader] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            epochs: int = 5,
            callback: Optional[Callable] = None,
            verbose: bool = True) -> Dict:
        """
        Train the QLLM model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            epochs: Number of training epochs
            callback: Callback function
            verbose: Whether to print progress
            
        Returns:
            dict: Training history
        """
        # Default optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            if verbose:
                train_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            else:
                train_iter = train_dataloader
            
            for batch in train_iter:
                input_ids, labels = batch
                
                # Forward pass
                logits = self.forward(input_ids)
                loss = self.loss_fn(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            self.training_history['loss'].append(train_loss)
            
            # Calculate perplexity
            train_perplexity = np.exp(train_loss)
            self.training_history['perplexity'].append(train_perplexity)
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_perplexity = self.evaluate(val_dataloader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_perplexity'].append(val_perplexity)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Loss = {train_loss:.4f}, Perplexity = {train_perplexity:.4f}, "
                          f"Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Loss = {train_loss:.4f}, Perplexity = {train_perplexity:.4f}")
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, self.params, train_loss)
        
        self.trained = True
        return self.training_history
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader
            
        Returns:
            tuple: (loss, perplexity)
        """
        self.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, labels = batch
                
                # Forward pass
                logits = self.forward(input_ids)
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def generate(self, 
                input_ids: torch.Tensor, 
                max_length: int = 50, 
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text from the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            
        Returns:
            torch.Tensor: Generated token IDs [batch_size, max_length]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # Initialize with input_ids
        generated = input_ids.clone()
        
        # Generate tokens
        for i in range(max_length - input_ids.shape[1]):
            # Get predictions for the last token
            logits = self.forward(generated)[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = torch.topk(logits, k=top_k, dim=-1)[0][:, -1].unsqueeze(-1)
                logits[logits < indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def parameters(self):
        """
        Get model parameters for optimization.
        
        Returns:
            list: List of parameters
        """
        # Convert quantum parameters to PyTorch parameters
        params_tensor = torch.nn.Parameter(torch.tensor(self.params, requires_grad=True))
        
        # Return both classical and quantum parameters
        return list(self.embedding.parameters()) + [params_tensor]
    
    def train(self, mode: bool = True):
        """
        Set the model to training mode.
        
        Args:
            mode: Whether to set training mode
        """
        self.training = mode
        self.embedding.train(mode)
        return self
    
    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.train(False)
        return self
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        state_dict = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'max_seq_length': self.max_seq_length,
            'device_name': self.device_name,
            'embedding_type': self.embedding_type,
            'circuit_type': self.circuit_type,
            'params': self.params,
            'embedding_state': self.embedding.state_dict(),
            'trained': self.trained,
            'training_history': self.training_history,
            'creation_time': self.creation_time
        }
        
        torch.save(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            device: Quantum device to use
            
        Returns:
            QLLMBase: Loaded model
        """
        state_dict = torch.load(filepath)
        
        # Create new instance
        model = cls(
            vocab_size=state_dict['vocab_size'],
            embedding_dim=state_dict['embedding_dim'],
            n_qubits=state_dict['n_qubits'],
            n_layers=state_dict['n_layers'],
            n_heads=state_dict['n_heads'],
            max_seq_length=state_dict['max_seq_length'],
            device=device or state_dict['device_name'],
            embedding_type=state_dict['embedding_type'],
            circuit_type=state_dict['circuit_type']
        )
        
        # Load parameters
        model.params = state_dict['params']
        model.embedding.load_state_dict(state_dict['embedding_state'])
        model.trained = state_dict['trained']
        model.training_history = state_dict['training_history']
        model.creation_time = state_dict['creation_time']
        
        return model