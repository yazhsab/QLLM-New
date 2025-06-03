"""
Advanced implementation of Quantum Large Language Model (QLLM).
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
from tqdm import tqdm

from quantum_llm.qllm_base import QLLMBase
from variational_circuits.variational_circuits import create_variational_circuit
from basic_quantum_circuits.data_encoding import create_data_embedding_circuit


class QLLMAdvanced(QLLMBase):
    """
    Advanced Quantum Large Language Model with state-of-the-art quantum algorithms.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 n_qubits: int,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 max_seq_length: int = 128,
                 device: str = "default.qubit",
                 embedding_type: str = "fourier",
                 circuit_type: str = "quantum_transformer",
                 use_quantum_attention: bool = True,
                 use_rotary_embeddings: bool = True,
                 use_quantum_ffn: bool = True,
                 use_quantum_layernorm: bool = True):
        """
        Initialize the advanced QLLM.
        
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
            use_quantum_attention: Whether to use quantum attention mechanism
            use_rotary_embeddings: Whether to use rotary embeddings
            use_quantum_ffn: Whether to use quantum feed-forward network
            use_quantum_layernorm: Whether to use quantum layer normalization
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_length=max_seq_length,
            device=device,
            embedding_type=embedding_type,
            circuit_type=circuit_type
        )
        
        self.use_quantum_attention = use_quantum_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_quantum_ffn = use_quantum_ffn
        self.use_quantum_layernorm = use_quantum_layernorm
        
        # Additional classical components
        self.output_projection = nn.Linear(n_qubits, vocab_size)
        
        # Positional embeddings
        if use_rotary_embeddings:
            self.rotary_dim = embedding_dim // 2
            self.max_freq = 10000.0
        else:
            self.positional_embedding = nn.Embedding(max_seq_length, embedding_dim)
    
    def _create_quantum_transformer(self, seq_idx: int):
        """
        Create an advanced quantum transformer circuit for a specific sequence position.
        
        Args:
            seq_idx: Index in the sequence
            
        Returns:
            function: Quantum circuit function
        """
        @qml.qnode(self.devices[seq_idx])
        def circuit(inputs, params):
            # Apply rotary embeddings if enabled
            if self.use_rotary_embeddings:
                # Apply phase based on position
                for i in range(self.n_qubits):
                    qml.PhaseShift(seq_idx / (10000 ** (i / self.n_qubits)), wires=i)
            
            # Embed classical data into quantum state
            self.quantum_embedding(inputs)
            
            param_idx = 0
            
            # Transformer layers
            for layer in range(self.n_layers):
                # Quantum Layer Normalization (if enabled)
                if self.use_quantum_layernorm:
                    for wire in range(self.n_qubits):
                        qml.RY(params[param_idx % len(params)], wires=wire)
                        param_idx += 1
                
                # Multi-head Quantum Attention
                qubits_per_head = self.n_qubits // self.n_heads
                
                for head in range(self.n_heads):
                    head_wires = list(range(head * qubits_per_head, (head + 1) * qubits_per_head))
                    
                    # Query, Key, Value transformations
                    for wire in head_wires:
                        qml.Rot(params[param_idx % len(params)], 
                               params[(param_idx+1) % len(params)], 
                               params[(param_idx+2) % len(params)], wires=wire)
                        param_idx += 3
                    
                    if self.use_quantum_attention:
                        # Advanced quantum attention mechanism
                        # Implements a quantum version of scaled dot-product attention
                        
                        # Split head wires into Q, K, V sections
                        q_size = k_size = v_size = qubits_per_head // 3
                        q_wires = head_wires[:q_size]
                        k_wires = head_wires[q_size:q_size+k_size]
                        v_wires = head_wires[q_size+k_size:]
                        
                        # Q-K interaction (attention scores)
                        for q in q_wires:
                            for k in k_wires:
                                qml.CNOT(wires=[q, k])
                                qml.RZ(params[param_idx % len(params)], wires=k)
                                qml.CNOT(wires=[q, k])
                                param_idx += 1
                        
                        # Apply softmax-like operation (approximated in quantum)
                        for k in k_wires:
                            qml.RY(params[param_idx % len(params)], wires=k)
                            param_idx += 1
                        
                        # K-V interaction (weighted values)
                        for k in k_wires:
                            for v in v_wires:
                                qml.CNOT(wires=[k, v])
                                qml.RZ(params[param_idx % len(params)], wires=v)
                                qml.CNOT(wires=[k, v])
                                param_idx += 1
                    else:
                        # Simple entanglement as attention
                        for i in range(len(head_wires)):
                            for j in range(i+1, len(head_wires)):
                                qml.CNOT(wires=[head_wires[i], head_wires[j]])
                                qml.RZ(params[param_idx % len(params)], wires=head_wires[j])
                                qml.CNOT(wires=[head_wires[i], head_wires[j]])
                                param_idx += 1
                
                # Quantum Feed-Forward Network
                if self.use_quantum_ffn:
                    # First FFN layer (expansion)
                    for wire in range(self.n_qubits):
                        qml.Rot(params[param_idx % len(params)], 
                               params[(param_idx+1) % len(params)], 
                               params[(param_idx+2) % len(params)], wires=wire)
                        param_idx += 3
                    
                    # Non-linearity (approximated by controlled operations)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                        qml.RZ(params[param_idx % len(params)], wires=i+1)
                        qml.CNOT(wires=[i, i+1])
                        param_idx += 1
                    
                    # Second FFN layer (projection)
                    for wire in range(self.n_qubits):
                        qml.Rot(params[param_idx % len(params)], 
                               params[(param_idx+1) % len(params)], 
                               params[(param_idx+2) % len(params)], wires=wire)
                        param_idx += 3
                else:
                    # Simple rotation layer
                    for wire in range(self.n_qubits):
                        qml.Rot(params[param_idx % len(params)], 
                               params[(param_idx+1) % len(params)], 
                               params[(param_idx+2) % len(params)], wires=wire)
                        param_idx += 3
                
                # Final Layer Normalization
                if self.use_quantum_layernorm and layer == self.n_layers - 1:
                    for wire in range(self.n_qubits):
                        qml.RY(params[param_idx % len(params)], wires=wire)
                        param_idx += 1
            
            # Return expectation values for all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the advanced QLLM.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Classical embedding
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional embeddings if not using rotary
        if not self.use_rotary_embeddings:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.positional_embedding(positions)
            embeddings = embeddings + pos_embeddings
        
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
                
                # Project to vocabulary size using classical layer
                logits = self.output_projection(result_tensor)
                
                seq_outputs.append(logits)
            
            outputs.append(torch.stack(seq_outputs))
        
        return torch.stack(outputs)
    
    def init_params(self):
        """Initialize circuit parameters with improved initialization."""
        super().init_params()
        
        # Use Glorot/Xavier initialization for better convergence
        limit = np.sqrt(6.0 / (self.n_qubits + self.n_qubits))
        self.params = np.random.uniform(-limit, limit, size=self.n_params)
        
        # Scale to [0, 2π] range for rotation gates
        self.params = 2 * np.pi * (self.params - np.min(self.params)) / (np.max(self.params) - np.min(self.params))
    
    def parameters(self):
        """
        Get model parameters for optimization.
        
        Returns:
            list: List of parameters
        """
        # Convert quantum parameters to PyTorch parameters
        params_tensor = torch.nn.Parameter(torch.tensor(self.params, requires_grad=True))
        
        # Return all parameters
        param_list = list(self.embedding.parameters()) + [params_tensor] + list(self.output_projection.parameters())
        
        if not self.use_rotary_embeddings:
            param_list += list(self.positional_embedding.parameters())
            
        return param_list
    
    def train(self, mode: bool = True):
        """
        Set the model to training mode.
        
        Args:
            mode: Whether to set training mode
        """
        self.training = mode
        self.embedding.train(mode)
        self.output_projection.train(mode)
        
        if not self.use_rotary_embeddings:
            self.positional_embedding.train(mode)
            
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
            'use_quantum_attention': self.use_quantum_attention,
            'use_rotary_embeddings': self.use_rotary_embeddings,
            'use_quantum_ffn': self.use_quantum_ffn,
            'use_quantum_layernorm': self.use_quantum_layernorm,
            'params': self.params,
            'embedding_state': self.embedding.state_dict(),
            'output_projection_state': self.output_projection.state_dict(),
            'trained': self.trained,
            'training_history': self.training_history,
            'creation_time': self.creation_time
        }
        
        if not self.use_rotary_embeddings:
            state_dict['positional_embedding_state'] = self.positional_embedding.state_dict()
        
        torch.save(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            device: Quantum device to use
            
        Returns:
            QLLMAdvanced: Loaded model
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
            circuit_type=state_dict['circuit_type'],
            use_quantum_attention=state_dict['use_quantum_attention'],
            use_rotary_embeddings=state_dict['use_rotary_embeddings'],
            use_quantum_ffn=state_dict['use_quantum_ffn'],
            use_quantum_layernorm=state_dict['use_quantum_layernorm']
        )
        
        # Load parameters
        model.params = state_dict['params']
        model.embedding.load_state_dict(state_dict['embedding_state'])
        model.output_projection.load_state_dict(state_dict['output_projection_state'])
        
        if not model.use_rotary_embeddings:
            model.positional_embedding.load_state_dict(state_dict['positional_embedding_state'])
        
        model.trained = state_dict['trained']
        model.training_history = state_dict['training_history']
        model.creation_time = state_dict['creation_time']
        
        return model


class QLLMWithKVCache(QLLMAdvanced):
    """
    QLLM with key-value cache for efficient inference.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the QLLM with KV cache."""
        super().__init__(*args, **kwargs)
        self.kv_cache = None
    
    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with KV cache support.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            use_cache: Whether to use and update KV cache
            
        Returns:
            tuple: (output logits, updated KV cache)
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize or get KV cache
        if use_cache and self.kv_cache is None:
            self.kv_cache = {
                'last_seq_len': 0,
                'quantum_states': []
            }
        
        # Classical embedding
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional embeddings if not using rotary
        if not self.use_rotary_embeddings:
            if use_cache:
                # Use correct positions based on cache
                positions = torch.arange(
                    self.kv_cache['last_seq_len'], 
                    self.kv_cache['last_seq_len'] + seq_len, 
                    device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                
            pos_embeddings = self.positional_embedding(positions)
            embeddings = embeddings + pos_embeddings
        
        # Process each sequence position with quantum circuits
        outputs = []
        
        for b in range(batch_size):
            seq_outputs = []
            
            for s in range(seq_len):
                # Get embedding for this position
                emb = embeddings[b, s].detach().numpy()
                
                # Reduce dimension if needed
                if len(emb) > self.n_qubits:
                    emb = np.array([np.mean(emb[i:i+len(emb)//self.n_qubits]) 
                                   for i in range(0, len(emb), len(emb)//self.n_qubits)])[:self.n_qubits]
                
                # Normalize to [0, 2π]
                emb = 2 * np.pi * (emb - np.min(emb)) / (np.max(emb) - np.min(emb) + 1e-8)
                
                # Get absolute position for rotary embeddings or cache
                abs_pos = self.kv_cache['last_seq_len'] + s if use_cache else s
                
                # Create and run quantum circuit for this position
                circuit = self.quantum_circuit_fn(abs_pos)
                result = circuit(emb, self.params)
                
                # Store in KV cache if using cache
                if use_cache:
                    if len(self.kv_cache['quantum_states']) <= abs_pos:
                        self.kv_cache['quantum_states'].append(result)
                    else:
                        self.kv_cache['quantum_states'][abs_pos] = result
                
                # Convert to tensor
                result_tensor = torch.tensor(result, dtype=torch.float32)
                
                # Project to vocabulary size using classical layer
                logits = self.output_projection(result_tensor)
                
                seq_outputs.append(logits)
            
            outputs.append(torch.stack(seq_outputs))
        
        # Update cache length
        if use_cache:
            self.kv_cache['last_seq_len'] += seq_len
        
        return torch.stack(outputs), self.kv_cache if use_cache else None
    
    def generate(self, 
                input_ids: torch.Tensor, 
                max_length: int = 50, 
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                use_cache: bool = True) -> torch.Tensor:
        """
        Generate text with KV cache for efficiency.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            use_cache: Whether to use KV cache
            
        Returns:
            torch.Tensor: Generated token IDs [batch_size, max_length]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # Reset KV cache
        if use_cache:
            self.kv_cache = None
        
        # Initialize with input_ids
        generated = input_ids.clone()
        
        # Process initial input
        logits, _ = self.forward(generated, use_cache=use_cache)
        
        # Generate tokens
        for i in range(max_length - input_ids.shape[1]):
            # Get predictions for the last token
            last_token_logits = logits[:, -1, :]
            
            # Apply temperature
            last_token_logits = last_token_logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = torch.topk(last_token_logits, k=top_k, dim=-1)[0][:, -1].unsqueeze(-1)
                last_token_logits[last_token_logits < indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    last_token_logits[b, indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Forward pass on only the new token
            logits, _ = self.forward(next_token, use_cache=use_cache)
        
        return generated