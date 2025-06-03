"""
Multimodal Quantum LLM model implementation.

This module provides a multimodal extension of the Quantum LLM that can process
different types of input modalities (text, images, audio) using quantum computing.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
from tqdm import tqdm

from quantum_llm.qllm_advanced import QLLMAdvanced
from quantum_llm.multimodal.encoders import (
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    MultimodalFusion
)
from quantum_llm.multimodal.data_encoding import (
    create_multimodal_embedding_circuit,
    create_modality_specific_embedding,
    quantum_multimodal_fusion
)


class MultimodalQLLM(QLLMAdvanced):
    """
    Multimodal Quantum Large Language Model.
    
    This class extends the advanced QLLM to support multiple input modalities
    (text, images, audio) using quantum computing for multimodal learning.
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
                 use_quantum_layernorm: bool = True,
                 supported_modalities: List[str] = ["text"],
                 modality_embedding_types: Optional[Dict[str, str]] = None,
                 qubit_allocations: Optional[Dict[str, int]] = None,
                 fusion_dim: int = 512):
        """
        Initialize the multimodal QLLM.
        
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
            supported_modalities: List of supported modalities
            modality_embedding_types: Dictionary mapping modality names to embedding types
            qubit_allocations: Dictionary mapping modality names to qubit allocations
            fusion_dim: Dimension of the fused embedding
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
            circuit_type=circuit_type,
            use_quantum_attention=use_quantum_attention,
            use_rotary_embeddings=use_rotary_embeddings,
            use_quantum_ffn=use_quantum_ffn,
            use_quantum_layernorm=use_quantum_layernorm
        )
        
        self.supported_modalities = supported_modalities
        
        # Default embedding types for each modality
        self.modality_embedding_types = modality_embedding_types or {
            "text": "angle",
            "image": "amplitude",
            "audio": "fourier"
        }
        
        # Default qubit allocations
        if qubit_allocations is None:
            # Allocate qubits evenly among supported modalities
            qubits_per_modality = n_qubits // len(supported_modalities)
            self.qubit_allocations = {
                modality: qubits_per_modality for modality in supported_modalities
            }
            
            # Allocate any remaining qubits to the first modality
            remaining = n_qubits - (qubits_per_modality * len(supported_modalities))
            if remaining > 0 and supported_modalities:
                self.qubit_allocations[supported_modalities[0]] += remaining
        else:
            self.qubit_allocations = qubit_allocations
        
        # Feature dimensions for each modality
        self.feature_dims = {
            "text": embedding_dim,
            "image": embedding_dim,
            "audio": embedding_dim
        }
        
        # Create modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        
        if "text" in supported_modalities:
            self.modality_encoders["text"] = TextEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                output_dim=embedding_dim
            )
        
        if "image" in supported_modalities:
            self.modality_encoders["image"] = ImageEncoder(
                input_channels=3,  # RGB
                output_dim=embedding_dim
            )
        
        if "audio" in supported_modalities:
            self.modality_encoders["audio"] = AudioEncoder(
                input_channels=1,  # Mono
                output_dim=embedding_dim
            )
        
        # Create multimodal fusion module
        self.classical_fusion = MultimodalFusion(
            modality_dims={modality: embedding_dim for modality in supported_modalities},
            fusion_dim=fusion_dim
        )
        
        # Create quantum embedding circuit for multimodal data
        self.quantum_multimodal_embedding = create_multimodal_embedding_circuit(
            modality_types=self.modality_embedding_types,
            n_qubits=n_qubits,
            qubit_allocations=self.qubit_allocations,
            feature_dims=self.feature_dims
        )
        
        # Create quantum fusion circuit
        self.quantum_fusion = quantum_multimodal_fusion(
            n_qubits=n_qubits,
            n_modalities=len(supported_modalities)
        )
        
        # Additional parameters for quantum fusion
        self.fusion_params = np.random.uniform(
            low=0, high=2*np.pi, size=n_qubits * 6
        )
    
    def _create_quantum_transformer(self, seq_idx: int):
        """
        Create a multimodal quantum transformer circuit for a specific sequence position.
        
        Args:
            seq_idx: Index in the sequence
            
        Returns:
            function: Quantum circuit function
        """
        @qml.qnode(self.devices[seq_idx])
        def circuit(inputs, params):
            # Apply multimodal embedding
            if isinstance(inputs, dict):
                # If inputs is a dictionary of modality features
                self.quantum_multimodal_embedding(inputs)
            else:
                # Fallback to standard embedding for backward compatibility
                self.quantum_embedding(inputs)
            
            # Apply quantum fusion if multiple modalities
            if len(self.supported_modalities) > 1:
                self.quantum_fusion(self.fusion_params)
            
            # Apply rotary embeddings if enabled
            if self.use_rotary_embeddings:
                # Apply phase based on position
                for i in range(self.n_qubits):
                    qml.PhaseShift(seq_idx / (10000 ** (i / self.n_qubits)), wires=i)
            
            param_idx = 0
            
            # Transformer layers (same as in QLLMAdvanced)
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
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the multimodal QLLM.
        
        Args:
            inputs: Dictionary mapping modality names to their inputs
                - 'text': [batch_size, seq_len] tensor of token IDs
                - 'image': [batch_size, channels, height, width] tensor of images
                - 'audio': [batch_size, channels, time] tensor of audio
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        # Process each modality with its encoder
        modality_embeddings = {}
        
        for modality, modality_input in inputs.items():
            if modality in self.modality_encoders:
                modality_embeddings[modality] = self.modality_encoders[modality](modality_input)
        
        # Get batch size and sequence length from text input (if available)
        if 'text' in inputs:
            batch_size, seq_len = inputs['text'].shape
        else:
            # For non-text inputs, assume batch size from first modality and seq_len=1
            batch_size = next(iter(inputs.values())).shape[0]
            seq_len = 1
        
        # Process each sequence position with quantum circuits
        outputs = []
        
        for b in range(batch_size):
            seq_outputs = []
            
            for s in range(seq_len):
                # Prepare quantum inputs for each modality
                quantum_inputs = {}
                
                for modality, embedding in modality_embeddings.items():
                    # Get embedding for this position and batch
                    if modality == 'text':
                        # For text, we have sequence dimension
                        emb = embedding[b, s].detach().numpy()
                    else:
                        # For other modalities, we might not have sequence dimension
                        emb = embedding[b].detach().numpy()
                    
                    # Normalize to [0, 2π]
                    emb = 2 * np.pi * (emb - np.min(emb)) / (np.max(emb) - np.min(emb) + 1e-8)
                    
                    quantum_inputs[modality] = emb
                
                # Create and run quantum circuit for this position
                circuit = self.quantum_circuit_fn(s)
                result = circuit(quantum_inputs, self.params)
                
                # Convert to tensor
                result_tensor = torch.tensor(result, dtype=torch.float32)
                
                # Project to vocabulary size using classical layer
                logits = self.output_projection(result_tensor)
                
                seq_outputs.append(logits)
            
            outputs.append(torch.stack(seq_outputs))
        
        return torch.stack(outputs)
    
    def parameters(self):
        """
        Get model parameters for optimization.
        
        Returns:
            list: List of parameters
        """
        # Convert quantum parameters to PyTorch parameters
        params_tensor = torch.nn.Parameter(torch.tensor(self.params, requires_grad=True))
        fusion_params_tensor = torch.nn.Parameter(torch.tensor(self.fusion_params, requires_grad=True))
        
        # Get parameters from all components
        param_list = [params_tensor, fusion_params_tensor]
        
        # Add parameters from modality encoders
        for encoder in self.modality_encoders.values():
            param_list.extend(list(encoder.parameters()))
        
        # Add parameters from classical fusion
        param_list.extend(list(self.classical_fusion.parameters()))
        
        # Add parameters from output projection
        param_list.extend(list(self.output_projection.parameters()))
        
        # Add positional embedding parameters if not using rotary
        if not self.use_rotary_embeddings:
            param_list.extend(list(self.positional_embedding.parameters()))
            
        return param_list
    
    def train(self, mode: bool = True):
        """
        Set the model to training mode.
        
        Args:
            mode: Whether to set training mode
        """
        self.training = mode
        
        # Set training mode for all components
        for encoder in self.modality_encoders.values():
            encoder.train(mode)
        
        self.classical_fusion.train(mode)
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
            'supported_modalities': self.supported_modalities,
            'modality_embedding_types': self.modality_embedding_types,
            'qubit_allocations': self.qubit_allocations,
            'params': self.params,
            'fusion_params': self.fusion_params,
            'output_projection_state': self.output_projection.state_dict(),
            'trained': self.trained,
            'training_history': self.training_history,
            'creation_time': self.creation_time
        }
        
        # Save modality encoder states
        state_dict['modality_encoders_state'] = {
            modality: encoder.state_dict()
            for modality, encoder in self.modality_encoders.items()
        }
        
        # Save classical fusion state
        state_dict['classical_fusion_state'] = self.classical_fusion.state_dict()
        
        # Save positional embedding if not using rotary
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
            MultimodalQLLM: Loaded model
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
            use_quantum_layernorm=state_dict['use_quantum_layernorm'],
            supported_modalities=state_dict['supported_modalities'],
            modality_embedding_types=state_dict['modality_embedding_types'],
            qubit_allocations=state_dict['qubit_allocations']
        )
        
        # Load parameters
        model.params = state_dict['params']
        model.fusion_params = state_dict['fusion_params']
        model.output_projection.load_state_dict(state_dict['output_projection_state'])
        
        # Load modality encoder states
        for modality, encoder_state in state_dict['modality_encoders_state'].items():
            if modality in model.modality_encoders:
                model.modality_encoders[modality].load_state_dict(encoder_state)
        
        # Load classical fusion state
        model.classical_fusion.load_state_dict(state_dict['classical_fusion_state'])
        
        # Load positional embedding if not using rotary
        if not model.use_rotary_embeddings:
            model.positional_embedding.load_state_dict(state_dict['positional_embedding_state'])
        
        model.trained = state_dict['trained']
        model.training_history = state_dict['training_history']
        model.creation_time = state_dict['creation_time']
        
        return model