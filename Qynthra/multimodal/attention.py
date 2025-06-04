"""
Cross-modal attention mechanisms for Quantum LLM.

This module provides quantum and classical cross-modal attention mechanisms
for better fusion of information from different modalities.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable


class ClassicalCrossModalAttention(nn.Module):
    """Classical cross-modal attention mechanism."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        """
        Initialize the cross-modal attention module.
        
        Args:
            embedding_dim: Dimension of the embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def add_modality(self, modality: str):
        """
        Add a modality to the cross-modal attention.
        
        Args:
            modality: Name of the modality
        """
        if modality not in self.projections:
            self.projections[modality] = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, query_modality: str, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query_modality: Name of the query modality
            embeddings: Dictionary mapping modality names to their embeddings
                Each embedding should have shape [batch_size, seq_len, embedding_dim]
                
        Returns:
            torch.Tensor: Attended embeddings [batch_size, seq_len, embedding_dim]
        """
        # Ensure all modalities have projections
        for modality in embeddings:
            self.add_modality(modality)
        
        # Apply projections to each modality
        projected = {
            modality: self.projections[modality](embedding)
            for modality, embedding in embeddings.items()
        }
        
        # Get query embedding
        if query_modality not in projected:
            raise ValueError(f"Query modality '{query_modality}' not found in embeddings")
        
        query = projected[query_modality]
        
        # Concatenate all other modalities as keys and values
        other_modalities = [
            projected[modality] for modality in projected
            if modality != query_modality
        ]
        
        if not other_modalities:
            # If no other modalities, return query as is
            return query
        
        # Concatenate other modalities
        # For simplicity, we'll just concatenate along the sequence dimension
        # In a more sophisticated implementation, you might want to use a different approach
        key_value = torch.cat(other_modalities, dim=1)
        
        # Apply multi-head attention
        attended, _ = self.mha(
            query=query,
            key=key_value,
            value=key_value
        )
        
        # Residual connection and layer normalization
        attended = self.layer_norm1(query + attended)
        
        # Feed-forward network
        output = self.layer_norm2(attended + self.ffn(attended))
        
        return output


def quantum_cross_modal_attention(
    n_qubits: int,
    n_modalities: int
) -> Callable:
    """
    Create a quantum circuit for cross-modal attention.
    
    Args:
        n_qubits: Number of qubits
        n_modalities: Number of modalities
        
    Returns:
        function: Quantum cross-modal attention function
    """
    # Qubits per modality
    qubits_per_modality = n_qubits // n_modalities
    
    def attention_circuit(params: np.ndarray, query_modality_idx: int) -> None:
        """
        Apply quantum cross-modal attention circuit.
        
        Args:
            params: Circuit parameters
            query_modality_idx: Index of the query modality
        """
        param_idx = 0
        
        # Get qubit ranges for each modality
        modality_qubits = []
        for m in range(n_modalities):
            start_qubit = m * qubits_per_modality
            end_qubit = start_qubit + qubits_per_modality
            modality_qubits.append(list(range(start_qubit, end_qubit)))
        
        # Query modality qubits
        query_qubits = modality_qubits[query_modality_idx]
        
        # Apply controlled operations from query qubits to other modalities
        for m in range(n_modalities):
            if m != query_modality_idx:
                target_qubits = modality_qubits[m]
                
                # Apply controlled operations
                for q_idx, q in enumerate(query_qubits):
                    for t_idx, t in enumerate(target_qubits):
                        # Apply CNOT from query to target
                        qml.CNOT(wires=[q, t])
                        
                        # Apply controlled rotation
                        qml.CRZ(
                            params[param_idx % len(params)],
                            wires=[q, t]
                        )
                        param_idx += 1
                        
                        # Apply CNOT again to disentangle
                        qml.CNOT(wires=[q, t])
        
        # Apply final rotations to query qubits
        for q in query_qubits:
            qml.Rot(
                params[param_idx % len(params)],
                params[(param_idx + 1) % len(params)],
                params[(param_idx + 2) % len(params)],
                wires=q
            )
            param_idx += 3
    
    return attention_circuit