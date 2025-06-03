"""
Data encoding circuits for Quantum Large Language Models.

This module provides various methods for embedding classical data into quantum states,
which is a crucial first step in quantum machine learning.
"""

import pennylane as qml
import numpy as np
from typing import List, Callable, Optional, Union


def create_data_embedding_circuit(embedding_type: str, n_qubits: int, n_features: int) -> Callable:
    """
    Create a data embedding circuit based on the specified type.
    
    Args:
        embedding_type: Type of embedding ('angle', 'amplitude', 'iqp', 'fourier')
        n_qubits: Number of qubits
        n_features: Number of classical features to embed
        
    Returns:
        function: Data embedding function
    """
    if embedding_type == "angle":
        return angle_embedding(n_qubits, n_features)
    elif embedding_type == "amplitude":
        return amplitude_embedding(n_qubits)
    elif embedding_type == "iqp":
        return iqp_embedding(n_qubits, n_features)
    elif embedding_type == "fourier":
        return fourier_embedding(n_qubits, n_features)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def angle_embedding(n_qubits: int, n_features: int) -> Callable:
    """
    Create an angle embedding circuit.
    
    Angle embedding encodes classical data as rotation angles of qubits.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of classical features to embed
        
    Returns:
        function: Angle embedding function
    """
    def embedding(features: np.ndarray) -> None:
        """
        Apply angle embedding to qubits.
        
        Args:
            features: Classical features to embed [n_features]
        """
        # Ensure features match expected size
        if len(features) < n_features:
            # Pad with zeros if needed
            features = np.pad(features, (0, n_features - len(features)))
        elif len(features) > n_features:
            # Truncate if too many features
            features = features[:n_features]
            
        # Apply rotations to qubits
        for i in range(min(n_qubits, n_features)):
            qml.RX(features[i], wires=i)
            qml.RY(features[i], wires=i)
            
    return embedding


def amplitude_embedding(n_qubits: int) -> Callable:
    """
    Create an amplitude embedding circuit.
    
    Amplitude embedding encodes classical data as amplitudes of the quantum state.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        function: Amplitude embedding function
    """
    def embedding(features: np.ndarray) -> None:
        """
        Apply amplitude embedding to qubits.
        
        Args:
            features: Classical features to embed [2^n_qubits]
        """
        # Normalize features
        normalized_features = features / np.linalg.norm(features)
        
        # Pad or truncate to match 2^n_qubits
        required_size = 2**n_qubits
        if len(normalized_features) < required_size:
            normalized_features = np.pad(normalized_features, (0, required_size - len(normalized_features)))
        elif len(normalized_features) > required_size:
            normalized_features = normalized_features[:required_size]
            
        # Renormalize after padding/truncation
        normalized_features = normalized_features / np.linalg.norm(normalized_features)
        
        # Apply amplitude embedding
        qml.AmplitudeEmbedding(normalized_features, wires=range(n_qubits), normalize=True)
            
    return embedding


def iqp_embedding(n_qubits: int, n_features: int) -> Callable:
    """
    Create an IQP (Instantaneous Quantum Polynomial) embedding circuit.
    
    IQP embedding uses diagonal unitaries to encode classical data.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of classical features to embed
        
    Returns:
        function: IQP embedding function
    """
    def embedding(features: np.ndarray) -> None:
        """
        Apply IQP embedding to qubits.
        
        Args:
            features: Classical features to embed [n_features]
        """
        # Ensure features match expected size
        if len(features) < n_features:
            features = np.pad(features, (0, n_features - len(features)))
        elif len(features) > n_features:
            features = features[:n_features]
            
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            
        # Apply diagonal unitaries
        feature_idx = 0
        for i in range(n_qubits):
            # Single-qubit rotations
            qml.RZ(features[feature_idx % n_features], wires=i)
            feature_idx += 1
            
            # Two-qubit interactions
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(features[feature_idx % n_features], wires=j)
                qml.CNOT(wires=[i, j])
                feature_idx += 1
                
        # Apply Hadamard again to all qubits
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            
    return embedding


def fourier_embedding(n_qubits: int, n_features: int) -> Callable:
    """
    Create a Fourier embedding circuit.
    
    Fourier embedding uses multiple frequencies to encode classical data.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of classical features to embed
        
    Returns:
        function: Fourier embedding function
    """
    def embedding(features: np.ndarray) -> None:
        """
        Apply Fourier embedding to qubits.
        
        Args:
            features: Classical features to embed [n_features]
        """
        # Ensure features match expected size
        if len(features) < n_features:
            features = np.pad(features, (0, n_features - len(features)))
        elif len(features) > n_features:
            features = features[:n_features]
            
        # Apply Fourier embedding
        for i in range(n_qubits):
            # Apply Hadamard to create superposition
            qml.Hadamard(wires=i)
            
            # Apply different frequency rotations
            for j in range(min(n_features, 3)):  # Use up to 3 frequencies per qubit
                freq = j + 1
                qml.RZ(freq * features[j % n_features], wires=i)
                qml.RY(freq * features[(j+1) % n_features], wires=i)
                qml.RZ(freq * features[(j+2) % n_features], wires=i)
                
    return embedding


def hybrid_embedding(n_qubits: int, n_features: int, primary_type: str = "angle", 
                    secondary_type: str = "fourier", ratio: float = 0.5) -> Callable:
    """
    Create a hybrid embedding circuit that combines two embedding methods.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of classical features to embed
        primary_type: Primary embedding type
        secondary_type: Secondary embedding type
        ratio: Ratio of qubits to use for primary embedding (0.0-1.0)
        
    Returns:
        function: Hybrid embedding function
    """
    # Calculate number of qubits for each embedding
    n_primary = int(n_qubits * ratio)
    n_secondary = n_qubits - n_primary
    
    # Create embedding functions
    primary_embedding = create_data_embedding_circuit(primary_type, n_primary, n_features)
    secondary_embedding = create_data_embedding_circuit(secondary_type, n_secondary, n_features)
    
    def embedding(features: np.ndarray) -> None:
        """
        Apply hybrid embedding to qubits.
        
        Args:
            features: Classical features to embed [n_features]
        """
        # Apply primary embedding to first set of qubits
        with qml.tape.QuantumTape() as tape:
            primary_embedding(features)
        
        # Extract operations and apply with adjusted wires
        for op in tape.operations:
            # Create new operation with adjusted wires
            new_wires = [w for w in op.wires]
            qml.apply(op.name, new_wires, op.parameters)
            
        # Apply secondary embedding to second set of qubits
        with qml.tape.QuantumTape() as tape:
            secondary_embedding(features)
            
        # Extract operations and apply with adjusted wires
        for op in tape.operations:
            # Create new operation with adjusted wires
            new_wires = [w + n_primary for w in op.wires]
            qml.apply(op.name, new_wires, op.parameters)
            
    return embedding


def quantum_embedding_layer(n_qubits: int, n_features: int, embedding_type: str = "angle") -> Callable:
    """
    Create a quantum embedding layer that can be used in a quantum neural network.
    
    Args:
        n_qubits: Number of qubits
        n_features: Number of classical features to embed
        embedding_type: Type of embedding
        
    Returns:
        function: Quantum embedding layer function
    """
    # Create base embedding
    base_embedding = create_data_embedding_circuit(embedding_type, n_qubits, n_features)
    
    def embedding_layer(features: np.ndarray, params: np.ndarray) -> None:
        """
        Apply quantum embedding layer.
        
        Args:
            features: Classical features to embed [n_features]
            params: Trainable parameters for the embedding layer [n_params]
        """
        # Apply base embedding
        base_embedding(features)
        
        # Apply trainable transformations
        param_idx = 0
        for i in range(n_qubits):
            if param_idx + 3 <= len(params):
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=i)
                param_idx += 3
                
        # Apply entangling layer
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
            
        # Apply final rotations
        for i in range(n_qubits):
            if param_idx + 3 <= len(params):
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=i)
                param_idx += 3
                
    return embedding_layer