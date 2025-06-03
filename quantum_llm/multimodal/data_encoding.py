"""
Multimodal data encoding circuits for Quantum LLM.

This module extends the basic data encoding circuits to support different input modalities
(text, images, audio) and provides methods for encoding multimodal data into quantum states.
"""

import pennylane as qml
import numpy as np
import torch
from typing import Dict, List, Callable, Optional, Union, Tuple

from quantum_llm.basic_quantum_circuits.data_encoding import (
    create_data_embedding_circuit,
    angle_embedding,
    amplitude_embedding,
    iqp_embedding,
    fourier_embedding
)


def create_multimodal_embedding_circuit(
    modality_types: Dict[str, str],
    n_qubits: int,
    qubit_allocations: Dict[str, int],
    feature_dims: Dict[str, int]
) -> Callable:
    """
    Create a multimodal embedding circuit that allocates different qubits to different modalities.
    
    Args:
        modality_types: Dictionary mapping modality names to their embedding types
        n_qubits: Total number of qubits
        qubit_allocations: Dictionary mapping modality names to their qubit allocations
        feature_dims: Dictionary mapping modality names to their feature dimensions
        
    Returns:
        function: Multimodal embedding function
    """
    # Validate qubit allocations
    total_allocated = sum(qubit_allocations.values())
    if total_allocated > n_qubits:
        raise ValueError(f"Total allocated qubits ({total_allocated}) exceeds available qubits ({n_qubits})")
    
    # Create embedding circuits for each modality
    embedding_circuits = {}
    qubit_offsets = {}
    
    offset = 0
    for modality, n_modality_qubits in qubit_allocations.items():
        if modality in modality_types:
            embedding_type = modality_types[modality]
            n_features = feature_dims.get(modality, n_modality_qubits)
            
            embedding_circuits[modality] = create_data_embedding_circuit(
                embedding_type, n_modality_qubits, n_features
            )
            qubit_offsets[modality] = offset
            offset += n_modality_qubits
    
    def embedding(features: Dict[str, np.ndarray]) -> None:
        """
        Apply multimodal embedding to qubits.
        
        Args:
            features: Dictionary mapping modality names to their features
        """
        for modality, modality_features in features.items():
            if modality in embedding_circuits:
                # Get the embedding circuit and offset for this modality
                circuit = embedding_circuits[modality]
                offset = qubit_offsets[modality]
                
                # Apply the embedding circuit with adjusted wires
                with qml.tape.QuantumTape() as tape:
                    circuit(modality_features)
                
                # Extract operations and apply with adjusted wires
                for op in tape.operations:
                    # Create new operation with adjusted wires
                    new_wires = [w + offset for w in op.wires]
                    qml.apply(op.name, new_wires, op.parameters)
    
    return embedding


def image_to_quantum_encoding(
    n_qubits: int,
    embedding_type: str = "amplitude"
) -> Callable:
    """
    Create a specialized embedding circuit for image data.
    
    Args:
        n_qubits: Number of qubits
        embedding_type: Type of embedding
        
    Returns:
        function: Image embedding function
    """
    def embedding(image_features: np.ndarray) -> None:
        """
        Apply image embedding to qubits.
        
        Args:
            image_features: Image features to embed
        """
        # For amplitude encoding, we need to reshape to match 2^n_qubits
        if embedding_type == "amplitude":
            required_size = 2**n_qubits
            
            # Reshape or pad/truncate as needed
            if len(image_features) != required_size:
                # Pad or truncate
                if len(image_features) < required_size:
                    image_features = np.pad(image_features, (0, required_size - len(image_features)))
                else:
                    image_features = image_features[:required_size]
            
            # Normalize
            image_features = image_features / np.linalg.norm(image_features)
            
            # Apply amplitude embedding
            qml.AmplitudeEmbedding(image_features, wires=range(n_qubits), normalize=True)
        
        # For angle encoding, use spatial patterns
        elif embedding_type == "angle":
            # Ensure we have enough features
            if len(image_features) < n_qubits * 2:
                image_features = np.pad(image_features, (0, n_qubits * 2 - len(image_features)))
            
            # Apply angle embedding with spatial patterns
            for i in range(n_qubits):
                qml.RX(image_features[i % len(image_features)], wires=i)
                qml.RY(image_features[(i + n_qubits) % len(image_features)], wires=i)
        
        # For Fourier encoding, use frequency components
        elif embedding_type == "fourier":
            # Apply Fourier embedding
            for i in range(n_qubits):
                # Apply Hadamard to create superposition
                qml.Hadamard(wires=i)
                
                # Apply different frequency rotations based on image features
                for j in range(min(len(image_features), 3)):
                    freq = j + 1
                    feature_idx = (i * 3 + j) % len(image_features)
                    qml.RZ(freq * image_features[feature_idx], wires=i)
    
    return embedding


def audio_to_quantum_encoding(
    n_qubits: int,
    embedding_type: str = "fourier"
) -> Callable:
    """
    Create a specialized embedding circuit for audio data.
    
    Args:
        n_qubits: Number of qubits
        embedding_type: Type of embedding
        
    Returns:
        function: Audio embedding function
    """
    def embedding(audio_features: np.ndarray) -> None:
        """
        Apply audio embedding to qubits.
        
        Args:
            audio_features: Audio features to embed
        """
        # For Fourier encoding (good for audio), use frequency components
        if embedding_type == "fourier":
            # Apply Fourier embedding
            for i in range(n_qubits):
                # Apply Hadamard to create superposition
                qml.Hadamard(wires=i)
                
                # Apply different frequency rotations based on audio features
                for j in range(min(len(audio_features), 3)):
                    freq = j + 1
                    feature_idx = (i * 3 + j) % len(audio_features)
                    qml.RZ(freq * audio_features[feature_idx], wires=i)
                    qml.RY(freq * audio_features[(feature_idx + 1) % len(audio_features)], wires=i)
        
        # For amplitude encoding
        elif embedding_type == "amplitude":
            required_size = 2**n_qubits
            
            # Reshape or pad/truncate as needed
            if len(audio_features) != required_size:
                if len(audio_features) < required_size:
                    audio_features = np.pad(audio_features, (0, required_size - len(audio_features)))
                else:
                    audio_features = audio_features[:required_size]
            
            # Normalize
            audio_features = audio_features / np.linalg.norm(audio_features)
            
            # Apply amplitude embedding
            qml.AmplitudeEmbedding(audio_features, wires=range(n_qubits), normalize=True)
    
    return embedding


def create_modality_specific_embedding(
    modality: str,
    n_qubits: int,
    embedding_type: Optional[str] = None
) -> Callable:
    """
    Create a modality-specific embedding circuit.
    
    Args:
        modality: Type of modality ('text', 'image', 'audio')
        n_qubits: Number of qubits
        embedding_type: Type of embedding (if None, uses default for modality)
        
    Returns:
        function: Modality-specific embedding function
    """
    if modality == "text":
        # Default to angle embedding for text
        embedding_type = embedding_type or "angle"
        return create_data_embedding_circuit(embedding_type, n_qubits, n_qubits)
    
    elif modality == "image":
        # Default to amplitude embedding for images
        embedding_type = embedding_type or "amplitude"
        return image_to_quantum_encoding(n_qubits, embedding_type)
    
    elif modality == "audio":
        # Default to Fourier embedding for audio
        embedding_type = embedding_type or "fourier"
        return audio_to_quantum_encoding(n_qubits, embedding_type)
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


def quantum_multimodal_fusion(
    n_qubits: int,
    n_modalities: int
) -> Callable:
    """
    Create a quantum circuit for fusing multiple modalities.
    
    Args:
        n_qubits: Number of qubits
        n_modalities: Number of modalities to fuse
        
    Returns:
        function: Quantum fusion function
    """
    # Qubits per modality
    qubits_per_modality = n_qubits // n_modalities
    
    def fusion_circuit(params: np.ndarray) -> None:
        """
        Apply quantum fusion circuit.
        
        Args:
            params: Circuit parameters
        """
        param_idx = 0
        
        # Apply initial rotations to each modality section
        for m in range(n_modalities):
            start_qubit = m * qubits_per_modality
            end_qubit = start_qubit + qubits_per_modality
            
            for i in range(start_qubit, end_qubit):
                qml.Rot(
                    params[param_idx % len(params)],
                    params[(param_idx + 1) % len(params)],
                    params[(param_idx + 2) % len(params)],
                    wires=i
                )
                param_idx += 3
        
        # Apply cross-modality entanglement
        for m1 in range(n_modalities):
            for m2 in range(m1 + 1, n_modalities):
                # Connect qubits across modalities
                q1 = m1 * qubits_per_modality
                q2 = m2 * qubits_per_modality
                
                # Apply CNOT gates between modalities
                for i in range(qubits_per_modality):
                    if q1 + i < n_qubits and q2 + i < n_qubits:
                        qml.CNOT(wires=[q1 + i, q2 + i])
                        qml.RZ(params[param_idx % len(params)], wires=q2 + i)
                        qml.CNOT(wires=[q1 + i, q2 + i])
                        param_idx += 1
        
        # Apply final rotations to all qubits
        for i in range(n_qubits):
            qml.Rot(
                params[param_idx % len(params)],
                params[(param_idx + 1) % len(params)],
                params[(param_idx + 2) % len(params)],
                wires=i
            )
            param_idx += 3
    
    return fusion_circuit