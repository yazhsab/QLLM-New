"""
Variational quantum circuits for Quantum Large Language Models.

This module provides parameterized quantum circuits that can be trained
to perform specific tasks in quantum machine learning.
"""

import pennylane as qml
import numpy as np
from typing import List, Callable, Optional, Union, Tuple, Dict


def create_variational_circuit(circuit_type: str, n_qubits: int, n_layers: int) -> Callable:
    """
    Create a variational quantum circuit based on the specified type.
    
    Args:
        circuit_type: Type of variational circuit ('basic', 'strongly_entangling', 'hardware_efficient', 'custom')
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        
    Returns:
        function: Variational circuit function
    """
    if circuit_type == "basic":
        return basic_variational_circuit(n_qubits, n_layers)
    elif circuit_type == "strongly_entangling":
        return strongly_entangling_circuit(n_qubits, n_layers)
    elif circuit_type == "hardware_efficient":
        return hardware_efficient_circuit(n_qubits, n_layers)
    elif circuit_type == "custom":
        return custom_variational_circuit(n_qubits, n_layers)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def basic_variational_circuit(n_qubits: int, n_layers: int) -> Callable:
    """
    Create a basic variational circuit with alternating rotation and entanglement layers.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        
    Returns:
        function: Basic variational circuit function
    """
    def circuit(params: np.ndarray) -> None:
        """
        Apply basic variational circuit.
        
        Args:
            params: Circuit parameters [n_layers * n_qubits * 3]
        """
        # Check parameter shape
        expected_params = n_layers * n_qubits * 3
        if len(params) < expected_params:
            # Pad with zeros if needed
            params = np.pad(params, (0, expected_params - len(params)))
        elif len(params) > expected_params:
            # Truncate if too many parameters
            params = params[:expected_params]
            
        # Reshape parameters for easier indexing
        params = params.reshape(n_layers, n_qubits, 3)
        
        # Apply variational layers
        for layer in range(n_layers):
            # Rotation layer
            for qubit in range(n_qubits):
                qml.Rot(params[layer, qubit, 0], 
                       params[layer, qubit, 1], 
                       params[layer, qubit, 2], 
                       wires=qubit)
            
            # Entanglement layer (except after the last rotation layer)
            if layer < n_layers - 1:
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                # Connect the last qubit to the first one for circular entanglement
                qml.CNOT(wires=[n_qubits - 1, 0])
                
    return circuit


def strongly_entangling_circuit(n_qubits: int, n_layers: int) -> Callable:
    """
    Create a strongly entangling circuit with long-range entanglement.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        
    Returns:
        function: Strongly entangling circuit function
    """
    def circuit(params: np.ndarray) -> None:
        """
        Apply strongly entangling circuit.
        
        Args:
            params: Circuit parameters [n_layers * n_qubits * 3]
        """
        # Check parameter shape
        expected_params = n_layers * n_qubits * 3
        if len(params) < expected_params:
            params = np.pad(params, (0, expected_params - len(params)))
        elif len(params) > expected_params:
            params = params[:expected_params]
            
        # Reshape parameters for easier indexing
        params = params.reshape(n_layers, n_qubits, 3)
        
        # Apply variational layers
        for layer in range(n_layers):
            # Rotation layer
            for qubit in range(n_qubits):
                qml.Rot(params[layer, qubit, 0], 
                       params[layer, qubit, 1], 
                       params[layer, qubit, 2], 
                       wires=qubit)
            
            # Entanglement layer with long-range connections
            for qubit in range(n_qubits):
                # Connect each qubit with another qubit at increasing distance
                target = (qubit + layer + 1) % n_qubits
                qml.CNOT(wires=[qubit, target])
                
    return circuit


def hardware_efficient_circuit(n_qubits: int, n_layers: int) -> Callable:
    """
    Create a hardware-efficient circuit optimized for NISQ devices.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        
    Returns:
        function: Hardware-efficient circuit function
    """
    def circuit(params: np.ndarray) -> None:
        """
        Apply hardware-efficient circuit.
        
        Args:
            params: Circuit parameters [n_layers * n_qubits * 2]
        """
        # Check parameter shape
        expected_params = n_layers * n_qubits * 2
        if len(params) < expected_params:
            params = np.pad(params, (0, expected_params - len(params)))
        elif len(params) > expected_params:
            params = params[:expected_params]
            
        # Reshape parameters for easier indexing
        params = params.reshape(n_layers, n_qubits, 2)
        
        # Apply variational layers
        for layer in range(n_layers):
            # Single-qubit rotation layer (RY, RZ only for hardware efficiency)
            for qubit in range(n_qubits):
                qml.RY(params[layer, qubit, 0], wires=qubit)
                qml.RZ(params[layer, qubit, 1], wires=qubit)
            
            # Entanglement layer with nearest-neighbor connectivity
            for qubit in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[qubit, qubit + 1])
                
            # Staggered entanglement in next layer
            if layer < n_layers - 1:
                for qubit in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[qubit, qubit + 1])
                
    return circuit


def custom_variational_circuit(n_qubits: int, n_layers: int) -> Callable:
    """
    Create a custom variational circuit optimized for NLP tasks.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        
    Returns:
        function: Custom variational circuit function
    """
    def circuit(params: np.ndarray) -> None:
        """
        Apply custom variational circuit.
        
        Args:
            params: Circuit parameters [n_layers * (n_qubits * 3 + n_qubits - 1)]
        """
        # Calculate expected parameter count
        params_per_layer = n_qubits * 3 + n_qubits - 1  # 3 for rotations, n_qubits-1 for entanglement
        expected_params = n_layers * params_per_layer
        
        # Check parameter shape
        if len(params) < expected_params:
            params = np.pad(params, (0, expected_params - len(params)))
        elif len(params) > expected_params:
            params = params[:expected_params]
        
        param_idx = 0
        
        # Apply variational layers
        for layer in range(n_layers):
            # First rotation layer
            for qubit in range(n_qubits):
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=qubit)
                param_idx += 3
            
            # Entanglement layer with parameterized strength
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
                qml.RZ(params[param_idx], wires=qubit + 1)
                qml.CNOT(wires=[qubit, qubit + 1])
                param_idx += 1
                
            # Non-linear transformation (approximated in quantum)
            for qubit in range(n_qubits):
                qml.Hadamard(wires=qubit)
                qml.T(wires=qubit)
                qml.Hadamard(wires=qubit)
                
    return circuit


def quantum_transformer_circuit(n_qubits: int, n_heads: int = 2) -> Callable:
    """
    Create a quantum circuit that implements a transformer-like architecture.
    
    Args:
        n_qubits: Number of qubits
        n_heads: Number of attention heads
        
    Returns:
        function: Quantum transformer circuit function
    """
    def circuit(params: np.ndarray) -> None:
        """
        Apply quantum transformer circuit.
        
        Args:
            params: Circuit parameters
        """
        # Calculate qubits per head
        qubits_per_head = n_qubits // n_heads
        
        param_idx = 0
        
        # Multi-head attention
        for head in range(n_heads):
            head_start = head * qubits_per_head
            head_end = head_start + qubits_per_head
            head_wires = range(head_start, head_end)
            
            # Query transformation
            for wire in head_wires:
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
                param_idx += 3
            
            # Key-Query interaction (attention mechanism)
            for i, wire_i in enumerate(head_wires):
                for j, wire_j in enumerate(head_wires):
                    if i < j:  # Upper triangular to avoid redundancy
                        qml.CNOT(wires=[wire_i, wire_j])
                        qml.RZ(params[param_idx], wires=wire_j)
                        qml.CNOT(wires=[wire_i, wire_j])
                        param_idx += 1
            
            # Value transformation
            for wire in head_wires:
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
                param_idx += 3
        
        # Feed-forward network
        for wire in range(n_qubits):
            qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
            param_idx += 3
            
        # Final layer normalization (approximated)
        for wire in range(n_qubits):
            qml.RY(params[param_idx], wires=wire)
            param_idx += 1
            
    return circuit


def quantum_rnn_circuit(n_qubits: int, n_layers: int) -> Callable:
    """
    Create a quantum circuit that implements an RNN-like architecture.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of recurrent layers
        
    Returns:
        function: Quantum RNN circuit function
    """
    def circuit(params: np.ndarray) -> None:
        """
        Apply quantum RNN circuit.
        
        Args:
            params: Circuit parameters
        """
        # Split qubits into hidden and visible
        n_hidden = n_qubits // 2
        hidden_wires = range(n_hidden)
        visible_wires = range(n_hidden, n_qubits)
        
        param_idx = 0
        
        # Initialize hidden state
        for wire in hidden_wires:
            qml.Hadamard(wires=wire)
        
        # Apply recurrent layers
        for layer in range(n_layers):
            # Hidden-visible interaction
            for h_wire in hidden_wires:
                for v_wire in visible_wires:
                    qml.CNOT(wires=[v_wire, h_wire])
            
            # Hidden state update
            for wire in hidden_wires:
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
                param_idx += 3
            
            # Hidden-hidden interaction (recurrent connection)
            for i in range(len(hidden_wires) - 1):
                qml.CNOT(wires=[hidden_wires[i], hidden_wires[i+1]])
                qml.RZ(params[param_idx], wires=hidden_wires[i+1])
                qml.CNOT(wires=[hidden_wires[i], hidden_wires[i+1]])
                param_idx += 1
            
            # Hidden-visible update
            for h_wire in hidden_wires:
                for v_wire in visible_wires:
                    qml.CNOT(wires=[h_wire, v_wire])
            
            # Visible state update
            for wire in visible_wires:
                qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
                param_idx += 3
                
    return circuit


def quantum_memory_circuit(n_qubits: int, memory_size: int) -> Callable:
    """
    Create a quantum circuit that implements a memory mechanism.
    
    Args:
        n_qubits: Number of qubits
        memory_size: Size of quantum memory (in qubits)
        
    Returns:
        function: Quantum memory circuit function
    """
    def circuit(params: np.ndarray, memory_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply quantum memory circuit.
        
        Args:
            params: Circuit parameters
            memory_state: Previous memory state (if any)
            
        Returns:
            np.ndarray: Updated memory state
        """
        # Split qubits into memory and processing
        memory_wires = range(memory_size)
        proc_wires = range(memory_size, n_qubits)
        
        param_idx = 0
        
        # Initialize or load memory state
        if memory_state is not None:
            # Apply memory state preparation
            for i, wire in enumerate(memory_wires):
                if i < len(memory_state):
                    # Convert memory state value to rotation angle
                    angle = np.arccos(np.clip(memory_state[i], -1.0, 1.0))
                    qml.RY(angle, wires=wire)
        
        # Memory-processing interaction
        for m_wire in memory_wires:
            for p_wire in proc_wires:
                qml.CNOT(wires=[m_wire, p_wire])
                qml.RZ(params[param_idx], wires=p_wire)
                qml.CNOT(wires=[m_wire, p_wire])
                param_idx += 1
        
        # Processing update
        for wire in proc_wires:
            qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
            param_idx += 3
        
        # Processing-memory update
        for p_wire in proc_wires:
            for m_wire in memory_wires:
                qml.CNOT(wires=[p_wire, m_wire])
                qml.RZ(params[param_idx], wires=m_wire)
                qml.CNOT(wires=[p_wire, m_wire])
                param_idx += 1
        
        # Memory update
        for wire in memory_wires:
            qml.Rot(params[param_idx], params[param_idx+1], params[param_idx+2], wires=wire)
            param_idx += 3
        
        # Measure memory state
        return [qml.expval(qml.PauliZ(wire)) for wire in memory_wires]
                
    return circuit