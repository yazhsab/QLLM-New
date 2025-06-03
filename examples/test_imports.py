"""
Simple test script to verify that the Quantum LLM components can be imported correctly.
"""

def test_imports():
    """Test importing all key components."""
    print("Testing imports...")
    try:
        # Import core classes
        print("Importing core classes...")
        from quantum_llm import (
            QLLMBase,
            QLLMAdvanced,
            QLLMWithKVCache,
            QLLMTokenizer,
            QLLMDataset
        )
        print("Core classes imported successfully!")
        
        # Import quantum neural network models
        print("Importing quantum neural network models...")
        from quantum_llm import (
            QTransformer,
            QRNN,
            QMemoryNetwork
        )
        print("Quantum neural network models imported successfully!")
        
        # Import quantum circuit utilities
        print("Importing quantum circuit utilities...")
        from quantum_llm import (
            create_data_embedding_circuit,
            angle_embedding,
            amplitude_embedding,
            iqp_embedding,
            fourier_embedding,
            hybrid_embedding,
            quantum_embedding_layer
        )
        print("Quantum circuit utilities imported successfully!")
        
        print("Importing variational circuits...")
        from quantum_llm import (
            create_variational_circuit,
            basic_variational_circuit,
            strongly_entangling_circuit,
            hardware_efficient_circuit,
            custom_variational_circuit,
            quantum_transformer_circuit,
            quantum_rnn_circuit,
            quantum_memory_circuit
        )
        print("Variational circuits imported successfully!")
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_instantiation():
    """Test creating instances of key classes."""
    print("\nTesting instantiation...")
    try:
        print("Importing required classes...")
        from quantum_llm import QLLMTokenizer, QLLMBase, QTransformer
        import pennylane as qml
        
        # Create tokenizer
        print("Creating tokenizer...")
        tokenizer = QLLMTokenizer(vocab_size=100, max_length=32)
        print("✅ Created tokenizer")
        
        # Create basic model
        print("Creating basic model...")
        model = QLLMBase(
            vocab_size=100,
            embedding_dim=8,
            n_qubits=2,
            n_layers=1,
            max_seq_length=32,
            device="default.qubit",
            embedding_type="angle",
            circuit_type="quantum_transformer"
        )
        print("✅ Created basic model")
        
        # Create transformer
        print("Creating quantum transformer...")
        transformer = QTransformer(
            n_qubits=2,
            n_layers=1,
            n_heads=1,
            embedding_dim=8,
            embedding_type="angle",
            device="default.qubit"
        )
        print("✅ Created quantum transformer")
        
        return True
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Quantum LLM implementation...")
    import_success = test_imports()
    
    if import_success:
        print("\nTesting Quantum LLM instantiation...")
        instantiation_success = test_instantiation()
        
    if import_success and instantiation_success:
        print("\n🎉 All tests passed! The Quantum LLM implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")