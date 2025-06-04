"""
Functional test script for the Qynthra project.
This script tests the actual functionality of the project components.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any

# Add the current directory to the Python path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def test_basic_functionality():
    """Test basic functionality of the Qynthra package."""
    print_header("Testing Basic Functionality")
    
    try:
        # Import the package
        import Qynthra
        print("✅ Successfully imported Qynthra package")
        
        # Test basic components
        from Qynthra import QLLMBase, QLLMTokenizer
        print("✅ Successfully imported core components")
        
        # Create a simple tokenizer
        tokenizer = QLLMTokenizer(vocab_size=100, max_length=32)
        print("✅ Created tokenizer instance")
        
        # Train tokenizer on a small text
        sample_text = ["Quantum computing is the future of AI."]
        tokenizer.train_from_texts(sample_text)
        print("✅ Trained tokenizer on sample text")
        
        # Create a small model
        model = QLLMBase(
            vocab_size=100,
            embedding_dim=8,
            n_qubits=2,
            n_layers=1,
            max_seq_length=16,
            device="default.qubit",
            embedding_type="angle",
            circuit_type="basic"
        )
        print("✅ Created model instance")
        
        # Test model parameters
        print(f"  Model parameters: {model.n_params}")
        
        return True
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False

def test_advanced_functionality():
    """Test advanced functionality of the Qynthra package."""
    print_header("Testing Advanced Functionality")
    
    try:
        # Import advanced components
        from Qynthra import QLLMAdvanced
        print("✅ Successfully imported advanced components")
        
        # Create an advanced model
        model = QLLMAdvanced(
            vocab_size=100,
            embedding_dim=8,
            n_qubits=2,
            n_layers=1,
            n_heads=2,
            max_seq_length=16,
            device="default.qubit",
            embedding_type="angle",
            circuit_type="basic"
        )
        print("✅ Created advanced model instance")
        
        # Test model parameters
        print(f"  Model parameters: {model.n_params}")
        
        return True
    except Exception as e:
        print(f"❌ Error in advanced functionality test: {e}")
        return False

def test_multimodal_functionality():
    """Test multimodal functionality of the Qynthra package."""
    print_header("Testing Multimodal Functionality")
    
    try:
        # Import multimodal components
        from Qynthra import MultimodalQLLM
        from Qynthra.multimodal.encoders import TextEncoder, ImageEncoder, AudioEncoder
        print("✅ Successfully imported multimodal components")
        
        # Create encoders
        text_encoder = TextEncoder(
            vocab_size=100,
            embedding_dim=8,
            max_seq_length=16
        )
        print("✅ Created text encoder")
        
        image_encoder = ImageEncoder(
            input_channels=3,
            embedding_dim=8
        )
        print("✅ Created image encoder")
        
        audio_encoder = AudioEncoder(
            input_channels=1,
            embedding_dim=8
        )
        print("✅ Created audio encoder")
        
        # Create multimodal model
        model = MultimodalQLLM(
            vocab_size=100,
            embedding_dim=8,
            n_qubits=4,
            n_layers=1,
            n_heads=2,
            device="default.qubit",
            supported_modalities=["text", "image", "audio"],
            qubit_allocations={
                "text": 2,
                "image": 1,
                "audio": 1
            }
        )
        print("✅ Created multimodal model instance")
        
        return True
    except Exception as e:
        print(f"❌ Error in multimodal functionality test: {e}")
        return False

def test_client_functionality():
    """Test client functionality."""
    print_header("Testing Client Functionality")
    
    try:
        # Import Python client
        from Qynthra.client.python.qynthra_client import QynthraClient
        print("✅ Successfully imported Python client")
        
        # Create client instance
        client = QynthraClient(api_url="http://localhost:8000")
        print("✅ Created client instance")
        
        # Test client methods (without making actual API calls)
        methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]
        print(f"  Available client methods: {', '.join(methods)}")
        
        return True
    except Exception as e:
        print(f"❌ Error in client functionality test: {e}")
        return False

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Functional test for Qynthra")
    parser.add_argument("--basic", action="store_true", help="Run basic functionality tests")
    parser.add_argument("--advanced", action="store_true", help="Run advanced functionality tests")
    parser.add_argument("--multimodal", action="store_true", help="Run multimodal functionality tests")
    parser.add_argument("--client", action="store_true", help="Run client functionality tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run all tests
    if not any(vars(args).values()):
        args.all = True
    
    print_header("Qynthra Functional Test Suite")
    
    results = {}
    
    if args.all or args.basic:
        results["basic"] = test_basic_functionality()
    
    if args.all or args.advanced:
        results["advanced"] = test_advanced_functionality()
    
    if args.all or args.multimodal:
        results["multimodal"] = test_multimodal_functionality()
    
    if args.all or args.client:
        results["client"] = test_client_functionality()
    
    # Summary
    print_header("Test Summary")
    
    for test_name, result in results.items():
        if result:
            print(f"✅ {test_name.capitalize()} functionality: PASSED")
        else:
            print(f"❌ {test_name.capitalize()} functionality: FAILED")
    
    print("\nTo run specific tests:")
    print("  python functional_test.py --basic")
    print("  python functional_test.py --advanced")
    print("  python functional_test.py --multimodal")
    print("  python functional_test.py --client")
    print("  python functional_test.py --all")

if __name__ == "__main__":
    main()