"""
Test script for running the Qynthra example files.
This script runs the example files and checks if they execute without errors.
"""

import os
import sys
import subprocess
import argparse

def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def run_example(example_path):
    """Run an example file and check if it executes without errors."""
    print(f"\nRunning example: {example_path}")
    
    try:
        # Run the example file
        result = subprocess.run(
            [sys.executable, example_path],
            capture_output=True,
            text=True,
            timeout=60  # Timeout after 60 seconds
        )
        
        # Check if the example executed without errors
        if result.returncode == 0:
            print(f"✅ Example executed successfully")
            return True
        else:
            print(f"❌ Example failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Example timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running example: {e}")
        return False

def main():
    """Run all examples."""
    parser = argparse.ArgumentParser(description="Test Qynthra examples")
    parser.add_argument("--example", type=str, help="Run a specific example file")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run all examples
    if not args.example and not args.all:
        args.all = True
    
    print_header("Qynthra Example Test Suite")
    
    # Get all example files
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    example_files = [
        os.path.join(examples_dir, f)
        for f in os.listdir(examples_dir)
        if f.endswith(".py")
    ]
    
    # Sort example files
    example_files.sort()
    
    # Print available examples
    print("Available examples:")
    for i, example in enumerate(example_files):
        print(f"  {i+1}. {os.path.basename(example)}")
    
    # Run examples
    results = {}
    
    if args.all:
        # Run all examples
        for example in example_files:
            example_name = os.path.basename(example)
            results[example_name] = run_example(example)
    elif args.example:
        # Run a specific example
        example_path = os.path.join(examples_dir, args.example)
        if os.path.isfile(example_path):
            results[args.example] = run_example(example_path)
        else:
            print(f"❌ Example file not found: {args.example}")
    
    # Summary
    print_header("Test Summary")
    
    for example_name, result in results.items():
        if result:
            print(f"✅ {example_name}: PASSED")
        else:
            print(f"❌ {example_name}: FAILED")
    
    print("\nTo run specific examples:")
    print("  python test_examples.py --example basic_qllm_example.py")
    print("  python test_examples.py --all")

if __name__ == "__main__":
    main()