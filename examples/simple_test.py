"""
Simple test script to verify Python is working.
"""

print("Hello from simple_test.py!")

# Try importing one module
try:
    import pennylane as qml
    print("Successfully imported PennyLane!")
except ImportError as e:
    print(f"Failed to import PennyLane: {e}")

# Try importing from our package
try:
    from quantum_llm.basic_quantum_circuits import data_encoding
    print("Successfully imported data_encoding module!")
except ImportError as e:
    print(f"Failed to import data_encoding: {e}")
    import traceback
    traceback.print_exc()