import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

def steane_code_circuit() -> QuantumCircuit:
    """
    Constructs the Steane code circuit for encoding a single logical qubit on qubit index 3.

    Returns:
        QuantumCircuit: A quantum circuit that encodes a single logical qubit using the Steane code.
    """
    # Create a quantum circuit with 7 qubits for the Steane code
    steane_circuit = QuantumCircuit(7)

    # Encoding steps for the Steane code
    # Step 1: Create superposition states
    steane_circuit.h(0)
    steane_circuit.h(1)
    steane_circuit.h(2)

    # Step 2: Entangle qubits to form the logical |0_L> and |1_L> states
    steane_circuit.cx(3, 5)
    steane_circuit.cx(3, 6)
    steane_circuit.cx(0, 3)
    steane_circuit.cx(0, 4)
    steane_circuit.cx(0, 5)
    steane_circuit.cx(1, 4)
    steane_circuit.cx(1, 5)
    steane_circuit.cx(1, 6)
    steane_circuit.cx(2, 3)
    steane_circuit.cx(2, 4)
    steane_circuit.cx(2, 6)

    return steane_circuit

def steane_code_encoding_circuit(initial_state: QuantumCircuit) -> QuantumCircuit:
    """
    Constructs the Steane code encoding circuit for a single logical qubit.

    Args:
        initial_state (QuantumCircuit): A quantum circuit with one qubit representing the initial state to be encoded.

    Returns:
        QuantumCircuit: A quantum circuit that encodes the initial state into the Steane code.
    """
    if initial_state.num_qubits != 7:
        raise ValueError("Initial state circuit must have exactly seven qubits.")

    steane_circuit = steane_code_circuit()
    # steane_circuit.compose(initial_state, qubits=[0], inplace=True)
    # steane_circuit += steane_code_circuit()

    encoded_state = initial_state.compose(steane_circuit, inplace=False)

    return encoded_state

# initial_state = QuantumCircuit(7)
# # initial_state.h(3)  # Example initial state |+>
# # initial_state.t(0)  # Apply T-gate to the initial state

# steane_encoded_ht_state = steane_code_encoding_circuit(initial_state)

# for i in range(7):
#     steane_encoded_ht_state.h(i)

# state = Statevector(steane_encoded_ht_state)
# encoded = np.array(state.data).round(5)

# print("Encoded state dimension:", len(encoded))
# print("Encoded state vector:", encoded)