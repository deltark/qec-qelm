import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate, TGate
import pickle

perfect_hgate = HGate(label='h1')
perfect_cnot = CXGate(label='cnot1')

def steane_code_circuit() -> QuantumCircuit:
    """
    Constructs the Steane code circuit for encoding a single logical qubit on qubit index 6.

    Returns:
        QuantumCircuit: A quantum circuit that encodes a single logical qubit using the Steane code.
    """
    # Create a quantum circuit with 7 qubits for the Steane code
    steane_circuit = QuantumCircuit(7)

    # Encoding steps for the Steane code
    # Step 1: Create superposition states
    steane_circuit.append(perfect_hgate, [0])
    steane_circuit.append(perfect_hgate, [1])
    steane_circuit.append(perfect_hgate, [2])

    # Step 2: Entangle qubits to form the logical |0_L> and |1_L> states
    steane_circuit.append(perfect_cnot, (6, 5))
    steane_circuit.append(perfect_cnot, (6, 4))
    steane_circuit.append(perfect_cnot, (0, 6))
    steane_circuit.append(perfect_cnot, (0, 5))
    steane_circuit.append(perfect_cnot, (0, 3))
    steane_circuit.append(perfect_cnot, (1, 6))
    steane_circuit.append(perfect_cnot, (1, 4))
    steane_circuit.append(perfect_cnot, (1, 3))
    steane_circuit.append(perfect_cnot, (2, 5))
    steane_circuit.append(perfect_cnot, (2, 4))
    steane_circuit.append(perfect_cnot, (2, 3))

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

filename = f'results/circuit_runs/error_states_int.pkl'
with open(filename, 'rb') as f:
    error_states_int = pickle.load(f)

def t_gate_teleportation(qc, ancilla, logical_qubit_register, creg):
     # T-gate teleportation using ancilla qubits
            qc.reset(ancilla)
            qc.append(perfect_hgate, [ancilla[6]])
            qc.t(ancilla[6])
            qc.compose(steane_code_circuit(), qubits=ancilla, inplace=True)
            for i in range(7):
                qc.append(perfect_cnot, (ancilla[i], logical_qubit_register[i]))
            qc.measure(logical_qubit_register, creg)
            # Conditional SX-gate based on measurement outcome
            for error in error_states_int:
                with qc.if_test((creg, error)):
                    for i in range(7):
                        qc.x(ancilla[i])
                        for _ in range(3): # Transversal S-gate is 3 layers of S
                            qc.s(ancilla[i])
            # Swap ancilla back to the logical qubit
            for i in range(7):
                qc.swap(ancilla[i], logical_qubit_register[i])

# qreg = QuantumRegister(7)
# ancilla = QuantumRegister(7)
# creg = ClassicalRegister(7)

# qc = QuantumCircuit(qreg, ancilla, creg)

# qc.h(qreg[6])

# qc.compose(steane_code_circuit(), qreg, inplace=True)
# t_gate_teleportation(qc, ancilla, qreg, creg)
# qc.compose(steane_code_circuit().inverse(), qreg, inplace=True)
# qc.save_statevector()

# initial_state = QuantumCircuit(7)
# # # initial_state.append(perfect_hgate, 3)  # Example initial state |+>
# # # initial_state.t(0)  # Apply T-gate to the initial state

# qc = steane_code_encoding_circuit(initial_state)
# qc.measure_all()

# # for i in range(7):
# #     steane_encoded_ht_state.append(perfect_hgate, i)

# # state = Statevector(steane_encoded_ht_state)
# # encoded = np.array(state.data).round(5)

# # print("Encoded state dimension:", len(encoded))
# # print("Encoded state vector:", encoded)

# # Transpile for simulator
# simulator = Aer.get_backend('aer_simulator')
# simulator = AerSimulator(method = 'statevector')
# # qc = transpile(qc, simulator)
# # # Run and get unitary
# # result = simulator.run(qc, shots=1000).result()
# # counts = result.get_counts(qc)
# # print("Measurement results:", counts)

# result = simulator.run(qc).result()
# state = result.get_statevector()
# state = partial_trace(state, [0,1,2,3,4,5, 7,8,9,10,11,12,13])
# print(state)

# qc = QuantumCircuit(1)
# qc.h(0)
# qc.t(0)
# qc.save_statevector()

# result = simulator.run(qc).result()
# state = result.get_statevector()
# print(state)