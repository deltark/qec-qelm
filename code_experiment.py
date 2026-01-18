import numpy as np
import pickle
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from steane_code import steane_code_encoding_circuit, steane_code_circuit
from random_cliff_t_circuit_ergodicity_numpy import beta_k
import scipy.sparse as sp

Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))

filename = 'results/fourier_analysis/fourier_expressivity_from_sequence_n3.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
p_T_values = data['p_T_values']
max_expressivity_unitaries_per_pT = data['max_expressivity_unitaries_per_pT']
sequences_per_pT = data['max_expressivity_sequences_per_pT']

nqubits = 3
nphysical_qubits = 7 * nqubits  # Each logical qubit is encoded into 7 physical qubits using the Steane code

x = 0.76 # Classical data to encode

##bitstrings in logical 1 state of Steane code
logical_1_states = ['1111111', '0101010', '1001100', '0011001', '1110000', '0100101', '1000011', '0010110']
##bitstrings that are within distance 1 of logical 1 state
error_states = []
for state in logical_1_states:
    error_states.append(state)
    for i in range(7):
        flipped_bit = '1' if state[i] == '0' else '0'
        error_state = state[:i] + flipped_bit + state[i+1:]
        error_states.append(error_state)
#convert to integers
error_states_int = [int(state, 2) for state in error_states]

reg1 = QuantumRegister(7, 'q')
reg2 = QuantumRegister(7, 'r')
reg3 = QuantumRegister(7, 's')
ancilla = QuantumRegister(7, 'a')
creg = ClassicalRegister(7, 'c')

register_list = [reg1, reg2, reg3]

qc = QuantumCircuit(reg1, reg2, reg3, ancilla, creg)

# Steane code encoding for each logical qubit

initial_state = QuantumCircuit(7)
initial_state.h(6)
initial_state.rz(2*beta_k(0)*x, 6)
# Here we can set the initial state for each logical qubit if needed
steane_encoded = steane_code_encoding_circuit(initial_state)
qc.compose(steane_encoded, qubits=register_list[0], inplace=True)

for i in range(1, nqubits):
    initial_state = QuantumCircuit(7)
    # Here we can set the initial state for each logical qubit if needed
    steane_encoded = steane_code_encoding_circuit(initial_state)
    qc.compose(steane_encoded, qubits=register_list[i], inplace=True)

for inst in sequences_per_pT[4]:
    if inst[0] == 'H':
        for i in range(7):
            qc.h(register_list[inst[1]][i])
    elif inst[0] == 'S':
        for _ in range(3): # Transversal S-gate is 3 layers of S
            for i in range(7):
                qc.s(register_list[inst[1]][i])
    elif inst[0] == 'CNOT':
        for i in range(7):
            qc.cx(register_list[inst[1]][i], register_list[inst[2]][i])
    
    elif inst[0] == 'T':
        # T-gate teleportation using ancilla qubits
        qc.reset(ancilla)
        qc.h(ancilla[6])
        qc.t(ancilla[6])
        qc.compose(steane_code_circuit(), qubits=ancilla, inplace=True)
        for i in range(7):
            qc.cx(ancilla[i], register_list[inst[1]][i])
        qc.measure(register_list[inst[1]], creg)
        # Conditional S-gate based on measurement outcome
        for error in error_states_int:
            with qc.if_test((creg, error)):
                for i in range(7):
                    for _ in range(3): # Transversal S-gate is 3 layers of S
                        qc.s(ancilla[i])
                    qc.x(ancilla[i])
        # Swap ancilla back to the logical qubit
        for i in range(7):
            qc.swap(ancilla[i], register_list[inst[1]][i])

qc.measure(register_list[0], creg)


# print(qc)

# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
qc = transpile(qc, simulator)
# Run and get unitary
result = simulator.run(qc, shots=1).result()
counts = result.get_counts(qc)
print("Measurement results:", counts)