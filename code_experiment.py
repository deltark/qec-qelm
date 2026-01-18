import numpy as np
import pickle
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.quantum_info import Statevector
from steane_code import steane_code_encoding_circuit

filename = 'results/fourier_analysis/fourier_expressivity_from_sequence_n3.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
p_T_values = data['p_T_values']
max_expressivity_unitaries_per_pT = data['max_expressivity_unitaries_per_pT']
sequences_per_pT = data['max_expressivity_sequences_per_pT']

nqubits = 3
nphysical_qubits = 7 * nqubits  # Each logical qubit is encoded into 7 physical qubits using the Steane code

reg1 = QuantumRegister(7, 'q')
reg2 = QuantumRegister(7, 'r')
reg3 = QuantumRegister(7, 's')
ancilla = QuantumRegister(7, 'a')

register_list = [reg1, reg2, reg3]

qc = QuantumCircuit(reg1, reg2, reg3, ancilla)

# Steane code encoding for each logical qubit
for i in range(nqubits):
    initial_state = QuantumCircuit(7)
    # Here we can set the initial state for each logical qubit if needed
    steane_encoded = steane_code_encoding_circuit(initial_state)
    qc.compose(steane_encoded, qubits=register_list[i], inplace=True)

for inst in sequences_per_pT[4]:
    if inst[0] == 'H':
        for i in range(7):
            qc.h(register_list[inst[1]][i])
    elif inst[0] == 'S':
        for i in range(7):
            qc.s(register_list[inst[1]][i])
    elif inst[0] == 'CNOT':
        for i in range(7):
            qc.cx(register_list[inst[1]][i], register_list[inst[2]][i])
    
    # elif inst[0] == 'T':
        # T-gate teleportation usig ancilla qubits


print(qc)