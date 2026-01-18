import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit, synthesis
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
import pickle
from random_cliff_t_circuit_ergodicity_numpy import ising_hamiltonian, beta_k
import matplotlib.pyplot as plt

simulator = AerSimulator()

Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))

nqubits = 3
n_accessible = 1
n_hidden = nqubits - n_accessible

circuit = QuantumCircuit(nqubits)

#hidden qubit 0
#accessible qubit 1
circuit.h(1)

#Translate encoding to qiskit circuit
encoding_unitary = sp.linalg.expm(-1j * beta_k(0) * Z).toarray()
for i in range(n_accessible-1):
    encoding_unitary = np.kron(encoding_unitary, sp.linalg.expm(-1j * beta_k(i+1) * Z)).toarray()

#Apply encoding unitary to accessible qubits
circuit.unitary(encoding_unitary, range(n_hidden, nqubits))

#Translate unitary to qiskit circuit
ising_hamiltonian_chaotic = ising_hamiltonian(nqubits, J=-1.0, Bz=0.7, Bx=1.5)
ising_unitary = sp.linalg.expm(-1j * 1* ising_hamiltonian_chaotic).toarray()

# Apply ising unitary to all qubits
# reservoir_circuit = QuantumCircuit(nqubits)
# reservoir_circuit.unitary(ising_unitary, range(nqubits))


# Decompose unitary into basis gates
# reservoir_circuit = transpile(reservoir_circuit, basis_gates=['cx', 'h', 's', 't'], approximation_degree=1.0, optimization_level=3)
# print(reservoir_circuit)

#Run on noisy simulator (Pauli noise model)
# plt.figure()
# for error_prob in [0.0, 0.01, 0.03, 0.07, 0.1]:
#     error = pauli_error([('I', 1 - error_prob), ('X', error_prob / 3), ('Y', error_prob / 3), ('Z', error_prob / 3)])
#     noise_model = NoiseModel()
#     noise_model.add_all_qubit_quantum_error(error, ['unitary'])

#     compiled_circuit = simulator.compile(circuit)
#     result = simulator.run(compiled_circuit, noise_model=noise_model, shots=1024).result()
#     counts = result.get_counts()

#     plt.plot(range(len(counts)), list(counts.values()), label=f'Error prob: {error_prob}')
# plt.legend()