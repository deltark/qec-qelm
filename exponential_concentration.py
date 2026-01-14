import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
import pickle
from random_cliff_t_circuit_ergodicity_numpy import ising_hamiltonian, beta_k

Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))

nqubits = 2
n_accessible = 1
n_hidden = nqubits - n_accessible

circuit = QuantumCircuit(nqubits)

#hidden qubit 0
#accessible qubit 1
circuit.h(1)

#Translate encoding to qiskit circuit
encoding_unitary = np.linalg.exp(-1j * beta_k(0) * Z)
for i in range(n_accessible-1):
    encoding_unitary = np.kron(encoding_unitary, np.linalg.exp(-1j * beta_k(i+1) * Z))

circuit.unitary(encoding_unitary, range(n_hidden, nqubits))

#Translate unitary to qiskit circuit
ising_hamiltonian_chaotic = ising_hamiltonian(nqubits, J=-1.0, Bz=0.7, Bx=1.5)
ising_unitary = sp.linalg.expm(-1j * ising_hamiltonian_chaotic).toarray()


#Run on noisy simulator (Pauli noise model)