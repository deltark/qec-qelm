import pickle
import scipy.sparse as sp
import numpy as np
from random_cliff_t_circuit_ergodicity_numpy import compute_fourier_coeffs, lift_1q_gate, ising_hamiltonian
import matplotlib.pyplot as plt
import plothist

n_hidden = 4

X = sp.csc_matrix([[0, 1], [1, 0]])
Z = sp.csc_matrix([[1, 0], [0, -1]])
Y = sp.csc_matrix([[0, -1j], [1j, 0]])


for n_accessible in range(1, 5):
    nqubits = n_hidden + n_accessible

    ising_hamiltonian_chaotic = ising_hamiltonian(nqubits, J=-1.0, Bz=0.7, Bx=1.5)
    ising_unitary = sp.linalg.expm(-1j * ising_hamiltonian_chaotic)

    observables = []
    for pauli in [X, Y, Z]:
        observable = pauli
        for _ in range(n_accessible - 1):
            observable = sp.kron(observable, pauli, format='csc')
        for _ in range(n_hidden):
            observable = sp.kron(observable, sp.eye(2, format='csc'), format='csc')
        observables.append(observable)

    fourier_coeffs = compute_fourier_coeffs(nqubits, observables[0], ising_unitary.toarray(), n_accessible)

    plt.figure()
    plt.bar(range(len(fourier_coeffs)), np.abs(fourier_coeffs))
    plt.savefig(f'results/figs/ising_fourier_coeffs_naccessible{n_accessible}.png')