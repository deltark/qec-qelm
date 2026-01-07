import pickle
import scipy.sparse as sp
import numpy as np
from random_cliff_t_circuit_ergodicity_numpy import compute_fourier_coeffs
import matplotlib.pyplot as plt
import plothist

n_hidden = 4
n_accessible = 4
nqubits = n_hidden + n_accessible

X = sp.csc_matrix([[0, 1], [1, 0]])

observable = X
for _ in range(n_accessible - 1):
    observable = sp.kron(observable, X, format='csc')
for _ in range(n_hidden):
    observable = sp.kron(observable, sp.identity(2), format='csc')

for p_T in [0.1, 0.15, 0.2, 0.25]:
    filename = f'results/random_circuit_n{nqubits}_d{10*nqubits}_p{p_T}.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    unitaries = data['unitaries']
    for unitary in unitaries[:5]:  # Analyze first 5 unitaries
        fourier_coeffs = compute_fourier_coeffs(nqubits, observable, unitary, n_accessible)
        fig = plt.figure()
        plt.plot(np.abs(fourier_coeffs), marker='o')
        fig.savefig(f'results/figs/fourier_n{nqubits}_p{p_T}_coeffs.png')