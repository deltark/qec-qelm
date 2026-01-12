import pickle
import scipy.sparse as sp
import numpy as np
from random_cliff_t_circuit_ergodicity_numpy import compute_fourier_coeffs, lift_1q_gate
import matplotlib.pyplot as plt
import plothist

n_hidden = 4
n_accessible = 1
nqubits = n_hidden + n_accessible

X = sp.csc_matrix([[0, 1], [1, 0]])
Z = sp.csc_matrix([[1, 0], [0, -1]])
Y = sp.csc_matrix([[0, -1j], [1j, 0]])

observables = []
for pauli in [X, Y, Z]:
    observable = pauli
    for _ in range(n_accessible - 1):
        observable = sp.kron(observable, pauli, format='csc')
    for _ in range(n_hidden):
        observable = sp.kron(observable, sp.eye(2, format='csc'), format='csc')
    observables.append(observable)

fourier_expressivity_by_pT = []
max_expressivity_unitaries = []
for p_T in [0.1, 0.15, 0.2, 0.25]:
    
    filename = f'results/random_circuit_n{nqubits}_d{10*nqubits}_p{p_T}.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    unitaries = data['unitaries']
    fourier_expressivities = []
    for unitary in unitaries:
        fourier_expressivity_per_observable = []
        for observable in observables:
            fourier_coeffs = compute_fourier_coeffs(nqubits, observable, unitary, n_accessible)
            # print(f'Fourier coeffs: {fourier_coeffs}')
            fourier_expressivity = np.count_nonzero(fourier_coeffs)/len(fourier_coeffs)
            fourier_expressivity_per_observable.append(fourier_expressivity)
        fourier_expressivities.append(np.mean(fourier_expressivity_per_observable))

    print(f'Fourier expressivities for p_T={p_T}: {fourier_expressivities}')

    max_expressivity = np.max(fourier_expressivities)
    fourier_expressivity_by_pT.append(max_expressivity)

    max_index = np.argmax(fourier_expressivities)
    max_expressivity_unitaries.append(unitaries[max_index])
    print(f'Max Fourier expressivity: {max_expressivity} at index {max_index}')


pickle_filename = f'results/fourier_analysis/fourier_expressivity_n{nqubits}_d{10*nqubits}.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump({'p_T_values': [0.1, 0.15, 0.2, 0.25],
                 'unitaries_per_pT': max_expressivity_unitaries,
                 'fourier_expressivity_by_pT': fourier_expressivity_by_pT}, f)
    
plt.figure()
plt.plot([0.1, 0.15, 0.2, 0.25], fourier_expressivity_by_pT, marker='o')
plt.xlabel('p_T')
plt.ylabel('Max Fourier Expressivity')
plt.title('Max Fourier Expressivity vs p_T')
plt.savefig(f'results/fourier_analysis/fourier_expressivity_n{nqubits}_d{10*nqubits}.png')