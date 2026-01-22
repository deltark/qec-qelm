import pickle
import scipy.sparse as sp
import numpy as np
from random_cliff_t_circuit_ergodicity_numpy import compute_fourier_coeffs, lift_1q_gate, generate_all_pauli_observables
import matplotlib.pyplot as plt
import plothist

nqubits = 3

# X = sp.csc_matrix([[0, 1], [1, 0]])
# Z = sp.csc_matrix([[1, 0], [0, -1]])
# Y = sp.csc_matrix([[0, -1j], [1j, 0]])

observables = generate_all_pauli_observables(nqubits)

list_p_T = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
max_acc = []
res_acc=[]
var_acc=[]
for n_accessible in [1]:
    n_hidden = nqubits - n_accessible

    # for pauli in pauli_list:
    #     observable = pauli
    #     for _ in range(n_accessible - 1):
    #         observable = sp.kron(observable, pauli, format='csc')
    #     for _ in range(n_hidden):
    #         observable = sp.kron(observable, sp.eye(2, format='csc'), format='csc')
    #     observables.append(observable)

    fourier_expressivity_by_pT = []
    max_expressivity_unitaries = []
    max_expressivity_sequence = []
    average_expressivity_by_pT = []
    variance_expressivity_by_pT = []
    for p_T in list_p_T:

        filename = f'results/random_circuit_from_sequence_n{nqubits}_ngates{10*nqubits**2}_p{p_T}_new.pkl'
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

        average_ratios = data['avg_ratio']

        print(f'Fourier expressivities for p_T={p_T}: {fourier_expressivities}')

        print(f'Average ratios for p_T={p_T}: {average_ratios}')
        

        max_expressivity = np.max(fourier_expressivities)
        fourier_expressivity_by_pT.append(max_expressivity)
        avg_expressivity = np.mean(fourier_expressivities)
        average_expressivity_by_pT.append(avg_expressivity)
        variance_expressivity = np.var(fourier_expressivities)
        variance_expressivity_by_pT.append(variance_expressivity)

        max_index = np.argmax(fourier_expressivities)
        max_expressivity_unitaries.append(unitaries[max_index])
        max_expressivity_sequence.append(data['sequences'][max_index])
        print(f'Max expressivity sequence for p_T={p_T}: {data["sequences"][max_index]}')
        print(f'Max Fourier expressivity: {max_expressivity} at index {max_index}')


    pickle_filename = f'results/fourier_analysis/fourier_expressivity_from_sequence_n{nqubits}.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump({'p_T_values': list_p_T,
                    'max_expressivity_unitaries_per_pT': max_expressivity_unitaries,
                    'max_expressivity_sequences_per_pT': max_expressivity_sequence,
                    'average_expressivity_by_pT': average_expressivity_by_pT,
                    'variance_expressivity_by_pT': variance_expressivity_by_pT,
                    'max_fourier_expressivity_by_pT': fourier_expressivity_by_pT}, f)
    res_acc.append(average_expressivity_by_pT)
    var_acc.append(variance_expressivity_by_pT)
    max_acc.append(fourier_expressivity_by_pT) 
plt.figure()
for i in range(len(res_acc)):
    plt.errorbar(list_p_T, res_acc[i], marker='o', yerr=var_acc[i], label=f'n_accessible={i+1}')
    plt.plot(list_p_T, max_acc[i], marker='x')
plt.xlabel('p_T')
plt.ylabel('Average Fourier Expressivity')
plt.title('Average Fourier Expressivity vs p_T')
plt.legend()
plt.savefig(f'results/fourier_analysis/avg_fourier_expressivity_from_sequence_n{nqubits}_allobservables_new2.png')