import pickle
from random_cliff_t_circuit_ergodicity_numpy import get_average_ratio_of_adjacent_gaps, random_clifford_T_unitary_from_sequence

def createanddumprandomcircuits(nqubits, ngates, p_T):
    num_unitaries = 20
    unitaries = []
    avg_ratio = []
    for _ in range(num_unitaries):
        U = random_clifford_T_unitary_from_sequence(nqubits, ngates, p_T).toarray()
        print(U)
        avg_ratio_value = get_average_ratio_of_adjacent_gaps(U)
        unitaries.append(U)
        avg_ratio.append(avg_ratio_value)

    filename = f'results/random_circuit_from_sequence_n{nqubits}_ngates{ngates}_p{p_T}.pkl'
    dict = {
        "unitaries": unitaries,
        "avg_ratio": avg_ratio
    }
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)


for n in [3]:
    ngates = 10*n**2
    for p_T in [0.1, 0.15, 0.2, 0.25]:
        createanddumprandomcircuits(nqubits=n, ngates=ngates, p_T=p_T)