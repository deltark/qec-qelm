import pickle
from random_cliff_t_circuit_ergodicity_numpy import get_average_ratio_of_adjacent_gaps, random_clifford_T_unitary_from_sequence

def createanddumprandomcircuits(nqubits, ngates, p_T):
    num_unitaries = 100
    unitaries = []
    avg_ratio = []
    sequences = []
    for _ in range(num_unitaries):
        U, sequence = random_clifford_T_unitary_from_sequence(nqubits, ngates, p_T)
        print("Sequence:", sequence)
        U = U.toarray()
        print(U)
        avg_ratio_value = get_average_ratio_of_adjacent_gaps(U)
        unitaries.append(U)
        avg_ratio.append(avg_ratio_value)
        sequences.append(sequence)

    filename = f'results/random_circuit_from_sequence_n{nqubits}_ngates{ngates}_p{p_T}_new.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    dict = {
        "unitaries": data['unitaries'] + unitaries,
        "sequences": data['sequences'] + sequences,
        "avg_ratio": data['avg_ratio'] + avg_ratio
    }
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)


for n in [3]:
    for p_T in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        createanddumprandomcircuits(nqubits=n, ngates=10*n**2, p_T=p_T)