import pickle
from random_cliff_t_circuit_ergodicity_numpy import random_clifford_T_unitary, get_average_ratio_of_adjacent_gaps

def createanddumprandomcircuits(nqubits, depth, p_T):
    num_unitaries = 20
    unitaries = []
    avg_ratio = []
    for i in range(num_unitaries):
        U = random_clifford_T_unitary(nqubits, depth, p_T).toarray()
        print(U)
        avg_ratio_value = get_average_ratio_of_adjacent_gaps(U)
        unitaries.append(U)
        avg_ratio.append(avg_ratio_value)

    filename = f'results/random_circuit_n{nqubits}_d{depth}_p{p_T}.pkl'
    dict = {
        "unitaries": unitaries,
        "avg_ratio": avg_ratio
    }
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)


for n in [6, 7, 8]:
    depth = 10*n
    createanddumprandomcircuits(nqubits=n, depth=depth, p_T=0.1)
    createanddumprandomcircuits(nqubits=n, depth=depth, p_T=0.15)
    createanddumprandomcircuits(nqubits=n, depth=depth, p_T=0.2)
    createanddumprandomcircuits(nqubits=n, depth=depth, p_T=0.25)