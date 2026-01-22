import numpy as np
import pickle
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate, TGate
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from random_cliff_t_circuit_ergodicity_numpy import beta_k
import scipy.sparse as sp

nqubits = 3
pT_index = 3
Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))
# observable = sp.kron(Z, sp.eye(2))
observable = sp.kron(sp.eye(2), Z)
observable = sp.kron(sp.eye(2), observable)

# simulator = AerSimulator(method = 'statevector')
simulator = AerSimulator()

perfect_hgate = HGate(label='h1')
perfect_cnot = CXGate(label='cnot1')

filename = f'results/fourier_analysis/fourier_expressivity_from_sequence_n{nqubits}.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
p_T_values = data['p_T_values']
max_expressivity_unitaries_per_pT = data['max_expressivity_unitaries_per_pT']
sequences_per_pT = data['max_expressivity_sequences_per_pT']

# for x in np.arange(0, 1, 0.2):
for x in [0.4]:

    qreg = QuantumRegister(nqubits, 'q')
    creg = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qreg, creg)
    
    # Encoding classical data
    qc.append(perfect_hgate, [qreg[0]])
    qc.rz(2*np.pi*beta_k(0)*x, qreg[0])

    # Apply reservoir circuit
    for inst in sequences_per_pT[pT_index]:
        if inst [0] == 'H':
            qc.h(inst[1])
        elif inst[0] == 'CNOT':
            qc.cx(inst[1], inst[2])
        elif inst[0] == 'T':
            qc.t(inst[1])
        elif inst[0] == 'S':
            qc.s(inst[1])

    # qc.h(qreg[0])
    # qc.save_statevector()
    qc.measure(qreg[0], creg[0])

    # print(qc)


    for error_prob in [0.0, 0.0001, 0.001, 0.01, 0.1]:
    # for error_prob in [0.0]:
            error = depolarizing_error(error_prob, 1)
            error2 = depolarizing_error(error_prob, 2)
            # error = pauli_error([('I', 1 - error_prob), ('X', error_prob / 3), ('Y', error_prob / 3), ('Z', error_prob / 3)])
            # error2 = error.tensor(error)
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(error, ['h', 's', 't'])
            noise_model.add_all_qubit_quantum_error(error2, ['cx'])

            # print(compiled_circuit.data)
            nshots = 500
            result = simulator.run(qc, noise_model=noise_model, shots=nshots).result()
            # result = simulator.run(qc).result()
            counts = result.get_counts()
            # state = result.get_statevector()
            # expect_value = state.expectation_value(observable.toarray())
            # print(expect_value)
            # print(state)

            filename = f'results/circuit_runs/raw_{nqubits}qubits_pT{p_T_values[pT_index]}_errorprob{error_prob}_x{x:.2f}_{nshots}shots.pkl'
            # filename = f'results/circuit_runs/raw_{nqubits}qubits_pT{p_T_values[pT_index]}_errorprob{error_prob}_x{x:.2f}_statevector.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(counts, f)
                # pickle.dump(expect_value, f)
            
            # print(f'Error prob: {error_prob}, x: {x:.2f}, Measurement results:', counts)