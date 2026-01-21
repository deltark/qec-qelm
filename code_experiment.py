import numpy as np
import pickle
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate, TGate
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.classical import expr
from steane_code import steane_code_encoding_circuit, steane_code_circuit, t_gate_teleportation
from random_cliff_t_circuit_ergodicity_numpy import beta_k
import scipy.sparse as sp

nqubits = 2
nshots = 10
simulator = Aer.get_backend('aer_simulator')

Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))

perfect_hgate = HGate(label='h1')
perfect_cnot = CXGate(label='cnot1')

filename = f'results/fourier_analysis/fourier_expressivity_from_sequence_n{nqubits}.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
p_T_values = data['p_T_values']
max_expressivity_unitaries_per_pT = data['max_expressivity_unitaries_per_pT']
sequences_per_pT = data['max_expressivity_sequences_per_pT']

# nphysical_qubits = 7 * nqubits  # Each logical qubit is encoded into 7 physical qubits using the Steane code

filename = f'results/circuit_runs/error_states_int.pkl'
with open(filename, 'rb') as f:
    error_states_int = pickle.load(f)

# for x in np.arange(0.0, 1.0, 0.2):
for x in [0.4]:

# def run_steane_code_experiment(x, prob_error, p_T_index):

    # reg1 = QuantumRegister(7, 'q')
    # reg2 = QuantumRegister(7, 'r')
    # reg3 = QuantumRegister(7, 's')
    #define as many registers as logical qubits
    register_list = []
    for i in range(nqubits):
        register_list.append(QuantumRegister(7, f'q{i}'))
    ancilla = QuantumRegister(7, 'a')
    creg = ClassicalRegister(7, 'c')

    # register_list = [reg1, reg2]

    qc = QuantumCircuit(*register_list, ancilla, creg)

    # #create a classical register for every error state value
    # for error in error_states_int:
    #     qc.add_var(f'c{error}', error)

    # expr.iter_vars()

    # Steane code encoding for each logical qubit

    initial_state = QuantumCircuit(7)
    initial_state.append(perfect_hgate, [6])
    initial_state.rz(2*np.pi*beta_k(0)*x, 6)
    # Here we can set the initial state for each logical qubit if needed
    steane_encoded = steane_code_encoding_circuit(initial_state)
    qc.compose(steane_encoded, qubits=register_list[0], inplace=True)

    for i in range(1, nqubits):
        initial_state = QuantumCircuit(7)
        # Here we can set the initial state for each logical qubit if needed
        steane_encoded = steane_code_encoding_circuit(initial_state)
        qc.compose(steane_encoded, qubits=register_list[i], inplace=True)

    for inst in sequences_per_pT[5]:
        if inst[0] == 'H':
            for i in range(7):
                qc.h(register_list[inst[1]][i])
        elif inst[0] == 'S':
            for _ in range(3): # Transversal S-gate is 3 layers of S
                for i in range(7):
                    qc.s(register_list[inst[1]][i])
        elif inst[0] == 'CNOT':
            for i in range(7):
                qc.cx(register_list[inst[1]][i], register_list[inst[2]][i])
        
        elif inst[0] == 'T':
            # T-gate teleportation using ancilla qubits
            t_gate_teleportation(qc, ancilla, register_list[inst[1]], creg)

    # for reg in register_list:
    #     qc.compose(steane_code_circuit().inverse(), reg, inplace=True)
    # qc.save_statevector()
    qc.measure(register_list[0], creg)


    # print(qc)

    # Transpile for simulator
    # simulator = AerSimulator(method = 'statevector')
    # qc = transpile(qc, simulator)
    #
    # result = simulator.run(qc, shots=1).result()
    # counts = result.get_counts(qc)
    # print("Measurement results:", counts)


    # Run on noisy simulator (Pauli noise model)

    # for error_prob in [0.0, 0.01, 0.03, 0.07, 0.1]:
    # for error_prob in [0.5, 0.9]:
    # for error_prob in [0.0, 0.0001, 0.001, 0.01, 0.1]:
    for error_prob in [0.1]:
        error = depolarizing_error(error_prob, 1)
        error2 = depolarizing_error(error_prob, 2)
        # error = pauli_error([('I', 1 - error_prob), ('X', error_prob / 3), ('Y', error_prob / 3), ('Z', error_prob / 3)])
        # error2 = error.tensor(error)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['h', 's', 'x'])
        noise_model.add_all_qubit_quantum_error(error2, ['cx'])

        # compiled_circuit = transpile(qc, simulator)
        # print(compiled_circuit.data)
        result = simulator.run(qc, noise_model=noise_model, shots=nshots).result()
        counts = result.get_counts()

        # filename = f'results/circuit_runs/steane_code_{nqubits}logqubits_errorprob{error_prob}_x{x:.2f}.pkl'
        # with open(filename, 'rb') as f:
        #     data = pickle.load(f)

        # #add counts to existing data
        # for key, value in counts.items():
        #     if key in data:
        #         data[key] += value
        #     else:
        #         data[key] = value

        # with open(filename, 'wb') as f:
        #     pickle.dump(data, f)

        print(f'Error prob: {error_prob}, x: {x:.2f}, Measurement results:', counts)

        # plt.plot(range(len(counts)), list(counts.values()), label=f'Error prob: {error_prob}')
