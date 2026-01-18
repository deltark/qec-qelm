import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile, QuantumRegister
from random_cliff_t_circuit_ergodicity_numpy import *
# import os
# os.environ['RUST_BACKTRACE'] = 'full' # For debugging synthesis errors

nqubits = 3
observables = generate_all_pauli_observables(nqubits)
epsilon = 0.38
list_nsteps = [1, 2, 3]

for n_accessible in range(1, nqubits):
    n_hidden = nqubits - n_accessible

    for nsteps in list_nsteps:
        print(f'\n=== n_accessible={n_accessible}, nsteps={nsteps} ===')

        ising_circuit = ising_unitary_circuit_trotterstep_approximated(nqubits, J=-1.0, Bz=0.7, Bx=1.5, timestep=0.5, nsteps=nsteps, epsilon=epsilon)
        ising_circuit = ising_circuit.decompose('circuit-*', reps=1)
        ising_circuit.save_unitary()
        # print("Generated Ising circuit:")
        # print(ising_circuit)
        # print(ising_circuit.data)

        n_gates = len(ising_circuit.data)
        print(f'Number of gates in the circuit: {n_gates}')

        n_tgates = count_n_Tgates(ising_circuit)
        print(f'Number of T-gates in the circuit: {n_tgates}')

        p_T = n_tgates / n_gates
        print(f'Proportion of T-gates in the circuit: {p_T}')

        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator')
        # Another option to create the simulator
        # simulator = AerSimulator(method = 'unitary')
        ising_circuit = transpile(ising_circuit, simulator)

        # Run and get unitary
        result = simulator.run(ising_circuit).result()
        unitary = result.get_unitary(ising_circuit).to_matrix()
        # print("Circuit unitary:\n", np.asarray(unitary).round(5))

        
        fourier_expressivity_per_observable = []
        for observable in observables:
            fourier_coeffs = compute_fourier_coeffs(nqubits, observable, unitary, n_accessible)
            # print(f'Fourier coeffs: {fourier_coeffs}')
            fourier_expressivity = np.count_nonzero(fourier_coeffs)/len(fourier_coeffs)
            fourier_expressivity_per_observable.append(fourier_expressivity)
        fourier_expressivity = np.mean(fourier_expressivity_per_observable)
        print(f'Fourier expressivity for n_accessible={n_accessible}: {fourier_expressivity}')
