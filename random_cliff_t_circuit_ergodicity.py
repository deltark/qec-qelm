import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile, QuantumRegister
from scipy import linalg

nqubits = 7
depth = 8
t_proportion = 0.2
timestep = 1

def random_clifford_t_circuit(nqubits, depth, t_proportion):
    circuit = QuantumCircuit(nqubits)
    for _ in range(depth):
        for q in range(nqubits):
            r = np.random.rand()
            if r < t_proportion:
                circuit.t(q)
            elif r < (1-t_proportion)/3:
                circuit.h(q)
            elif r < 2*(1-t_proportion)/3:
                circuit.s(q)
            else:
                circuit.cx(q, (q+1) % nqubits) 

    circuit.save_unitary()   
        
    return circuit


print("Generating random Clifford+T circuit...")
print(f"Number of qubits: {nqubits}, Depth: {depth}, T-gate proportion: {t_proportion}")
random_circuit = random_clifford_t_circuit(nqubits, depth, t_proportion)
print(random_circuit)

# Transpile for simulator
simulator = Aer.get_backend('aer_simulator')
# Another option to create the simulator
# simulator = AerSimulator(method = 'unitary')
random_circuit = transpile(random_circuit, simulator)

# Run and get unitary
result = simulator.run(random_circuit).result()
unitary = result.get_unitary(random_circuit)
print("Circuit unitary:\n", np.asarray(unitary).round(5))

# Take log of unitary to get Hamiltonian:
ham = 1j/timestep*linalg.logm(unitary)
print("Log of unitary (Hamiltonian):\n", np.asarray(ham).round(5))

# Get eigenvalues
eigenvalues, _ = linalg.eig(ham)
eigenvalues = np.real_if_close(eigenvalues)
eigenvalues = np.sort(eigenvalues)
#positive
eigenvalues = eigenvalues[eigenvalues >= 0]
print("Eigenvalues of the Hamiltonian:\n", np.asarray(eigenvalues).round(5))

# Get energy gaps
energy_gaps = np.diff(eigenvalues)
print("Energy gaps:\n", np.asarray(energy_gaps).round(5))

# Ratio of adjacent gaps
ratios = energy_gaps[1:] / energy_gaps[:-1]
# invert if needed to have ratios <= 1
ratios = np.minimum(ratios, 1/ratios)
print("Ratios of adjacent gaps:\n", np.asarray(ratios).round(5))

# Average ratio
avg_ratio = np.mean(ratios)
print("Average ratio of adjacent gaps:", round(avg_ratio, 5))