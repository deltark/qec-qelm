from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Statevector
import numpy as np

def steane_encode():
    # Create 7-qubit register and 1 classical bit for measurement (optional)
    qr = QuantumRegister(7, 'q')
    circuit = QuantumCircuit(qr, name="Steane Encoder")

    # The input state (logical qubit) is in q0: |ψ⟩ = α|0⟩ + β|1⟩
    # The other six qubits are initialized in |0⟩

    # Start encoding process (constructing |0_L⟩ and |1_L⟩ superpositions)

    # Step 1: Create equal superposition on q0, q1, q2
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.h(qr[2])

    # Step 2: Add parity checks for the classical [7,4,3] Hamming code structure
    circuit.cx(qr[0], qr[3])
    circuit.cx(qr[0], qr[4])
    circuit.cx(qr[1], qr[3])
    circuit.cx(qr[1], qr[5])
    circuit.cx(qr[2], qr[4])
    circuit.cx(qr[2], qr[5])

    # Step 3: Final parity qubit q6 depends on q0, q1, q2
    circuit.cx(qr[0], qr[6])
    circuit.cx(qr[1], qr[6])
    circuit.cx(qr[2], qr[6])

    # Step 4: Now entangle with the logical qubit information (X stabilizer encoding)
    # Apply CNOTs from the logical qubit (q0) to others to propagate logical |ψ⟩
    circuit.cx(qr[0], qr[3])
    circuit.cx(qr[1], qr[5])
    circuit.cx(qr[2], qr[6])

    # Step 5: Apply Hadamards to first three qubits (CSS structure)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.h(qr[2])

    return circuit


# # Example usage
# steane_circuit = steane_encode()
# steane_circuit.draw('mpl')


# Create the encoding circuit
circuit = steane_encode()

# Prepare |ψ> = cos(θ)|0⟩ + sin(θ)|1⟩, for example θ = π/4 gives |+⟩
theta = np.pi / 4
prep = QuantumCircuit(7)
prep.ry(2*theta, 0)

# Combine: prepare state then encode
full_circuit = prep.compose(circuit)
full_circuit.draw('mpl')

# Simulate encoded state
state = Statevector(full_circuit)
encoded = state.data
print("Encoded state dimension:", len(encoded))
print("Encoded state vector:", encoded)
