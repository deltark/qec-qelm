import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from qiskit import QuantumCircuit, synthesis

# def random_2_qubit_gate_sequence(i, j, depth, p_T):
#     """Generate a random sequence of gates."""
#     gates = ['H', 'S', 'CNOT', 'T']
#     sequence = []
#     for _ in range(depth):
#         gate = np.random.choice(gates, p=[0.4, 0.4, 3/depth])
#         if gate == 'CNOT':
#             control = i
#             target = j
#             sequence.append((gate, control, target))
#         else:
#             qubit = np.random.randint(0, nqubits)
#             sequence.append((gate, qubit))
#     return sequence

def random_clifford_sequence(nqubits, ngates):
    """Generate a random sequence of Clifford gates."""
    gates = ['H', 'S', 'CNOT']
    sequence = []
    for _ in range(ngates):
        i, j = np.random.choice(nqubits, size=2, replace=False)
        gate = np.random.choice(gates, p=[0.4, 0.4, 0.2])
        if gate == 'CNOT':
            # control = np.random.randint(0, nqubits)
            # target = (control + np.random.randint(1, nqubits)) % nqubits
            sequence.append((gate, i, j))
        else:
            qubit = np.random.choice([i,j])
            sequence.append((gate, qubit))
    return sequence

def t_doping(sequence, t_proportion):
    """Insert T-gates into the Clifford sequence based on the given proportion."""
    t_count = int(len(sequence) * t_proportion)
    doped_sequence = sequence.copy()
    positions_to_replace = np.random.choice(len(doped_sequence), t_count, replace=False)
    finaltcount = 0
    for pos in positions_to_replace:
        if doped_sequence[pos][0] == 'CNOT':
            doped_sequence[pos] = ('T', doped_sequence[pos][2])  # Replace CNOT with T on target qubit
            finaltcount += 1
        else:
            doped_sequence[pos]= ('T', doped_sequence[pos][1])  # Replace single qubit gate with T
            finaltcount += 1
    print(f"Inserted {finaltcount} T-gates into the sequence.")
    
    return doped_sequence

# nqubits = 8
# ngates = 10*nqubits**2
# # depth = 80
# t_proportion = 0.2

# print("Generating random Clifford+T sequence...")
# print(f"Number of qubits: {nqubits}, Number of gates: {ngates}, T-gate proportion: {t_proportion}")
# clifford_sequence = random_clifford_sequence(nqubits, ngates)
# print("Random Clifford sequence:")
# print(clifford_sequence)
# doped_sequence = t_doping(clifford_sequence, t_proportion)
# print("Random Clifford+T sequence:")
# print(doped_sequence)

I2 = sp.identity(2, format="csc", dtype=complex)
H = sp.csc_matrix(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
S = sp.csc_matrix(np.array([[1, 0], [0, 1j]]))
T = sp.csc_matrix(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]))
X = sp.csc_matrix(np.array([[0, 1], [1, 0]]))
Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))
Y = sp.csc_matrix(np.array([[0, -1j], [1j, 0]]))
zero_projector = sp.csc_matrix(np.array([[1, 0], [0, 0]]))
one_projector = sp.csc_matrix(np.array([[0, 0], [0, 1]]))

CNOT_01 = sp.csc_matrix(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=complex
)


def full_matrix(nqubits, sequence):
    """Construct the full unitary matrix for the given sequence."""
    U = sp.identity(2**nqubits, format='csc')
    for gate in sequence:
        if gate[0] != 'CNOT':
            if gate[0] == 'H':
                single_gate = H
            elif gate[0] == 'S':
                single_gate = S
            elif gate[0] == 'T':
                single_gate = T
            qubit = gate[1]
            op = sp.identity(1, format='csc')
            for i in range(nqubits):
                if i == qubit:
                    op = sp.kron(op, single_gate, format='csc')
                else:
                    op = sp.kron(op, sp.identity(2, format='csc'), format='csc')
        else:
            control = gate[1]
            target = gate[2]
            op1 = sp.identity(1, format='csc')
            for i in range(nqubits):
                if i == control:
                    op1 = sp.kron(op1, zero_projector, format='csc')
                # elif i == target:
                #     op1 = sp.kron(op, sp.identity(2, format='csc'), format='csc')
                else:
                    op1 = sp.kron(op1, sp.identity(2, format='csc'), format='csc')
            op2 = sp.identity(1, format='csc')
            for i in range(nqubits):
                if i == control:
                    op2 = sp.kron(op2, one_projector, format='csc')
                elif i == target:
                    op2 = sp.kron(op2, X, format='csc')
                else:
                    op2 = sp.kron(op2, sp.identity(2, format='csc'), format='csc')
            op = op1 + op2
            
        U = op @ U
        
    return U

def lift_1q_gate(gate, q, n):
    ops = []
    for i in range(n):
        ops.append(gate if i == q else I2)
    U = ops[0]
    for op in ops[1:]:
        U = sp.kron(U, op, format="csc")
    return U

def lift_2q_gate(gate, q, n):
    ops = []
    i = 0
    while i < n:
        if i == q:
            ops.append(gate)
            i += 2
        else:
            ops.append(I2)
            i += 1
    U = ops[0]
    for op in ops[1:]:
        U = sp.kron(U, op, format="csc")
    return U


# def random_clifford_layer(n):
#     U = sp.identity(2**n, format="csc", dtype=complex)

#     # Random H / S on each qubit
#     for q in range(n):
#         if np.random.rand() < 0.5:
#             U = lift_1q_gate(H, q, n) @ U
#         if np.random.rand() < 0.5:
#             U = lift_1q_gate(S, q, n) @ U

#     # Brickwork CNOTs
#     for q in range(0, n - 1, 2):
#         if np.random.rand() < 0.5:
#             U = lift_2q_gate(CNOT_01, q, n) @ U

#     return U

# def random_clifford_T_unitary(n, depth, p_T):
#     """
#     n     : number of qubits
#     depth : number of layers
#     p_T   : probability of T gate per qubit per layer
#     """
#     D = 2**n
#     U = sp.identity(D, format="csc", dtype=complex)

#     for _ in range(depth):
#         # Clifford
#         U = random_clifford_layer(n) @ U

#         # Random H / S on each qubit
#         for q in range(n):
#             if np.random.rand() < 0.5:
#                 U = lift_1q_gate(H, q, n) @ U
#             if np.random.rand() < 0.5:
#                 U = lift_1q_gate(S, q, n) @ U
#             if np.random.rand() < p_T:
#                 U = lift_1q_gate(T, q, n) @ U

#         # Brickwork CNOTs
#         for q in range(0, n - 1, 2):
#             if np.random.rand() < 0.5:
#                 U = lift_2q_gate(CNOT_01, q, n) @ U

#         # T layer
#         for q in range(n):
#             if np.random.rand() < p_T:
#                 U = lift_1q_gate(T, q, n) @ U

#     return U


# unitary = random_clifford_T_unitary(nqubits, depth, t_proportion).toarray()
# print("Circuit unitary:\n", np.asarray(unitary).round(5))

def random_clifford_T_unitary_from_sequence(nqubits, ngates, t_proportion):
    """Generate the unitary matrix for a given Clifford+T gate sequence."""
    sequence = random_clifford_sequence(nqubits, ngates)
    doped_sequence = t_doping(sequence, t_proportion)
    U = full_matrix(nqubits, doped_sequence)
    return U, doped_sequence

# unitary = full_matrix(nqubits, doped_sequence)
# print("Circuit unitary:\n", np.asarray(unitary).round(5))

def get_hamiltonian(unitary, timestep):
    """Compute the Hamiltonian from the unitary."""
    ham = 1j / timestep * sla.logm(unitary)
    return ham

def get_average_ratio_of_adjacent_gaps(unitary):
    """Compute the average ratio of adjacent energy gaps."""
    ham = get_hamiltonian(unitary, timestep=1)
    # print("Log of unitary (Hamiltonian):\n", np.asarray(ham).round(5))

    # Get eigenvalues
    eigenvalues, _ = sla.eig(ham)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)
    #positive
    eigenvalues = eigenvalues[eigenvalues >= 0]
    # print("Eigenvalues of the Hamiltonian:\n", np.asarray(eigenvalues).round(5))

    # Get energy gaps
    energy_gaps = np.diff(eigenvalues)
    # print("Energy gaps:\n", np.asarray(energy_gaps).round(5))

    # Ratio of adjacent gaps
    ratios = energy_gaps[1:] / energy_gaps[:-1]
    # invert if needed to have ratios <= 1
    ratios = np.minimum(ratios, 1/ratios)
    # print("Ratios of adjacent gaps:\n", np.asarray(ratios).round(5))

    # Average ratio
    avg_ratio = np.mean(ratios)
    # print("Average ratio of adjacent gaps:", round(avg_ratio, 5))
    return round(avg_ratio, 5)


##### FOURIER ANALYSIS #####

def beta_k(qubit_index):
    """Compute beta_k for the given qubit index."""
    return 3**qubit_index #indexing from 0 unlike Zoe Holmes paper

# def encoding_hamiltonian_eigenvalues_array(nqubits):
#     """Create a list of eigenvalues that encode Z on each qubit."""
#     eigenvalues_array = np.array([])
#     for i in range(nqubits):
#         beta_k_val = beta_k(i)
#         eigenvalues_array = np.append(eigenvalues_array, np.array([beta_k_val/2, -beta_k_val/2]))
#     return eigenvalues_array

statevector_0 = sp.csc_matrix([[1], [0]])
statevector_1 = sp.csc_matrix([[0], [1]])

def compute_fourier_coeffs(nqubits, observable, reservoir_unitary, accessible_qubits):
    """Compute the Fourier coefficients of the energy differences."""
    hidden_qubits = nqubits - accessible_qubits
    fourier_coeffs = np.zeros((beta_k(accessible_qubits)-1)//2+1)
    # eigenvalues_array = encoding_hamiltonian_eigenvalues_array(nqubits)
    statevector_hidden = sp.csc_matrix(([1], ([0], [0])), shape=(int(2**hidden_qubits), 1))
    # Initialize hidden qubits to |0...0>
    # print("Statevector hidden:")
    # print(statevector_hidden)

    for i in range(2**accessible_qubits):
        binary_i = format(i, f'0{accessible_qubits}b')
        # statevector_i = sp.csc_matrix(statevector_0 if binary_i[0] == '0' else statevector_1)
        # print("Statevector i before:")
        # print(statevector_i)
        # for bit in binary_i[1:]:
        #     statevector_i = sp.kron(statevector_i, statevector_0 if bit == '0' else statevector_1, format='csc') 
        
        statevector_i = sp.csc_matrix(([1], ([i], [0])), shape=(int(2**accessible_qubits), 1))
        statevector_i = sp.kron(statevector_i, statevector_hidden, format='csc')

        # print("Statevector i:")
        # print(statevector_i)

        for j in range(i, 2**accessible_qubits):
            binary_j = format(j, f'0{accessible_qubits}b')
            # statevector_j = sp.csc_matrix(statevector_0 if binary_j[0] == '0' else statevector_1)
            # for bit in binary_j[1:]:
            #     statevector_j = sp.kron(statevector_j, statevector_0 if bit == '0' else statevector_1, format='csc')
            statevector_j = sp.csc_matrix(([1], ([j], [0])), shape=(int(2**accessible_qubits), 1))
            statevector_j = sp.kron(statevector_j, statevector_hidden, format='csc')

            # print("Statevector j:")
            # print(statevector_j)

            # print("Observable:\n", observable)

            # print("Components of coefficient calculation:")
            # print("statevector_j.T:\n", statevector_j.T)
            # print("reservoir_unitary.conj().T:\n", reservoir_unitary.conj().T)
            # print("observable:\n", observable)
            # print("statevector_i:\n", statevector_i)

            coeff = (statevector_j.T @ reservoir_unitary.conj().T @ observable @ reservoir_unitary @ statevector_i)[0,0]

            # print(f"Coefficient for states {binary_i} and {binary_j}: {coeff}")

            freq = 0
            if j != i:
                for q in range(accessible_qubits):
                    if binary_i[accessible_qubits-q-1] == '0' and binary_j[accessible_qubits-q-1] == '1':
                        freq -= beta_k(q)
                    elif binary_i[accessible_qubits-q-1] == '1' and binary_j[accessible_qubits-q-1] == '0':
                        freq += beta_k(q)

            # print(f"Frequency for states {i} and {j}: {freq}")

            index = abs(freq)
            fourier_coeffs[index] += coeff/int(2**accessible_qubits)

        # Source - https://stackoverflow.com/a
        # Posted by FBruzzesi
        # Retrieved 2025-12-27, License - CC BY-SA 4.0

        fourier_coeffs[np.isclose(fourier_coeffs, 0, atol=1e-15)] = 0

    return fourier_coeffs

def pauli_operator(pauli, qubit_index, nqubits):
    """
    Build a many-qubit operator equal to pauli on qubit_index
    and identity elsewhere.
    """
    ops = []
    for q in range(nqubits):
        ops.append(pauli if q == qubit_index else I2)
    # Tensor product from left (most significant) to right (least)
    mat = ops[0]
    for next_op in ops[1:]:
        mat = sp.kron(mat, next_op, format='csc')
    return mat

def ising_hamiltonian(nqubits, J=1.0, Bz=0.0, Bx=1.0):
    """
    Build a transverse-field Ising Hamiltonian for nqubits:

    H =  J * sum_{j=0..N-2} Z_j Z_{j+1}
        + Bz * sum_{j=0..N-1} Z_j
        + Bx * sum_{j=0..N-1} X_j
    """
    H = sp.csc_matrix((2**nqubits, 2**nqubits), dtype=complex)

    # Nearest-neighbour ZZ terms
    for j in range(nqubits - 1):
        ZZ = pauli_operator(Z, j, nqubits) @ pauli_operator(Z, j+1, nqubits)
        H += J * ZZ

    # Transverse field Z terms
    for j in range(nqubits):
        H += Bz * pauli_operator(Z, j, nqubits)

    # Transverse field X terms
    for j in range(nqubits):
        H += Bx * pauli_operator(X, j, nqubits)
    return H

def ising_unitary_circuit_trotterstep_approximated(nqubits, J = -1.0, Bz = 0.7, Bx = 1.5, timestep = 1.0, nsteps=1, epsilon=1e-8):
    """
    Build a transverse-field Ising Hamiltonian evolution for nqubits:

    H =  J * sum_{j=0..N-2} Z_j Z_{j+1}
        + Bz * sum_{j=0..N-1} Z_j
        + Bx * sum_{j=0..N-1} X_j
    """
    qc = QuantumCircuit(nqubits)

    for _ in range(nsteps):

        # ZZ interactions
        for j in range(nqubits - 1):
            qc.cx(j, j + 1)
            rzsynth = synthesis.gridsynth_rz(2 * J * timestep, epsilon=epsilon)
            qc.append(rzsynth.to_instruction(), [j + 1])
            qc.cx(j, j + 1)

        # Bz Z fields
        for j in range(nqubits):
            rzsynth = synthesis.gridsynth_rz(2 * Bz * timestep, epsilon=epsilon)
            qc.append(rzsynth.to_instruction(), [j])

        # Bx X fields
        for j in range(nqubits):
            qc.h(j)
            rzsynth = synthesis.gridsynth_rz(2 * Bx * timestep, epsilon=epsilon)
            qc.append(rzsynth.to_instruction(), [j])
            qc.h(j)
        
    return qc

def count_n_Tgates(circuit):
    """Count the number of T-gates in a Qiskit circuit."""
    t_count = 0
    for instr, qargs, cargs in circuit.data:
        if instr.name == 't':
            t_count += 1
    return t_count

def generate_all_pauli_observables(nqubits):
    """Generate all Pauli observables for nqubits."""
    paulis = [I2, X, Y, Z]
    # pauli_names = ['I', 'X', 'Y', 'Z']
    observables = []
    for i in range(1, 4**nqubits):
        ops = []
        # ops_names = []
        index = i
        for _ in range(nqubits):
            ops.append(paulis[index % 4])
            # ops_names.append(pauli_names[index % 4])
            index //= 4
        # print(ops_names)
        obs = ops[0]
        for op in ops[1:]:
            obs = sp.kron(obs, op, format='csc')
        observables.append(obs)
    return observables

# n_accessible = 4
# n_hidden = nqubits - n_accessible
# print(f"Computing Fourier coefficients with {n_accessible} accessible qubits and {n_hidden} hidden qubits...")  
# Z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))
# # observable = Z.copy()
# observable = X.copy()
# for _ in range(n_accessible-1):
#     observable = sp.kron(observable, X.copy(), format='csc')
    # observable = sp.kron(observable, I2, format='csc')
# observable = sp.kron(observable, sp.identity(2**(n_hidden), format='csc'), format='csc')

# print("Observable:\n", observable)
# ising_hamiltonian_chaotic = ising_hamiltonian(nqubits, J=-1.0, Bz=0.7, Bx=1.5)
# ising_unitary = sp.linalg.expm(-1j * ising_hamiltonian_chaotic)

# fourier_coeffs = compute_fourier_coeffs(nqubits, observable, ising_unitary, n_accessible)
# print("Fourier coefficients:\n", np.asarray(fourier_coeffs))

# print("For random Clifford+T circuit:")
# fourier_coeffs_circuit = compute_fourier_coeffs(nqubits, observable, unitary, n_accessible)
# print("Fourier coefficients:\n", np.asarray(fourier_coeffs_circuit))

# #TODO : Check big endian/little endian consistency