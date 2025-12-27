import numpy as np
import scipy as sp

def random_clifford_sequence(nqubits, ngates):
    """Generate a random sequence of Clifford gates."""
    gates = ['H', 'S', 'CNOT']
    sequence = []
    for _ in range(ngates):
        gate = np.random.choice(gates)
        if gate == 'CNOT':
            control = np.random.randint(0, nqubits)
            target = (control + np.random.randint(1, nqubits)) % nqubits
            sequence.append((gate, control, target))
        else:
            qubit = np.random.randint(0, nqubits)
            sequence.append((gate, qubit))
    return sequence

def t_doping(sequence, t_proportion):
    """Insert T-gates into the Clifford sequence based on the given proportion."""
    t_count = int(len(sequence) * t_proportion)
    doped_sequence = sequence.copy()
    positions_to_replace = np.random.choice(len(doped_sequence), t_count, replace=False)
    for pos in positions_to_replace:
        if doped_sequence[pos][0] == 'CNOT':
            doped_sequence[pos] = ('T', doped_sequence[pos][2])  # Replace CNOT with T on target qubit
        else:
            doped_sequence[pos]= ('T', doped_sequence[pos][1])  # Replace single qubit gate with T
    return doped_sequence

nqubits = 8
ngates = 90
t_proportion = 0.25

print("Generating random Clifford+T sequence...")
print(f"Number of qubits: {nqubits}, Number of gates: {ngates}, T-gate proportion: {t_proportion}")
clifford_sequence = random_clifford_sequence(nqubits, ngates)
print("Random Clifford sequence:")
print(clifford_sequence)
doped_sequence = t_doping(clifford_sequence, t_proportion)
print("Random Clifford+T sequence:")
print(doped_sequence)

H = sp.sparse.csc_matrix(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
S = sp.sparse.csc_matrix(np.array([[1, 0], [0, 1j]]))
T = sp.sparse.csc_matrix(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]))
X = sp.sparse.csc_matrix(np.array([[0, 1], [1, 0]]))
zero_projector = sp.sparse.csc_matrix(np.array([[1, 0], [0, 0]]))
one_projector = sp.sparse.csc_matrix(np.array([[0, 0], [0, 1]]))

def full_matrix(nqubits, sequence):
    """Construct the full unitary matrix for the given sequence."""
    U = sp.sparse.identity(2**nqubits, format='csc')
    for gate in sequence:
        if gate[0] != 'CNOT':
            if gate[0] == 'H':
                single_gate = H
            elif gate[0] == 'S':
                single_gate = S
            elif gate[0] == 'T':
                single_gate = T
            qubit = gate[1]
            op = sp.sparse.identity(1, format='csc')
            for i in range(nqubits):
                if i == qubit:
                    op = sp.sparse.kron(op, single_gate, format='csc')
                else:
                    op = sp.sparse.kron(op, sp.sparse.identity(2, format='csc'), format='csc')
        else:
            control = gate[1]
            target = gate[2]
            op1 = sp.sparse.identity(1, format='csc')
            for i in range(nqubits):
                if i == control:
                    op1 = sp.sparse.kron(op1, zero_projector, format='csc')
                # elif i == target:
                #     op1 = sp.sparse.kron(op, sp.sparse.identity(2, format='csc'), format='csc')
                else:
                    op1 = sp.sparse.kron(op1, sp.sparse.identity(2, format='csc'), format='csc')
            op2 = sp.sparse.identity(1, format='csc')
            for i in range(nqubits):
                if i == control:
                    op2 = sp.sparse.kron(op2, one_projector, format='csc')
                elif i == target:
                    op2 = sp.sparse.kron(op2, X, format='csc')
                else:
                    op2 = sp.sparse.kron(op2, sp.sparse.identity(2, format='csc'), format='csc')
            op = op1 + op2
            
        U = op @ U
        
    return U

unitary = full_matrix(nqubits, doped_sequence).toarray()
print("Circuit unitary:\n", np.asarray(unitary).round(5))

def get_hamiltonian(unitary, timestep):
    """Compute the Hamiltonian from the unitary."""
    ham = 1j / timestep * sp.linalg.logm(unitary)
    return ham

ham = get_hamiltonian(unitary, timestep=1)
print("Log of unitary (Hamiltonian):\n", np.asarray(ham).round(5))

# Get eigenvalues
eigenvalues, _ = sp.linalg.eig(ham)
eigenvalues = np.real(eigenvalues)
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

statevector_0 = sp.sparse.csc_matrix([[1], [0]])
statevector_1 = sp.sparse.csc_matrix([[0], [1]])

def compute_fourier_coeffs(nqubits, observable, reservoir_unitary, accessible_qubits):
    """Compute the Fourier coefficients of the energy differences."""
    hidden_qubits = nqubits - accessible_qubits
    fourier_coeffs = np.zeros((beta_k(accessible_qubits)-1)//2+1)
    # eigenvalues_array = encoding_hamiltonian_eigenvalues_array(nqubits)
    statevector_hidden = sp.sparse.csc_matrix(([1], ([0], [0])), shape=(int(2**hidden_qubits), 1))
    # Initialize hidden qubits to |0...0>
    # print("Statevector hidden:")
    # print(statevector_hidden)

    for i in range(2**accessible_qubits):
        binary_i = format(i, f'0{accessible_qubits}b')
        # statevector_i = sp.sparse.csc_matrix(statevector_0 if binary_i[0] == '0' else statevector_1)
        # print("Statevector i before:")
        # print(statevector_i)
        # for bit in binary_i[1:]:
        #     statevector_i = sp.sparse.kron(statevector_i, statevector_0 if bit == '0' else statevector_1, format='csc') 
        
        statevector_i = sp.sparse.csc_matrix(([1], ([i], [0])), shape=(int(2**accessible_qubits), 1))
        statevector_i = sp.sparse.kron(statevector_i, statevector_hidden, format='csc')

        # print("Statevector i:")
        # print(statevector_i)

        for j in range(i, 2**accessible_qubits):
            binary_j = format(j, f'0{accessible_qubits}b')
            # statevector_j = sp.sparse.csc_matrix(statevector_0 if binary_j[0] == '0' else statevector_1)
            # for bit in binary_j[1:]:
            #     statevector_j = sp.sparse.kron(statevector_j, statevector_0 if bit == '0' else statevector_1, format='csc')
            statevector_j = sp.sparse.csc_matrix(([1], ([j], [0])), shape=(int(2**accessible_qubits), 1))
            statevector_j = sp.sparse.kron(statevector_j, statevector_hidden, format='csc')

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
        ops.append(pauli if q == qubit_index else sp.sparse.identity(2, format='csc'))
    # Tensor product from left (most significant) to right (least)
    mat = ops[0]
    for next_op in ops[1:]:
        mat = sp.sparse.kron(mat, next_op, format='csc')
    return mat

def ising_hamiltonian(nqubits, J=1.0, Bz=0.0, Bx=1.0):
    """
    Build a transverse-field Ising Hamiltonian for nqubits:

    H =  J * sum_{j=0..N-2} Z_j Z_{j+1}
        + Bz * sum_{j=0..N-1} Z_j
        + Bx * sum_{j=0..N-1} X_j
    """
    H = sp.sparse.csc_matrix((2**nqubits, 2**nqubits), dtype=complex)

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

n_accessible = 4
n_hidden = nqubits - n_accessible
print(f"Computing Fourier coefficients with {n_accessible} accessible qubits and {n_hidden} hidden qubits...")  
Z = sp.sparse.csc_matrix(np.array([[1, 0], [0, -1]]))
# observable = Z.copy()
observable = X.copy()
for _ in range(n_accessible-1):
    observable = sp.sparse.kron(observable, X.copy(), format='csc')
    # observable = sp.sparse.kron(observable, sp.sparse.identity(2, format='csc'), format='csc')
observable = sp.sparse.kron(observable, sp.sparse.identity(2**(n_hidden), format='csc'), format='csc')

# print("Observable:\n", observable)
ising_hamiltonian_chaotic = ising_hamiltonian(nqubits, J=-1.0, Bz=0.7, Bx=1.5)
ising_unitary = sp.sparse.linalg.expm(-1j * ising_hamiltonian_chaotic)

fourier_coeffs = compute_fourier_coeffs(nqubits, observable, ising_unitary, n_accessible)
print("Fourier coefficients:\n", np.asarray(fourier_coeffs))

print("For random Clifford+T circuit:")
fourier_coeffs_circuit = compute_fourier_coeffs(nqubits, observable, unitary, n_accessible)
print("Fourier coefficients:\n", np.asarray(fourier_coeffs_circuit))

#TODO : Check big endian/little endian consistency