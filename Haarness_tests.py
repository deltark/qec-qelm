import numpy as np
import itertools
import scipy.sparse as sp
from random_cliff_t_circuit_ergodicity_numpy import *

def frame_potential_2(unitaries):
    """
    unitaries : list of (D x D) matrices (dense or sparse)
    """
    M = len(unitaries)
    vals = []

    for i in range(M):
        Ui = unitaries[i].conj().T
        for j in range(M):
            Uj = unitaries[j]
            tr = (Ui @ Uj).diagonal().sum()
            vals.append(np.abs(tr)**4)

    return np.mean(vals)

def partial_trace(rho, keep, n):
    """
    rho  : density matrix (2^n x 2^n)
    keep : list of qubits to keep
    """
    keep = sorted(keep)
    traced = [i for i in range(n) if i not in keep]

    dims = [2] * n
    reshaped = rho.reshape(dims + dims)

    for q in reversed(traced):
        reshaped = reshaped.trace(axis1=q, axis2=q+n)

    d_keep = 2 ** len(keep)
    return reshaped.reshape((d_keep, d_keep))


def subsystem_purity(U, n, keep):
    """
    U    : unitary matrix
    keep : qubits to keep
    """
    D = 2**n

    # |psi> = U |0>
    psi0 = np.zeros(D)
    psi0[0] = 1
    psi = U @ psi0

    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace(rho, keep, n)

    return np.real(np.trace(rho_A @ rho_A))

def average_purity(unitaries, n, keep):
    return np.mean([subsystem_purity(U.toarray(), n, keep) for U in unitaries])

def page_purity(n, k):
    dA = 2**k
    dB = 2**(n-k)
    return (dA + dB) / (dA*dB + 1)


I = sp.csr_matrix([[1,0],[0,1]], dtype=complex)
X = sp.csr_matrix([[0,1],[1,0]], dtype=complex)
Y = sp.csr_matrix([[0,-1j],[1j,0]], dtype=complex)
Z = sp.csr_matrix([[1,0],[0,-1]], dtype=complex)

pauli_1q = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def pauli_string(paulis):
    U = pauli_1q[paulis[0]]
    for p in paulis[1:]:
        U = sp.kron(U, pauli_1q[p], format="csr")
    return U

def pauli_weight(paulis):
    return sum(p != 'I' for p in paulis)

def pauli_growth(U, n, P_string):
    """
    U        : unitary (sparse)
    P_string : e.g. ['X','I','I','I']
    """
    D = 2**n
    P = pauli_string(P_string)

    P_t = U.conj().T @ P @ U

    weights = {}
    norm = 0.0

    for paulis in itertools.product(['I','X','Y','Z'], repeat=n):
        Q = pauli_string(paulis)
        coeff = (Q.multiply(P_t)).diagonal().sum() / D
        prob = np.abs(coeff)**2

        w = pauli_weight(paulis)
        weights[w] = weights.get(w, 0) + prob
        norm += prob

    # normalize (numerical safety)
    for k in weights:
        weights[k] /= norm

    return weights


nqubits = 8
depth = 50
t_proportion = 0.2

purity = subsystem_purity(random_clifford_T_unitary(nqubits, depth, t_proportion).toarray(), nqubits, keep=[0,1,2,3])
print("Subsystem purity of qubits [0,1,2,3]:", purity)