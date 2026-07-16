"""
prism.gates
===========
Gate taxonomy for the PRISM framework.

Everything PRISM needs to reason about a gate *before* it touches a
statevector lives here: how many qubits it acts on (arity), whether it is
diagonal / particle-preserving / Clifford, its two-qubit operator Schmidt
rank, and the quasi-probability-decomposition (QPD) sampling overhead
``gamma`` with and without symmetry reduction.

These tables are the structural priors used by the partition-cost terms
(C3 ``log gamma`` overhead and the operator-entanglement edge weights).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#  Arity  (number of qubits a gate acts on)
# ---------------------------------------------------------------------------
GATE_ARITY: dict[str, int] = {
    # Single-qubit
    'I': 1, 'X': 1, 'Y': 1, 'Z': 1, 'H': 1, 'S': 1, 'SDG': 1, 'T': 1, 'TDG': 1,
    'RX': 1, 'RY': 1, 'RZ': 1, 'U1': 1, 'U2': 1, 'U3': 1, 'PHASE': 1, 'P': 1,
    'SX': 1, 'SXDG': 1, 'R': 1,
    # Two-qubit
    'CNOT': 2, 'CX': 2, 'CY': 2, 'CZ': 2, 'CH': 2,
    'CRX': 2, 'CRY': 2, 'CRZ': 2, 'CU1': 2, 'CU3': 2, 'CPHASE': 2,
    'SWAP': 2, 'ISWAP': 2,
    # Three-qubit
    'CCX': 3, 'CSWAP': 3,
    # Four-qubit (3 controls + 1 target)
    'C3X': 4, 'C3Z': 4,
}

# Backwards-friendly alias (legacy code imported ``quantum_gates``)
quantum_gates = GATE_ARITY

SINGLE = frozenset(g for g, a in GATE_ARITY.items() if a == 1)
TWO = frozenset(g for g, a in GATE_ARITY.items() if a == 2)
MULTI = frozenset(g for g, a in GATE_ARITY.items() if a > 2)

# ---------------------------------------------------------------------------
#  Symmetry / structural classes
# ---------------------------------------------------------------------------
DIAGONAL_GATES = frozenset({'Z', 'RZ', 'CZ', 'T', 'S', 'PHASE', 'U1', 'CU1', 'CPHASE', 'CRZ'})
PARTICLE_PRESERVING = frozenset({'SWAP', 'ISWAP', 'CZ', 'CRZ', 'CU1', 'CPHASE', 'U1', 'PHASE'})
CLIFFORD = frozenset({'H', 'X', 'Y', 'Z', 'S', 'CZ', 'CNOT', 'CX', 'CY', 'SWAP', 'CCX', 'CSWAP'})

# ---------------------------------------------------------------------------
#  Two-qubit operator Schmidt rank  (chi)  — knitting cost scales with chi,
#  not with the raw CNOT count (paper, observation 1).
# ---------------------------------------------------------------------------
SCHMIDT_RANK: dict[str, int] = {
    'CNOT': 2, 'CX': 2, 'CY': 2, 'CZ': 2, 'CH': 2,
    'CRX': 2, 'CRY': 2, 'CRZ': 2, 'CU1': 2, 'CU3': 4, 'CPHASE': 2,
    'SWAP': 4, 'ISWAP': 4,
    'CCX': 4, 'CSWAP': 8, 'C3X': 4, 'C3Z': 4,
}

# ---------------------------------------------------------------------------
#  QPD sampling overhead  gamma   (generic vs symmetry-reduced)
# ---------------------------------------------------------------------------
QPD_GAMMA: dict[str, float] = {
    'CNOT': 3.0, 'CX': 3.0, 'CY': 3.0, 'CZ': 3.0, 'CH': 3.0,
    'CRX': 3.0, 'CRY': 3.0, 'CRZ': 3.0, 'CU1': 3.0, 'CU3': 7.0, 'CPHASE': 3.0,
    'SWAP': 7.0, 'ISWAP': 7.0,
    'CCX': 9.0, 'CSWAP': 9.0, 'C3X': 9.0, 'C3Z': 9.0,
}
QPD_GAMMA_SYM: dict[str, float] = {
    'CNOT': 2.0, 'CX': 2.0, 'CY': 2.0, 'CZ': 2.0, 'CH': 2.0,
    'CRX': 2.0, 'CRY': 2.0, 'CRZ': 2.0, 'CU1': 2.0, 'CU3': 4.0, 'CPHASE': 2.0,
    'SWAP': 3.0, 'ISWAP': 3.0,
    'CCX': 4.0, 'CSWAP': 4.0, 'C3X': 4.0, 'C3Z': 4.0,
}

# Explicit quasi-probability decompositions for the common cut gates.
QPD_DECOMP: dict[str, list[tuple]] = {
    'CNOT': [(+0.50, 'I_A', 'I_B'), (+0.50, 'I_A', 'X_B'), (+0.50, 'Z_A', 'I_B'), (-0.50, 'Z_A', 'X_B')],
    'CZ':   [(+0.50, 'I_A', 'I_B'), (+0.50, 'I_A', 'Z_B'), (+0.50, 'Z_A', 'I_B'), (-0.50, 'Z_A', 'Z_B')],
    'SWAP': [(+0.25, 'I_A', 'I_B'), (+0.25, 'X_A', 'X_B'), (-0.25, 'Y_A', 'Y_B'), (+0.25, 'Z_A', 'Z_B')],
    'ISWAP': [(+0.25, 'I_A', 'I_B'), (+0.25, 'X_A', 'X_B'), (+0.25, 'Y_A', 'Y_B'), (-0.25, 'Z_A', 'Z_B')],
    'CRZ':  [(+0.50, 'I_A', 'I_B'), (+0.50, 'I_A', 'RZ_B'), (+0.50, 'Z_A', 'I_B'), (-0.50, 'Z_A', 'RZ_B')],
    'CRX':  [(+0.50, 'I_A', 'I_B'), (+0.50, 'I_A', 'RX_B'), (+0.50, 'Z_A', 'I_B'), (-0.50, 'Z_A', 'RX_B')],
    'CRY':  [(+0.50, 'I_A', 'I_B'), (+0.50, 'I_A', 'RY_B'), (+0.50, 'Z_A', 'I_B'), (-0.50, 'Z_A', 'RY_B')],
}

# Gates whose parameters negate under inversion (used when building U-dagger).
SIGN_FLIP_ON_INVERSE = frozenset({'RX', 'RY', 'RZ', 'U1', 'PHASE', 'P',
                                  'CRX', 'CRY', 'CRZ', 'CU1', 'CPHASE'})


def gate_arity(name: str) -> int:
    """Number of qubits the named gate acts on (defaults to 1 if unknown)."""
    return GATE_ARITY.get(str(name).upper(), 1)


def is_two_qubit(name: str) -> bool:
    return gate_arity(name) == 2


def schmidt_rank(name: str) -> int:
    """Two-qubit operator Schmidt rank (chi). Defaults to 4 for unknown 2q gates."""
    return int(SCHMIDT_RANK.get(str(name).upper(), 4))


def qpd_gamma(name: str, symmetry_reduced: bool = False) -> float:
    """QPD sampling factor gamma for one cut of this gate."""
    name = str(name).upper()
    table = QPD_GAMMA_SYM if symmetry_reduced else QPD_GAMMA
    return float(table.get(name, 3.0 if symmetry_reduced else 9.0))


def gate_type_code(name: str) -> int:
    """0 idle, 1 single, 2 two-qubit, 3 multi — used for occupancy heatmaps."""
    n = str(name).upper()
    if n in SINGLE:
        return 1
    if n in TWO:
        return 2
    if n in MULTI:
        return 3
    return 0


__all__ = [
    'GATE_ARITY', 'quantum_gates', 'SINGLE', 'TWO', 'MULTI',
    'DIAGONAL_GATES', 'PARTICLE_PRESERVING', 'CLIFFORD',
    'SCHMIDT_RANK', 'QPD_GAMMA', 'QPD_GAMMA_SYM', 'QPD_DECOMP',
    'SIGN_FLIP_ON_INVERSE',
    'gate_arity', 'is_two_qubit', 'schmidt_rank', 'qpd_gamma', 'gate_type_code',
]
