"""
prism.distributed
==================
Exact distributed execution by **gate teleportation** (the entanglement-assisted
paradigm), as a foil to PRISM's quasi-probability circuit cutting.

A two-qubit gate whose qubits live on different QPUs can be executed *exactly*
with a shared Bell pair (one **ebit**) plus two classical bits — the textbook
non-local CNOT (cat-entangler / cat-disentangler, Eisert et al. 2000):

    Bell pair on (e1,e2);  A: CX(c,e1), measure e1 -> m1;  send m1 to B;
    B: if m1 X(e2), CX(e2,t), H(e2), measure e2 -> m2;     send m2 to A;
    A: if m2 Z(c).                                          # = CNOT(c,t)

It is local-operations-and-classical-communication (LOCC) plus one ebit, so it
respects QPU boundaries. :func:`nonlocal_cnot` appends it (with mid-circuit
measurement + classical feedforward); :func:`build_distributed_circuit` realises
a whole layout under a bipartition, teleporting every cross CNOT/CZ and reusing
one ebit pair via reset. The ebit cost equals the number of cross gates — which
is exactly what a good PRISM partition minimises.
"""
from __future__ import annotations

import numpy as np

from .graph import classify_gates

_TWO_LOCAL_CROSS = {'CNOT', 'CX', 'CZ'}


def nonlocal_cnot(qc, c, t, e1, e2, cbit1, cbit2, reset_ancillas=True):
    """Append a teleported CNOT(c -> t) using ebit ancillas (e1, e2) and two
    classical bits. Mid-circuit measurement + classical feedforward (LOCC + 1
    ebit). With ``reset_ancillas`` the ancillas are reset to |0> for reuse."""
    qc.h(e1)
    qc.cx(e1, e2)                       # shared Bell pair (the ebit)
    qc.cx(c, e1)                        # A-local: entangle control into e1
    qc.measure(e1, cbit1)               # A measures e1  -> m1 (1 cbit A->B)
    with qc.if_test((cbit1, 1)):
        qc.x(e2)                        # B applies correction
    qc.cx(e2, t)                        # B-local: controlled-X on the target
    qc.h(e2)
    qc.measure(e2, cbit2)               # B measures e2 -> m2 (1 cbit B->A)
    with qc.if_test((cbit2, 1)):
        qc.z(c)                         # A applies correction
    if reset_ancillas:
        qc.reset(e1)
        qc.reset(e2)


def nonlocal_cz(qc, q1, q2, e1, e2, cbit1, cbit2, reset_ancillas=True):
    """Teleported CZ via H . CNOT . H on the target side (one ebit)."""
    qc.h(q2)
    nonlocal_cnot(qc, q1, q2, e1, e2, cbit1, cbit2, reset_ancillas=reset_ancillas)
    qc.h(q2)


def _aer_counts(qc, shots, seed=0):
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator()
    tq = transpile(qc, sim)
    return sim.run(tq, shots=shots, seed_simulator=seed).result().get_counts()


def verify_nonlocal_cnot(shots=20000, seed=0):
    """Sample-verify the teleported CNOT against a local CNOT over all four
    computational inputs and a superposition. Returns (ok, max_TVD)."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    max_tvd = 0.0
    inputs = [('00', []), ('01', [('x', 1)]), ('10', [('x', 0)]),
              ('11', [('x', 0), ('x', 1)]), ('+0', [('h', 0)]), ('+1', [('h', 0), ('x', 1)])]
    for _, prep in inputs:
        # distributed
        q = QuantumRegister(4, 'q'); cm = ClassicalRegister(2, 'm'); cr = ClassicalRegister(2, 'r')
        qc = QuantumCircuit(q, cm, cr)
        for g, w in prep:
            getattr(qc, g)(q[w])
        nonlocal_cnot(qc, q[0], q[1], q[2], q[3], cm[0], cm[1])
        qc.measure(q[0], cr[0]); qc.measure(q[1], cr[1])
        dist_counts = _aer_counts(qc, shots, seed)
        # reference local CNOT
        q2 = QuantumRegister(2, 'q'); c2 = ClassicalRegister(2, 'r')
        ref = QuantumCircuit(q2, c2)
        for g, w in prep:
            getattr(ref, g)(q2[w])
        ref.cx(q2[0], q2[1]); ref.measure(q2[0], c2[0]); ref.measure(q2[1], c2[1])
        ref_counts = _aer_counts(ref, shots, seed)

        def to_p(counts):
            p = np.zeros(4)
            for k, v in counts.items():
                bits = k.split()[0]              # readout register (last-added => first token)
                p[int(bits, 2)] += v
            return p / p.sum()
        max_tvd = max(max_tvd, 0.5 * np.sum(np.abs(to_p(dist_counts) - to_p(ref_counts))))
    return (max_tvd < 0.03), float(max_tvd)


def build_distributed_circuit(layout, A_set, B_set, n_qubits, measure=True):
    """Build a dynamic circuit executing ``layout`` under bipartition (A,B),
    teleporting every cross CNOT/CZ with one reused ebit pair.

    Returns ``(qc, info)``; ``info`` has the ebit/cbit cost. ``qc`` uses
    mid-circuit measurement + feedforward, so simulate it by sampling
    (:func:`distributed_sample_probs`), not statevector."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from .compiler import append_gate
    A_set, B_set = set(A_set), set(B_set)
    cross = [g for layer in layout for g in layer
             if len(g['qubits']) >= 2 and not (set(g['qubits']).issubset(A_set)
                                               or set(g['qubits']).issubset(B_set))]
    n_cross = len(cross)
    qr = QuantumRegister(n_qubits, 'q')
    er = QuantumRegister(2, 'e')
    regs = [qr, er]
    prot = ClassicalRegister(max(2, 2), 'm')
    regs.append(prot)
    rd = ClassicalRegister(n_qubits, 'r') if measure else None
    if rd is not None:
        regs.append(rd)
    qc = QuantumCircuit(*regs)
    ebits = unsupported = 0
    for li, layer in enumerate(layout):
        for gi, gate in enumerate(layer):
            name = gate['gate'].upper()
            qs = gate['qubits']
            crosses = len(qs) >= 2 and not (set(qs).issubset(A_set) or set(qs).issubset(B_set))
            if not crosses:
                g = {**gate, 'qubits': [qr[q] for q in qs],
                     'source_layer': gate.get('source_layer', li),
                     'source_gate_index': gate.get('source_gate_index', gi)}
                append_gate(qc, g)
                continue
            if name in ('CNOT', 'CX') and len(qs) == 2:
                nonlocal_cnot(qc, qr[qs[0]], qr[qs[1]], er[0], er[1], prot[0], prot[1]); ebits += 1
            elif name == 'CZ' and len(qs) == 2:
                nonlocal_cz(qc, qr[qs[0]], qr[qs[1]], er[0], er[1], prot[0], prot[1]); ebits += 1
            else:
                unsupported += 1
    if measure:
        for q in range(n_qubits):
            qc.measure(qr[q], rd[q])
    info = {'n_ebits': ebits, 'n_cbits': 2 * ebits, 'cross_gates': n_cross,
            'cross_unsupported': unsupported, 'total_qubits': n_qubits + 2}
    return qc, info


def distributed_sample_probs(layout, A_set, B_set, n_qubits, shots=20000, seed=0):
    """Sample the teleported distributed circuit; return (data distribution, info)."""
    qc, info = build_distributed_circuit(layout, A_set, B_set, n_qubits, measure=True)
    if info['cross_unsupported'] > 0:
        return None, info
    counts = _aer_counts(qc, shots, seed)
    p = np.zeros(2 ** n_qubits)
    for k, v in counts.items():
        bits = k.split()[0]                      # readout register 'r' is the last-added => first token
        p[int(bits, 2)] += v
    s = p.sum()
    return (p / s if s > 0 else p), info


def count_cross_gates(layout, A_set, B_set):
    """Cross-partition gate count = ebits for exact distributed execution."""
    return len(classify_gates(layout, A_set, B_set)['cross'])


__all__ = [
    'nonlocal_cnot', 'nonlocal_cz', 'verify_nonlocal_cnot',
    'build_distributed_circuit', 'distributed_sample_probs', 'count_cross_gates',
]
