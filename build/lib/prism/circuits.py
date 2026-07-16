"""
prism.circuits
==============
Circuit-family generators and the random-ansatz builder.

Every generator returns a *layout* (``list[list[dict]]``) and is fully
reproducible given a ``random.Random`` instance.  The nine families mirror
the benchmark in the PRISM paper:

    QFT, QPE, VQE-HEA, VQE-UCC, QAOA, QNN, MERA, MPS, OTOC

plus ``RQC`` (random quantum circuit) via :func:`generate_ansatz_layout`.
"""
from __future__ import annotations

import math
import random as _random

from .gates import GATE_ARITY, SINGLE, TWO, MULTI


# ---------------------------------------------------------------------------
#  Structured circuit families
# ---------------------------------------------------------------------------
def gen_qft(n, rng=None):
    layout = []
    for i in range(n):
        layout.append([{'gate': 'H', 'qubits': [i]}])
        for j in range(i + 1, n):
            layout.append([{'gate': 'CPHASE', 'qubits': [j, i],
                            'params': [math.pi / (2 ** (j - i))]}])
    for i in range(n // 2):
        layout.append([{'gate': 'SWAP', 'qubits': [i, n - 1 - i]}])
    return layout


def gen_qpe(n, rng=None):
    n_anc = max(n // 2, 2)
    n_sys = n - n_anc
    layout = [[{'gate': 'H', 'qubits': [i]} for i in range(n_anc)]]
    if n_sys > 0:
        layout.append([{'gate': 'X', 'qubits': [n_anc]}])
    rng = rng or _random
    for k in range(n_anc):
        for s in range(n_sys):
            angle = rng.uniform(0.5, 1.5) * math.pi / (2 ** k)
            layout.append([{'gate': 'CRZ', 'qubits': [k, n_anc + s], 'params': [angle]}])
    for i in range(n_anc - 1, -1, -1):
        for j in range(i + 1, n_anc):
            layout.append([{'gate': 'CPHASE', 'qubits': [j, i],
                            'params': [-math.pi / (2 ** (j - i))]}])
        layout.append([{'gate': 'H', 'qubits': [i]}])
    return layout


def gen_vqe_hea(n, rng=None, depth=3):
    rng = rng or _random
    layout = []
    for _ in range(depth):
        layout.append([{'gate': 'RY', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
        layout.append([{'gate': 'RZ', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
        for i in range(n - 1):
            layout.append([{'gate': 'CNOT', 'qubits': [i, i + 1]}])
    layout.append([{'gate': 'RY', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
    return layout


def gen_vqe_ucc(n, rng=None):
    rng = rng or _random
    layout = []
    for i in range(n // 2):
        layout.append([{'gate': 'X', 'qubits': [i]}])
    for i in range(n - 1):
        layout.append([{'gate': 'CNOT', 'qubits': [i, i + 1]}])
        layout.append([{'gate': 'RY', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]}])
        layout.append([{'gate': 'CNOT', 'qubits': [i, i + 1]}])
    for i in range(0, n - 3, 2):
        layout.append([{'gate': 'CNOT', 'qubits': [i, i + 1]}])
        layout.append([{'gate': 'CNOT', 'qubits': [i + 2, i + 3]}])
        layout.append([{'gate': 'CNOT', 'qubits': [i + 1, i + 2]}])
        layout.append([{'gate': 'RZ', 'qubits': [i + 2], 'params': [rng.uniform(0, 2 * math.pi)]}])
        layout.append([{'gate': 'CNOT', 'qubits': [i + 1, i + 2]}])
        layout.append([{'gate': 'CNOT', 'qubits': [i + 2, i + 3]}])
        layout.append([{'gate': 'CNOT', 'qubits': [i, i + 1]}])
    return layout


def gen_qaoa(n, rng=None, p=2):
    """Ring-topology MaxCut QAOA (generic family member). For an arbitrary
    problem graph use :func:`prism.qml.qaoa_maxcut_layout`."""
    rng = rng or _random
    layout = [[{'gate': 'H', 'qubits': [i]} for i in range(n)]]
    for _ in range(p):
        gamma, beta = rng.uniform(0, math.pi), rng.uniform(0, math.pi)
        for i in range(n):
            j = (i + 1) % n
            if i < j:
                layout.append([{'gate': 'CNOT', 'qubits': [i, j]}])
                layout.append([{'gate': 'RZ', 'qubits': [j], 'params': [2 * gamma]}])
                layout.append([{'gate': 'CNOT', 'qubits': [i, j]}])
        layout.append([{'gate': 'RX', 'qubits': [i], 'params': [2 * beta]} for i in range(n)])
    return layout


def gen_qnn(n, rng=None, depth=3):
    rng = rng or _random
    layout = [[{'gate': 'RY', 'qubits': [i], 'params': [rng.uniform(0, math.pi)]} for i in range(n)]]
    for _ in range(depth):
        for axis in ('RX', 'RY', 'RZ'):
            layout.append([{'gate': axis, 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
        for i in range(n - 1):
            layout.append([{'gate': 'CZ', 'qubits': [i, i + 1]}])
        if n > 2:
            layout.append([{'gate': 'CZ', 'qubits': [0, n - 1]}])
    return layout


def gen_mera(n, rng=None):
    rng = rng or _random
    layout = [[{'gate': 'H', 'qubits': [i]} for i in range(n)]]
    scale = 1
    while scale < n:
        d_layer, used = [], set()
        for i in range(scale, n - 1, 2 * scale):
            q1, q2 = i, i + 1
            if q2 < n and q1 not in used and q2 not in used:
                d_layer.append({'gate': 'CRZ', 'qubits': [q1, q2], 'params': [rng.uniform(0, 2 * math.pi)]})
                used.update([q1, q2])
        if d_layer:
            layout.append(d_layer)
        i_layer, used = [], set()
        for i in range(0, n - scale, 2 * scale):
            q1, q2 = i, i + scale
            if q2 < n and q1 not in used and q2 not in used:
                i_layer.append({'gate': 'CRY', 'qubits': [q1, q2], 'params': [rng.uniform(0, 2 * math.pi)]})
                used.update([q1, q2])
        if i_layer:
            layout.append(i_layer)
        scale *= 2
    return layout


def gen_mps(n, rng=None, depth=3):
    rng = rng or _random
    layout = [[{'gate': 'RY', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)]]
    for _ in range(depth):
        even = [{'gate': 'CNOT', 'qubits': [i, i + 1]} for i in range(0, n - 1, 2)]
        if even:
            layout.append(even)
        layout.append([{'gate': 'RY', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
        odd = [{'gate': 'CNOT', 'qubits': [i, i + 1]} for i in range(1, n - 1, 2)]
        if odd:
            layout.append(odd)
        layout.append([{'gate': 'RZ', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
    return layout


def gen_otoc(n, rng=None, depth=2, q_butterfly=None, q_perturb=None):
    """Out-of-Time-Order Correlator: |+>^n . U . W . U-dagger . V . U.

    The forward-inverse-forward time symmetry makes this the only family with
    near-cancelling deep dynamics — a stress test for light-cone methods.
    """
    rng = rng or _random
    if q_butterfly is None:
        q_butterfly = n // 2
    if q_perturb is None:
        q_perturb = 0
    layout = [[{'gate': 'H', 'qubits': [i]} for i in range(n)]]

    def _build_U():
        sub = []
        for _ in range(depth):
            sub.append([{'gate': 'RZ', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
            sub.append([{'gate': 'RX', 'qubits': [i], 'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
            even = [{'gate': 'CZ', 'qubits': [i, i + 1]} for i in range(0, n - 1, 2)]
            if even:
                sub.append(even)
            odd = [{'gate': 'CZ', 'qubits': [i, i + 1]} for i in range(1, n - 1, 2)]
            if odd:
                sub.append(odd)
        return sub

    def _copy(layers):
        out = []
        for layer in layers:
            nl = []
            for g in layer:
                gd = {'gate': g['gate'], 'qubits': list(g['qubits'])}
                if 'params' in g:
                    gd['params'] = list(g['params'])
                nl.append(gd)
            out.append(nl)
        return out

    def _invert(layers):
        from .gates import SIGN_FLIP_ON_INVERSE
        inv = []
        for layer in reversed(layers):
            nl = []
            for g in reversed(layer):
                name = g['gate'].upper()
                gd = {'gate': g['gate'], 'qubits': list(g['qubits'])}
                if g.get('params'):
                    gd['params'] = [-p for p in g['params']] if name in SIGN_FLIP_ON_INVERSE else list(g['params'])
                nl.append(gd)
            inv.append(nl)
        return inv

    U = _build_U()
    Ud = _invert(U)
    layout.extend(_copy(U))
    layout.append([{'gate': 'X', 'qubits': [q_butterfly]}])
    layout.extend(Ud)
    layout.append([{'gate': 'X', 'qubits': [q_perturb]}])
    layout.extend(_copy(U))
    return layout


# ---------------------------------------------------------------------------
#  Random ansatz  (RQC family)
# ---------------------------------------------------------------------------
def generate_ansatz_layout(n_qubits, depth, max_qubits_per_layer=4, rng=None):
    """Random-circuit layout. Numeric params are sampled once and stored, so
    the same gate yields the same unitary in full and sub circuits."""
    R = rng if rng is not None else _random
    layout = []
    gate_items = list(GATE_ARITY.items())
    for _ in range(depth):
        layer, used = [], set()
        items = list(gate_items)
        R.shuffle(items)
        for gate, arity in items:
            available = [q for q in range(n_qubits) if q not in used]
            if len(available) >= arity:
                selected = sorted(R.sample(available, arity))
                proposed = used.union(selected)
                if len(proposed) <= max_qubits_per_layer:
                    gd = {'gate': gate, 'qubits': selected}
                    if gate in ('RX', 'RY', 'RZ', 'PHASE', 'U1', 'CU1', 'CPHASE', 'CRX', 'CRY', 'CRZ'):
                        gd['params'] = [R.uniform(0, 2 * math.pi)]
                    elif gate == 'U2':
                        gd['params'] = [R.uniform(0, 2 * math.pi), R.uniform(0, 2 * math.pi)]
                    elif gate in ('U3', 'CU3'):
                        gd['params'] = [R.uniform(0, 2 * math.pi) for _ in range(3)]
                    elif gate == 'R':
                        gd['params'] = [R.uniform(0, 1), R.uniform(0, 2 * math.pi)]
                    layer.append(gd)
                    used = proposed
        layout.append(layer)
    return layout


# ---------------------------------------------------------------------------
#  Registry & helpers
# ---------------------------------------------------------------------------
RQC_DEPTH = 12

CIRCUIT_FAMILIES = {
    'QFT': gen_qft, 'QPE': gen_qpe, 'VQE-HEA': gen_vqe_hea, 'VQE-UCC': gen_vqe_ucc,
    'QAOA': gen_qaoa, 'QNN': gen_qnn, 'MERA': gen_mera, 'MPS': gen_mps, 'OTOC': gen_otoc,
}

FAMILY_SEEDS = {
    'QFT': 100, 'QPE': 200, 'VQE-HEA': 300, 'VQE-UCC': 400, 'QAOA': 500,
    'QNN': 600, 'MERA': 700, 'MPS': 800, 'RQC': 900, 'OTOC': 1000,
}


def make_layout(family, n, seed, rqc_depth=RQC_DEPTH):
    """Reproducible layout for ``family`` at ``n`` qubits.

    The RNG convention matches the paper benchmark:
    ``rng = Random(seed + n*17 + family_seed)``.
    """
    fam_seed = FAMILY_SEEDS.get(family, 0)
    rng = _random.Random(seed + n * 17 + fam_seed)
    if family == 'RQC':
        return generate_ansatz_layout(n, rqc_depth, max_qubits_per_layer=n, rng=rng)
    gen = CIRCUIT_FAMILIES[family]
    return gen(n, rng)


def layout_gate_count(layout):
    return sum(len(layer) for layer in layout)


def layout_depth(layout):
    return len(layout)


def layout_stats(layout, n_qubits=None):
    """Summary dict: gate count, depth, single/two/multi-qubit gate counts."""
    s = t = m = 0
    for layer in layout:
        for g in layer:
            name = g['gate'].upper()
            if name in SINGLE:
                s += 1
            elif name in TWO:
                t += 1
            elif name in MULTI:
                m += 1
    return {
        'n_qubits': n_qubits,
        'depth': len(layout),
        'total_gates': s + t + m,
        'single_qubit_gates': s,
        'two_qubit_gates': t,
        'multi_qubit_gates': m,
    }


__all__ = [
    'gen_qft', 'gen_qpe', 'gen_vqe_hea', 'gen_vqe_ucc', 'gen_qaoa',
    'gen_qnn', 'gen_mera', 'gen_mps', 'gen_otoc', 'generate_ansatz_layout',
    'CIRCUIT_FAMILIES', 'FAMILY_SEEDS', 'RQC_DEPTH',
    'make_layout', 'layout_gate_count', 'layout_depth', 'layout_stats',
]
