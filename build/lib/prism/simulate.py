"""
prism.simulate
==============
Exact statevector simulation, subcircuit reconstruction, and the
distribution-comparison metrics used to score a partition.

For a bipartition (A, B) we simulate the two local subcircuits (cross gates
dropped), form the product reconstruction ``p_recon = p_A x p_B`` on the full
register, and compare it to the ideal full-circuit distribution with six
metrics (TVD, fidelity, KL, Hellinger, JS, cross-entropy) and the unified
``Q_Score`` from the paper.
"""
from __future__ import annotations

import numpy as np

from .compiler import materialize_layout
from .graph import build_subcircuit_layout


# ---------------------------------------------------------------------------
#  Statevector helpers
# ---------------------------------------------------------------------------
def get_statevector(qc):
    """Exact statevector of a QuantumCircuit (numpy array), or None on failure."""
    try:
        from qiskit.quantum_info import Statevector
        return np.array(Statevector(qc).data)
    except Exception:
        pass
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        sim = AerSimulator(method='statevector')
        q = qc.copy()
        q.save_statevector()
        return np.array(sim.run(transpile(q, sim)).result().get_statevector())
    except Exception:
        return None


def prob_from_statevector(sv):
    p = np.abs(sv) ** 2
    s = p.sum()
    return p / s if s > 0 else p


def statevector_probabilities(layout, n_qubits):
    """Convenience: ideal output distribution of a layout via exact SV."""
    sv = get_statevector(materialize_layout(layout, n_qubits))
    return prob_from_statevector(sv) if sv is not None else None


def entanglement_entropy(statevector, A_qubits, n_qubits):
    """Von Neumann entropy S(A) = -Tr rho_A log2 rho_A across the cut."""
    n = n_qubits
    A_qubits = list(A_qubits)
    nA = len(A_qubits)
    nB = n - nA
    B_qubits = [q for q in range(n) if q not in A_qubits]
    order = A_qubits + B_qubits
    t = statevector.reshape([2] * n)
    m = np.transpose(t, order).reshape(2 ** nA, 2 ** nB)
    s = np.linalg.svd(m, compute_uv=False)
    probs = s ** 2
    probs = probs[probs > 1e-14]
    return round(max(float(-np.sum(probs * np.log2(probs + 1e-15))), 0.0), 6)


def build_recon_index(sorted_nodes, n_qubits):
    """Map every full-register basis index to the local index of a subset."""
    n_full = 2 ** n_qubits
    all_idx = np.arange(n_full, dtype=np.int64)
    sub_idx = np.zeros(n_full, dtype=np.int64)
    for k, q in enumerate(sorted_nodes):
        sub_idx += ((all_idx >> q) & 1) << k
    return sub_idx


def simulate_subcircuit(A_set, B_set, layout, n_qubits, side):
    """Exact marginal distribution of one local subcircuit (cross gates dropped)."""
    node_set = A_set if side == 'A' else B_set
    sub_layout, sorted_nodes, _ = build_subcircuit_layout(layout, node_set, side)
    qc = materialize_layout(sub_layout, len(node_set))
    sv = get_statevector(qc)
    return (prob_from_statevector(sv), sorted_nodes) if sv is not None else (None, sorted_nodes)


def reconstruct_product(A_set, B_set, layout, n_qubits):
    """Uncorrected product reconstruction  p_hat = p_A (x) p_B  on the full
    register (cross-partition gates dropped) — the partition-quality probe."""
    pA, sA = simulate_subcircuit(A_set, B_set, layout, n_qubits, 'A')
    pB, sB = simulate_subcircuit(A_set, B_set, layout, n_qubits, 'B')
    if pA is None or pB is None:
        return None
    p = pA[build_recon_index(sA, n_qubits)] * pB[build_recon_index(sB, n_qubits)]
    s = p.sum()
    return p / s if s > 0 else None


def reconstruct_product_kway(parts, layout, n_qubits):
    """Product reconstruction over ``k`` fragments: p_hat = p_1 (x) ... (x) p_k
    on the full register (all inter-fragment gates dropped)."""
    from .graph import build_subcircuit_layout
    p = np.ones(2 ** n_qubits, dtype=float)
    for S in parts:
        S = set(S)
        if not S:
            continue
        sub_layout, sorted_nodes, _ = build_subcircuit_layout(layout, S, 'S')
        sv = get_statevector(materialize_layout(sub_layout, len(S)))
        if sv is None:
            return None
        pS = prob_from_statevector(sv)
        p = p * pS[build_recon_index(sorted_nodes, n_qubits)]
    s = p.sum()
    return p / s if s > 0 else None


# ---------------------------------------------------------------------------
#  Distribution metrics
# ---------------------------------------------------------------------------
def distribution_metrics(p_ideal, p_recon):
    """Six standard distribution-comparison metrics between two distributions."""
    eps = 1e-15
    p_i = np.clip(p_ideal, eps, None)
    p_r = np.clip(p_recon, eps, None)
    tvd = float(0.5 * np.sum(np.abs(p_ideal - p_recon)))
    fid = float(np.sum(np.sqrt(np.clip(p_ideal, 0, None) * np.clip(p_recon, 0, None))))
    mask = p_ideal > eps
    kl = float(np.sum(p_ideal[mask] * np.log(p_ideal[mask] / p_r[mask])))
    hell = float(np.sqrt(0.5 * np.sum((np.sqrt(p_ideal) - np.sqrt(p_recon)) ** 2)))
    m = 0.5 * (p_i + p_r)
    js = float(0.5 * (np.sum(p_i * np.log(p_i / m)) + np.sum(p_r * np.log(p_r / m))))
    js = max(js, 0.0)
    ce = float(-np.sum(p_ideal * np.log(p_r)))
    return {'tvd': tvd, 'fidelity': fid, 'kl_divergence': kl,
            'hellinger': hell, 'js_divergence': js, 'cross_entropy': ce}


def q_score(metrics):
    """Paper's unified Q_Score (absolute, in [0,1], higher is better):

        Q = (Fid . A . B . C)^(1/6) . D^(1/3)
        A = 1 - TVD, B = 1 - He, C = 1 - JS/log2, D = e^(-KL).
    """
    import math
    tvd = metrics['tvd']
    he = metrics['hellinger']
    js = metrics['js_divergence']
    kl = metrics['kl_divergence']
    fid = metrics['fidelity']
    A = max(1.0 - tvd, 0.0)
    B = max(1.0 - he, 0.0)
    C = max(1.0 - js / math.log(2), 0.0)
    D = math.exp(-max(kl, 0.0))
    inner = max(fid * A * B * C, 0.0)
    return float((inner ** (1.0 / 6.0)) * (D ** (1.0 / 3.0)))


def compute_unified_scores(results_dict):
    """Relative min-max-normalised weighted score + rank across methods
    (used for per-instance ranking; complements the absolute q_score)."""
    weights = {'tvd': 0.25, 'fidelity': 0.25, 'kl_divergence': 0.15,
               'hellinger': 0.10, 'js_divergence': 0.15, 'cross_entropy': 0.10}
    direction = {'tvd': False, 'fidelity': True, 'kl_divergence': False,
                 'hellinger': False, 'js_divergence': False, 'cross_entropy': False}
    valid = {k: v for k, v in results_dict.items() if v.get('error') is None and 'tvd' in v}
    if len(valid) < 2:
        for k in valid:
            valid[k]['unified_score'] = 1.0
            valid[k]['rank'] = 1
        return valid
    metrics = list(weights.keys())
    raw = {met: [valid[m][met] for m in valid] for met in metrics}
    for met in metrics:
        vals = np.array(raw[met], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        normed = np.full_like(vals, 0.5) if vmax - vmin < 1e-12 else (vals - vmin) / (vmax - vmin)
        if not direction[met]:
            normed = 1.0 - normed
        for i, m in enumerate(valid):
            valid[m].setdefault('_norm', {})[met] = float(normed[i])
    for m in valid:
        score = sum(weights[met] * valid[m]['_norm'][met] for met in metrics)
        valid[m]['unified_score'] = round(float(score), 6)
        del valid[m]['_norm']
    ranked = sorted(valid.items(), key=lambda kv: kv[1]['unified_score'], reverse=True)
    for rank, (m, data) in enumerate(ranked, start=1):
        data['rank'] = rank
    return valid


__all__ = [
    'get_statevector', 'prob_from_statevector', 'statevector_probabilities',
    'entanglement_entropy', 'build_recon_index', 'simulate_subcircuit',
    'reconstruct_product', 'reconstruct_product_kway',
    'distribution_metrics', 'q_score', 'compute_unified_scores',
]
