"""
prism.partition.cost
=====================
The multi-objective partition cost (paper eq. 5 / 23).

A candidate bipartition (A, B) is scored by a convex aggregation of five
terms:

    C1  S(A)            entanglement entropy across the cut
    C2  I(A:B)          classical mutual information of the marginals
    C3  log gamma_sym   symmetry-reduced QPD sampling overhead
    C4  W_cut(A,B)      weighted interaction-graph cut
    C5  Delta(A,B)      balance penalty

When the full statevector is available (n <= ~22) the physics terms C1/C2 are
exact; otherwise a structural surrogate (C3-C5 only) is used.  Lower is better.
"""
from __future__ import annotations

import math
import numpy as np

from ..gates import QPD_GAMMA_SYM
from ..graph import classify_gates
from ..simulate import (prob_from_statevector, entanglement_entropy, build_recon_index)

# default convex weights (paper eq. 23)
W_ENT, W_MI, W_GAMMA, W_CUT, W_CROSS, W_BAL = 1.00, 0.70, 0.45, 0.20, 0.12, 0.08


def _cut_weight(G, A_set):
    cw = 0.0
    for u, v, d in G.edges(data=True):
        if (u in A_set) != (v in A_set):
            cw += float(d.get('weight', 1.0))
    return cw


def _gamma_log(cross_gates):
    g = 0.0
    for gate in cross_gates:
        g += math.log(max(QPD_GAMMA_SYM.get(gate['gate'].upper(), 3.0), 1.0))
    return g


def partition_cost_surrogate(A_set, B_set, G, layout, n_qubits):
    """Structural surrogate cost (no statevector): C3 + C4 + C5 only."""
    A_set = set(A_set)
    cut_weight = _cut_weight(G, A_set)
    gc = classify_gates(layout, A_set, B_set)
    cross = len(gc['cross'])
    gamma_log = _gamma_log(gc['cross'])
    balance = abs(len(A_set) - len(set(B_set))) / max(n_qubits, 1)
    return float(0.55 * math.log1p(cut_weight) + 0.25 * cross + 0.10 * gamma_log + 0.10 * balance)


def compute_partition_cost(A_set, B_set, G, layout, n_qubits, sv_full=None, p_full=None):
    """Return ``(score, detail)``. ``detail`` carries every C-term so callers
    can inspect the multi-objective breakdown. Lower score is better."""
    A_sorted = sorted(A_set)
    B_sorted = sorted(B_set)
    A_set, B_set = set(A_set), set(B_set)
    gc = classify_gates(layout, A_set, B_set)

    cut_weight = _cut_weight(G, A_set)
    gamma_log = _gamma_log(gc['cross'])
    balance = abs(len(A_set) - len(B_set)) / max(n_qubits, 1)
    total_gates = sum(len(layer) for layer in layout)
    cross_frac = len(gc['cross']) / max(total_gates, 1)

    detail = {
        'cut_weight': float(cut_weight),
        'n_cross_gates': len(gc['cross']),
        'gamma_log_sym': float(gamma_log),
        'balance_penalty': float(balance),
        'cross_gate_fraction': float(cross_frac),
        'entanglement_entropy': None,
        'mutual_information': None,
        'mode': 'surrogate',
    }

    if sv_full is None:
        score = partition_cost_surrogate(A_set, B_set, G, layout, n_qubits)
        detail['score'] = float(score)
        return float(score), detail

    if p_full is None:
        p_full = prob_from_statevector(sv_full)

    ent = entanglement_entropy(sv_full, A_sorted, n_qubits)
    idxA = build_recon_index(A_sorted, n_qubits)
    idxB = build_recon_index(B_sorted, n_qubits)
    pA = np.bincount(idxA, weights=p_full, minlength=1 << len(A_sorted)).astype(float)
    pB = np.bincount(idxB, weights=p_full, minlength=1 << len(B_sorted)).astype(float)
    if pA.sum() > 0:
        pA /= pA.sum()
    if pB.sum() > 0:
        pB /= pB.sum()
    eps = 1e-15
    pA_nz, pB_nz, pF_nz = pA[pA > eps], pB[pB > eps], p_full[p_full > eps]
    HA = float(-np.sum(pA_nz * np.log(pA_nz + eps))) if pA_nz.size else 0.0
    HB = float(-np.sum(pB_nz * np.log(pB_nz + eps))) if pB_nz.size else 0.0
    H = float(-np.sum(pF_nz * np.log(pF_nz + eps))) if pF_nz.size else 0.0
    mi = max(HA + HB - H, 0.0)

    score = (W_ENT * ent + W_MI * mi + W_GAMMA * gamma_log
             + W_CUT * math.log1p(cut_weight) + W_CROSS * cross_frac + W_BAL * balance)

    detail.update({'score': float(score), 'entanglement_entropy': float(ent),
                   'mutual_information': float(mi), 'mode': 'physics'})
    return float(score), detail


def partition_cost_terms(A_set, B_set, G, layout, n_qubits, sv_full=None, p_full=None):
    """Just the five C-terms (C1..C5) as a dict — for ablation / insight plots."""
    _, d = compute_partition_cost(A_set, B_set, G, layout, n_qubits, sv_full, p_full)
    return {
        'C1_entanglement_entropy': d['entanglement_entropy'],
        'C2_mutual_information': d['mutual_information'],
        'C3_log_gamma_sym': d['gamma_log_sym'],
        'C4_cut_weight': d['cut_weight'],
        'C5_balance_penalty': d['balance_penalty'],
        'cross_gate_fraction': d['cross_gate_fraction'],
        'mode': d['mode'],
    }


__all__ = ['compute_partition_cost', 'partition_cost_surrogate', 'partition_cost_terms']
