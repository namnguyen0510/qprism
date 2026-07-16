"""
prism.partition.ladder
=======================
The PRISM ablation ladder — four single-mechanism variants that isolate the
contribution of each idea, culminating in PRISM-LCT (in ``lct.py``):

    PRISM-KL   Kernighan-Lin on the gate-count interaction graph.
               No entanglement awareness, no symmetry, no SA.
    PRISM-OE   + Operator-Entanglement edge weighting, then structural SA.
    PRISM-MI   + physics cost with entanglement entropy + Mutual Information
               (statevector-aware SA).
    PRISM-BF   + Boundary-Focused move proposals (swaps drawn from cut-incident
               qubits), same physics cost.

All variants share the simulated-annealing engine :func:`optimize_partition_sa`
and the multi-objective cost in :mod:`prism.partition.cost`; only the search
graph and move policy change rung to rung.
"""
from __future__ import annotations

import math
import random

from ..graph import estimate_edge_entanglement
from ..simulate import get_statevector, prob_from_statevector
from .cost import compute_partition_cost

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
#  PRISM-KL : Kernighan-Lin bisection on the gate-count interaction graph
# ---------------------------------------------------------------------------
def kl_interaction_partition(G, n_qubits, seed=DEFAULT_SEED):
    import networkx as nx
    connected = set(G.nodes())
    isolated = [n for n in range(n_qubits) if n not in connected or G.degree(n) == 0]
    active = [n for n in range(n_qubits) if n not in isolated]
    if len(active) >= 2:
        H = G.subgraph(active).copy()
        comps = list(nx.connected_components(H))
        if len(comps) > 1:
            for i in range(len(comps) - 1):
                u, v = next(iter(comps[i])), next(iter(comps[i + 1]))
                H.add_edge(u, v, weight=0.001, gates=['virtual'])
        A, B = nx.algorithms.community.kernighan_lin_bisection(H, weight='weight', seed=seed)
    else:
        half = n_qubits // 2
        A, B = set(range(half)), set(range(half, n_qubits))
    A, B = set(A) | set(isolated[0::2]), set(B) | set(isolated[1::2])
    while abs(len(A) - len(B)) > 2:
        if len(A) > len(B):
            node = next(iter(A)); A.remove(node); B.add(node)
        else:
            node = next(iter(B)); B.remove(node); A.add(node)
    return A, B


# legacy-name alias
partition_graph = kl_interaction_partition


def _entanglement_weighted_graph(G, layout):
    H = G.copy()
    for (u, v), w in estimate_edge_entanglement(layout).items():
        if H.has_edge(u, v):
            H[u][v]['weight'] = max(float(w), 0.1)
        else:
            H.add_edge(u, v, weight=max(float(w), 0.1), gates=['virtual'])
    return H


# ---------------------------------------------------------------------------
#  Shared simulated-annealing engine
# ---------------------------------------------------------------------------
def optimize_partition_sa(G, layout, n_qubits, qc_full=None, start_partition=None,
                          iters=180, boundary_focus=False, seed=DEFAULT_SEED):
    """SA refinement of a bipartition under the multi-objective cost.

    Uses the physics-aligned cost (entropy + MI + gamma + cut + balance) when a
    full statevector is available, else the structural surrogate.
    """
    rng = random.Random(seed)
    if start_partition is None:
        A, B = kl_interaction_partition(G, n_qubits, seed=seed)
    else:
        A, B = set(start_partition[0]), set(start_partition[1])

    sv_full = p_full = None
    if qc_full is not None:
        sv_full = get_statevector(qc_full)
        if sv_full is not None:
            p_full = prob_from_statevector(sv_full)

    def cost(pa, pb):
        return compute_partition_cost(pa, pb, G, layout, n_qubits, sv_full=sv_full, p_full=p_full)[0]

    cur = cost(A, B)
    best_A, best_B, best = set(A), set(B), cur
    temp0 = max(1.0, abs(cur) + 1.0)

    for t in range(max(1, iters)):
        T = temp0 * (0.975 ** t)
        bA = [q for q in A if any(nb in B for nb in G.neighbors(q))]
        bB = [q for q in B if any(nb in A for nb in G.neighbors(q))]
        thresh = 0.72 if boundary_focus else 0.60
        use_swap = bool(bA and bB and rng.random() < thresh)

        A_new, B_new = set(A), set(B)
        if use_swap:
            qa, qb = rng.choice(bA), rng.choice(bB)
            A_new.discard(qa); A_new.add(qb)
            B_new.discard(qb); B_new.add(qa)
        else:
            move_from_A = (rng.random() < 0.5 and len(A) > 1) or len(B) <= 1
            if move_from_A and len(A) > 1:
                q = rng.choice(tuple(A)); A_new.remove(q); B_new.add(q)
            elif len(B) > 1:
                q = rng.choice(tuple(B)); B_new.remove(q); A_new.add(q)
            else:
                continue
        if abs(len(A_new) - len(B_new)) > 2:
            continue

        new = cost(A_new, B_new)
        if new <= cur or rng.random() < math.exp(-(new - cur) / max(T, 1e-12)):
            A, B, cur = A_new, B_new, new
            if new < best:
                best_A, best_B, best = set(A), set(B), new

    # greedy boundary cleanup
    improved, steps = True, 0
    while improved and steps < 12:
        improved = False
        steps += 1
        bA = [q for q in best_A if any(nb in best_B for nb in G.neighbors(q))]
        bB = [q for q in best_B if any(nb in best_A for nb in G.neighbors(q))]
        cands = [(qa, qb) for qa in bA[:6] for qb in bB[:6]]
        rng.shuffle(cands)
        for qa, qb in cands[:24]:
            A_new, B_new = set(best_A), set(best_B)
            A_new.discard(qa); A_new.add(qb)
            B_new.discard(qb); B_new.add(qa)
            if abs(len(A_new) - len(B_new)) > 2:
                continue
            s = cost(A_new, B_new)
            if s + 1e-9 < best:
                best_A, best_B, best = A_new, B_new, s
                improved = True
                break
    return best_A, best_B


# ---------------------------------------------------------------------------
#  Ladder rungs
# ---------------------------------------------------------------------------
def prism_kl(G, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED):
    """Rung 1 — KL on the gate-count graph (structural only)."""
    return kl_interaction_partition(G, n_qubits, seed=seed)


def prism_oe(G, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED):
    """Rung 2 — operator-entanglement edge weighting + structural SA."""
    H = _entanglement_weighted_graph(G, layout)
    start = kl_interaction_partition(H, n_qubits, seed=seed)
    return optimize_partition_sa(H, layout, n_qubits, qc_full=None,
                                 start_partition=start, iters=120,
                                 boundary_focus=False, seed=seed)


def prism_mi(G, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED):
    """Rung 3 — physics cost (entropy + mutual information), statevector-aware SA."""
    H = _entanglement_weighted_graph(G, layout)
    start = kl_interaction_partition(H, n_qubits, seed=seed)
    return optimize_partition_sa(H, layout, n_qubits, qc_full=qc_full,
                                 start_partition=start, iters=220,
                                 boundary_focus=False, seed=seed)


def prism_bf(G, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED):
    """Rung 4 — boundary-focused move proposals, same physics cost."""
    if n_qubits > 24:
        return optimize_partition_sa(G, layout, n_qubits, qc_full=None,
                                     start_partition=kl_interaction_partition(G, n_qubits, seed=seed),
                                     iters=160, boundary_focus=True, seed=seed)
    H = _entanglement_weighted_graph(G, layout)
    start = kl_interaction_partition(H, n_qubits, seed=seed)
    return optimize_partition_sa(H, layout, n_qubits, qc_full=qc_full,
                                 start_partition=start, iters=260,
                                 boundary_focus=True, seed=seed)


__all__ = [
    'kl_interaction_partition', 'partition_graph', 'optimize_partition_sa',
    'prism_kl', 'prism_oe', 'prism_mi', 'prism_bf',
]
