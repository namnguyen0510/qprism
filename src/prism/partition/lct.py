"""
prism.partition.lct
====================
PRISM-LCT — Light-Cone Tempering, the main PRISM method.

A single-shot heuristic solver for the multi-objective partition problem,
built as a seven-stage pipeline (paper Fig. 1):

    S1  Light-cone augmented graph construction
    S2  Multi-start seeding (KL, spectral, Louvain, METIS, cone-BFS, random)
    S3  Parallel-tempered replica ensemble (geometric temperature ladder)
    S4  Tabu-guided move operators (boundary swap, single move, double swap,
        cluster move)
    S5  Adaptive operator-weight rebalancing
    S6  Inverse-score consensus combine
    S7  Greedy boundary polish

The objective is unchanged from the ablation ladder — the gains come from
optimisation quality (escaping the rugged landscape), not a different cost.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict, deque

from ..graph import build_lightcone_graph
from ..simulate import get_statevector, prob_from_statevector
from .cost import compute_partition_cost
from .baselines import spectral_partition, louvain_partition, metis_partition
from .ladder import kl_interaction_partition

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
#  S2 — seeding helpers
# ---------------------------------------------------------------------------
def _light_cone_seed(layout, n_qubits):
    """Grow a partition from the qubit with the smallest forward light-cone."""
    cone = {q: {q} for q in range(n_qubits)}
    for layer in layout:
        for gate in layer:
            qs = gate.get('qubits', [])
            if len(qs) >= 2:
                merged = set()
                for q in qs:
                    merged.update(cone[q])
                for q in merged:
                    cone[q] = merged
    seed = min(range(n_qubits), key=lambda q: len(cone[q]))
    A = {seed}
    frontier = list(cone[seed] - A)
    while len(A) < n_qubits // 2 and frontier:
        A.add(frontier.pop(0))
    while len(A) < n_qubits // 2:
        for q in range(n_qubits):
            if q not in A:
                A.add(q)
                break
    return A, set(range(n_qubits)) - A


def _random_balanced(n_qubits, rng):
    qs = list(range(n_qubits))
    rng.shuffle(qs)
    half = n_qubits // 2
    return set(qs[:half]), set(qs[half:])


def _gather_starts(G_aug, layout, n_qubits, rng):
    raw = []
    for fn in (kl_interaction_partition, spectral_partition, louvain_partition, metis_partition):
        try:
            res = fn(G_aug, n_qubits)
            if isinstance(res, tuple) and len(res) == 2:
                raw.append(res)
        except Exception:
            pass
    try:
        raw.append(_light_cone_seed(layout, n_qubits))
    except Exception:
        pass
    raw.append(_random_balanced(n_qubits, rng))
    raw.append(_random_balanced(n_qubits, rng))

    seen, unique = set(), []
    for A, B in raw:
        A, B = set(A), set(B)
        if not A or not B:
            continue
        key = frozenset(A) if min(A) < min(B) else frozenset(B)
        if key not in seen:
            seen.add(key)
            unique.append((A, B))
    return unique


# ---------------------------------------------------------------------------
#  S4 — move operators
# ---------------------------------------------------------------------------
def _propose_move(A, B, G, tabu, rng, weights):
    op = rng.choices(list(weights.keys()), weights=list(weights.values()))[0]

    if op == 'boundary_swap':
        bA = [q for q in A if any(nb in B for nb in G.neighbors(q))]
        bB = [q for q in B if any(nb in A for nb in G.neighbors(q))]
        if not bA or not bB:
            return None, None, op, []
        for _ in range(8):
            qa, qb = rng.choice(bA), rng.choice(bB)
            if qa not in tabu or qb not in tabu:
                return (A - {qa}) | {qb}, (B - {qb}) | {qa}, op, [qa, qb]
        return None, None, op, []

    if op == 'single_move':
        if len(A) > len(B):
            cand, src = tuple(A), 'A'
        elif len(B) > len(A):
            cand, src = tuple(B), 'B'
        else:
            cand = tuple(A) if rng.random() < 0.5 else tuple(B)
            src = 'A' if cand and cand[0] in A else 'B'
        for _ in range(8):
            q = rng.choice(cand)
            if q not in tabu:
                if src == 'A':
                    return A - {q}, B | {q}, op, [q]
                return A | {q}, B - {q}, op, [q]
        return None, None, op, []

    if op == 'double_swap':
        if len(A) < 2 or len(B) < 2:
            return None, None, op, []
        qas = rng.sample(tuple(A), 2)
        qbs = rng.sample(tuple(B), 2)
        return (A - set(qas)) | set(qbs), (B - set(qbs)) | set(qas), op, qas + qbs

    if op == 'cluster_move':
        if len(A) <= 3 or len(B) <= 3:
            return None, None, op, []
        from_A = rng.random() < 0.5
        side, other = (A, B) if from_A else (B, A)
        seed = rng.choice(tuple(side))
        cluster = {seed}
        for nb in G.neighbors(seed):
            if nb in side and len(cluster) < 3:
                cluster.add(nb)
        if len(cluster) >= len(side) - 1:
            return None, None, op, []
        side_n, other_n = side - cluster, other | cluster
        if from_A:
            return side_n, other_n, op, list(cluster)
        return other_n, side_n, op, list(cluster)

    return None, None, op, []


# ---------------------------------------------------------------------------
#  S6 — consensus combine
# ---------------------------------------------------------------------------
def _consensus_partition(chains, n_qubits):
    s_min = min(c['score'] for c in chains)
    weights = [1.0 / (1e-6 + ch['score'] - s_min + 1e-3) for ch in chains]
    a_score = [0.0] * n_qubits
    for ch, w in zip(chains, weights):
        for q in ch['A']:
            a_score[q] += w
    ranked = sorted(range(n_qubits), key=lambda q: -a_score[q])
    half = n_qubits // 2
    return set(ranked[:half]), set(ranked[half:])


# ---------------------------------------------------------------------------
#  S7 — greedy polish
# ---------------------------------------------------------------------------
def _greedy_polish(A_in, B_in, G, layout, n_qubits, sv_full=None, p_full=None, max_passes=25):
    A, B = set(A_in), set(B_in)
    best, _ = compute_partition_cost(A, B, G, layout, n_qubits, sv_full=sv_full, p_full=p_full)
    for _ in range(max_passes):
        improved = False
        bA = sorted(q for q in A if any(nb in B for nb in G.neighbors(q)))
        bB = sorted(q for q in B if any(nb in A for nb in G.neighbors(q)))
        for qa in bA:
            for qb in bB:
                A_new, B_new = (A - {qa}) | {qb}, (B - {qb}) | {qa}
                if abs(len(A_new) - len(B_new)) > 2:
                    continue
                s, _ = compute_partition_cost(A_new, B_new, G, layout, n_qubits,
                                              sv_full=sv_full, p_full=p_full)
                if s + 1e-9 < best:
                    A, B, best, improved = A_new, B_new, s, True
                    break
            if improved:
                break
        if not improved:
            break
    return A, B


# ---------------------------------------------------------------------------
#  Main entry
# ---------------------------------------------------------------------------
def prism_lct(G_base, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED, budget_iters=150):
    """PRISM-LCT: light-cone augmentation + parallel-tempered tabu search +
    consensus combine + greedy polish. Returns ``(A_set, B_set)``."""
    rng = random.Random(seed)
    G_aug = build_lightcone_graph(G_base, layout, n_qubits)

    sv_full = p_full = None
    if qc_full is not None and n_qubits <= 22:
        try:
            sv_full = get_statevector(qc_full)
            p_full = prob_from_statevector(sv_full) if sv_full is not None else None
        except Exception:
            sv_full = p_full = None

    cache = {}

    def cost(A, B):
        key = frozenset(A) if min(A) < min(B) else frozenset(B)
        if key in cache:
            return cache[key]
        s, _ = compute_partition_cost(A, B, G_aug, layout, n_qubits, sv_full=sv_full, p_full=p_full)
        cache[key] = s
        return s

    starts = _gather_starts(G_aug, layout, n_qubits, rng) or [_random_balanced(n_qubits, rng)]

    n_chains = min(4, max(2, len(starts)))
    init = cost(*starts[0])
    T_max = max(2.0, abs(init) * 0.5)
    T_levels = [T_max * (0.5 ** k) for k in range(n_chains)]

    chains = []
    for i in range(n_chains):
        A0, B0 = starts[i % len(starts)]
        chains.append({
            'A': set(A0), 'B': set(B0), 'T': T_levels[i], 'score': cost(A0, B0),
            'tabu': deque(maxlen=max(6, n_qubits // 3)),
            'op_w': {'boundary_swap': 0.45, 'single_move': 0.30, 'double_swap': 0.15, 'cluster_move': 0.10},
            'accept': defaultdict(lambda: [0, 0]),
        })

    best_A, best_B = chains[0]['A'].copy(), chains[0]['B'].copy()
    best = chains[0]['score']
    for ch in chains:
        if ch['score'] < best:
            best_A, best_B, best = ch['A'].copy(), ch['B'].copy(), ch['score']

    exchange_period = max(8, budget_iters // 25)
    rebalance_period = max(20, budget_iters // 12)
    cooling_period = max(15, budget_iters // 15)

    for it in range(budget_iters):
        for ch in chains:
            A_new, B_new, op, moved = _propose_move(ch['A'], ch['B'], G_aug, ch['tabu'], rng, ch['op_w'])
            if A_new is None or abs(len(A_new) - len(B_new)) > 2:
                continue
            ch['accept'][op][1] += 1
            new = cost(A_new, B_new)
            delta = new - ch['score']
            if delta <= 0 or rng.random() < math.exp(-delta / max(ch['T'], 1e-9)):
                ch['A'], ch['B'], ch['score'] = A_new, B_new, new
                ch['accept'][op][0] += 1
                for q in moved:
                    ch['tabu'].append(q)
                if new < best:
                    best_A, best_B, best = ch['A'].copy(), ch['B'].copy(), new

        if it > 0 and it % cooling_period == 0:
            for ch in chains:
                ch['T'] *= 0.92

        if it > 0 and it % exchange_period == 0:
            for i in range(len(chains) - 1):
                c1, c2 = chains[i], chains[i + 1]
                d = (c1['score'] - c2['score']) * (1.0 / c1['T'] - 1.0 / c2['T'])
                if rng.random() < min(1.0, math.exp(-d)):
                    c1['A'], c2['A'] = c2['A'], c1['A']
                    c1['B'], c2['B'] = c2['B'], c1['B']
                    c1['score'], c2['score'] = c2['score'], c1['score']

        if it > 0 and it % rebalance_period == 0:
            for ch in chains:
                rates = {op: (acc + 1) / (att + 1) for op, (acc, att) in ch['accept'].items()}
                tot = sum(rates.values())
                if tot > 0:
                    for op in ch['op_w']:
                        if op in rates:
                            ch['op_w'][op] = 0.7 * ch['op_w'][op] + 0.3 * (rates[op] / tot)
                    s = sum(ch['op_w'].values())
                    if s > 0:
                        for op in ch['op_w']:
                            ch['op_w'][op] /= s
                ch['accept'].clear()

    cA, cB = _consensus_partition(chains, n_qubits)
    cs = cost(cA, cB)
    if cs < best:
        best_A, best_B, best = cA, cB, cs

    best_A, best_B = _greedy_polish(best_A, best_B, G_aug, layout, n_qubits,
                                    sv_full=sv_full, p_full=p_full, max_passes=20)
    return best_A, best_B


# legacy alias
mosaic_lct = prism_lct

__all__ = ['prism_lct', 'mosaic_lct']
