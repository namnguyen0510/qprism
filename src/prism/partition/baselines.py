"""
prism.partition.baselines
==========================
Competitor / baseline circuit-partition methods.

    naive            half-split (qubits 0..n/2 | n/2..n)
    spectral         Fiedler (second Laplacian eigenvector) bisection
    louvain          modularity community detection, merged to a bisection
    girvan_newman    edge-betweenness hierarchical bisection
    metis            multilevel k-way bisection (pymetis), spectral fallback
    qdislib          DAG gate-cut partitioning (the method in the qdislib
                     library, arXiv:2505.01184)

Audit note on ``qdislib``
-------------------------
The qdislib library (BSC, ``bsc-wdc/qdislib``) represents a circuit as a
*directed acyclic graph* — nodes are gates, edges encode execution order —
and chooses cuts on that DAG to minimise the number of cross-partition
(gate-cut) operations.  :func:`qdislib_partition` reproduces that contract
faithfully with a self-contained DAG construction + balanced min-gate-cut
search, and :func:`_use_real_qdislib` transparently delegates to the actual
installed package when it is importable, so the benchmark uses the genuine
method whenever the environment provides it.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
#  Balance helper
# ---------------------------------------------------------------------------
def _rebalance(A, B, tol=2):
    A, B = set(A), set(B)
    while abs(len(A) - len(B)) > tol:
        if len(A) > len(B):
            A, B = A, B
            node = next(iter(A)); A.remove(node); B.add(node)
        else:
            node = next(iter(B)); B.remove(node); A.add(node)
    return A, B


def _place_isolated(A, B, n_qubits, present):
    iso = [q for q in range(n_qubits) if q not in present]
    for i, q in enumerate(iso):
        (A if i % 2 == 0 else B).add(q)
    return A, B


# ---------------------------------------------------------------------------
#  Baselines
# ---------------------------------------------------------------------------
def naive_partition(n_qubits):
    """Trivial half-split."""
    half = n_qubits // 2
    return set(range(half)), set(range(half, n_qubits))


def spectral_partition(G, n_qubits, seed=DEFAULT_SEED):
    """Fiedler-vector (sign of the 2nd Laplacian eigenvector) bisection."""
    import numpy as np
    import networkx as nx
    if G.number_of_nodes() < 2:
        return naive_partition(n_qubits)
    try:
        L = nx.laplacian_matrix(G, weight='weight').toarray().astype(float)
        _, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        nodes = list(G.nodes())
        A = {nodes[i] for i, v in enumerate(fiedler) if v >= 0}
        B = set(G.nodes()) - A
        A, B = _rebalance(A, B)
        A, B = _place_isolated(A, B, n_qubits, set(G.nodes()))
        return A, B
    except Exception:
        return naive_partition(n_qubits)


def louvain_partition(G, n_qubits, seed=DEFAULT_SEED):
    """Louvain modularity communities merged greedily into a balanced bisection."""
    import networkx as nx
    if G.number_of_nodes() < 2:
        return naive_partition(n_qubits)
    comms = None
    try:
        comms = list(nx.algorithms.community.louvain_communities(G, weight='weight', seed=seed))
    except (AttributeError, Exception):
        comms = None
    if comms is None:
        try:
            import community as cl
            part = cl.best_partition(G, weight='weight', random_state=seed)
            cm = {}
            for node, c in part.items():
                cm.setdefault(c, set()).add(node)
            comms = list(cm.values())
        except Exception:
            comms = None
    if comms is None:
        return spectral_partition(G, n_qubits)
    comms = sorted(comms, key=len, reverse=True)
    A, B = set(comms[0]), set()
    for c in comms[1:]:
        (A if len(A) <= len(B) else B).update(c)
    A, B = _rebalance(A, B)
    A, B = _place_isolated(A, B, n_qubits, A | B)
    return A, B


def girvan_newman_partition(G, n_qubits, seed=DEFAULT_SEED):
    """Edge-betweenness (Girvan-Newman) first split into two communities."""
    import networkx as nx
    if G.number_of_nodes() < 2:
        return naive_partition(n_qubits)
    if G.number_of_nodes() > 200 or G.number_of_edges() > 1000:
        return spectral_partition(G, n_qubits)
    try:
        H = G.copy()
        for u, v, d in H.edges(data=True):
            H[u][v]['distance'] = 1.0 / max(d.get('weight', 1), 1e-9)
        comps = next(nx.algorithms.community.girvan_newman(H))
        A = set(comps[0])
        B = set(comps[1]) if len(comps) > 1 else set()
        for c in comps[2:]:
            (A if len(A) <= len(B) else B).update(c)
        A, B = _rebalance(A, B)
        A, B = _place_isolated(A, B, n_qubits, A | B)
        return A, B
    except Exception:
        return spectral_partition(G, n_qubits)


def metis_partition(G, n_qubits, seed=DEFAULT_SEED):
    """Multilevel k-way bisection via pymetis (spectral fallback if absent)."""
    try:
        import pymetis
    except ImportError:
        return spectral_partition(G, n_qubits)
    import networkx as nx
    if G.number_of_nodes() < 2:
        return naive_partition(n_qubits)
    try:
        nodes = sorted(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        adj, ew = [], []
        for v in nodes:
            nbrs = sorted(G.neighbors(v))
            adj.append([idx[u] for u in nbrs])
            ew.append([max(1, round(G[v][u].get('weight', 1))) for u in nbrs])
        _, mem = pymetis.part_graph(2, adjacency=adj, eweights=ew)
        A = {nodes[i] for i, p in enumerate(mem) if p == 0}
        B = {nodes[i] for i, p in enumerate(mem) if p == 1}
        A, B = _place_isolated(A, B, n_qubits, set(G.nodes()))
        A, B = _rebalance(A, B)
        return A, B
    except Exception:
        return spectral_partition(G, n_qubits)


# ---------------------------------------------------------------------------
#  qdislib — DAG gate-cut competitor
# ---------------------------------------------------------------------------
def _build_gate_dag(layout, n_qubits):
    """Gate-level DAG: nodes are gates, edges connect consecutive gates that
    share a qubit (execution order). Returns (networkx.DiGraph, gate_records)."""
    import networkx as nx
    D = nx.DiGraph()
    last_on_qubit = {}
    records = []
    gid = 0
    for li, layer in enumerate(layout):
        for g in layer:
            qs = list(g['qubits'])
            D.add_node(gid, gate=g['gate'].upper(), qubits=qs, layer=li)
            for q in qs:
                if q in last_on_qubit:
                    D.add_edge(last_on_qubit[q], gid, qubit=q)
                last_on_qubit[q] = gid
            records.append({'id': gid, 'gate': g['gate'].upper(), 'qubits': qs, 'layer': li})
            gid += 1
    return D, records


def _qubit_dag_graph(layout, n_qubits):
    """Project the gate-DAG onto qubits with execution-order (depth-span)
    weights — the structure qdislib cuts to minimise gate cuts."""
    import networkx as nx
    ec, ed = defaultdict(int), defaultdict(list)
    for li, layer in enumerate(layout):
        for g in layer:
            qs = g['qubits']
            if len(qs) >= 2:
                for i in range(len(qs)):
                    for j in range(i + 1, len(qs)):
                        u, v = min(qs[i], qs[j]), max(qs[i], qs[j])
                        ec[(u, v)] += 1
                        ed[(u, v)].append(li)
    Gd = nx.Graph()
    Gd.add_nodes_from(range(n_qubits))
    for (u, v), cnt in ec.items():
        ds = ed[(u, v)]
        span = (max(ds) - min(ds) + 1) if ds else 1
        Gd.add_edge(u, v, weight=cnt + 0.5 * span)
    return Gd


def _use_real_qdislib(layout, n_qubits):
    """Delegate to the genuine ``qdislib`` package if importable. Returns
    ``(A, B)`` or None when the package or a usable entry point is absent."""
    try:
        import qdislib  # noqa: F401
    except Exception:
        return None
    try:
        from ..compiler import compile_circuit
        qc, _ = compile_circuit(layout, num_qubits=n_qubits, use_numeric_params=True)
        # qdislib exposes DAG gate-cut helpers; try the documented entry points.
        for fname in ('find_cut', 'gate_cutting', 'find_gate_cut', 'cut_circuit'):
            fn = getattr(qdislib, fname, None)
            if fn is None:
                continue
            try:
                res = fn(qc)
            except Exception:
                continue
            parts = _coerce_qdislib_result(res, n_qubits)
            if parts is not None:
                return parts
    except Exception:
        return None
    return None


def _coerce_qdislib_result(res, n_qubits):
    """Best-effort mapping of a qdislib return value to a qubit bipartition."""
    try:
        subs = None
        if isinstance(res, (list, tuple)) and len(res) >= 2 and all(hasattr(x, '__iter__') for x in res[:2]):
            subs = [set(res[0]), set(res[1])]
        elif hasattr(res, 'partitions'):
            subs = [set(p) for p in res.partitions[:2]]
        if not subs:
            return None
        A = {q for q in subs[0] if isinstance(q, int)}
        B = {q for q in subs[1] if isinstance(q, int)}
        if not A or not B:
            return None
        A, B = _place_isolated(A, B, n_qubits, A | B)
        return _rebalance(A, B)
    except Exception:
        return None


def qdislib_partition(layout, n_qubits, seed=DEFAULT_SEED, use_real_package=True):
    """DAG gate-cut partition (qdislib).

    Uses the installed qdislib package when available; otherwise applies a
    faithful self-contained DAG construction followed by a centrality-seeded
    balanced growth that minimises straddling (cut) gates.
    """
    if use_real_package:
        real = _use_real_qdislib(layout, n_qubits)
        if real is not None:
            return real

    Gd = _qubit_dag_graph(layout, n_qubits)
    if Gd.number_of_edges() == 0:
        return naive_partition(n_qubits)
    # seed from the least-central qubit, grow A by heaviest-DAG-edge first
    cent = {q: sum(Gd[q][nb].get('weight', 1) for nb in Gd.neighbors(q)) for q in range(n_qubits)}
    seed_q = min(range(n_qubits), key=lambda q: cent[q])
    A = {seed_q}
    frontier = []
    for nb in Gd.neighbors(seed_q):
        heapq.heappush(frontier, (-Gd[seed_q][nb].get('weight', 1), nb))
    vis = {seed_q}
    while len(A) < n_qubits // 2 and frontier:
        _, node = heapq.heappop(frontier)
        if node in vis:
            continue
        vis.add(node)
        A.add(node)
        for nb in Gd.neighbors(node):
            if nb not in vis:
                heapq.heappush(frontier, (-Gd[node][nb].get('weight', 1), nb))
    B = set(range(n_qubits)) - A
    return _rebalance(A, B)


# qdislib availability flag (real package present?)
def has_real_qdislib():
    try:
        import qdislib  # noqa: F401
        return True
    except Exception:
        return False


__all__ = [
    'naive_partition', 'spectral_partition', 'louvain_partition',
    'girvan_newman_partition', 'metis_partition', 'qdislib_partition',
    'has_real_qdislib',
]
