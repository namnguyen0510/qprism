"""
prism.graph
===========
Interaction-graph construction and the light-cone augmentation (Stage 1 of
the PRISM-LCT pipeline).

* :func:`build_interaction_graph`  — static qubit-interaction graph G_int
* :func:`estimate_edge_entanglement` — operator-entanglement edge weights
* :func:`build_lightcone_graph`    — G_aug: G_int + depth-decayed causal-cone
                                     overlap + operator-entanglement weights
* :func:`classify_gates` / :func:`compute_cut_stats` / :func:`build_subcircuit_layout`
"""
from __future__ import annotations

import math
from collections import defaultdict

from .gates import SCHMIDT_RANK, PARTICLE_PRESERVING


def build_interaction_graph(layout, n_qubits):
    """Static qubit-interaction graph: an edge (u,v) with weight = number of
    two-qubit (or multi-qubit) gates acting jointly on u and v."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    for layer in layout:
        for gate in layer:
            qs = gate['qubits']
            if len(qs) >= 2:
                for i in range(len(qs)):
                    for j in range(i + 1, len(qs)):
                        u, v = qs[i], qs[j]
                        if G.has_edge(u, v):
                            G[u][v]['weight'] += 1
                            G[u][v]['gates'].append(gate['gate'])
                        else:
                            G.add_edge(u, v, weight=1, gates=[gate['gate']])
    return G


def estimate_edge_entanglement(layout):
    """Pairwise operator-entanglement pressure per qubit pair (eq. 8 surrogate).

    Each two-qubit gate contributes ``(1 + 0.05*layer) * log(chi)``, damped by
    0.8 for particle-preserving gates (their cuts are cheaper)."""
    weights = defaultdict(float)
    for layer_idx, layer in enumerate(layout):
        depth_factor = 1.0 + 0.05 * layer_idx
        for gate in layer:
            qs = gate.get('qubits', [])
            if len(qs) < 2:
                continue
            name = gate['gate'].upper()
            sr = max(int(SCHMIDT_RANK.get(name, len(qs))), 2)
            gw = depth_factor * math.log(sr)
            if name in PARTICLE_PRESERVING:
                gw *= 0.8
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    u, v = sorted((qs[i], qs[j]))
                    weights[(u, v)] += gw
    return dict(weights)


def build_lightcone_graph(G_base, layout, n_qubits):
    """Stage 1 of PRISM-LCT: augment G_base with depth-decayed light-cone
    overlap weights plus operator-entanglement weights (paper eqs. 6-9)."""
    import networkx as nx
    H = G_base.copy()
    n_layers = max(len(layout), 1)
    cone = {q: {q} for q in range(n_qubits)}
    pair_w = defaultdict(float)

    for layer_idx, layer in enumerate(layout):
        for gate in layer:
            qs = gate.get('qubits', [])
            if len(qs) >= 2:
                merged = set()
                for q in qs:
                    merged.update(cone[q])
                for q in merged:
                    cone[q] = merged
        decay = math.exp(-layer_idx / max(n_layers / 3.0, 1.0))
        for q in range(n_qubits):
            for r in cone[q]:
                if r > q:
                    pair_w[(q, r)] += decay

    ent_w = estimate_edge_entanglement(layout)

    for (u, v), w in pair_w.items():
        existing = float(H[u][v]['weight']) if H.has_edge(u, v) else 0.0
        new_w = existing + 0.5 * float(w) + float(ent_w.get((u, v), 0.0))
        if H.has_edge(u, v):
            H[u][v]['weight'] = max(new_w, 0.1)
        else:
            H.add_edge(u, v, weight=max(new_w, 0.1), gates=['lightcone'])

    for (u, v), w in ent_w.items():
        if not H.has_edge(u, v):
            H.add_edge(u, v, weight=max(float(w), 0.1), gates=['ent'])
    return H


def classify_gates(layout, A_set, B_set):
    """Partition every gate into A-local, B-local, or cross (cut) by its support.

    Returns a dict with keys ``local_A``, ``local_B``, ``cross`` and the
    aliases ``A_only`` / ``B_only`` for the local lists."""
    A_set, B_set = set(A_set), set(B_set)
    local_A, local_B, cross = [], [], []
    for layer_idx, layer in enumerate(layout):
        for gate in layer:
            qs = set(gate['qubits'])
            entry = {**gate, 'layer': layer_idx}
            if qs.issubset(A_set):
                local_A.append(entry)
            elif qs.issubset(B_set):
                local_B.append(entry)
            else:
                cross.append(entry)
    return {'local_A': local_A, 'local_B': local_B, 'cross': cross,
            'A_only': local_A, 'B_only': local_B}


def compute_cut_stats(G, A_set, B_set):
    """Cut statistics for a bipartition over interaction graph ``G``."""
    A_set = set(A_set)
    cut_edges, inner_A, inner_B = [], [], []
    cut_weight = 0
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1)
        if (u in A_set) == (v in A_set):
            (inner_A if u in A_set else inner_B).append((u, v, d))
        else:
            cut_edges.append((u, v, d))
            cut_weight += w
    total_w = sum(d.get('weight', 1) for _, _, d in G.edges(data=True))
    return {
        'n_cut_edges': len(cut_edges),
        'cut_weight': int(cut_weight),
        'total_edge_weight': int(total_w),
        'cut_fraction': round(cut_weight / max(total_w, 1), 4),
        'inner_A_edges': len(inner_A),
        'inner_B_edges': len(inner_B),
        'cut_edges': cut_edges,
        'inner_A': inner_A,
        'inner_B': inner_B,
    }


def build_subcircuit_layout(layout, node_set, name='A', gate_class=None):
    """Project ``layout`` onto ``node_set`` (cross gates dropped). Returns
    ``(sub_layout, sorted_nodes, remap)`` where remap maps global->local index."""
    node_set = set(node_set)
    sorted_nodes = sorted(node_set)
    remap = {q: i for i, q in enumerate(sorted_nodes)}
    sub_layout = []
    for layer_idx, layer in enumerate(layout):
        sub_layer = []
        for gate_idx, gate in enumerate(layer):
            qs = gate['qubits']
            if all(q in node_set for q in qs):
                entry = {**gate, 'source_layer': layer_idx, 'source_gate_index': gate_idx,
                         'qubits': [remap[q] for q in qs], 'source': 'local'}
                sub_layer.append(entry)
            # cross-partition gates are omitted from the local subcircuit
        if sub_layer:
            sub_layout.append(sub_layer)
    return sub_layout, sorted_nodes, remap


__all__ = [
    'build_interaction_graph', 'estimate_edge_entanglement', 'build_lightcone_graph',
    'classify_gates', 'compute_cut_stats', 'build_subcircuit_layout',
]
