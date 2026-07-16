"""
prism.partition.kway
=====================
Multi-QPU (k-way) partitioning by recursive bisection.

The PRISM objective is defined for a single cut (k=2). To target a cluster of
``k`` smaller QPUs we recursively bisect the largest current fragment with the
chosen bipartition method, each time solving the *induced* sub-problem (the
sub-circuit restricted to that fragment), until ``k`` parts remain. This keeps
fragments balanced (~n/k qubits) so a circuit too wide for one device fits on
several.
"""
from __future__ import annotations

from ..graph import build_interaction_graph, build_subcircuit_layout, classify_gates
from ..compiler import compile_circuit


def kway_partition(method, layout, n_qubits, k, qc_full=None, seed=42, simulate=True):
    """Split the circuit into ``k`` qubit groups by recursive bisection.

    Returns a list of ``k`` disjoint qubit sets covering ``range(n_qubits)``.
    Each recursive bisection runs ``method`` on the induced sub-circuit of the
    fragment being split.
    """
    from . import partition as _partition  # lazy import (avoids circular import)

    parts = [set(range(n_qubits))]
    while len(parts) < k:
        parts.sort(key=len, reverse=True)
        S = parts.pop(0)
        if len(S) < 2:
            parts.append(S)
            break
        # induced sub-problem on S: gates fully inside S, qubits remapped 0..|S|-1
        sub_layout, sorted_nodes, remap = build_subcircuit_layout(layout, S, 'S')
        nS = len(S)
        Gsub = build_interaction_graph(sub_layout, nS)
        sub_qc = None
        if simulate and nS <= 22:
            try:
                sub_qc, _ = compile_circuit(sub_layout, num_qubits=nS, use_numeric_params=True)
            except Exception:
                sub_qc = None
        try:
            A_sub, B_sub = _partition(method, Gsub, sub_layout, nS, qc_full=sub_qc, seed=seed)
        except Exception:
            # fall back to an even split of S
            half = nS // 2
            A_sub, B_sub = set(range(half)), set(range(half, nS))
        inv = {i: q for q, i in remap.items()}
        A = {inv[i] for i in A_sub if i in inv}
        B = {inv[i] for i in B_sub if i in inv}
        # any qubit of S not placed (isolated in the sub-graph) -> smaller side
        leftover = S - A - B
        for q in leftover:
            (A if len(A) <= len(B) else B).add(q)
        parts.append(A)
        parts.append(B)
    return [p for p in parts if p]


def kway_cross_gates(layout, parts):
    """Number of gates that straddle two or more fragments (the cut gates)."""
    part_of = {}
    for i, S in enumerate(parts):
        for q in S:
            part_of[q] = i
    cross = 0
    for layer in layout:
        for g in layer:
            ps = {part_of.get(q, -1) for q in g['qubits']}
            if len(ps) > 1:
                cross += 1
    return cross


def kway_stats(layout, parts, n_qubits):
    """Summary: number of parts, max fragment width, cross-gate count, balance."""
    sizes = [len(S) for S in parts]
    return {
        'k': len(parts),
        'fragment_sizes': sorted(sizes, reverse=True),
        'max_fragment': max(sizes) if sizes else 0,
        'cross_gates': kway_cross_gates(layout, parts),
        'balance': (min(sizes) / max(sizes)) if sizes and max(sizes) else 0.0,
    }


__all__ = ['kway_partition', 'kway_cross_gates', 'kway_stats']
