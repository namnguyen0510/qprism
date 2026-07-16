"""
prism.partition
===============
Unified access to every circuit-partition method PRISM ships.

Method registry (paper order)::

    Baselines        Naive, Spectral, Louvain, Girvan-Newman, METIS, qdislib
    PRISM ladder     PRISM-KL, PRISM-OE, PRISM-MI, PRISM-BF
    Main method      PRISM-LCT   (aliased "PRISM")

Every method is wrapped to a uniform signature
``fn(G, layout, n_qubits, qc_full, seed) -> (A_set, B_set)`` so they are
interchangeable.  Use :func:`partition` to call one by name, or
:func:`run_all_partition_methods` to run them all with timing.
"""
from __future__ import annotations

import time

from .cost import compute_partition_cost, partition_cost_surrogate, partition_cost_terms
from .baselines import (naive_partition, spectral_partition, louvain_partition,
                        girvan_newman_partition, metis_partition, qdislib_partition,
                        has_real_qdislib)
from .ladder import (kl_interaction_partition, partition_graph, optimize_partition_sa,
                     prism_kl, prism_oe, prism_mi, prism_bf)
from .lct import prism_lct, mosaic_lct
from .kway import kway_partition, kway_cross_gates, kway_stats

DEFAULT_SEED = 42


# Uniform adapters: every entry takes (G, layout, n, qc_full, seed).
PARTITION_METHODS = {
    'Naive':         lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: naive_partition(n),
    'Spectral':      lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: spectral_partition(G, n, seed=seed),
    'Louvain':       lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: louvain_partition(G, n, seed=seed),
    'Girvan-Newman': lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: girvan_newman_partition(G, n, seed=seed),
    'METIS':         lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: metis_partition(G, n, seed=seed),
    'qdislib':       lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: qdislib_partition(layout, n, seed=seed),
    'PRISM-KL':      lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: prism_kl(G, layout, n, qc_full=qc_full, seed=seed),
    'PRISM-OE':      lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: prism_oe(G, layout, n, qc_full=qc_full, seed=seed),
    'PRISM-MI':      lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: prism_mi(G, layout, n, qc_full=qc_full, seed=seed),
    'PRISM-BF':      lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: prism_bf(G, layout, n, qc_full=qc_full, seed=seed),
    'PRISM-LCT':     lambda G, layout, n, qc_full=None, seed=DEFAULT_SEED: prism_lct(G, layout, n, qc_full=qc_full, seed=seed),
}

# Canonical display order and groupings.
METHOD_ORDER = list(PARTITION_METHODS.keys())
BASELINE_METHODS = ['Naive', 'Spectral', 'Louvain', 'Girvan-Newman', 'METIS', 'qdislib']
PRISM_LADDER = ['PRISM-KL', 'PRISM-OE', 'PRISM-MI', 'PRISM-BF', 'PRISM-LCT']
MAIN_METHOD = 'PRISM-LCT'

# "PRISM" is the headline name for the main method.
PARTITION_METHODS['PRISM'] = PARTITION_METHODS['PRISM-LCT']


def partition(method, G, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED):
    """Run a single named partition method, returning ``(A_set, B_set)``."""
    if method not in PARTITION_METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {METHOD_ORDER}")
    return PARTITION_METHODS[method](G, layout, n_qubits, qc_full, seed)


def run_all_partition_methods(G, layout, n_qubits, qc_full=None, seed=DEFAULT_SEED,
                              methods=None):
    """Run every (or a chosen subset of) partition method.

    Returns ``(results, failures)`` where ``results[name] = (A, B, runtime_s)``
    and ``failures[name]`` holds the error string for any method that failed
    or returned a degenerate cut.
    """
    names = methods if methods is not None else METHOD_ORDER
    results, failures = {}, {}
    for name in names:
        fn = PARTITION_METHODS.get(name)
        if fn is None:
            failures[name] = 'unknown method'
            continue
        t0 = time.time()
        try:
            res = fn(G, layout, n_qubits, qc_full, seed)
            dt = time.time() - t0
            if isinstance(res, tuple) and len(res) == 2:
                A, B = set(res[0]), set(res[1])
            elif isinstance(res, dict) and res.get('available') and 'A_set' in res:
                A, B = set(res['A_set']), set(res['B_set'])
            else:
                failures[name] = 'unrecognised return'
                continue
            if A and B:
                results[name] = (A, B, dt)
            else:
                failures[name] = 'degenerate (empty side)'
        except Exception as e:
            failures[name] = f'{type(e).__name__}: {e}'
    return results, failures


__all__ = [
    'PARTITION_METHODS', 'METHOD_ORDER', 'BASELINE_METHODS', 'PRISM_LADDER', 'MAIN_METHOD',
    'partition', 'run_all_partition_methods',
    'compute_partition_cost', 'partition_cost_surrogate', 'partition_cost_terms',
    'naive_partition', 'spectral_partition', 'louvain_partition',
    'girvan_newman_partition', 'metis_partition', 'qdislib_partition', 'has_real_qdislib',
    'kl_interaction_partition', 'partition_graph', 'optimize_partition_sa',
    'prism_kl', 'prism_oe', 'prism_mi', 'prism_bf', 'prism_lct', 'mosaic_lct',
    'kway_partition', 'kway_cross_gates', 'kway_stats',
]
