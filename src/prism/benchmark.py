"""
prism.benchmark
===============
The end-to-end benchmark driver: run *all* partition methods on *all* circuit
families × qubit counts × seeds, evaluate each cut by exact statevector
reconstruction, and collect a tidy results table.

    evaluate_partition   structural + reconstruction metrics for one cut
    benchmark_instance   all methods on one (family, n, seed) instance
    run_benchmark        the full grid -> list of row dicts (or DataFrame)
    summarise_benchmark  per-method aggregate (mean Q_Score, rank, runtime)
"""
from __future__ import annotations

import math
import time

from .circuits import make_layout, CIRCUIT_FAMILIES
from .compiler import compile_circuit
from .graph import build_interaction_graph, classify_gates, compute_cut_stats
from .symmetry import compute_qpd_overhead
from .simulate import (get_statevector, prob_from_statevector, entanglement_entropy,
                       reconstruct_product, distribution_metrics, q_score,
                       compute_unified_scores)
from .partition import run_all_partition_methods, METHOD_ORDER

DEFAULT_QUBITS = [6, 8, 10, 12]
DEFAULT_SEEDS = [19, 23, 29, 31, 37]
DEFAULT_FAMILIES = ['QFT', 'QPE', 'VQE-HEA', 'VQE-UCC', 'QAOA', 'QNN', 'MERA', 'MPS', 'RQC']


def evaluate_partition(A, B, G, layout, n_qubits, sv_full=None, p_ideal=None):
    """Structural + reconstruction metrics for a single bipartition (A, B)."""
    A, B = set(A), set(B)
    cs = compute_cut_stats(G, A, B)
    gc = classify_gates(layout, A, B)
    qpd = compute_qpd_overhead(gc['cross'], sym_reduction=True)
    row = {
        'n_A': len(A), 'n_B': len(B),
        'balance': round(min(len(A), len(B)) / max(len(A), len(B), 1), 4),
        'n_cut_edges': cs['n_cut_edges'], 'cut_weight': cs['cut_weight'],
        'cut_fraction': cs['cut_fraction'], 'n_cross_gates': len(gc['cross']),
        'gamma_sym': qpd['total_gamma_sym'],
        'log10_gamma_sym': math.log10(max(qpd['total_gamma_sym'], 1e-12)),
        'total_schmidt_rank': qpd['total_schmidt_rank'],
        'entanglement_entropy': None,
        'tvd': None, 'fidelity': None, 'kl_divergence': None,
        'hellinger': None, 'js_divergence': None, 'cross_entropy': None,
        'q_score': None, 'error': None,
    }
    if sv_full is not None:
        try:
            row['entanglement_entropy'] = entanglement_entropy(sv_full, sorted(A), n_qubits)
        except Exception:
            pass
    if p_ideal is not None:
        p_rec = reconstruct_product(A, B, layout, n_qubits)
        if p_rec is not None:
            dm = distribution_metrics(p_ideal, p_rec)
            row.update(dm)
            row['q_score'] = q_score(dm)
        else:
            row['error'] = 'reconstruction failed'
    return row


def benchmark_instance(family, n, seed, methods=None, simulate=True, rqc_depth=None):
    """Run every method on one (family, n, seed) instance. Returns a list of
    per-method row dicts including a relative ``unified_score``/``rank``."""
    layout = make_layout(family, n, seed) if rqc_depth is None else make_layout(family, n, seed, rqc_depth)
    qc_full, _ = compile_circuit(layout, num_qubits=n, use_numeric_params=True)
    G = build_interaction_graph(layout, n)
    sv = p_ideal = None
    if simulate and n <= 20:
        sv = get_statevector(qc_full)
        p_ideal = prob_from_statevector(sv) if sv is not None else None

    results, failures = run_all_partition_methods(G, layout, n, qc_full=qc_full,
                                                  seed=seed, methods=methods)
    rows, recon = [], {}
    for name, (A, B, dt) in results.items():
        ev = evaluate_partition(A, B, G, layout, n, sv_full=sv, p_ideal=p_ideal)
        ev.update({'family': family, 'n_qubits': n, 'seed': seed, 'method': name,
                   'runtime_s': dt})
        rows.append(ev)
        if ev.get('tvd') is not None:
            recon[name] = {k: ev[k] for k in ('tvd', 'fidelity', 'kl_divergence',
                                              'hellinger', 'js_divergence', 'cross_entropy')}
            recon[name]['error'] = None
    scored = compute_unified_scores(recon)
    for r in rows:
        s = scored.get(r['method'], {})
        r['unified_score'] = s.get('unified_score')
        r['rank'] = s.get('rank')
    return rows, failures


def run_benchmark(families=None, qubits=None, seeds=None, methods=None,
                  simulate=True, out=None, verbose=True, as_dataframe=True):
    """Run the full benchmark grid.

    Parameters
    ----------
    families, qubits, seeds : lists (defaults are laptop-friendly)
    methods : subset of method names, or None for all
    simulate : exact statevector reconstruction (needs n <= 20)
    out : optional path; writes ``<out>.csv`` (and ``.json`` if pandas absent)
    as_dataframe : return a pandas DataFrame when pandas is available

    Returns the results as a DataFrame (or list of dicts).
    """
    families = families or DEFAULT_FAMILIES
    qubits = qubits or DEFAULT_QUBITS
    seeds = seeds or DEFAULT_SEEDS
    rows, fail_log = [], {}
    t0 = time.time()
    for family in families:
        for n in qubits:
            for seed in seeds:
                inst, fails = benchmark_instance(family, n, seed, methods=methods, simulate=simulate)
                rows.extend(inst)
                for k, v in fails.items():
                    fail_log[k] = fail_log.get(k, 0) + 1
        if verbose:
            print(f'  {family:8s} done  ({time.time() - t0:6.0f} s, {len(rows)} rows)')
    if verbose and fail_log:
        print('  method failures:', fail_log)

    if out is not None:
        _write_results(rows, out)

    if as_dataframe:
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            pass
    return rows


def _write_results(rows, out):
    import json
    import os
    base = os.fspath(out)
    if base.endswith('.csv'):
        base = base[:-4]
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(base + '.csv', index=False)
    except ImportError:
        with open(base + '.json', 'w') as f:
            json.dump(rows, f, indent=2)


def summarise_benchmark(df):
    """Per-method aggregate: mean Q_Score, mean rank, mean runtime, win rate."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError('summarise_benchmark requires pandas')
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    g = df.groupby('method')
    summary = pd.DataFrame({
        'mean_q_score': g['q_score'].mean(),
        'mean_rank': g['rank'].mean(),
        'mean_tvd': g['tvd'].mean(),
        'mean_fidelity': g['fidelity'].mean(),
        'mean_runtime_s': g['runtime_s'].mean(),
        'n_instances': g.size(),
    })
    if 'rank' in df.columns:
        wins = df[df['rank'] == 1].groupby('method').size()
        summary['win_count'] = wins.reindex(summary.index).fillna(0).astype(int)
    order = [m for m in METHOD_ORDER if m in summary.index]
    return summary.reindex(order)


__all__ = [
    'evaluate_partition', 'benchmark_instance', 'run_benchmark', 'summarise_benchmark',
    'DEFAULT_QUBITS', 'DEFAULT_SEEDS', 'DEFAULT_FAMILIES',
]
