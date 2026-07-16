#!/usr/bin/env python3
"""
run_otoc_circuit.py
===================
Focused, CLI-driven runner for benchmarking circuit-partition methods on a
single circuit family — defaulting to the OTOC (Out-of-Time-Order Correlator)
family. Designed as a lightweight companion to ``run_benchmark_circuit_partition.py``:
same partition methods, same metrics, same per-seed output layout, but
scoped to one family at a time and parameterised from the command line.

Examples
--------
    # Default: OTOC across n ∈ {8..12}, 3 seeds, depth 2
    python run_otoc_circuit.py

    # Different family, custom qubit list and seeds
    python run_otoc_circuit.py --family QFT --qubits 8,10,12 --seeds 19,23

    # OTOC with deeper scrambling unitary
    python run_otoc_circuit.py --otoc-depth 3

    # Explicit output directory (otherwise auto-named per seed)
    python run_otoc_circuit.py --outdir my_otoc_run

Output layout (per seed, identical to the main benchmark):
    <outdir>_seed<SEED>/
        circuits/<FAMILY>/n<N>/
            full/                full_layout.pkl, full_diagram.png
            methods/<METHOD>/    A/B_layout.pkl, A/B_diagram.png, stats.json
        results.csv
        results.json
        plots/                   research-grade box plots, heatmaps, Pareto, win-counts
"""
import argparse
import csv
import datetime
import json
import math
import os
import pickle
import random
import sys
import threading
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ── Project infrastructure (same imports as run_benchmark_circuit_partition) ──
from compiler import compile_quantum_circuit
from quantum_circuits import CIRCUIT_FAMILIES as _CIRCUIT_FAMILIES_BASE
from quantum_circuits import FAMILY_SEEDS as _FAMILY_SEEDS_BASE
from quantum_circuits import gen_otoc

from run_random_circuits import (
    build_interaction_graph,
    _naive_bisection, _spectral_bisection, _louvain_bisection,
    _girvan_newman_bisection, _metis_bisection, _try_qdislib,
    classify_gates, compute_cut_stats, compute_qpd_overhead,
    build_subcircuit_layout, compile_subcircuit,
    _entanglement_entropy, _get_statevector,
    save_circuit_diagram_qiskit, _save_circuit_diagram_fallback,
    _simulate_subcircuit, _build_recon_index, _prob_from_statevector,
    _materialize_layout_circuit,
    _compute_distribution_metrics, _compute_unified_scores,
    # mosaic ablation variants
    partition_graph,           # mosaic-KL
    _mosaic_syment_static,     # mosaic-OE
    mosaic_plus_plus,          # mosaic-MI
    _mosaic_adaptive,          # mosaic-BF
)
from mosaic_lct import mosaic_lct

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_QUBITS = [11]
DEFAULT_SEEDS  = [2,3,5,7,11,13,17,42,43,44,19,23,29,31,37,41,47,53,59,61,67,71,73,79,83,89,97,101,103,107]
DEFAULT_FAMILY = 'OTOC'
HIGHLIGHT_METHOD = 'mosaic-LCT'

# Available families — start with the registry from quantum_circuits.py and
# allow callers to extend. (The main benchmark also exposes 'RQC', which we do
# not include here because it pulls a heavier generator chain; users wanting
# RQC sweeps should use run_benchmark_circuit_partition.py.)
CIRCUIT_FAMILIES = dict(_CIRCUIT_FAMILIES_BASE)
FAMILY_SEEDS     = dict(_FAMILY_SEEDS_BASE)

# Method order — canonical for plotting (baselines → mosaic ladder → mosaic-LCT)
METHOD_ORDER = ['Naive', 'Spectral', 'Louvain', 'Girvan-Newman', 'METIS', 'qdislib',
                'mosaic-KL', 'mosaic-OE', 'mosaic-MI', 'mosaic-BF', 'mosaic-LCT']

BASELINE_COLOURS = {
    'Naive':         '#95A5A6',
    'Spectral':      '#7F8C8D',
    'Louvain':       '#5D6D7E',
    'Girvan-Newman': '#85929E',
    'METIS':         '#34495E',
    'qdislib':       '#566573',
}
MOSAIC_RAMP = {
    'mosaic-KL':  '#A9DFBF',
    'mosaic-OE':  '#7DCEA0',
    'mosaic-MI':  '#52BE80',
    'mosaic-BF':  '#2ECC71',
    'mosaic-LCT': '#1E8449',
}


# ──────────────────────────────────────────────────────────────────────────────
#  PARTITION-METHOD DISPATCH  (mirrors benchmark)
# ──────────────────────────────────────────────────────────────────────────────
def get_all_partition_methods(G, layout, n_qubits, qc_full, seed, outdir):
    """Return {method_name: (A_set, B_set, runtime_sec)}."""
    methods = {}

    def _time_call(name, fn):
        t0 = time.time()
        try:
            res = fn()
            dt = time.time() - t0
            if isinstance(res, tuple) and len(res) == 2:
                A, B = set(res[0]), set(res[1])
                if A and B:
                    methods[name] = (A, B, dt)
            elif isinstance(res, dict) and res.get('available') and 'A_set' in res:
                A, B = set(res['A_set']), set(res['B_set'])
                if A and B:
                    methods[name] = (A, B, dt)
        except Exception as e:
            print(f"      [warn] {name:14s} failed: {e}")

    # Non-mosaic baselines
    _time_call('Naive',         lambda: _naive_bisection(n_qubits))
    _time_call('Spectral',      lambda: _spectral_bisection(G, n_qubits))
    _time_call('Louvain',       lambda: _louvain_bisection(G, n_qubits))
    _time_call('Girvan-Newman', lambda: _girvan_newman_bisection(G, n_qubits))
    _time_call('METIS',         lambda: _metis_bisection(G, n_qubits))
    _time_call('qdislib',       lambda: _try_qdislib(layout, n_qubits))

    # mosaic ablation ladder
    _time_call('mosaic-KL', lambda: partition_graph(G, n_qubits))
    _time_call('mosaic-OE', lambda: _mosaic_syment_static(G, layout, n_qubits))
    _time_call('mosaic-MI', lambda: mosaic_plus_plus(G, layout, n_qubits, qc_full=qc_full))
    _time_call('mosaic-BF', lambda: _mosaic_adaptive(G, layout, n_qubits, qc_full, outdir))

    # Main method: mosaic-LCT
    _time_call('mosaic-LCT', lambda: mosaic_lct(G, layout, n_qubits,
                                                qc_full=qc_full, seed=seed))
    return methods


# ──────────────────────────────────────────────────────────────────────────────
#  DIAGRAM HELPER
# ──────────────────────────────────────────────────────────────────────────────
def save_diagram(qc, filepath, title=''):
    try:
        save_circuit_diagram_qiskit(qc, filepath, title)
    except Exception:
        try:
            _save_circuit_diagram_fallback(qc, filepath, title)
        except Exception as e:
            print(f"      [warn] diagram failed for {filepath.name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  SIMULATION-BASED COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_full(layout, n_qubits):
    if n_qubits > 24:
        return None
    try:
        sv = _get_statevector(_materialize_layout_circuit(layout, n_qubits))
        return _prob_from_statevector(sv) if sv is not None else None
    except Exception:
        return None


def _simulate_method_partitions(method_partitions, layout, n_qubits, p_ideal):
    """Simulate every method's A & B subcircuits in parallel and compute
    reconstruction-distance metrics. Returns {method: metrics_dict}."""
    results = {}
    if p_ideal is None or n_qubits > 24:
        return {m: {'sim_error': 'skipped (n_qubits>24 or full SV failed)'}
                for m in method_partitions}

    sim_data = {}
    lock = threading.Lock()

    def _run(lbl, A, B):
        try:
            pA, sA = _simulate_subcircuit(A, B, layout, n_qubits, 'A')
            pB, sB = _simulate_subcircuit(A, B, layout, n_qubits, 'B')
            with lock:
                sim_data[lbl] = (pA, sA, pB, sB)
        except Exception as e:
            with lock:
                sim_data[lbl] = ('error', str(e))

    max_w = min(len(method_partitions), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futs = [pool.submit(_run, lbl, A, B)
                for lbl, (A, B, _) in method_partitions.items()]
        for f in as_completed(futs):
            try: f.result()
            except Exception: pass

    for lbl, data in sim_data.items():
        if (isinstance(data, tuple) and len(data) == 4
                and data[0] is not None and data[2] is not None):
            pA, sA, pB, sB = data
            try:
                idxA = _build_recon_index(sA, n_qubits)
                idxB = _build_recon_index(sB, n_qubits)
                p_rec = pA[idxA] * pB[idxB]
                s = float(p_rec.sum())
                if s <= 0:
                    results[lbl] = {'sim_error': 'reconstruction sum is zero'}
                    continue
                p_rec /= s
                m = _compute_distribution_metrics(p_ideal, p_rec)
                results[lbl] = {**m, 'error': None}
            except Exception as e:
                results[lbl] = {'sim_error': str(e)}
        else:
            err = (data[1] if isinstance(data, tuple) and data[0] == 'error'
                   else 'sim failed')
            results[lbl] = {'sim_error': err}

    valid = {k: v for k, v in results.items()
             if v.get('error') is None and 'tvd' in v}
    if valid:
        for k, v in _compute_unified_scores(valid).items():
            results[k] = v
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  BENCHMARK ONE  (family, n_qubits, seed)  →  list of row dicts
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_one(family, gen_fn, n_qubits, family_dir, seed, outdir,
                  family_kwargs=None):
    family_kwargs = family_kwargs or {}
    rng = random.Random(seed + n_qubits * 17 + FAMILY_SEEDS[family])
    layout = gen_fn(n_qubits, rng, **family_kwargs)
    qc_full, _ = compile_quantum_circuit(
        layout, num_qubits=n_qubits, use_numeric_params=True)
    G = build_interaction_graph(layout, n_qubits)

    n_dir       = family_dir / f"n{n_qubits}"
    full_dir    = n_dir / "full"
    methods_dir = n_dir / "methods"
    full_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)

    with open(full_dir / "full_layout.pkl", 'wb') as f:
        pickle.dump({'layout': layout, 'n_qubits': n_qubits, 'family': family,
                     'depth': qc_full.depth(), 'n_gates': qc_full.size()}, f)
    save_diagram(qc_full, full_dir / "full_diagram.png",
                 f"{family} (n={n_qubits})")

    sv = None
    if n_qubits <= 24:
        try:
            sv = _get_statevector(qc_full)
        except Exception:
            sv = None

    methods = get_all_partition_methods(G, layout, n_qubits, qc_full, seed, outdir)

    print(f"      [sim]  full SV  + {len(methods)} subcircuit reconstructions ...")
    p_ideal = _simulate_full(layout, n_qubits)
    sim_metrics = _simulate_method_partitions(methods, layout, n_qubits, p_ideal)

    rows = []
    total_gates = sum(len(l) for l in layout)
    n_two_full  = sum(1 for l in layout for g in l if len(g['qubits']) >= 2)

    for mname, (A, B, dt) in methods.items():
        try:
            cs   = compute_cut_stats(G, A, B)
            gc   = classify_gates(layout, A, B)
            qpd  = compute_qpd_overhead(gc['cross'], sym_reduction=True)
            entropy = (_entanglement_entropy(sv, sorted(A), n_qubits)
                       if sv is not None else None)

            slA, snA, _ = build_subcircuit_layout(layout, A, 'A', gc)
            slB, snB, _ = build_subcircuit_layout(layout, B, 'B', gc)
            qcA, _ = compile_subcircuit(slA, len(A))
            qcB, _ = compile_subcircuit(slB, len(B))

            safe = mname.replace('/', '_').replace(' ', '_')
            mdir = methods_dir / safe
            mdir.mkdir(parents=True, exist_ok=True)
            for pn, sl, sn, qc, ns in [('A', slA, snA, qcA, A),
                                       ('B', slB, snB, qcB, B)]:
                with open(mdir / f"{pn}_layout.pkl", 'wb') as f:
                    pickle.dump({'layout': sl, 'n_qubits': len(ns),
                                 'sorted_nodes': sn, 'partition': sorted(ns),
                                 'method': mname, 'family': family,
                                 'depth': qc.depth(), 'n_gates': qc.size()}, f)
                save_diagram(
                    qc, mdir / f"{pn}_diagram.png",
                    f"{family} | {mname} | {pn} "
                    f"({len(ns)}q, {qc.size()}g, d={qc.depth()})")

            sm = sim_metrics.get(mname, {})
            row = {
                'family': family, 'n_qubits': n_qubits, 'method': mname,
                'n_A': len(A), 'n_B': len(B),
                'balance_ratio': round(min(len(A), len(B)) / max(len(A), len(B), 1), 4),
                'n_total_gates': total_gates, 'n_two_qubit_full': n_two_full,
                'n_local_A': len(gc['local_A']), 'n_local_B': len(gc['local_B']),
                'n_cross_gates': len(gc['cross']),
                'n_cut_edges': cs['n_cut_edges'],
                'cut_weight': cs['cut_weight'],
                'cut_fraction': cs['cut_fraction'],
                'gamma_generic': qpd['total_gamma_generic'],
                'gamma_sym': qpd['total_gamma_sym'],
                'log10_gamma_sym':     math.log10(max(qpd['total_gamma_sym'],     1e-12)),
                'log10_gamma_generic': math.log10(max(qpd['total_gamma_generic'], 1e-12)),
                'gamma_reduction_ratio': round(
                    qpd['total_gamma_sym'] / max(qpd['total_gamma_generic'], 1e-30), 6),
                'schmidt_total': qpd['total_schmidt_rank'],
                'entanglement_entropy': entropy,
                'partition_runtime_sec': round(dt, 4),
                'depth_A': qcA.depth(), 'depth_B': qcB.depth(),
                'gates_A': qcA.size(), 'gates_B': qcB.size(),
                'tvd':            sm.get('tvd'),
                'fidelity':       sm.get('fidelity'),
                'kl_divergence':  sm.get('kl_divergence'),
                'hellinger':      sm.get('hellinger'),
                'js_divergence':  sm.get('js_divergence'),
                'cross_entropy':  sm.get('cross_entropy'),
                'unified_score':  sm.get('unified_score'),
                'sim_rank':       sm.get('rank'),
                'sim_error':      sm.get('sim_error'),
            }
            rows.append(row)

            ent_s = f"{entropy:.3f}" if entropy is not None else "  -- "
            tvd_s = f"{sm['tvd']:.4f}"      if sm.get('tvd') is not None else "  --  "
            fid_s = f"{sm['fidelity']:.4f}" if sm.get('fidelity') is not None else "  --  "
            us_s  = f"{sm['unified_score']:.3f}" if sm.get('unified_score') is not None else " -- "
            print(f"      {mname:14s}  cut={cs['n_cut_edges']:3d}  "
                  f"cross={len(gc['cross']):3d}  "
                  f"γs={qpd['total_gamma_sym']:.2e}  S={ent_s}  "
                  f"TVD={tvd_s}  F={fid_s}  U={us_s}  t={dt:.2f}s")

            with open(mdir / "stats.json", 'w') as f:
                json.dump(row, f, indent=2, default=str)
        except Exception as e:
            print(f"      [warn] eval {mname} failed: {e}")
            traceback.print_exc()

    return rows


# ──────────────────────────────────────────────────────────────────────────────
#  CSV WRITER
# ──────────────────────────────────────────────────────────────────────────────
def save_csv(rows, csv_path):
    if not rows:
        print("   [warn] No rows to save to CSV"); return
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f"   Saved CSV: {csv_path}  ({len(rows)} rows × {len(keys)} cols)")


# ──────────────────────────────────────────────────────────────────────────────
#  PLOTS  (same format as the main benchmark; trivially degrades when there is
#          only one family — the per-family panels collapse to a single panel,
#          but the box-plots, Pareto, win-counts and ranking bar all still work)
# ──────────────────────────────────────────────────────────────────────────────
def _gather(rows, key, method=None, family=None, n=None):
    return [r[key] for r in rows
            if (method is None or r['method'] == method)
            and (family is None or r['family'] == family)
            and (n is None or r['n_qubits'] == n)
            and r.get(key) is not None]


def make_plots(rows, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    families = sorted(set(r['family'] for r in rows),
                      key=lambda f: list(CIRCUIT_FAMILIES).index(f)
                                    if f in CIRCUIT_FAMILIES else 999)
    seen_methods = set(r['method'] for r in rows)
    methods = ([m for m in METHOD_ORDER if m in seen_methods]
               + [m for m in seen_methods if m not in METHOD_ORDER])
    qubits = sorted(set(r['n_qubits'] for r in rows))

    method_colors = {}
    for m in methods:
        if m in BASELINE_COLOURS:
            method_colors[m] = BASELINE_COLOURS[m]
        elif m in MOSAIC_RAMP:
            method_colors[m] = MOSAIC_RAMP[m]
        else:
            method_colors[m] = '#BDC3C7'

    metrics = [
        ('n_cut_edges',          'Cut Edges',                  False),
        ('n_cross_gates',        'Cross Gates',                False),
        ('log10_gamma_sym',      'log₁₀(γ_sym)',               False),
        ('cut_fraction',         'Cut Fraction',               False),
        ('balance_ratio',        'Balance Ratio',              True),
        ('entanglement_entropy', 'Cut Entanglement S',         False),
        ('partition_runtime_sec','Runtime [s]',                False),
        ('tvd',                  'TVD vs ideal',               False),
        ('fidelity',             'Fidelity vs ideal',          True),
        ('kl_divergence',        'KL divergence',              False),
        ('hellinger',            'Hellinger distance',         False),
        ('js_divergence',        'JS divergence',              False),
        ('unified_score',        'Unified Score',              True),
    ]
    n_plots = 0

    # 1. GLOBAL BOX-PLOT GRID
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows))
    axf = axes.flatten()
    for ax, (key, title, hb) in zip(axf, metrics):
        m_vals = {m: _gather(rows, key, method=m) for m in methods}
        m_vals = {m: v for m, v in m_vals.items() if v}
        if not m_vals:
            ax.axis('off'); continue
        order = sorted(m_vals.keys(),
                       key=lambda m: -np.median(m_vals[m]) if hb else np.median(m_vals[m]))
        data = [m_vals[m] for m in order]
        cols = [method_colors[m] for m in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6,
                        showmeans=True,
                        meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                   'markeredgecolor': 'black', 'markersize': 5})
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(cols[i]); patch.set_alpha(0.78)
            if order[i] == HIGHLIGHT_METHOD:
                patch.set_edgecolor('#1B5E20'); patch.set_linewidth(2.4)
        for med in bp['medians']:
            med.set_color('black'); med.set_linewidth(1.5)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    for ax in axf[len(metrics):]:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(plots_dir / "global_boxplot_grid.png", dpi=130, bbox_inches='tight')
    plt.close(fig); n_plots += 1

    # 2. SCALING BOX PLOTS (per n_qubits, per metric)
    gap = 1
    for key, title, hb in metrics:
        m_sorted = methods
        positions, data_seq, labels, colors_seq = [], [], [], []
        for qi, n in enumerate(qubits):
            for mi, m in enumerate(m_sorted):
                v = _gather(rows, key, method=m, n=n)
                if v:
                    positions.append(qi * (len(m_sorted) + gap) + mi)
                    labels.append(m)
                    colors_seq.append(method_colors[m])
                    data_seq.append(v)
        if not data_seq:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(qubits) * (len(m_sorted) + gap) * 0.55), 5.5))
        bp = ax.boxplot(data_seq, positions=positions, patch_artist=True, widths=0.6,
                        showmeans=True,
                        meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                   'markeredgecolor': 'black', 'markersize': 4})
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors_seq[i]); patch.set_alpha(0.78)
            if labels[i] == HIGHLIGHT_METHOD:
                patch.set_edgecolor('#1B5E20'); patch.set_linewidth(2.0)
        center_per_n = [qi * (len(m_sorted) + gap) + (len(m_sorted) - 1) / 2
                        for qi in range(len(qubits))]
        ax.set_xticks(center_per_n)
        ax.set_xticklabels([f'n={n}' for n in qubits], fontsize=11)
        legend_h = [Patch(facecolor=method_colors[m], edgecolor='white', label=m)
                    for m in m_sorted]
        ax.legend(handles=legend_h, loc='best', fontsize=8, ncol=2)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'Scaling — {title} per n_qubits  '
                     f'({"higher=better" if hb else "lower=better"})',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout()
        fig.savefig(plots_dir / f"scaling_box_{key}.png", dpi=130, bbox_inches='tight')
        plt.close(fig); n_plots += 1

    # 3. PARETO  (fidelity vs log10 γ_sym)
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in methods:
        xs = [r['log10_gamma_sym'] for r in rows
              if r['method'] == m and r.get('fidelity') is not None
              and r.get('log10_gamma_sym') is not None]
        ys = [r['fidelity'] for r in rows
              if r['method'] == m and r.get('fidelity') is not None
              and r.get('log10_gamma_sym') is not None]
        if xs:
            sz = 75 if m == HIGHLIGHT_METHOD else 45
            ec = 'black' if m == HIGHLIGHT_METHOD else 'white'
            ax.scatter(xs, ys, color=method_colors[m], label=m, s=sz, alpha=0.78,
                       edgecolors=ec, linewidths=0.6)
    ax.set_xlabel('log₁₀(γ_sym)  [overhead, lower is better]', fontsize=11)
    ax.set_ylabel('Bhattacharyya Fidelity vs ideal  [higher is better]', fontsize=11)
    ax.set_title('Pareto Trade-off: Fidelity vs Sampling Overhead',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "pareto_fidelity_vs_gamma.png", dpi=130, bbox_inches='tight')
    plt.close(fig); n_plots += 1

    # 4. WIN-COUNT MATRIX  (best unified score per family/n)
    sim_win = np.zeros((len(methods), len(families)), dtype=int)
    for j, fam in enumerate(families):
        for n in qubits:
            sub = [(r['method'], r['unified_score']) for r in rows
                   if r['family'] == fam and r['n_qubits'] == n
                   and r.get('unified_score') is not None]
            if not sub:
                continue
            sub.sort(key=lambda x: -x[1])
            best = sub[0][0]
            if best in methods:
                sim_win[methods.index(best), j] += 1
    if sim_win.sum() > 0:
        order = list(np.argsort(-sim_win.sum(axis=1)))
        sim_win = sim_win[order]
        m_disp = [methods[i] for i in order]
        fig, ax = plt.subplots(figsize=(max(len(families) * 1.5, 6),
                                        max(len(m_disp) * 0.55, 4.5)))
        im = ax.imshow(sim_win, aspect='auto', cmap='Greens',
                       vmin=0, vmax=max(sim_win.max(), 1))
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=20, ha='right')
        ax.set_yticks(range(len(m_disp))); ax.set_yticklabels(m_disp)
        for i in range(len(m_disp)):
            for j in range(len(families)):
                v = sim_win[i, j]
                ax.text(j, i, str(v), ha='center', va='center', fontsize=10,
                        color='white' if v > sim_win.max() / 2 else 'black',
                        fontweight='bold')
        plt.colorbar(im, ax=ax, label='# wins (highest unified score)')
        ax.set_title('Win-Count — # times method achieves best reconstruction',
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        fig.savefig(plots_dir / "win_count_unified_score.png", dpi=130, bbox_inches='tight')
        plt.close(fig); n_plots += 1

    # 5. AGGREGATE RANKING BAR  (mean unified score)
    score_data = defaultdict(list)
    for r in rows:
        if r.get('unified_score') is not None:
            score_data[r['method']].append(r['unified_score'])
    if score_data:
        m_score = {m: float(np.mean(v)) for m, v in score_data.items()}
        sorted_m = sorted(m_score.keys(), key=lambda m: -m_score[m])
        fig, ax = plt.subplots(figsize=(max(len(sorted_m) * 1.4, 10), 6))
        vals = [m_score[m] for m in sorted_m]
        cols = [method_colors[m] for m in sorted_m]
        bars = ax.bar(range(len(sorted_m)), vals, color=cols, edgecolor='white')
        for i, m in enumerate(sorted_m):
            if m == HIGHLIGHT_METHOD:
                bars[i].set_edgecolor('#1B5E20'); bars[i].set_linewidth(2.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(vals) * 0.01, f'{v:.3f}',
                    ha='center', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(sorted_m)))
        ax.set_xticklabels(sorted_m, rotation=15, ha='right')
        ax.set_ylabel('Mean Unified Score (higher = better)', fontsize=11)
        ax.set_title('Aggregate Ranking — Reconstruction Quality (Unified Score)',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout()
        fig.savefig(plots_dir / "ranking_unified_score.png", dpi=130, bbox_inches='tight')
        plt.close(fig); n_plots += 1

    print(f"   Saved {n_plots} plots to {plots_dir}")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI + DRIVER
# ──────────────────────────────────────────────────────────────────────────────
def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--family', default=DEFAULT_FAMILY,
                   choices=sorted(CIRCUIT_FAMILIES.keys()),
                   help=f'Circuit family to run (default: {DEFAULT_FAMILY}). '
                        f'Available: {sorted(CIRCUIT_FAMILIES.keys())}')
    p.add_argument('--qubits', default=','.join(map(str, DEFAULT_QUBITS)),
                   help=f'Comma-separated qubit counts '
                        f'(default: {",".join(map(str, DEFAULT_QUBITS))})')
    p.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)),
                   help=f'Comma-separated PRNG seeds '
                        f'(default: {",".join(map(str, DEFAULT_SEEDS))})')
    p.add_argument('--outdir', default=None,
                   help='Base output directory (the script appends _seed<SEED> per run). '
                        'Default: benchmark_<TS>_<FAMILY>')
    p.add_argument('--otoc-depth', type=int, default=2,
                   help='Brick-wall depth of the OTOC scrambling unitary U '
                        '(only used when --family OTOC). Default: 2')
    return p.parse_args(argv)


def run_single_seed(seed, family, qubits, base_outdir, family_kwargs):
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"{base_outdir}_seed{seed}_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    random.seed(seed); np.random.seed(seed)

    print('=' * 78)
    print(f' OTOC-runner   |   seed = {seed}')
    print(f'   Family:           {family}')
    print(f'   Qubit counts:     {qubits}')
    if family_kwargs:
        print(f'   Family kwargs:    {family_kwargs}')
    print(f'   Output directory: {outdir.resolve()}')
    print(f'   Baselines:        Naive, Spectral, Louvain, Girvan-Newman, METIS, qdislib')
    print(f'   mosaic ablation:  mosaic-KL, mosaic-OE, mosaic-MI, mosaic-BF')
    print(f'   Main method:      mosaic-LCT  (Light-Cone Tempering)')
    print('=' * 78)

    gen_fn = CIRCUIT_FAMILIES[family]
    family_dir = outdir / "circuits" / family
    family_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    t0 = time.time()
    print(f"\n[{family}]")
    for n in qubits:
        print(f"   ── n_qubits = {n} ──")
        try:
            rows = benchmark_one(family, gen_fn, n, family_dir, seed, outdir,
                                 family_kwargs=family_kwargs)
            all_rows.extend(rows)
        except Exception as e:
            print(f"   [error] {family} n={n}: {e}")
            traceback.print_exc()

    save_csv(all_rows, outdir / "results.csv")
    with open(outdir / "results.json", 'w') as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"   Saved JSON: {outdir / 'results.json'}")

    print("\n[Plots] Generating box-plot / Pareto / win-count figures ...")
    make_plots(all_rows, outdir / "plots")

    elapsed = time.time() - t0
    print('\n' + '=' * 78)
    print(f' ✓ seed {seed} complete in {elapsed:.1f}s   |   total runs: {len(all_rows)}')
    print(f'   Output: {outdir.resolve()}')
    print('=' * 78)
    return outdir, len(all_rows)


def main(argv=None):
    args = parse_args(argv)
    qubits = _parse_int_list(args.qubits)
    seeds  = _parse_int_list(args.seeds)
    family = args.family

    base_outdir = args.outdir or f"benchmark_{family}"

    # Family-specific knobs. Only OTOC is parameterised here; other families
    # use whatever defaults their generator already has.
    family_kwargs = {}
    if family == 'OTOC':
        family_kwargs['depth'] = args.otoc_depth

    grand_total = 0
    grand_t0 = time.time()
    out_dirs = []
    for seed in seeds:
        out, nrows = run_single_seed(seed, family, qubits,
                                     base_outdir, family_kwargs)
        out_dirs.append(out)
        grand_total += nrows

    print('\n' + '#' * 78)
    print(f'# All seeds done. Total rows: {grand_total} across {len(seeds)} seed(s).')
    for d in out_dirs:
        print(f'#   {d}')
    print(f'# Elapsed: {time.time() - grand_t0:.1f}s')
    print('#' * 78)


if __name__ == '__main__':
    main()
