#!/usr/bin/env python3
"""
benchmark_partitioning.py
=========================
Independent benchmark of mosaic-LCT (the main method) against the mosaic
ablation ladder and a battery of non-mosaic baselines, on 8 circuit families
× 4 qubit counts.

Circuit families: QFT, QPE, VQE-HEA, VQE-UCC, QAOA, QNN, MERA, MPS, OTOC
Qubit counts:     {10, 12, 14, 16}

Methods compared:

  Baselines (non-mosaic):
    Naive          half-split
    Spectral       Fiedler bisection
    Louvain        modularity community detection
    Girvan-Newman  betweenness-based hierarchical cuts
    METIS          multilevel k-way (if pymetis available)
    qdislib        DAG-cut (if available)

  mosaic ablation ladder (each adds one mechanism):
    mosaic-KL        Kernighan–Lin on gate-count graph              (structural)
    mosaic-OE        + Operator-Entanglement edge weighting         (structural SA)
    mosaic-MI        + physics cost with Mutual Information         (SV-aware SA)
    mosaic-BF        + Boundary-Focused move proposals              (SV-aware SA)

  Main method:
    mosaic-LCT       + Light-Cone graph + Parallel Tempering + Tabu
                   + Consensus + Greedy Polish

For each (family, n_qubits, method):
    * partition + structural metrics (cut edges, cross gates, γ_sym, ...)
    * subcircuit reconstruction simulation:
          full statevector → p_ideal
          per-method  pA·pB → p_recon
          compute TVD / fidelity / KL / Hellinger / JS / cross-entropy
          unified score + rank
    * pickle + diagram for full circuit and each method's A/B subcircuit

Output:
    benchmark_<TS>/
        circuits/<FAMILY>/n<N>/
            full/                     full_layout.pkl, full_diagram.png
            methods/<METHOD>/         A/B_layout.pkl, A/B_diagram.png, stats.json
        results.csv     ← meta-results for downstream analysis
        results.json
        plots/          ← box-plot-driven research-grade figures
"""
import os, sys, json, pickle, datetime, math, random, time, csv, traceback, warnings, threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ── Existing project infrastructure ───────────────────────────────────────────
from compiler import compile_quantum_circuit
from qcirc_generator import generate_ansatz_layout
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
    # ── mosaic ablation variants (existing implementations in run_random_circuits) ──
    partition_graph,         # mosaic-KL : Kernighan–Lin on gate-count graph
    _mosaic_syment_static,     # mosaic-OE : Operator-Entanglement weighting + structural SA
    mosaic_plus_plus,          # mosaic-MI : physics cost (entropy + Mutual Information) + SA
    _mosaic_adaptive,          # mosaic-BF : Boundary-Focused refinement
)
# ── mosaic-LCT : Light-Cone Tempering — main method ────────────────────────────
from mosaic_lct import mosaic_lct

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
QUBIT_LIST = [8,9,10,11,12]
SEEDS = [19, 23, 29, 31, 37, 41, 47, 53, 59, 61,
         67, 71, 73, 79, 83, 89, 97, 101, 103, 107]
for SEED in SEEDS:
    TS         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTDIR     = Path(f"benchmark_{TS}_seed{SEED}")
    random.seed(SEED); np.random.seed(SEED)

    FAMILY_SEEDS = {'QFT':100,'QPE':200,'VQE-HEA':300,'VQE-UCC':400,
                    'QAOA':500,'QNN':600,'MERA':700,'MPS':800,'RQC':900,
                    'OTOC':1000}

    # RQC generator settings (same generator as the original run_random_circuits.py)
    RQC_DEPTH           = 8        # circuit depth for the RQC family
    RQC_MAX_Q_PER_LAYER = None     # None → uncapped (set to n at call time)

    HIGHLIGHT_METHOD = 'mosaic-LCT'

    # ──────────────────────────────────────────────────────────────────────────────
    #  CIRCUIT FAMILY GENERATORS  →  return layout (list of layers of gate dicts)
    # ──────────────────────────────────────────────────────────────────────────────
    def gen_qft(n, rng):
        layout = []
        for i in range(n):
            layout.append([{'gate':'H','qubits':[i]}])
            for j in range(i+1, n):
                layout.append([{'gate':'CPHASE','qubits':[j,i],
                                'params':[math.pi / (2 ** (j - i))]}])
        for i in range(n // 2):
            layout.append([{'gate':'SWAP','qubits':[i, n-1-i]}])
        return layout

    def gen_qpe(n, rng):
        n_anc = max(n // 2, 2); n_sys = n - n_anc
        layout = [[{'gate':'H','qubits':[i]} for i in range(n_anc)]]
        if n_sys > 0:
            layout.append([{'gate':'X','qubits':[n_anc]}])
        for k in range(n_anc):
            for s in range(n_sys):
                angle = rng.uniform(0.5, 1.5) * math.pi / (2**k)
                layout.append([{'gate':'CRZ','qubits':[k, n_anc+s],'params':[angle]}])
        for i in range(n_anc-1, -1, -1):
            for j in range(i+1, n_anc):
                layout.append([{'gate':'CPHASE','qubits':[j,i],
                                'params':[-math.pi / (2 ** (j - i))]}])
            layout.append([{'gate':'H','qubits':[i]}])
        return layout

    def gen_vqe_hea(n, rng, depth=3):
        layout = []
        for _ in range(depth):
            layout.append([{'gate':'RY','qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)])
            layout.append([{'gate':'RZ','qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)])
            for i in range(n - 1):
                layout.append([{'gate':'CNOT','qubits':[i,i+1]}])
        layout.append([{'gate':'RY','qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)])
        return layout

    def gen_vqe_ucc(n, rng):
        layout = []
        for i in range(n // 2):
            layout.append([{'gate':'X','qubits':[i]}])
        for i in range(n - 1):
            layout.append([{'gate':'CNOT','qubits':[i,i+1]}])
            layout.append([{'gate':'RY','qubits':[i],'params':[rng.uniform(0,2*math.pi)]}])
            layout.append([{'gate':'CNOT','qubits':[i,i+1]}])
        for i in range(0, n - 3, 2):
            layout.append([{'gate':'CNOT','qubits':[i,i+1]}])
            layout.append([{'gate':'CNOT','qubits':[i+2,i+3]}])
            layout.append([{'gate':'CNOT','qubits':[i+1,i+2]}])
            layout.append([{'gate':'RZ','qubits':[i+2],'params':[rng.uniform(0,2*math.pi)]}])
            layout.append([{'gate':'CNOT','qubits':[i+1,i+2]}])
            layout.append([{'gate':'CNOT','qubits':[i+2,i+3]}])
            layout.append([{'gate':'CNOT','qubits':[i,i+1]}])
        return layout

    def gen_qaoa(n, rng, p=2):
        layout = [[{'gate':'H','qubits':[i]} for i in range(n)]]
        for _ in range(p):
            gamma, beta = rng.uniform(0, math.pi), rng.uniform(0, math.pi)
            for i in range(n):
                j = (i + 1) % n
                if i < j:
                    layout.append([{'gate':'CNOT','qubits':[i,j]}])
                    layout.append([{'gate':'RZ','qubits':[j],'params':[2*gamma]}])
                    layout.append([{'gate':'CNOT','qubits':[i,j]}])
            layout.append([{'gate':'RX','qubits':[i],'params':[2*beta]} for i in range(n)])
        return layout

    def gen_qnn(n, rng, depth=3):
        layout = [[{'gate':'RY','qubits':[i],'params':[rng.uniform(0,math.pi)]} for i in range(n)]]
        for _ in range(depth):
            for axis in ('RX','RY','RZ'):
                layout.append([{'gate':axis,'qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)])
            for i in range(n - 1):
                layout.append([{'gate':'CZ','qubits':[i,i+1]}])
            if n > 2:
                layout.append([{'gate':'CZ','qubits':[0,n-1]}])
        return layout

    def gen_mera(n, rng):
        layout = [[{'gate':'H','qubits':[i]} for i in range(n)]]
        scale = 1
        while scale < n:
            d_layer, used = [], set()
            for i in range(scale, n - 1, 2*scale):
                q1, q2 = i, i + 1
                if q2 < n and q1 not in used and q2 not in used:
                    d_layer.append({'gate':'CRZ','qubits':[q1,q2],
                                    'params':[rng.uniform(0,2*math.pi)]})
                    used.update([q1, q2])
            if d_layer: layout.append(d_layer)
            i_layer, used = [], set()
            for i in range(0, n - scale, 2*scale):
                q1, q2 = i, i + scale
                if q2 < n and q1 not in used and q2 not in used:
                    i_layer.append({'gate':'CRY','qubits':[q1,q2],
                                    'params':[rng.uniform(0,2*math.pi)]})
                    used.update([q1, q2])
            if i_layer: layout.append(i_layer)
            scale *= 2
        return layout

    def gen_mps(n, rng, depth=3):
        layout = [[{'gate':'RY','qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)]]
        for _ in range(depth):
            even = [{'gate':'CNOT','qubits':[i,i+1]} for i in range(0, n-1, 2)]
            if even: layout.append(even)
            layout.append([{'gate':'RY','qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)])
            odd = [{'gate':'CNOT','qubits':[i,i+1]} for i in range(1, n-1, 2)]
            if odd: layout.append(odd)
            layout.append([{'gate':'RZ','qubits':[i],'params':[rng.uniform(0,2*math.pi)]} for i in range(n)])
        return layout

    def gen_rqc(n, rng, depth=None, max_q_per_layer=None):
        """Random Quantum Circuit — heterogeneous gate set sampled by qcirc_generator.

        This is the same generator used in the original run_random_circuits.py
        (Phase 0): a randomized layered ansatz that mixes 1-, 2-, 3-, and 4-qubit
        gates drawn from the project's gate taxonomy. Numeric parameters are
        sampled in-place from `rng` and stored on each gate dict, so the resulting
        layout is fully reproducible.
        """
        return generate_ansatz_layout(
            n,
            depth or RQC_DEPTH,
            max_qubits_per_layer=(max_q_per_layer or RQC_MAX_Q_PER_LAYER or n),
            rng=rng,
        )

    def gen_otoc(n, rng, depth=2, q_butterfly=None, q_perturb=None):
        """Out-of-Time-Order Correlator (OTOC) circuit.

        Canonical scrambling-echo protocol used to probe quantum information
        scrambling and many-body chaos:

                |+>^n ── U ── W ── U† ── V ── U ──

        where U is a brick-wall scrambling unitary (random Rz/Rx rotations
        alternated with even/odd CZ layers), W is a Pauli-X "butterfly" on
        qubit `q_butterfly` (default: middle qubit), and V is a perturbation
        Pauli-X on `q_perturb` (default: qubit 0). U† is the exact inverse of
        U (reversed layer/gate order; rotation angles negated; CZ, H, X
        self-inverse). The U/U†/U time-symmetric structure is structurally
        distinct from every other family in this benchmark and is a stress
        test for partition methods that exploit causal / light-cone structure.

        Parameters
        ----------
        n     : int            number of qubits
        rng   : random.Random  reproducible RNG (used for U rotation angles)
        depth : int            brick-wall depth of U (default 2)
        q_butterfly : int      qubit for W (default: n // 2)
        q_perturb   : int      qubit for V (default: 0)
        """
        if q_butterfly is None:
            q_butterfly = n // 2
        if q_perturb is None:
            q_perturb = 0

        layout = []
        layout.append([{'gate': 'H', 'qubits': [i]} for i in range(n)])

        def _build_U():
            sub = []
            for _ in range(depth):
                sub.append([{'gate': 'RZ', 'qubits': [i],
                             'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
                sub.append([{'gate': 'RX', 'qubits': [i],
                             'params': [rng.uniform(0, 2 * math.pi)]} for i in range(n)])
                even = [{'gate': 'CZ', 'qubits': [i, i + 1]}
                        for i in range(0, n - 1, 2)]
                if even: sub.append(even)
                odd = [{'gate': 'CZ', 'qubits': [i, i + 1]}
                       for i in range(1, n - 1, 2)]
                if odd: sub.append(odd)
            return sub

        def _copy_layers(layers):
            return [[{'gate': g['gate'], 'qubits': list(g['qubits']),
                      **({'params': list(g['params'])} if 'params' in g else {})}
                     for g in layer] for layer in layers]

        def _invert_U(U_layers):
            SIGN_FLIP = {'RX', 'RY', 'RZ', 'U1', 'PHASE',
                         'CRX', 'CRY', 'CRZ', 'CU1', 'CPHASE'}
            inv = []
            for layer in reversed(U_layers):
                new_layer = []
                for g in reversed(layer):
                    name = g['gate'].upper()
                    gd = {'gate': g['gate'], 'qubits': list(g['qubits'])}
                    if 'params' in g and g['params']:
                        gd['params'] = ([-p for p in g['params']]
                                        if name in SIGN_FLIP else list(g['params']))
                    new_layer.append(gd)
                inv.append(new_layer)
            return inv

        U_layers  = _build_U()
        Ud_layers = _invert_U(U_layers)

        layout.extend(_copy_layers(U_layers))                        # U
        layout.append([{'gate': 'X', 'qubits': [q_butterfly]}])      # W
        layout.extend(Ud_layers)                                     # U†
        layout.append([{'gate': 'X', 'qubits': [q_perturb]}])        # V
        layout.extend(_copy_layers(U_layers))                        # U
        return layout

    CIRCUIT_FAMILIES = {
        'QFT':     gen_qft,
        'QPE':     gen_qpe,
        'VQE-HEA': gen_vqe_hea,
        'VQE-UCC': gen_vqe_ucc,
        'QAOA':    gen_qaoa,
        'QNN':     gen_qnn,
        'MERA':    gen_mera,
        'MPS':     gen_mps,
        'RQC':     gen_rqc,
        'OTOC':    gen_otoc,
    }

    # ──────────────────────────────────────────────────────────────────────────────
    #  PARTITION METHODS — non-mosaic baselines + mosaic ablation ladder + mosaic-LCT
    # ──────────────────────────────────────────────────────────────────────────────
    def get_all_partition_methods(G, layout, n_qubits, qc_full):
        """Return {method_name: (A_set, B_set, runtime_sec)}.

        Method order is fixed for plot consistency:
        Baselines  →  mosaic ablation ladder (KL, OE, MI, BF)  →  mosaic-LCT (main).
        """
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

        # ── Non-mosaic baselines ────────────────────────────────────────────────────
        _time_call('Naive',         lambda: _naive_bisection(n_qubits))
        _time_call('Spectral',      lambda: _spectral_bisection(G, n_qubits))
        _time_call('Louvain',       lambda: _louvain_bisection(G, n_qubits))
        _time_call('Girvan-Newman', lambda: _girvan_newman_bisection(G, n_qubits))
        _time_call('METIS',         lambda: _metis_bisection(G, n_qubits))
        _time_call('qdislib',       lambda: _try_qdislib(layout, n_qubits))

        # ── mosaic ablation ladder ──────────────────────────────────────────────────
        # mosaic-KL  : Kernighan–Lin on gate-count interaction graph (no entanglement,
        #            no SV-aware cost — purely structural).
        _time_call('mosaic-KL',       lambda: partition_graph(G, n_qubits))
        # mosaic-OE  : Operator-Entanglement edge weighting + structural simulated
        #            annealing (still no SV-aware cost).
        _time_call('mosaic-OE',       lambda: _mosaic_syment_static(G, layout, n_qubits))
        # mosaic-MI  : Physics cost (entropy + Mutual Information + log-γ + cut +
        #            balance) with simulated annealing — SV-aware when feasible.
        _time_call('mosaic-MI',       lambda: mosaic_plus_plus(G, layout, n_qubits, qc_full=qc_full))
        # mosaic-BF  : Boundary-Focused move proposals on the same physics cost.
        _time_call('mosaic-BF',       lambda: _mosaic_adaptive(G, layout, n_qubits, qc_full, OUTDIR))

        # ── Main method ───────────────────────────────────────────────────────────
        # mosaic-LCT : Light-Cone graph augmentation + parallel-tempering with tabu,
        #            adaptive operator weights, consensus combine, and greedy polish.
        _time_call('mosaic-LCT',      lambda: mosaic_lct(G, layout, n_qubits,
                                                    qc_full=qc_full, seed=SEED))
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
        reconstruction-distance metrics.  Returns {method: metrics_dict}."""
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
            if isinstance(data, tuple) and len(data) == 4 \
                    and data[0] is not None and data[2] is not None:
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
                err = data[1] if isinstance(data, tuple) and data[0] == 'error' else 'sim failed'
                results[lbl] = {'sim_error': err}

        valid = {k: v for k, v in results.items() if v.get('error') is None and 'tvd' in v}
        if valid:
            for k, v in _compute_unified_scores(valid).items():
                results[k] = v
        return results

    # ──────────────────────────────────────────────────────────────────────────────
    #  BENCHMARK ONE  (family, n_qubits)
    # ──────────────────────────────────────────────────────────────────────────────
    def benchmark_one(family, gen_fn, n_qubits, family_dir):
        rng = random.Random(SEED + n_qubits * 17 + FAMILY_SEEDS[family])
        layout = gen_fn(n_qubits, rng)
        qc_full, _ = compile_quantum_circuit(layout, num_qubits=n_qubits, use_numeric_params=True)
        G = build_interaction_graph(layout, n_qubits)

        n_dir       = family_dir / f"n{n_qubits}"
        full_dir    = n_dir / "full"
        methods_dir = n_dir / "methods"
        full_dir.mkdir(parents=True, exist_ok=True)
        methods_dir.mkdir(parents=True, exist_ok=True)

        with open(full_dir / "full_layout.pkl", 'wb') as f:
            pickle.dump({'layout': layout, 'n_qubits': n_qubits, 'family': family,
                        'depth': qc_full.depth(), 'n_gates': qc_full.size()}, f)
        save_diagram(qc_full, full_dir / "full_diagram.png", f"{family} (n={n_qubits})")

        sv = None
        if n_qubits <= 24:
            try: sv = _get_statevector(qc_full)
            except Exception: sv = None

        methods = get_all_partition_methods(G, layout, n_qubits, qc_full)

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
                entropy = _entanglement_entropy(sv, sorted(A), n_qubits) if sv is not None else None

                slA, snA, _ = build_subcircuit_layout(layout, A, 'A', gc)
                slB, snB, _ = build_subcircuit_layout(layout, B, 'B', gc)
                qcA, _ = compile_subcircuit(slA, len(A))
                qcB, _ = compile_subcircuit(slB, len(B))

                safe = mname.replace('/', '_').replace(' ', '_')
                mdir = methods_dir / safe
                mdir.mkdir(parents=True, exist_ok=True)
                for pn, sl, sn, qc, ns in [('A', slA, snA, qcA, A), ('B', slB, snB, qcB, B)]:
                    with open(mdir / f"{pn}_layout.pkl", 'wb') as f:
                        pickle.dump({'layout': sl, 'n_qubits': len(ns), 'sorted_nodes': sn,
                                    'partition': sorted(ns), 'method': mname, 'family': family,
                                    'depth': qc.depth(), 'n_gates': qc.size()}, f)
                    save_diagram(qc, mdir / f"{pn}_diagram.png",
                                f"{family} | {mname} | {pn} ({len(ns)}q, {qc.size()}g, d={qc.depth()})")

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
                    'log10_gamma_sym': math.log10(max(qpd['total_gamma_sym'], 1e-12)),
                    'log10_gamma_generic': math.log10(max(qpd['total_gamma_generic'], 1e-12)),
                    'gamma_reduction_ratio': round(qpd['total_gamma_sym'] / max(qpd['total_gamma_generic'], 1e-30), 6),
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
                print(f"      {mname:14s}  cut={cs['n_cut_edges']:3d}  cross={len(gc['cross']):3d}  "
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
    #  RESEARCH-GRADE PLOTS  (box plots + heatmaps + Pareto + win-counts)
    # ──────────────────────────────────────────────────────────────────────────────
    def _gather(rows, key, method=None, family=None, n=None):
        return [r[key] for r in rows
                if (method is None or r['method'] == method)
                and (family is None or r['family'] == family)
                and (n is None or r['n_qubits'] == n)
                and r.get(key) is not None]

    def _box_axes(ax, data, labels, colors, title, ylabel,
                highlight=None, higher_better=False):
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        showmeans=True,
                        meanprops={'marker':'D','markerfacecolor':'white',
                                'markeredgecolor':'black','markersize':5})
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i]); patch.set_alpha(0.78)
            if labels[i] == highlight:
                patch.set_edgecolor('#1B5E20'); patch.set_linewidth(2.4)
        for med in bp['medians']:
            med.set_color('black'); med.set_linewidth(1.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)

    def make_plots(rows, plots_dir):
        plots_dir.mkdir(parents=True, exist_ok=True)
        if not rows:
            return

        families = sorted(set(r['family'] for r in rows),
                        key=lambda f: list(CIRCUIT_FAMILIES).index(f))
        # Canonical method order: baselines → mosaic ablation ladder → mosaic-LCT (main)
        METHOD_ORDER = ['Naive', 'Spectral', 'Louvain', 'Girvan-Newman', 'METIS', 'qdislib',
                        'mosaic-KL', 'mosaic-OE', 'mosaic-MI', 'mosaic-BF', 'mosaic-LCT']
        seen_methods = set(r['method'] for r in rows)
        methods = [m for m in METHOD_ORDER if m in seen_methods] + \
                [m for m in seen_methods if m not in METHOD_ORDER]
        qubits   = sorted(set(r['n_qubits'] for r in rows))

        # Colour scheme reflects the narrative:
        #   non-mosaic baselines → cool greys/blues; mosaic ablation ladder → green ramp;
        #   mosaic-LCT (main)   → deep saturated green, also used for highlight edges.
        BASELINE_COLOURS = {
            'Naive':         '#95A5A6',
            'Spectral':      '#7F8C8D',
            'Louvain':       '#5D6D7E',
            'Girvan-Newman': '#85929E',
            'METIS':         '#34495E',
            'qdislib':       '#566573',
        }
        mosaic_RAMP = {
            'mosaic-KL':  '#A9DFBF',
            'mosaic-OE':  '#7DCEA0',
            'mosaic-MI':  '#52BE80',
            'mosaic-BF':  '#2ECC71',
            'mosaic-LCT': '#1E8449',  # main — also used for highlight edge
        }
        method_colors = {}
        for m in methods:
            if m in BASELINE_COLOURS:  method_colors[m] = BASELINE_COLOURS[m]
            elif m in mosaic_RAMP:       method_colors[m] = mosaic_RAMP[m]
            else:                      method_colors[m] = '#BDC3C7'

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

        # ─── 1. GLOBAL BOX-PLOT GRID  (one panel per metric) ─────────────────────
        nrows, ncols = 4, 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.4*nrows))
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
            _box_axes(ax, data, order, cols,
                    f"{title}  ({'higher=better' if hb else 'lower=better'})",
                    title, highlight=HIGHLIGHT_METHOD, higher_better=hb)
        for ax in axf[len(metrics):]:
            ax.axis('off')
        fig.suptitle('Global Method Comparison — Box plots across all (family, n_qubits) pairs',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        fig.savefig(plots_dir / "global_boxplot_grid.png", dpi=130, bbox_inches='tight')
        plt.close(fig); n_plots += 1

        # ─── 2. PER-FAMILY BOX-PLOT GRID ─────────────────────────────────────────
        for fam in families:
            fam_rows = [r for r in rows if r['family'] == fam]
            if not fam_rows: continue
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.4*nrows))
            axf = axes.flatten()
            for ax, (key, title, hb) in zip(axf, metrics):
                m_vals = {m: _gather(fam_rows, key, method=m) for m in methods}
                m_vals = {m: v for m, v in m_vals.items() if v}
                if not m_vals:
                    ax.axis('off'); continue
                order = sorted(m_vals.keys(),
                            key=lambda m: -np.median(m_vals[m]) if hb else np.median(m_vals[m]))
                data = [m_vals[m] for m in order]
                cols = [method_colors[m] for m in order]
                _box_axes(ax, data, order, cols,
                        f"{title}  ({'higher=better' if hb else 'lower=better'})",
                        title, highlight=HIGHLIGHT_METHOD, higher_better=hb)
            for ax in axf[len(metrics):]:
                ax.axis('off')
            fig.suptitle(f'{fam} — Per-Family Box-plot Grid (across n∈{qubits})',
                        fontsize=15, fontweight='bold')
            plt.tight_layout()
            fig.savefig(plots_dir / f"family_{fam}_boxplot_grid.png", dpi=130, bbox_inches='tight')
            plt.close(fig); n_plots += 1

        # ─── 3. CROSS-FAMILY HEATMAPS  (method × family, mean value) ─────────────
        #   lo_good = True  → lower is better (use reversed cmap, sort ascending)
        #   lo_good = False → higher is better (use forward cmap, sort descending)
        for metric_key, title, lo_good in [
                ('log10_gamma_sym',      'Mean log₁₀(γ_sym)',          True),
                ('n_cross_gates',        'Mean Cross Gates',           True),
                ('n_cut_edges',          'Mean Cut Edges',             True),
                ('cut_fraction',         'Mean Cut Fraction',          True),
                ('entanglement_entropy', 'Mean Entanglement Entropy',  True),
                ('tvd',                  'Mean TVD',                   True),
                ('fidelity',             'Mean Fidelity',              False),
                ('kl_divergence',        'Mean KL',                    True),
                ('unified_score',        'Mean Unified Score',         False)]:
            mat = np.full((len(methods), len(families)), np.nan)
            for i, m in enumerate(methods):
                for j, fam in enumerate(families):
                    v = _gather(rows, metric_key, method=m, family=fam)
                    if v: mat[i, j] = np.mean(v)
            if np.all(np.isnan(mat)): continue
            # Sort rows so the best method per this metric is at row 0 (top).
            row_score = np.nanmean(mat, axis=1)
            finite_rows = ~np.isnan(row_score)
            if finite_rows.any():
                keys = np.where(finite_rows,
                                row_score if lo_good else -row_score, np.inf)
                order = list(np.argsort(keys))
            else:
                order = list(range(len(methods)))
            mat = mat[order]
            m_disp = [methods[i] for i in order]
            fig, ax = plt.subplots(figsize=(max(len(families)*1.2, 9),
                                            max(len(m_disp)*0.55, 4.5)))
            finite = mat[~np.isnan(mat)]
            vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
            cmap = 'RdYlGn_r' if lo_good else 'RdYlGn'
            im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(families))); ax.set_xticklabels(families, rotation=20, ha='right')
            ax.set_yticks(range(len(m_disp)));  ax.set_yticklabels(m_disp)
            for i in range(len(m_disp)):
                for j in range(len(families)):
                    if not np.isnan(mat[i, j]):
                        norm = (mat[i, j] - vmin) / (vmax - vmin + 1e-9)
                        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8,
                                color='white' if norm > 0.55 else 'black')
            plt.colorbar(im, ax=ax, label=title)
            ax.set_title(f'{title} — Method × Circuit Family  (best method at top)',
                        fontweight='bold')
            plt.tight_layout()
            fig.savefig(plots_dir / f"heatmap_{metric_key}.png", dpi=130, bbox_inches='tight')
            plt.close(fig); n_plots += 1

        # ─── 4. SCALING BOX PLOTS  (metric vs n_qubits, methods sorted best→worst) ─
        for key, title, hb in [('log10_gamma_sym', 'log₁₀(γ_sym)',     False),
                            ('unified_score',   'Unified Score',    True),
                            ('tvd',             'TVD vs ideal',     False),
                            ('fidelity',        'Fidelity vs ideal',True)]:
            # Decide the per-metric ordering ONCE using the overall median across
            # all n_qubits, so each n group reads left-to-right best → worst.
            m_score = {}
            for m in methods:
                v = _gather(rows, key, method=m)
                m_score[m] = float(np.median(v)) if v else (-np.inf if hb else np.inf)
            m_sorted = sorted(methods, key=lambda m: -m_score[m] if hb else m_score[m])

            fig, ax = plt.subplots(figsize=(max(len(qubits)*len(m_sorted)*0.45, 12), 6))
            positions, labels, colors_seq, data_seq = [], [], [], []
            gap = 1.5
            for qi, n in enumerate(qubits):
                for mi, m in enumerate(m_sorted):
                    pos = qi * (len(m_sorted) + gap) + mi
                    v = _gather(rows, key, method=m, n=n)
                    if not v: continue
                    positions.append(pos); labels.append(m)
                    colors_seq.append(method_colors[m]); data_seq.append(v)
            if not data_seq:
                plt.close(fig); continue
            bp = ax.boxplot(data_seq, positions=positions, patch_artist=True, widths=0.6,
                            showmeans=True,
                            meanprops={'marker':'D','markerfacecolor':'white',
                                    'markeredgecolor':'black','markersize':4})
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors_seq[i]); patch.set_alpha(0.78)
                if labels[i] == HIGHLIGHT_METHOD:
                    patch.set_edgecolor('#1B5E20'); patch.set_linewidth(2.0)
            center_per_n = [qi * (len(m_sorted) + gap) + (len(m_sorted) - 1)/2 for qi in range(len(qubits))]
            ax.set_xticks(center_per_n)
            ax.set_xticklabels([f'n={n}' for n in qubits], fontsize=11)
            legend_h = [Patch(facecolor=method_colors[m], edgecolor='white', label=m) for m in m_sorted]
            ax.legend(handles=legend_h, loc='best', fontsize=8, ncol=2)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'Scaling — {title} per n_qubits  '
                        f'({"higher=better" if hb else "lower=better"};  best at left of each n group)',
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout()
            fig.savefig(plots_dir / f"scaling_box_{key}.png", dpi=130, bbox_inches='tight')
            plt.close(fig); n_plots += 1

        # ─── 5. PARETO  (fidelity vs log10 γ_sym) ────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 7))
        for m in methods:
            xs = [r['log10_gamma_sym'] for r in rows
                if r['method']==m and r.get('fidelity') is not None
                and r.get('log10_gamma_sym') is not None]
            ys = [r['fidelity'] for r in rows
                if r['method']==m and r.get('fidelity') is not None
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

        # ─── 6. WIN-COUNT MATRIX  (best unified score per family/n) ──────────────
        sim_win = np.zeros((len(methods), len(families)), dtype=int)
        for j, fam in enumerate(families):
            for n in qubits:
                sub = [(r['method'], r['unified_score']) for r in rows
                    if r['family']==fam and r['n_qubits']==n
                    and r.get('unified_score') is not None]
                if not sub: continue
                sub.sort(key=lambda x: -x[1])
                best = sub[0][0]
                if best in methods:
                    sim_win[methods.index(best), j] += 1
        if sim_win.sum() > 0:
            # Sort rows: most wins at top.
            order = list(np.argsort(-sim_win.sum(axis=1)))
            sim_win = sim_win[order]
            m_disp = [methods[i] for i in order]
            fig, ax = plt.subplots(figsize=(max(len(families)*1.2, 9),
                                            max(len(m_disp)*0.55, 4.5)))
            im = ax.imshow(sim_win, aspect='auto', cmap='Greens', vmin=0, vmax=max(sim_win.max(), 1))
            ax.set_xticks(range(len(families))); ax.set_xticklabels(families, rotation=20, ha='right')
            ax.set_yticks(range(len(m_disp)));  ax.set_yticklabels(m_disp)
            for i in range(len(m_disp)):
                for j in range(len(families)):
                    v = sim_win[i, j]
                    ax.text(j, i, str(v), ha='center', va='center', fontsize=10,
                            color='white' if v > sim_win.max()/2 else 'black', fontweight='bold')
            plt.colorbar(im, ax=ax, label='# wins (highest unified score)')
            ax.set_title('Win-Count — # times method achieves best reconstruction per family  (most wins at top)',
                        fontweight='bold', fontsize=12)
            plt.tight_layout()
            fig.savefig(plots_dir / "win_count_unified_score.png", dpi=130, bbox_inches='tight')
            plt.close(fig); n_plots += 1

        # ─── 7. AGGREGATE RANKING BAR  (mean unified score) ──────────────────────
        score_data = defaultdict(list)
        for r in rows:
            if r.get('unified_score') is not None:
                score_data[r['method']].append(r['unified_score'])
        if score_data:
            m_score = {m: float(np.mean(v)) for m, v in score_data.items()}
            sorted_m = sorted(m_score.keys(), key=lambda m: -m_score[m])
            fig, ax = plt.subplots(figsize=(max(len(sorted_m)*1.4, 10), 6))
            vals = [m_score[m] for m in sorted_m]
            cols = [method_colors[m] for m in sorted_m]
            bars = ax.bar(range(len(sorted_m)), vals, color=cols, edgecolor='white')
            for i, m in enumerate(sorted_m):
                if m == HIGHLIGHT_METHOD:
                    bars[i].set_edgecolor('#1B5E20'); bars[i].set_linewidth(2.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.01, f'{v:.3f}',
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

        print(f"   Saved {n_plots} research-grade plots to {plots_dir}")

    # ──────────────────────────────────────────────────────────────────────────────
    #  MAIN
    # ──────────────────────────────────────────────────────────────────────────────
    def main():
        OUTDIR.mkdir(parents=True, exist_ok=True)
        print('='*78)
        print(' mosaic-LCT Independent Benchmark  (with mosaic-KL/OE/MI/BF ablation ladder)')
        print(f'   Output directory: {OUTDIR.resolve()}')
        print(f'   Families:         {list(CIRCUIT_FAMILIES)}')
        print(f'   Qubit counts:     {QUBIT_LIST}')
        print(f'   Baselines:        Naive, Spectral, Louvain, Girvan-Newman, METIS, qdislib')
        print(f'   mosaic ablation:    mosaic-KL, mosaic-OE, mosaic-MI, mosaic-BF')
        print(f'   Main method:      mosaic-LCT  (Light-Cone Tempering)')
        print('='*78)

        all_rows = []
        t0 = time.time()
        for family, gen_fn in CIRCUIT_FAMILIES.items():
            family_dir = OUTDIR / "circuits" / family
            family_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{family}]")
            for n in QUBIT_LIST:
                print(f"   ── n_qubits = {n} ──")
                try:
                    rows = benchmark_one(family, gen_fn, n, family_dir)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"   [error] {family} n={n}: {e}")
                    traceback.print_exc()

        save_csv(all_rows, OUTDIR / "results.csv")
        with open(OUTDIR / "results.json", 'w') as f:
            json.dump(all_rows, f, indent=2, default=str)
        print(f"   Saved JSON: {OUTDIR/'results.json'}")

        print("\n[Plots] Generating research-grade box-plot figures ...")
        make_plots(all_rows, OUTDIR / "plots")

        elapsed = time.time() - t0
        print('\n' + '='*78)
        print(f' ✓ Benchmark complete in {elapsed:.1f}s   |   total runs: {len(all_rows)}')
        print(f'   Output: {OUTDIR.resolve()}')
        print('='*78)

    if __name__ == '__main__':
        main()