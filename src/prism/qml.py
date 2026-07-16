"""
prism.qml
=========
Application layer: QAOA-MaxCut and Quantum Neural Networks, plus a converter
that lets PRISM partition *any* Qiskit circuit.

This module supplies everything the two application notebooks need:

QAOA-MaxCut
    * random problem graphs, the QAOA ansatz as a PRISM layout,
    * exact MaxCut expectation / brute-force optimum / approximation ratio,
    * a registry of many classical optimizers (scipy + SPSA + gradient
      descent + differential evolution),
    * partition-aware evaluation (does cutting the QAOA circuit preserve the
      MaxCut objective?).

QNN
    * Iris (4 features -> 4 qubits, angle encoding) and Digits
      (all 64 pixels -> 8 qubits, data re-uploading; no feature reduction),
    * a trainable variational classifier,
    * partition-aware evaluation of the trained circuits.

Qiskit bridge
    * :func:`qiskit_to_layout` converts a bound Qiskit circuit into a PRISM
      layout, so amplitude-encoded QNNs and arbitrary library circuits can be
      partitioned by every method in :mod:`prism.partition`.
"""
from __future__ import annotations

import math
import random
import numpy as np

from .compiler import materialize_layout, compile_circuit
from .graph import build_interaction_graph
from .simulate import get_statevector, prob_from_statevector
from .partition import run_all_partition_methods, partition


# ===========================================================================
#  Qiskit  ->  PRISM layout
# ===========================================================================
_QISKIT_GATE_MAP = {
    'id': 'I', 'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 's': 'S', 'sdg': 'SDG',
    't': 'T', 'tdg': 'TDG', 'sx': 'SX', 'sxdg': 'SXDG',
    'rx': 'RX', 'ry': 'RY', 'rz': 'RZ', 'p': 'PHASE', 'u1': 'PHASE',
    'u2': 'U2', 'u': 'U3', 'u3': 'U3',
    'cx': 'CNOT', 'cnot': 'CNOT', 'cy': 'CY', 'cz': 'CZ', 'ch': 'CH',
    'swap': 'SWAP', 'iswap': 'ISWAP',
    'crx': 'CRX', 'cry': 'CRY', 'crz': 'CRZ', 'cp': 'CPHASE', 'cu1': 'CPHASE',
    'ccx': 'CCX', 'cswap': 'CSWAP',
}
_SKIP = {'barrier', 'measure', 'snapshot', 'delay', 'reset'}


def qiskit_to_layout(qc, basis_gates=('u', 'cx'), pack=True):
    """Convert a Qiskit ``QuantumCircuit`` into a PRISM layout.

    Unsupported / composite instructions (``initialize``, ``state_preparation``,
    ``unitary`` ...) are decomposed by transpiling to ``basis_gates`` first.
    Parameters must be numeric (bind them before calling). When ``pack`` is
    True, gates are greedily packed into layers (a faithful depth model);
    otherwise each gate gets its own layer.
    """
    from qiskit import transpile
    names = {instr.operation.name.lower() for instr in qc.data}
    if not names.issubset(set(_QISKIT_GATE_MAP) | _SKIP):
        qc = transpile(qc, basis_gates=list(basis_gates), optimization_level=0)

    n = qc.num_qubits
    flat = []
    for instr in qc.data:
        name = instr.operation.name.lower()
        if name in _SKIP:
            continue
        gate = _QISKIT_GATE_MAP.get(name)
        if gate is None:
            raise ValueError(f"cannot map Qiskit gate {name!r} to a PRISM gate")
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        params = [float(p) for p in instr.operation.params] if instr.operation.params else []
        flat.append({'gate': gate, 'qubits': qubits, 'params': params})

    if not pack:
        return [[g] for g in flat]

    layout, free = [], [0] * n
    for g in flat:
        lvl = max(free[q] for q in g['qubits'])
        while len(layout) <= lvl:
            layout.append([])
        layout[lvl].append(g)
        for q in g['qubits']:
            free[q] = lvl + 1
    return [layer for layer in layout if layer]


def partition_qiskit_circuit(qc, method='PRISM-LCT', seed=42):
    """Partition an arbitrary Qiskit circuit with any PRISM method.
    Returns ``(A_set, B_set, layout)``."""
    layout = qiskit_to_layout(qc)
    n = qc.num_qubits
    G = build_interaction_graph(layout, n)
    qc_full, _ = compile_circuit(layout, num_qubits=n, use_numeric_params=True)
    A, B = partition(method, G, layout, n, qc_full=qc_full, seed=seed)
    return A, B, layout


# ===========================================================================
#  QAOA  —  MaxCut
# ===========================================================================
def random_maxcut_graph(n, seed=0, kind='regular', degree=3, weighted=False):
    """Random MaxCut problem graph (networkx) with optional edge weights."""
    import networkx as nx
    rng = random.Random(seed)
    if kind == 'regular' and (n * degree) % 2 == 0 and degree < n:
        g = nx.random_regular_graph(degree, n, seed=seed)
    else:
        g = nx.gnp_random_graph(n, p=min(0.5, max(2.0 / n, degree / max(n - 1, 1))), seed=seed)
        if g.number_of_edges() == 0:
            g.add_edge(0, min(1, n - 1))
    for u, v in g.edges():
        g[u][v]['weight'] = rng.uniform(0.5, 1.5) if weighted else 1.0
    return g


def maxcut_qaoa_layout(graph, gammas, betas):
    """QAOA ansatz layout for a MaxCut graph and parameters (len = p layers).

    Initial Hadamards; per layer: cost unitary exp(-i gamma w Z_iZ_j) realised
    as CNOT-RZ(2 gamma w)-CNOT on each edge, then mixer RX(2 beta) on all."""
    n = graph.number_of_nodes()
    edges = [(u, v, graph[u][v].get('weight', 1.0)) for u, v in graph.edges()]
    layout = [[{'gate': 'H', 'qubits': [i]} for i in range(n)]]
    for gamma, beta in zip(gammas, betas):
        for (u, v, w) in edges:
            layout.append([{'gate': 'CNOT', 'qubits': [u, v]}])
            layout.append([{'gate': 'RZ', 'qubits': [v], 'params': [2.0 * gamma * w]}])
            layout.append([{'gate': 'CNOT', 'qubits': [u, v]}])
        layout.append([{'gate': 'RX', 'qubits': [i], 'params': [2.0 * beta]} for i in range(n)])
    return layout


def _edge_arrays(graph):
    e = np.array([(u, v) for u, v in graph.edges()], dtype=np.int64)
    w = np.array([graph[u][v].get('weight', 1.0) for u, v in graph.edges()], dtype=float)
    return e, w


def maxcut_value(bitstring_int, graph):
    """Cut value of one assignment (integer bitmask, qubit q = bit q)."""
    val = 0.0
    for u, v in graph.edges():
        if ((bitstring_int >> u) & 1) != ((bitstring_int >> v) & 1):
            val += graph[u][v].get('weight', 1.0)
    return val


def maxcut_expectation(prob, graph, n):
    """Expected cut value E[C] under distribution ``prob`` (length 2**n)."""
    idx = np.arange(len(prob), dtype=np.int64)
    e, w = _edge_arrays(graph)
    exp = 0.0
    for (u, v), wt in zip(e, w):
        diff = ((idx >> int(u)) & 1) != ((idx >> int(v)) & 1)
        exp += wt * float(prob[diff].sum())
    return exp


def maxcut_brute_force(graph, n):
    """Exact maximum cut value (n <= ~20)."""
    best = 0.0
    for z in range(2 ** n):
        v = maxcut_value(z, graph)
        if v > best:
            best = v
    return best


def qaoa_split_params(params):
    p = len(params) // 2
    return params[:p], params[p:]


def qaoa_expectation_exact(params, graph, n):
    """E[cut] for QAOA(params) via exact statevector."""
    gammas, betas = qaoa_split_params(np.asarray(params, float))
    layout = maxcut_qaoa_layout(graph, gammas, betas)
    sv = get_statevector(materialize_layout(layout, n))
    if sv is None:
        return 0.0
    return maxcut_expectation(prob_from_statevector(sv), graph, n)


# ---------------------------------------------------------------------------
#  Optimizer registry
# ---------------------------------------------------------------------------
def _spsa(fun, x0, maxiter=120, seed=0, a=0.2, c=0.15):
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    best_x, best_f = x.copy(), fun(x)
    hist = [best_f]
    nfev = 1
    for k in range(1, maxiter + 1):
        ak = a / (k + 10) ** 0.602
        ck = c / k ** 0.101
        delta = rng.integers(0, 2, size=x.shape) * 2 - 1
        fp = fun(x + ck * delta); fm = fun(x - ck * delta); nfev += 2
        ghat = (fp - fm) / (2.0 * ck) * delta
        x = x - ak * ghat
        fx = fun(x); nfev += 1
        hist.append(fx)
        if fx < best_f:
            best_f, best_x = fx, x.copy()
    return best_x, best_f, nfev, hist


def _gradient_descent(fun, x0, maxiter=120, seed=0, lr=0.1, eps=1e-3):
    x = np.array(x0, float)
    best_x, best_f = x.copy(), fun(x)
    hist = [best_f]
    nfev = 1
    for _ in range(maxiter):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (fun(xp) - fun(xm)) / (2 * eps)
            nfev += 2
        x = x - lr * grad
        fx = fun(x); nfev += 1
        hist.append(fx)
        if fx < best_f:
            best_f, best_x = fx, x.copy()
    return best_x, best_f, nfev, hist


def _scipy_minimizer(method):
    def run(fun, x0, maxiter=120, seed=0):
        from scipy.optimize import minimize
        hist = []
        wrapped = lambda x: (lambda v: (hist.append(v), v)[1])(fun(x))
        opts = {'maxiter': maxiter}
        if method == 'COBYLA':
            opts = {'maxiter': maxiter, 'rhobeg': 0.5}
        res = minimize(wrapped, np.array(x0, float), method=method, options=opts)
        return np.array(res.x, float), float(res.fun), int(getattr(res, 'nfev', len(hist))), hist
    return run


def _differential_evolution(fun, x0, maxiter=40, seed=0):
    from scipy.optimize import differential_evolution
    d = len(x0)
    bounds = [(-math.pi, math.pi)] * d
    hist = []
    wrapped = lambda x: (lambda v: (hist.append(v), v)[1])(fun(x))
    res = differential_evolution(wrapped, bounds, maxiter=maxiter, seed=seed,
                                 popsize=8, polish=True, tol=1e-4)
    return np.array(res.x, float), float(res.fun), int(res.nfev), hist


OPTIMIZERS = {
    'COBYLA': _scipy_minimizer('COBYLA'),
    'Nelder-Mead': _scipy_minimizer('Nelder-Mead'),
    'Powell': _scipy_minimizer('Powell'),
    'SLSQP': _scipy_minimizer('SLSQP'),
    'L-BFGS-B': _scipy_minimizer('L-BFGS-B'),
    'CG': _scipy_minimizer('CG'),
    'TNC': _scipy_minimizer('TNC'),
    'trust-constr': _scipy_minimizer('trust-constr'),
    'SPSA': _spsa,
    'GradientDescent': _gradient_descent,
    'DifferentialEvolution': _differential_evolution,
}


def optimize_qaoa(graph, n, p=1, optimizer='COBYLA', seed=0, maxiter=120, x0=None):
    """Optimize QAOA parameters with one optimizer. Returns a result dict with
    best params, energy, approximation ratio, evaluations and history."""
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = rng.uniform(0, math.pi, size=2 * p)
    opt = maxcut_brute_force(graph, n)
    neg = lambda params: -qaoa_expectation_exact(params, graph, n)
    runner = OPTIMIZERS[optimizer]
    x_best, f_best, nfev, hist = runner(neg, x0, maxiter=maxiter, seed=seed)
    energy = -f_best
    return {
        'optimizer': optimizer, 'p': p, 'x': x_best, 'energy': energy,
        'optimal_cut': opt, 'approx_ratio': energy / opt if opt > 0 else float('nan'),
        'nfev': nfev, 'history': [-h for h in hist], 'x0': np.asarray(x0),
    }


def run_all_optimizers(graph, n, p=1, optimizers=None, seed=0, maxiter=120,
                       maxiter_overrides=None):
    """Optimize QAOA with every (or chosen) optimizer; returns {name: result}.

    ``maxiter_overrides`` maps an optimizer name to a custom iteration budget
    (useful to cap the expensive global optimizers, e.g.
    ``{'DifferentialEvolution': 15}``)."""
    optimizers = optimizers or list(OPTIMIZERS)
    overrides = maxiter_overrides or {}
    out = {}
    x0 = np.random.default_rng(seed).uniform(0, math.pi, size=2 * p)
    for name in optimizers:
        try:
            out[name] = optimize_qaoa(graph, n, p=p, optimizer=name, seed=seed,
                                      maxiter=overrides.get(name, maxiter), x0=x0)
        except Exception as e:
            out[name] = {'optimizer': name, 'error': f'{type(e).__name__}: {e}'}
    return out


def evaluate_qaoa_partitions(graph, n, params, methods=None, seed=0):
    """For optimized QAOA(params), cut the circuit with every method and
    measure how well the product reconstruction preserves the output
    distribution and the recovered MaxCut expectation / approximation ratio."""
    from .simulate import reconstruct_product, distribution_metrics, q_score
    gammas, betas = qaoa_split_params(np.asarray(params, float))
    layout = maxcut_qaoa_layout(graph, gammas, betas)
    G = build_interaction_graph(layout, n)
    qc_full, _ = compile_circuit(layout, num_qubits=n, use_numeric_params=True)
    sv = get_statevector(qc_full)
    p_ideal = prob_from_statevector(sv)
    opt = maxcut_brute_force(graph, n)
    exp_ideal = maxcut_expectation(p_ideal, graph, n)

    from .graph import classify_gates, compute_cut_stats
    from .symmetry import compute_qpd_overhead
    results, fails = run_all_partition_methods(G, layout, n, qc_full=qc_full,
                                               seed=seed, methods=methods)
    rows = []
    for name, (A, B, dt) in results.items():
        p_rec = reconstruct_product(A, B, layout, n)
        gc = classify_gates(layout, A, B)
        qpd = compute_qpd_overhead(gc['cross'], sym_reduction=True)
        cs = compute_cut_stats(G, A, B)
        row = {'method': name, 'n_A': len(A), 'n_B': len(B), 'runtime_s': dt,
               'exp_ideal': exp_ideal, 'optimal_cut': opt,
               'n_cross_gates': len(gc['cross']), 'cut_fraction': cs['cut_fraction'],
               'log10_gamma_sym': math.log10(max(qpd['total_gamma_sym'], 1e-12)),
               'approx_ratio_ideal': exp_ideal / opt if opt > 0 else float('nan')}
        if p_rec is not None:
            dm = distribution_metrics(p_ideal, p_rec)
            exp_rec = maxcut_expectation(p_rec, graph, n)
            row.update(dm)
            row['q_score'] = q_score(dm)
            row['exp_recon'] = exp_rec
            row['approx_ratio_recon'] = exp_rec / opt if opt > 0 else float('nan')
            row['cut_expectation_error'] = abs(exp_rec - exp_ideal)
        rows.append(row)
    return rows, fails


# ===========================================================================
#  QNN  —  classification
# ===========================================================================
def load_iris_qnn(n_classes=2, seed=0):
    """Iris -> (X, y, n_qubits=4, 'angle'). 4 features map to 4 qubits, scaled
    to [0, pi]. No feature reduction."""
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import MinMaxScaler
    d = load_iris()
    X, y = d.data, d.target
    if n_classes == 2:
        mask = y < 2
        X, y = X[mask], y[mask]
    X = MinMaxScaler((0, math.pi)).fit_transform(X)
    return X, y, 4, 'angle'


def load_digits_qnn(n_classes=2, n_qubits=8, seed=0):
    """Digits (8x8 = 64 pixels) -> (X, y, n_qubits, 'reupload').

    All 64 pixels are used (no PCA / feature selection). The 64 features are
    fed across the qubits via *data re-uploading* layers, so the full image
    is encoded with no dimensionality reduction."""
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import MinMaxScaler
    d = load_digits()
    X, y = d.data, d.target
    mask = y < n_classes
    X, y = X[mask], y[mask]
    X = MinMaxScaler((0, math.pi)).fit_transform(X)
    return X, y, n_qubits, 'reupload'


def load_wine_qnn(n_classes=2, n_qubits=8, seed=0):
    """Wine (13 features) -> (X, y, n_qubits, 'reupload'). All 13 features used
    via data re-uploading; no reduction. Binary by default (classes 0,1)."""
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import MinMaxScaler
    d = load_wine()
    X, y = d.data, d.target
    mask = y < n_classes
    X, y = X[mask], y[mask]
    X = MinMaxScaler((0, math.pi)).fit_transform(X)
    return X, y, n_qubits, 'reupload'


def load_breast_cancer_qnn(n_qubits=8, seed=0):
    """Breast-cancer Wisconsin (30 features) -> (X, y, n_qubits, 'reupload').
    All 30 features used via data re-uploading; binary; no reduction."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import MinMaxScaler
    d = load_breast_cancer()
    X = MinMaxScaler((0, math.pi)).fit_transform(d.data)
    return X, d.target, n_qubits, 'reupload'


def load_synth_qnn(n_features=8, n_qubits=8, n_samples=200, seed=0):
    """Synthetic linearly-separable-ish set: n_features -> n_qubits (angle when
    n_features==n_qubits, else re-uploading). Fully controllable, learnable."""
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=max(2, n_features // 2), n_redundant=0,
                               n_classes=2, class_sep=2.0, random_state=seed)
    X = MinMaxScaler((0, math.pi)).fit_transform(X)
    enc = 'angle' if n_features == n_qubits else 'reupload'
    return X, y, n_qubits, enc


def load_qnn_dataset(name, n_qubits=8, n_classes=2):
    """Dispatch by name: 'iris' | 'wine' | 'breast_cancer' | 'digits' | 'synth'."""
    name = name.lower()
    if name == 'iris':
        return load_iris_qnn(n_classes=n_classes)
    if name == 'wine':
        return load_wine_qnn(n_classes=n_classes, n_qubits=n_qubits)
    if name in ('breast_cancer', 'cancer', 'bc'):
        return load_breast_cancer_qnn(n_qubits=n_qubits)
    if name == 'digits':
        return load_digits_qnn(n_classes=n_classes, n_qubits=n_qubits)
    if name in ('synth', 'synthetic'):
        return load_synth_qnn(n_features=n_qubits, n_qubits=n_qubits)
    raise ValueError(f'unknown dataset {name!r}')


def _angle_encode_layout(x, n_qubits):
    return [[{'gate': 'RY', 'qubits': [i], 'params': [float(x[i % len(x)])]} for i in range(n_qubits)]]


def _reupload_encode_layers(x, n_qubits):
    """Spread all features across qubits in successive RY/RZ re-upload layers."""
    x = np.asarray(x, float)
    nfeat = len(x)
    sublayers = int(math.ceil(nfeat / n_qubits))
    layers = []
    for s in range(sublayers):
        chunk = x[s * n_qubits:(s + 1) * n_qubits]
        axis = 'RY' if s % 2 == 0 else 'RZ'
        layer = []
        for i in range(len(chunk)):
            layer.append({'gate': axis, 'qubits': [i], 'params': [float(chunk[i])]})
        if layer:
            layers.append(layer)
    return layers


def _entangler_edges(n_qubits, entangler='ring'):
    """Edges for one entangling layer under the chosen topology.

    ring     CZ chain + wrap-around (default; most entangling)
    linear   CZ chain only (MPS-like; a mid cut severs one bond)
    blocks   dense CZ within two halves + a single inter-half CZ
             (partition-friendly: a block-aligned cut severs one bond)
    full     all-to-all CZ (most entangling)
    """
    half = n_qubits // 2
    if entangler == 'linear':
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if entangler == 'full':
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if entangler == 'blocks':
        # two interleaved blocks (even / odd qubits) joined by a single link, so
        # the contiguous naive split slices through both blocks while a
        # structure-aware method recovers the true {evens}|{odds} partition.
        A = list(range(0, n_qubits, 2))
        B = list(range(1, n_qubits, 2))
        e = [(A[i], A[i + 1]) for i in range(len(A) - 1)]
        e += [(B[i], B[i + 1]) for i in range(len(B) - 1)]
        if A and B:
            e.append((A[0], B[0]))                                     # single inter-block link
        return e
    # ring (default)
    e = [(i, i + 1) for i in range(n_qubits - 1)]
    if n_qubits > 2:
        e.append((0, n_qubits - 1))
    return e


def _variational_block(weights_layer, n_qubits, entangler='ring'):
    """One trainable block: RY+RZ per qubit then a CZ entangling layer."""
    block = [[{'gate': 'RY', 'qubits': [i], 'params': [float(weights_layer[i, 0])]} for i in range(n_qubits)],
             [{'gate': 'RZ', 'qubits': [i], 'params': [float(weights_layer[i, 1])]} for i in range(n_qubits)]]
    ent = [{'gate': 'CZ', 'qubits': [u, v]} for (u, v) in _entangler_edges(n_qubits, entangler)]
    if ent:
        block.append(ent)
    return block


def qnn_n_params(n_qubits, reps):
    return n_qubits * 2 * reps


def qnn_circuit_layout(x, weights, encoding, n_qubits, reps, entangler='ring'):
    """Full QNN layout: data encoding (+ re-uploading) interleaved with
    ``reps`` trainable variational blocks. ``entangler`` selects the entangling
    topology ('ring' | 'linear' | 'blocks' | 'full')."""
    w = np.asarray(weights, float).reshape(reps, n_qubits, 2)
    layout = []
    if encoding == 'angle':
        for r in range(reps):
            layout += _angle_encode_layout(x, n_qubits)        # data re-upload each rep
            layout += _variational_block(w[r], n_qubits, entangler)
    elif encoding == 'reupload':
        enc = _reupload_encode_layers(x, n_qubits)
        for r in range(reps):
            layout += enc                                       # all-feature re-upload
            layout += _variational_block(w[r], n_qubits, entangler)
    elif encoding == 'amplitude':
        layout += amplitude_encode_layout(x, n_qubits)
        for r in range(reps):
            layout += _variational_block(w[r], n_qubits, entangler)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    return layout


def amplitude_encode_layout(x, n_qubits):
    """Amplitude-encode a (2**n_qubits)-length vector into n_qubits via a
    Qiskit StatePreparation, converted to a PRISM layout (all values used)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    vec = np.asarray(x, float)
    dim = 2 ** n_qubits
    if len(vec) < dim:
        vec = np.concatenate([vec, np.zeros(dim - len(vec))])
    else:
        vec = vec[:dim]
    norm = np.linalg.norm(vec)
    vec = vec / norm if norm > 0 else np.ones(dim) / math.sqrt(dim)
    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(vec), range(n_qubits))
    return qiskit_to_layout(qc, basis_gates=('u', 'cx'))


def qnn_expectation(layout, n_qubits, readout=0):
    """<Z_readout> in [-1, 1] from the exact output distribution."""
    sv = get_statevector(materialize_layout(layout, n_qubits))
    if sv is None:
        return 0.0
    p = prob_from_statevector(sv)
    idx = np.arange(len(p), dtype=np.int64)
    s = 1 - 2 * ((idx >> readout) & 1)            # +1 if bit 0, -1 if bit 1
    return float(np.sum(p * s))


def qnn_predict_proba(x, weights, encoding, n_qubits, reps, readout=0, entangler='ring'):
    """Map <Z_readout> in [-1,1] to class-1 probability in [0,1]."""
    layout = qnn_circuit_layout(x, weights, encoding, n_qubits, reps, entangler)
    z = qnn_expectation(layout, n_qubits, readout)
    return 0.5 * (1.0 - z)


def train_qnn(X, y, encoding, n_qubits, reps=2, optimizer='COBYLA', maxiter=60,
              seed=0, readout=0, n_train=40, verbose=False, entangler='ring'):
    """Train the variational QNN classifier (square loss). Returns
    ``(weights, info)``. Kept small for laptop-friendly runtimes."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:min(n_train, len(X))]
    Xtr, ytr = X[idx], y[idx]
    n_params = qnn_n_params(n_qubits, reps)
    w0 = rng.uniform(0, 2 * math.pi, size=n_params)

    def loss(w):
        err = 0.0
        for xi, yi in zip(Xtr, ytr):
            pr = qnn_predict_proba(xi, w, encoding, n_qubits, reps, readout, entangler)
            err += (pr - float(yi)) ** 2
        return err / len(Xtr)

    runner = OPTIMIZERS[optimizer]
    w_best, f_best, nfev, hist = runner(loss, w0, maxiter=maxiter, seed=seed)
    if verbose:
        print(f'  QNN trained ({encoding}, {n_qubits}q, reps={reps}, {entangler}): '
              f'final loss {f_best:.4f} in {nfev} evals')
    return w_best, {'final_loss': f_best, 'nfev': nfev, 'history': hist,
                    'n_qubits': n_qubits, 'reps': reps, 'encoding': encoding, 'entangler': entangler}


def qnn_accuracy(X, y, weights, encoding, n_qubits, reps, readout=0, entangler='ring'):
    correct = 0
    for xi, yi in zip(X, y):
        pred = 1 if qnn_predict_proba(xi, weights, encoding, n_qubits, reps, readout, entangler) >= 0.5 else 0
        correct += int(pred == int(yi))
    return correct / len(X)


def _pred_from_dist(p, readout=0):
    idx = np.arange(len(p), dtype=np.int64)
    s = 1 - 2 * ((idx >> readout) & 1)
    z = float(np.sum(p * s))
    return int(0.5 * (1.0 - z) >= 0.5)


def qnn_cut_for(X_ref, weights, encoding, n_qubits, reps, method='PRISM-LCT', seed=42, entangler='ring'):
    """Choose a partition for a trained QNN once, from a representative sample
    (the circuit's entangling structure is sample-independent). Returns (A, B)."""
    layout = qnn_circuit_layout(X_ref, weights, encoding, n_qubits, reps, entangler)
    G = build_interaction_graph(layout, n_qubits)
    qc, _ = compile_circuit(layout, num_qubits=n_qubits, use_numeric_params=True)
    return partition(method, G, layout, n_qubits, qc_full=qc, seed=seed)


def qnn_distributed_accuracy(X, y, weights, encoding, n_qubits, reps, A, B, readout=0, entangler='ring'):
    """Test accuracy of the trained QNN when each sample's circuit is run
    *distributed* across the fixed cut (A, B) and product-reconstructed."""
    from .simulate import reconstruct_product
    correct = 0
    for xi, yi in zip(X, y):
        layout = qnn_circuit_layout(xi, weights, encoding, n_qubits, reps, entangler)
        p = reconstruct_product(A, B, layout, n_qubits)
        if p is None:
            continue
        correct += int(_pred_from_dist(p, readout) == int(yi))
    return correct / len(X)


def qnn_distributed_report(X, y, weights, encoding, n_qubits, reps, A, B, readout=0, entangler='ring'):
    """Deployment report for running the trained QNN across the cut (A, B):
    ``accuracy`` (vs true labels) and ``agreement`` (fraction of samples whose
    *distributed* prediction matches the *monolithic* prediction). Agreement is
    the deployment-fidelity metric — meaningful even if the model is imperfect."""
    from .simulate import reconstruct_product
    correct = agree = n = 0
    for xi, yi in zip(X, y):
        layout = qnn_circuit_layout(xi, weights, encoding, n_qubits, reps, entangler)
        mono = _pred_from_dist(prob_from_statevector(get_statevector(materialize_layout(layout, n_qubits))), readout)
        p = reconstruct_product(A, B, layout, n_qubits)
        if p is None:
            continue
        d = _pred_from_dist(p, readout)
        correct += int(d == int(yi)); agree += int(d == mono); n += 1
    return {'accuracy': correct / max(n, 1), 'agreement': agree / max(n, 1), 'n': n}


def qnn_distributed_accuracy_kway(X, y, weights, encoding, n_qubits, reps, parts, readout=0, entangler='ring'):
    """Test accuracy when each sample's circuit is split across ``parts`` (k QPUs)
    and reconstructed as a k-fold product."""
    from .simulate import reconstruct_product_kway
    correct = 0
    for xi, yi in zip(X, y):
        layout = qnn_circuit_layout(xi, weights, encoding, n_qubits, reps, entangler)
        p = reconstruct_product_kway(parts, layout, n_qubits)
        if p is None:
            continue
        correct += int(_pred_from_dist(p, readout) == int(yi))
    return correct / len(X)


def evaluate_qnn_partitions(X_eval, weights, encoding, n_qubits, reps,
                            methods=None, seed=0, readout=0, max_samples=12):
    """Cut each trained-QNN circuit (one per sample) with every method and
    measure reconstruction quality + whether the predicted label is preserved.
    Returns per-(sample, method) rows."""
    from .simulate import reconstruct_product, distribution_metrics, q_score
    rng = np.random.default_rng(seed)
    sel = rng.permutation(len(X_eval))[:min(max_samples, len(X_eval))]
    rows = []
    for si in sel:
        x = X_eval[si]
        layout = qnn_circuit_layout(x, weights, encoding, n_qubits, reps)
        G = build_interaction_graph(layout, n_qubits)
        qc_full, _ = compile_circuit(layout, num_qubits=n_qubits, use_numeric_params=True)
        sv = get_statevector(qc_full)
        p_ideal = prob_from_statevector(sv)
        idx = np.arange(len(p_ideal), dtype=np.int64)
        s = 1 - 2 * ((idx >> readout) & 1)
        z_ideal = float(np.sum(p_ideal * s))
        pred_ideal = int(0.5 * (1 - z_ideal) >= 0.5)

        from .graph import classify_gates, compute_cut_stats
        from .symmetry import compute_qpd_overhead
        results, _ = run_all_partition_methods(G, layout, n_qubits, qc_full=qc_full,
                                               seed=seed, methods=methods)
        for name, (A, B, dt) in results.items():
            p_rec = reconstruct_product(A, B, layout, n_qubits)
            gc = classify_gates(layout, A, B)
            qpd = compute_qpd_overhead(gc['cross'], sym_reduction=True)
            cs = compute_cut_stats(G, A, B)
            row = {'sample': int(si), 'method': name, 'n_A': len(A), 'n_B': len(B),
                   'runtime_s': dt, 'z_ideal': z_ideal, 'pred_ideal': pred_ideal,
                   'n_cross_gates': len(gc['cross']), 'cut_fraction': cs['cut_fraction'],
                   'log10_gamma_sym': math.log10(max(qpd['total_gamma_sym'], 1e-12))}
            if p_rec is not None:
                dm = distribution_metrics(p_ideal, p_rec)
                z_rec = float(np.sum(p_rec * s))
                row.update(dm)
                row['q_score'] = q_score(dm)
                row['z_recon'] = z_rec
                row['pred_recon'] = int(0.5 * (1 - z_rec) >= 0.5)
                row['pred_preserved'] = int(row['pred_recon'] == pred_ideal)
                row['readout_error'] = abs(z_rec - z_ideal)
            rows.append(row)
    return rows


__all__ = [
    # qiskit bridge
    'qiskit_to_layout', 'partition_qiskit_circuit',
    # QAOA
    'random_maxcut_graph', 'maxcut_qaoa_layout', 'maxcut_value', 'maxcut_expectation',
    'maxcut_brute_force', 'qaoa_expectation_exact', 'qaoa_split_params',
    'OPTIMIZERS', 'optimize_qaoa', 'run_all_optimizers', 'evaluate_qaoa_partitions',
    # QNN
    'load_iris_qnn', 'load_digits_qnn', 'load_wine_qnn', 'load_breast_cancer_qnn',
    'load_synth_qnn', 'load_qnn_dataset',
    'qnn_circuit_layout', 'amplitude_encode_layout',
    'qnn_n_params', 'qnn_expectation', 'qnn_predict_proba', 'train_qnn',
    'qnn_accuracy', 'qnn_cut_for', 'qnn_distributed_accuracy', 'qnn_distributed_accuracy_kway',
    'qnn_distributed_report', 'evaluate_qnn_partitions',
]
