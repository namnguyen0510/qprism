"""
PRISM test suite.

Covers the public API: gate taxonomy, circuit families, interaction graph,
the multi-objective cost, every partition method, exact reconstruction +
metrics, symmetry/QPD analysis, the QAOA/QNN application layer, the Qiskit
bridge, and the benchmark driver.

Run with:  pytest -q
"""
import math
import numpy as np
import pytest

import prism


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def small_instance():
    n = 8
    layout = prism.make_layout('QAOA', n, seed=19)
    G = prism.build_interaction_graph(layout, n)
    qc, _ = prism.compile_circuit(layout, num_qubits=n, use_numeric_params=True)
    return layout, G, qc, n


# ---------------------------------------------------------------------------
#  Package surface
# ---------------------------------------------------------------------------
def test_version_and_lists():
    assert isinstance(prism.__version__, str)
    assert prism.list_methods()[-1] == 'PRISM-LCT'
    assert set(['Naive', 'Spectral', 'METIS', 'qdislib']).issubset(prism.list_methods())
    assert 'RQC' in prism.list_families()
    assert len(prism.PRISM_LADDER) == 5


def test_gate_helpers():
    assert prism.gate_arity('CNOT') == 2
    assert prism.schmidt_rank('SWAP') == 4
    assert prism.qpd_gamma('CNOT', symmetry_reduced=True) == 2.0
    assert prism.qpd_gamma('CNOT', symmetry_reduced=False) == 3.0
    assert prism.gate_type_code('H') == 1 and prism.gate_type_code('CZ') == 2


# ---------------------------------------------------------------------------
#  Circuit families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('family', ['QFT', 'QPE', 'VQE-HEA', 'VQE-UCC', 'QAOA',
                                    'QNN', 'MERA', 'MPS', 'OTOC', 'RQC'])
def test_all_families_build_and_simulate(family):
    n = 6
    layout = prism.make_layout(family, n, seed=7)
    assert prism.layout_gate_count(layout) > 0
    qc, _ = prism.compile_circuit(layout, num_qubits=n, use_numeric_params=True)
    assert qc.num_qubits == n
    sv = prism.get_statevector(qc)
    assert sv is not None
    p = prism.prob_from_statevector(sv)
    assert abs(p.sum() - 1.0) < 1e-6


def test_make_layout_reproducible():
    a = prism.make_layout('QNN', 8, seed=3)
    b = prism.make_layout('QNN', 8, seed=3)
    assert prism.layout_gate_count(a) == prism.layout_gate_count(b)


# ---------------------------------------------------------------------------
#  Partition methods
# ---------------------------------------------------------------------------
def test_all_methods_run_and_balanced(small_instance):
    layout, G, qc, n = small_instance
    results, fails = prism.run_all_partition_methods(G, layout, n, qc_full=qc, seed=19)
    assert not fails, f"methods failed: {fails}"
    assert set(results) == set(prism.METHOD_ORDER)
    for name, (A, B, dt) in results.items():
        assert A and B
        assert A.isdisjoint(B)
        assert A | B == set(range(n))
        assert abs(len(A) - len(B)) <= 2, f"{name} unbalanced"


def test_partition_dispatch(small_instance):
    layout, G, qc, n = small_instance
    A, B = prism.partition('PRISM-LCT', G, layout, n, qc_full=qc, seed=19)
    assert A | B == set(range(n))
    # alias
    A2, B2 = prism.partition('PRISM', G, layout, n, qc_full=qc, seed=19)
    assert (A2, B2) == (A, B)


def test_unknown_method_raises(small_instance):
    layout, G, qc, n = small_instance
    with pytest.raises(KeyError):
        prism.partition('not-a-method', G, layout, n)


def test_cost_terms_present(small_instance):
    layout, G, qc, n = small_instance
    sv = prism.get_statevector(qc)
    A, B = prism.naive_partition(n)
    score, detail = prism.compute_partition_cost(A, B, G, layout, n, sv_full=sv)
    assert detail['mode'] == 'physics'
    assert detail['entanglement_entropy'] is not None
    terms = prism.partition_cost_terms(A, B, G, layout, n, sv_full=sv)
    assert terms['C1_entanglement_entropy'] is not None
    # surrogate path when no statevector
    s2, d2 = prism.compute_partition_cost(A, B, G, layout, n)
    assert d2['mode'] == 'surrogate'


# ---------------------------------------------------------------------------
#  Reconstruction + metrics
# ---------------------------------------------------------------------------
def test_reconstruction_and_metrics(small_instance):
    layout, G, qc, n = small_instance
    p_ideal = prism.prob_from_statevector(prism.get_statevector(qc))
    A, B = prism.partition('PRISM-LCT', G, layout, n, qc_full=qc, seed=19)
    p_rec = prism.reconstruct_product(A, B, layout, n)
    assert p_rec is not None and abs(p_rec.sum() - 1.0) < 1e-6
    m = prism.distribution_metrics(p_ideal, p_rec)
    for k in ('tvd', 'fidelity', 'kl_divergence', 'hellinger', 'js_divergence', 'cross_entropy'):
        assert k in m
    assert 0.0 <= m['tvd'] <= 1.0
    q = prism.q_score(m)
    assert 0.0 <= q <= 1.0


def test_self_reconstruction_is_exact_for_product_state():
    # A circuit with no cross gates reconstructs perfectly under the planted cut.
    n = 4
    layout = [[{'gate': 'H', 'qubits': [i]} for i in range(n)],
              [{'gate': 'CZ', 'qubits': [0, 1]}, {'gate': 'CZ', 'qubits': [2, 3]}]]
    p_ideal = prism.prob_from_statevector(prism.get_statevector(
        prism.compile_circuit(layout, num_qubits=n)[0]))
    p_rec = prism.reconstruct_product({0, 1}, {2, 3}, layout, n)
    assert prism.distribution_metrics(p_ideal, p_rec)['tvd'] < 1e-9


# ---------------------------------------------------------------------------
#  Symmetry / QPD
# ---------------------------------------------------------------------------
def test_symmetry_and_qpd(small_instance):
    layout, G, qc, n = small_instance
    rep = prism.detect_symmetry(layout, n)
    assert 'detected_symmetries' in rep
    A, B = prism.naive_partition(n)
    gc = prism.classify_gates(layout, A, B)
    qpd = prism.compute_qpd_overhead(gc['cross'], sym_reduction=True)
    assert qpd['total_gamma_sym'] <= qpd['total_gamma_generic']
    sw = prism.schur_weyl_analysis(n, depth=len(layout))
    assert 0.0 <= sw['truncation_error'] <= 1.0


# ---------------------------------------------------------------------------
#  QAOA application layer
# ---------------------------------------------------------------------------
def test_qaoa_maxcut_and_optimizers():
    from prism import qml
    g = qml.random_maxcut_graph(8, seed=1, kind='regular', degree=3)
    opt = qml.maxcut_brute_force(g, 8)
    assert opt > 0
    res = qml.optimize_qaoa(g, 8, p=1, optimizer='COBYLA', seed=1, maxiter=30)
    assert 0.0 < res['approx_ratio'] <= 1.0 + 1e-9
    assert len(qml.OPTIMIZERS) >= 10
    rows, fails = qml.evaluate_qaoa_partitions(g, 8, res['x'], seed=1)
    assert rows and any(r.get('q_score') is not None for r in rows)


def test_qiskit_bridge_roundtrip():
    from prism import qml
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.rz(0.5, 1); qc.cx(1, 2); qc.ry(0.3, 2)
    layout = qml.qiskit_to_layout(qc)
    assert prism.layout_gate_count(layout) >= 5
    # reconstructed circuit simulates
    sv = prism.get_statevector(prism.materialize_layout(layout, 3))
    assert sv is not None and abs((np.abs(sv) ** 2).sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
#  QNN application layer
# ---------------------------------------------------------------------------
def test_qnn_iris_train_and_partition():
    from prism import qml
    X, y, nq, enc = qml.load_iris_qnn(n_classes=2)
    assert nq == 4 and enc == 'angle' and X.shape[1] == 4
    w, info = qml.train_qnn(X, y, enc, nq, reps=1, optimizer='COBYLA', maxiter=15, n_train=20)
    acc = qml.qnn_accuracy(X, y, w, enc, nq, 1)
    assert 0.0 <= acc <= 1.0
    rows = qml.evaluate_qnn_partitions(X, w, enc, nq, 1, seed=0, max_samples=2)
    assert rows and 'pred_preserved' in rows[0]


def test_qnn_digits_amplitude_uses_all_pixels():
    from prism import qml
    X, y, nq, enc = qml.load_digits_qnn(n_classes=2, n_qubits=8)
    assert X.shape[1] == 64  # all pixels, no reduction
    amp = qml.amplitude_encode_layout(X[0], 6)   # 64 -> 6 qubits
    assert prism.layout_gate_count(amp) > 0


# ---------------------------------------------------------------------------
#  Benchmark driver
# ---------------------------------------------------------------------------
def test_benchmark_instance():
    rows, fails = prism.benchmark_instance('QAOA', 6, 19)
    assert rows
    methods = {r['method'] for r in rows}
    assert 'PRISM-LCT' in methods
    for r in rows:
        if r.get('q_score') is not None:
            assert 0.0 <= r['q_score'] <= 1.0


# ---------------------------------------------------------------------------
#  Multi-QPU (k-way) partitioning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('k', [2, 3, 4])
def test_kway_partition(k):
    n = 8
    layout = prism.make_layout('QNN', n, seed=19)
    qc, _ = prism.compile_circuit(layout, num_qubits=n)
    parts = prism.kway_partition('PRISM-LCT', layout, n, k, qc_full=qc, seed=19)
    assert len(parts) == k
    # disjoint cover of all qubits
    union = set()
    for S in parts:
        assert S and union.isdisjoint(S)
        union |= S
    assert union == set(range(n))
    st = prism.kway_stats(layout, parts, n)
    assert st['k'] == k and st['max_fragment'] <= n
    p = prism.reconstruct_product_kway(parts, layout, n)
    assert p is not None and abs(p.sum() - 1.0) < 1e-6


def test_kway_max_fragment_shrinks():
    n = 12
    layout = prism.make_layout('MPS', n, seed=23)
    qc, _ = prism.compile_circuit(layout, num_qubits=n)
    mf2 = prism.kway_stats(layout, prism.kway_partition('PRISM-LCT', layout, n, 2, qc_full=qc), n)['max_fragment']
    mf4 = prism.kway_stats(layout, prism.kway_partition('PRISM-LCT', layout, n, 4, qc_full=qc), n)['max_fragment']
    assert mf4 <= mf2  # more QPUs -> smaller fragments


# ---------------------------------------------------------------------------
#  Distributed execution (gate teleportation)
# ---------------------------------------------------------------------------
def test_nonlocal_cnot_equals_cnot():
    ok, tvd = prism.verify_nonlocal_cnot(shots=8000)
    assert ok, f"teleported CNOT TVD vs local CNOT too high: {tvd}"


def test_distributed_qaoa_matches_ideal():
    from prism import distributed as dist
    n = 6
    layout = prism.make_layout('QAOA', n, seed=19)
    qc, _ = prism.compile_circuit(layout, num_qubits=n)
    p_ideal = prism.prob_from_statevector(prism.get_statevector(qc))
    G = prism.build_interaction_graph(layout, n)
    A, B = prism.partition('PRISM-LCT', G, layout, n, qc_full=qc, seed=19)
    p_d, info = dist.distributed_sample_probs(layout, A, B, n, shots=40000, seed=1)
    assert info['n_ebits'] == prism.count_cross_gates(layout, A, B)
    tvd = 0.5 * float(np.sum(np.abs(p_ideal - p_d)))
    assert tvd < 0.05  # exact up to shot noise


# ---------------------------------------------------------------------------
#  QNN deployment (datasets, entangler topologies, distributed accuracy)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['iris', 'wine', 'breast_cancer', 'digits', 'synth'])
def test_qnn_datasets_load(name):
    from prism import qml
    X, y, nq, enc = qml.load_qnn_dataset(name, n_qubits=6)
    assert X.shape[0] == len(y) and X.shape[1] >= 4
    assert enc in ('angle', 'reupload')
    assert set(np.unique(y)).issubset({0, 1, 2})


@pytest.mark.parametrize('entangler', ['ring', 'linear', 'blocks', 'full'])
def test_qnn_entangler_layouts(entangler):
    from prism import qml
    nq = 6
    layout = qml.qnn_circuit_layout(np.zeros(nq), np.zeros(qml.qnn_n_params(nq, 2)),
                                    'angle', nq, 2, entangler)
    assert prism.layout_gate_count(layout) > 0
    sv = prism.get_statevector(prism.materialize_layout(layout, nq))
    assert sv is not None


def test_qnn_distributed_report():
    from prism import qml
    X, y, nq, enc = qml.load_iris_qnn(n_classes=2)
    w, _ = qml.train_qnn(X, y, enc, nq, reps=1, optimizer='COBYLA', maxiter=15,
                         n_train=20, entangler='blocks')
    A, B = qml.qnn_cut_for(X[0], w, enc, nq, 1, method='PRISM-LCT', seed=0, entangler='blocks')
    rep = qml.qnn_distributed_report(X[:20], y[:20], w, enc, nq, 1, A, B, entangler='blocks')
    assert 0.0 <= rep['accuracy'] <= 1.0 and 0.0 <= rep['agreement'] <= 1.0
