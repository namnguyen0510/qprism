"""
PRISM — Multi-Objective Symmetry-Aware Irrep-truncated Circuit-partition
========================================================================

A commercial-grade toolkit for partitioning quantum circuits for distributed
(multi-QPU) execution, with a multi-objective, symmetry-aware, light-cone
tempered solver (**PRISM-LCT**) benchmarked against six classical baselines
and a four-rung ablation ladder.

Quick start
-----------
>>> import prism
>>> layout = prism.make_layout('QAOA', n=10, seed=19)
>>> G = prism.build_interaction_graph(layout, 10)
>>> qc, _ = prism.compile_circuit(layout, num_qubits=10)
>>> A, B = prism.partition('PRISM-LCT', G, layout, 10, qc_full=qc, seed=19)
>>> results, fails = prism.run_all_partition_methods(G, layout, 10, qc_full=qc)

The public API re-exports every building block: gate taxonomy, circuit
families, graph construction, the multi-objective cost, all partition
methods, exact statevector reconstruction + metrics, symmetry/QPD analysis,
the QAOA/QNN application layer (:mod:`prism.qml`), and the benchmark driver.
"""
from __future__ import annotations

__version__ = "0.1.0"
__paper__ = "PRISM: Multi-Objective Symmetry-Aware Irrep-truncated Circuit-partition"

# ── gate taxonomy ──────────────────────────────────────────────────────────
from .gates import (
    GATE_ARITY, quantum_gates, SINGLE, TWO, MULTI,
    DIAGONAL_GATES, PARTICLE_PRESERVING, CLIFFORD,
    SCHMIDT_RANK, QPD_GAMMA, QPD_GAMMA_SYM, QPD_DECOMP,
    gate_arity, is_two_qubit, schmidt_rank, qpd_gamma, gate_type_code,
)

# ── compilation ────────────────────────────────────────────────────────────
from .compiler import (
    compile_circuit, compile_quantum_circuit,
    append_gate, materialize_layout, deterministic_param_values,
)

# ── circuit families ───────────────────────────────────────────────────────
from .circuits import (
    gen_qft, gen_qpe, gen_vqe_hea, gen_vqe_ucc, gen_qaoa,
    gen_qnn, gen_mera, gen_mps, gen_otoc, generate_ansatz_layout,
    CIRCUIT_FAMILIES, FAMILY_SEEDS, RQC_DEPTH,
    make_layout, layout_gate_count, layout_depth, layout_stats,
)

# ── interaction graph & cuts ───────────────────────────────────────────────
from .graph import (
    build_interaction_graph, estimate_edge_entanglement, build_lightcone_graph,
    classify_gates, compute_cut_stats, build_subcircuit_layout,
)

# ── symmetry / QPD ─────────────────────────────────────────────────────────
from .symmetry import (
    irrep_xi, detect_symmetry, compute_qpd_overhead,
    schur_weyl_analysis, build_cg_aggregation,
)

# ── simulation & metrics ───────────────────────────────────────────────────
from .simulate import (
    get_statevector, prob_from_statevector, statevector_probabilities,
    entanglement_entropy, build_recon_index, simulate_subcircuit,
    reconstruct_product, reconstruct_product_kway,
    distribution_metrics, q_score, compute_unified_scores,
)

# ── partition methods (all of them) ────────────────────────────────────────
from .partition import (
    PARTITION_METHODS, METHOD_ORDER, BASELINE_METHODS, PRISM_LADDER, MAIN_METHOD,
    partition, run_all_partition_methods,
    compute_partition_cost, partition_cost_surrogate, partition_cost_terms,
    naive_partition, spectral_partition, louvain_partition,
    girvan_newman_partition, metis_partition, qdislib_partition, has_real_qdislib,
    kl_interaction_partition, optimize_partition_sa,
    prism_kl, prism_oe, prism_mi, prism_bf, prism_lct,
    kway_partition, kway_cross_gates, kway_stats,
)

# ── benchmark driver ───────────────────────────────────────────────────────
from .benchmark import (
    evaluate_partition, benchmark_instance, run_benchmark, summarise_benchmark,
)

# ── noisy simulation & reconstruction ──────────────────────────────────────
from .noise import (
    depolarizing_noise_model, noisy_probabilities,
    noisy_layout_probabilities, noisy_reconstruct_product,
)

# ── distributed execution (gate teleportation) ─────────────────────────────
from .distributed import (
    nonlocal_cnot, nonlocal_cz, verify_nonlocal_cnot,
    build_distributed_circuit, distributed_sample_probs, count_cross_gates,
)

# Canonical colour scheme shared by the notebooks.
METHOD_COLOURS = {
    'Naive': '#95A5A6', 'Spectral': '#7F8C8D', 'Louvain': '#5D6D7E',
    'Girvan-Newman': '#85929E', 'METIS': '#34495E', 'qdislib': '#566573',
    'PRISM-KL': '#A9DFBF', 'PRISM-OE': '#7DCEA0', 'PRISM-MI': '#52BE80',
    'PRISM-BF': '#2ECC71', 'PRISM-LCT': '#1E8449', 'PRISM': '#1E8449',
}
FAMILY_COLOURS = {'QFT': '#16A085', 'QPE': '#2E86C1', 'VQE-HEA': '#8E44AD',
                  'VQE-UCC': '#CB4335', 'QAOA': '#D68910', 'QNN': '#8E44AD',
                  'MERA': '#E67E22', 'MPS': '#2980B9', 'RQC': '#C0392B', 'OTOC': '#117A65'}


def list_methods():
    """Names of all partition methods in canonical order."""
    return list(METHOD_ORDER)


def list_families():
    """Names of all circuit families (structured) plus RQC."""
    return list(CIRCUIT_FAMILIES.keys()) + ['RQC']


__all__ = [
    '__version__',
    # gates
    'GATE_ARITY', 'quantum_gates', 'SINGLE', 'TWO', 'MULTI',
    'DIAGONAL_GATES', 'PARTICLE_PRESERVING', 'CLIFFORD',
    'SCHMIDT_RANK', 'QPD_GAMMA', 'QPD_GAMMA_SYM', 'QPD_DECOMP',
    'gate_arity', 'is_two_qubit', 'schmidt_rank', 'qpd_gamma', 'gate_type_code',
    # compiler
    'compile_circuit', 'compile_quantum_circuit', 'append_gate',
    'materialize_layout', 'deterministic_param_values',
    # circuits
    'gen_qft', 'gen_qpe', 'gen_vqe_hea', 'gen_vqe_ucc', 'gen_qaoa',
    'gen_qnn', 'gen_mera', 'gen_mps', 'gen_otoc', 'generate_ansatz_layout',
    'CIRCUIT_FAMILIES', 'FAMILY_SEEDS', 'RQC_DEPTH',
    'make_layout', 'layout_gate_count', 'layout_depth', 'layout_stats',
    # graph
    'build_interaction_graph', 'estimate_edge_entanglement', 'build_lightcone_graph',
    'classify_gates', 'compute_cut_stats', 'build_subcircuit_layout',
    # symmetry
    'irrep_xi', 'detect_symmetry', 'compute_qpd_overhead',
    'schur_weyl_analysis', 'build_cg_aggregation',
    # simulate
    'get_statevector', 'prob_from_statevector', 'statevector_probabilities',
    'entanglement_entropy', 'build_recon_index', 'simulate_subcircuit',
    'reconstruct_product', 'reconstruct_product_kway',
    'distribution_metrics', 'q_score', 'compute_unified_scores',
    # partition
    'PARTITION_METHODS', 'METHOD_ORDER', 'BASELINE_METHODS', 'PRISM_LADDER', 'MAIN_METHOD',
    'partition', 'run_all_partition_methods',
    'compute_partition_cost', 'partition_cost_surrogate', 'partition_cost_terms',
    'naive_partition', 'spectral_partition', 'louvain_partition',
    'girvan_newman_partition', 'metis_partition', 'qdislib_partition', 'has_real_qdislib',
    'kl_interaction_partition', 'optimize_partition_sa',
    'prism_kl', 'prism_oe', 'prism_mi', 'prism_bf', 'prism_lct',
    'kway_partition', 'kway_cross_gates', 'kway_stats',
    # benchmark
    'evaluate_partition', 'benchmark_instance', 'run_benchmark', 'summarise_benchmark',
    # noise
    'depolarizing_noise_model', 'noisy_probabilities',
    'noisy_layout_probabilities', 'noisy_reconstruct_product',
    # distributed
    'nonlocal_cnot', 'nonlocal_cz', 'verify_nonlocal_cnot',
    'build_distributed_circuit', 'distributed_sample_probs', 'count_cross_gates',
    # extras
    'METHOD_COLOURS', 'FAMILY_COLOURS', 'list_methods', 'list_families',
]
