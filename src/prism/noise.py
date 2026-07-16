"""
prism.noise
===========
Noisy simulation and noisy fragment reconstruction.

The circuit-cutting argument for NISQ hardware is that each *fragment* runs
fewer gates than the monolithic circuit and therefore accumulates less noise;
a good partition can make ``reconstruct(noisy fragments)`` more faithful than
``noisy full circuit``. This module provides exact (density-matrix) noisy
probabilities so that argument can be measured without shot noise.

    depolarizing_noise_model   per-gate depolarizing (+ optional readout) model
    noisy_probabilities        exact noisy output distribution of a circuit
    noisy_layout_probabilities noisy distribution of a PRISM layout
    noisy_reconstruct_product  product reconstruction from *noisy* fragments
"""
from __future__ import annotations

import numpy as np

from .compiler import materialize_layout
from .graph import build_subcircuit_layout
from .simulate import build_recon_index

# Noise is attached to a fixed two-gate basis so it survives transpilation.
_BASIS = ['u', 'cx']


def depolarizing_noise_model(p1=0.001, p2=0.01, p_readout=0.0):
    """Depolarizing error ``p1`` on 1-qubit (``u``) and ``p2`` on 2-qubit
    (``cx``) gates, with optional symmetric readout error ``p_readout``."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    nm = NoiseModel(basis_gates=_BASIS)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['u'])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    if p_readout > 0:
        ro = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        nm.add_all_qubit_readout_error(ro)
    return nm


def noisy_probabilities(qc, noise_model, n_qubits=None):
    """Exact noisy output distribution of a circuit via density-matrix Aer."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    # Translate gates to the noisy basis FIRST (before adding the save
    # instruction, which the basis translator cannot rewrite), then run.
    tq = transpile(qc, basis_gates=_BASIS, optimization_level=0)
    tq.save_probabilities()
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    res = sim.run(tq).result()
    p = np.asarray(res.data(0)['probabilities'], dtype=float)
    s = p.sum()
    return p / s if s > 0 else p


def noisy_layout_probabilities(layout, n_qubits, noise_model):
    """Noisy distribution of a full PRISM layout."""
    return noisy_probabilities(materialize_layout(layout, n_qubits), noise_model, n_qubits)


def _noisy_fragment(node_set, layout, noise_model):
    sub_layout, sorted_nodes, _ = build_subcircuit_layout(layout, node_set, 'X')
    qc = materialize_layout(sub_layout, len(node_set))
    return noisy_probabilities(qc, noise_model, len(node_set)), sorted_nodes


def noisy_reconstruct_product(A_set, B_set, layout, n_qubits, noise_model):
    """Product reconstruction ``p_A (x) p_B`` from *noisy* fragment marginals."""
    pA, sA = _noisy_fragment(A_set, layout, noise_model)
    pB, sB = _noisy_fragment(B_set, layout, noise_model)
    if pA is None or pB is None:
        return None
    p = pA[build_recon_index(sA, n_qubits)] * pB[build_recon_index(sB, n_qubits)]
    s = p.sum()
    return p / s if s > 0 else None


__all__ = [
    'depolarizing_noise_model', 'noisy_probabilities',
    'noisy_layout_probabilities', 'noisy_reconstruct_product',
]
