"""
prism.symmetry
==============
Symmetry detection, Schur-Weyl irrep-sector estimation, QPD gate-cutting
overhead, and the classical Clebsch-Gordan aggregation map.

These quantify the C3 term of the PRISM objective (symmetry-reduced QPD
overhead ``log gamma_sym``) and underpin the claim that classical
recombination across QPUs is cheap when symmetry sectors survive.
"""
from __future__ import annotations

import math
from collections import Counter

from .gates import (SINGLE, TWO, MULTI, DIAGONAL_GATES, PARTICLE_PRESERVING,
                    CLIFFORD, SCHMIDT_RANK, QPD_GAMMA, QPD_GAMMA_SYM, QPD_DECOMP)


def irrep_xi(lam_size, depth):
    """Irrep correlation length xi_lambda ~ |lambda| / log D."""
    return max(lam_size / max(math.log(max(depth, 1)), 1e-9), 0.5)


def detect_symmetry(layout, n_qubits, depth=None):
    """Detect U(1)/Z2 charges, Clifford fraction, and a DLA / t-design estimate."""
    if depth is None:
        depth = len(layout)
    gate_counts = Counter()
    for layer in layout:
        for g in layer:
            gate_counts[g['gate'].upper()] += 1
    total = sum(gate_counts.values())
    n_single = sum(v for k, v in gate_counts.items() if k in SINGLE)
    n_two = sum(v for k, v in gate_counts.items() if k in TWO)
    n_multi = sum(v for k, v in gate_counts.items() if k in MULTI)
    two_used = {k for k in gate_counts if k in TWO}
    u1_ok = two_used.issubset(PARTICLE_PRESERVING)
    z2_ok = all(k in DIAGONAL_GATES | {'I', 'Z', 'CZ', 'CRZ'} or k in PARTICLE_PRESERVING
                for k in gate_counts)
    n_clifford = sum(v for k, v in gate_counts.items() if k in CLIFFORD)
    clifford_frac = n_clifford / max(total, 1)
    n_distinct = len(gate_counts)
    dla = min(n_distinct * 4, 4 ** n_qubits - 1)
    approx_t = 1 if depth < 2 * n_qubits else 2
    sym_list = []
    if u1_ok:
        sym_list.append('U(1) [particle-number]')
    if z2_ok:
        sym_list.append('Z2 [parity]')
    if clifford_frac > 0.8:
        sym_list.append('Clifford group (dominant)')
    return {
        'n_qubits': n_qubits, 'depth': depth, 'total_gates': total,
        'single_qubit_gates': n_single, 'two_qubit_gates': n_two, 'multi_qubit_gates': n_multi,
        'gate_type_counts': dict(gate_counts), 'u1_conserved_charge': bool(u1_ok),
        'z2_parity_symmetry': bool(z2_ok), 'clifford_fraction': round(clifford_frac, 4),
        'two_qubit_gate_types_used': sorted(two_used),
        'all_two_qubit_particle_preserving': bool(u1_ok),
        'dla_dimension_estimate': int(dla), 'approx_t_design_order': int(approx_t),
        'detected_symmetries': sym_list,
    }


def compute_qpd_overhead(cross_gates, sym_reduction=True, epsilon=0.05):
    """Multiplicative QPD sampling overhead gamma over all cut gates,
    generic and symmetry-reduced, plus total Schmidt rank and shot budget."""
    g_gen, g_sym, schmidt = 1.0, 1.0, 0
    per_gate = []
    for g in cross_gates:
        name = g['gate'].upper()
        sr = SCHMIDT_RANK.get(name, 4)
        gam = QPD_GAMMA.get(name, 9.0)
        gam_s = QPD_GAMMA_SYM.get(name, gam)
        g_gen *= gam
        g_sym *= gam_s
        schmidt += sr
        per_gate.append({'gate': name, 'layer': g.get('layer'), 'qubits': g['qubits'],
                         'schmidt_rank': sr, 'gamma_generic': gam, 'gamma_sym': gam_s,
                         'qpd_terms': QPD_DECOMP.get(name, [])})
    return {
        'n_cut_gates': len(cross_gates), 'per_gate': per_gate,
        'total_gamma_generic': g_gen, 'total_gamma_sym': g_sym,
        'gamma_reduction_ratio': round(g_sym / max(g_gen, 1e-30), 6),
        'total_schmidt_rank': schmidt,
        'shot_overhead_generic': g_gen / epsilon ** 2,
        'shot_overhead_sym': g_sym / epsilon ** 2,
        'epsilon': epsilon,
    }


def schur_weyl_analysis(n_qubits, depth, n_cut_gates=0):
    """Estimate Schur-Weyl irrep-sector weights and the depth-truncation error."""
    d, n = depth, n_qubits
    sec = [{'label': '(n)', 'name': 'trivial / fully symmetric', 'dim_Sn': 1,
            'xi': 999.0, 'weight': 1.0, 'keep': True,
            'role': 'dominant - local product reconstruction'}]
    xi1 = irrep_xi(1, d)
    w1 = (n - 1) * math.exp(-d / xi1)
    sec.append({'label': '(n-1,1)', 'name': 'standard / defining', 'dim_Sn': n - 1,
                'xi': round(xi1, 3), 'weight': round(min(w1, 1.0), 6), 'keep': True,
                'role': 'leading correction - CG coupling across cut'})
    xi2, dim2 = irrep_xi(2, d), n * (n - 3) // 2
    w2 = dim2 * math.exp(-d / xi2)
    sec.append({'label': '(n-2,2)', 'name': 'adjoint', 'dim_Sn': dim2,
                'xi': round(xi2, 3), 'weight': round(min(w2, 1.0), 6), 'keep': (w2 > 0.01),
                'role': f'truncate if depth >= 2xi2 (~{2 * xi2:.0f})'})
    xi3, dim3 = irrep_xi(2, d), (n - 1) * (n - 2) // 2
    w3 = dim3 * math.exp(-d / xi3)
    sec.append({'label': '(n-2,1,1)', 'name': 'hook', 'dim_Sn': dim3,
                'xi': round(xi3, 3), 'weight': round(min(w3, 1.0), 6), 'keep': False,
                'role': f'drop (eps-small for d >= {math.ceil(3 * xi3)})'})
    trunc_err = sum(s['weight'] for s in sec if not s['keep'])
    kept = [s for s in sec if s['keep']]
    n_cg = sum(s['dim_Sn'] ** 2 for s in kept)
    return {'sectors': sec, 'truncation_error': round(min(trunc_err, 1.0), 6),
            'n_sectors_kept': len(kept), 'n_cg_coefficients': int(n_cg),
            'depth': d, 'xi_defining': round(xi1, 3)}


def build_cg_aggregation(schur, sym_report, n_qubits):
    """Classical Clebsch-Gordan recombination map (U(1) charge sectors when a
    particle-number symmetry is present, else Schur-Weyl irreps)."""
    n = n_qubits
    nA, nB = n // 2, n - n // 2
    has_u1 = sym_report['u1_conserved_charge']
    sectors = []
    if has_u1:
        for k in range(n + 1):
            dim_full = math.comb(n, k)
            sub_pairs = [(kA, k - kA) for kA in range(min(nA, k) + 1) if 0 <= k - kA <= nB]
            sectors.append({'charge_k': k, 'dim_full': dim_full, 'sub_pairs': sub_pairs,
                            'weight': round(1.0 / max(dim_full, 1), 8)})
    else:
        for s in schur['sectors']:
            if s['keep']:
                sectors.append({'irrep': s['label'], 'dim_Sn': s['dim_Sn'],
                                'weight': s['weight'], 'cg_terms': s['dim_Sn'] ** 2})
    return {
        'reconstruction_type': 'U(1) charge-sector CG' if has_u1 else 'Schur-Weyl irrep CG',
        'n_A': nA, 'n_B': nB, 'n_sectors': min(len(sectors), 20), 'sectors_shown': sectors[:6],
        'total_cg_coefficients': int(schur['n_cg_coefficients']),
        'error_bound': round(schur['truncation_error'], 6),
    }


__all__ = [
    'irrep_xi', 'detect_symmetry', 'compute_qpd_overhead',
    'schur_weyl_analysis', 'build_cg_aggregation',
]
