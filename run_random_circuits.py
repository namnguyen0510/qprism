#!/usr/bin/env python3
"""
mosaic_partitioner.py
===================
Symmetry-Aware Circuit Partitioning (mosaic) framework.
Implements:
Phase 0  – Generate a 100-qubit RQC (no circuit execution)
Phase 1  – Symmetry detection: U(1)/Z2 charges, gate-set algebra scan,
           Clifford vs non-Clifford fraction, particle-preserving subgroup
Phase 2  – Hilbert-space decomposition: weighted qubit-interaction graph +
           advanced partitioning (KL / METIS / entanglement-SA); Schur-Weyl
           irrep sector estimation; operator Schmidt rank bounds
Phase 3  – Gate cutting: QPD quasi-probability decomposition of every
           cross-partition gate, overhead γ computed with and without
           symmetry reduction
Phase 4  – Classical aggregation map: Clebsch-Gordan reconstruction
           formula per U(1) sector; error bound vs depth
"""
import sys, os, json, pickle, datetime, math, random, textwrap, csv, inspect, hashlib
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

from qcirc_generator import generate_ansatz_layout
from compiler import compile_quantum_circuit
from qops import quantum_gates

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
N_QUBITS        = 16
DEPTH           = 15
MAX_Q_PER_LAYER = 40
SEED            = 42
TS              = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR          = Path(f"mosaic_output_{TS}")
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
#  GATE TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────
ARITY = quantum_gates
SINGLE = {g for g,a in ARITY.items() if a == 1}
TWO    = {g for g,a in ARITY.items() if a == 2}
MULTI  = {g for g,a in ARITY.items() if a > 2}

DIAGONAL_GATES = {'Z','RZ','CZ','T','S','PHASE','U1','CU1','CPHASE','CRZ'}
PARTICLE_PRESERVING = {'SWAP','ISWAP','FSWAP','CZ','CRZ','CU1','CPHASE','U1','PHASE'}
CLIFFORD = {'H','X','Y','Z','S','CZ','CNOT','CX','CY','SWAP','CCX','CSWAP'}

SCHMIDT_RANK = {
    'CNOT':2,'CX':2,'CY':2,'CZ':2,'CH':2,
    'CRX':2,'CRY':2,'CRZ':2,'CU1':2,'CU3':4,'CPHASE':2,
    'SWAP':4,'ISWAP':4,
    'CCX':4,'CSWAP':8,'C3X':4,'C3Z':4,
}
QPD_GAMMA = {
    'CNOT':3.0,'CX':3.0,'CY':3.0,'CZ':3.0,'CH':3.0,
    'CRX':3.0,'CRY':3.0,'CRZ':3.0,'CU1':3.0,'CU3':7.0,'CPHASE':3.0,
    'SWAP':7.0,'ISWAP':7.0,
    'CCX':9.0,'CSWAP':9.0,'C3X':9.0,'C3Z':9.0,
}
QPD_GAMMA_SYM = {
    'CNOT':2.0,'CX':2.0,'CY':2.0,'CZ':2.0,'CH':2.0,
    'CRX':2.0,'CRY':2.0,'CRZ':2.0,'CU1':2.0,'CU3':4.0,'CPHASE':2.0,
    'SWAP':3.0,'ISWAP':3.0,
    'CCX':4.0,'CSWAP':4.0,'C3X':4.0,'C3Z':4.0,
}
QPD_DECOMP = {
    'CNOT': [(+0.50,'I_A','I_B'), (+0.50,'I_A','X_B'), (+0.50,'Z_A','I_B'), (-0.50,'Z_A','X_B')],
    'CZ':   [(+0.50,'I_A','I_B'), (+0.50,'I_A','Z_B'), (+0.50,'Z_A','I_B'), (-0.50,'Z_A','Z_B')],
    'SWAP': [(+0.25,'I_A','I_B'),(+0.25,'X_A','X_B'), (-0.25,'Y_A','Y_B'),(+0.25,'Z_A','Z_B')],
    'ISWAP':[(+0.25,'I_A','I_B'),(+0.25,'X_A','X_B'), (+0.25,'Y_A','Y_B'),(-0.25,'Z_A','Z_B')],
    'CRZ':  [(+0.50,'I_A','I_B'),(+0.50,'I_A','RZ_B'), (+0.50,'Z_A','I_B'),(-0.50,'Z_A','RZ_B')],
    'CRX':  [(+0.50,'I_A','I_B'),(+0.50,'I_A','RX_B'), (+0.50,'Z_A','I_B'),(-0.50,'Z_A','RX_B')],
    'CRY':  [(+0.50,'I_A','I_B'),(+0.50,'I_A','RY_B'), (+0.50,'Z_A','I_B'),(-0.50,'Z_A','RY_B')],
}

def irrep_xi(lam_size: int, d: int = DEPTH) -> float:
    return max(lam_size / max(math.log(max(d, 1)), 1e-9), 0.5)

IRREP_LABELS = [
    ('(n)',      'trivial / fully symmetric',   1),
    ('(n-1,1)', 'defining / standard',          N_QUBITS - 1),
    ('(n-2,2)', 'adjoint',                      (N_QUBITS*(N_QUBITS-3))//2),
    ('(n-2,1,1)','hook',                        (N_QUBITS-1)*(N_QUBITS-2)//2),
]

# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 0 — CIRCUIT GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_circuit(n_qubits: int, depth: int, max_q: int):
    print(f"[Phase 0] Generating {n_qubits}-qubit RQC  depth={depth}  max_q_per_layer={max_q}")
    layout = generate_ansatz_layout(n_qubits, depth, max_qubits_per_layer=max_q)
    qc, param_list = compile_quantum_circuit(layout)
    print(f"         Gates: {sum(len(l) for l in layout)}  Parameters: {len(param_list)}")
    return layout, qc, param_list

# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — SYMMETRY DETECTION
# ──────────────────────────────────────────────────────────────────────────────
def detect_symmetry(layout: list, n_qubits: int) -> dict:
    print("[Phase 1] Symmetry detection …")
    gate_counts = Counter()
    for layer in layout:
        for g in layer:
            gate_counts[g['gate'].upper()] += 1
    total = sum(gate_counts.values())
    n_single  = sum(v for k,v in gate_counts.items() if k in SINGLE)
    n_two     = sum(v for k,v in gate_counts.items() if k in TWO)
    n_multi   = sum(v for k,v in gate_counts.items() if k in MULTI)
    two_gates_used = {k for k in gate_counts if k in TWO}
    u1_ok = two_gates_used.issubset(PARTICLE_PRESERVING)
    z2_ok = all(k in DIAGONAL_GATES | {'I','Z','CZ','CRZ'} or k in PARTICLE_PRESERVING for k in gate_counts)
    n_clifford = sum(v for k,v in gate_counts.items() if k in CLIFFORD)
    clifford_frac = n_clifford / max(total, 1)
    n_distinct_types = len(gate_counts)
    dla_estimate = min(n_distinct_types * 4, 4**n_qubits - 1)
    t_design_depth_threshold = 2 * n_qubits
    approx_t = 1 if DEPTH < t_design_depth_threshold else 2
    report = {
        'n_qubits': n_qubits, 'depth': DEPTH, 'total_gates': total,
        'single_qubit_gates': n_single, 'two_qubit_gates': n_two, 'multi_qubit_gates': n_multi,
        'gate_type_counts': dict(gate_counts), 'u1_conserved_charge': bool(u1_ok),
        'z2_parity_symmetry': bool(z2_ok), 'clifford_fraction': round(clifford_frac, 4),
        'two_qubit_gate_types_used': sorted(two_gates_used),
        'all_two_qubit_particle_preserving': bool(u1_ok),
        'dla_dimension_estimate': int(dla_estimate), 'approx_t_design_order': int(approx_t),
        'weingarten_avg_nonzero_blocks': int(2**min(approx_t, 3)),
    }
    sym_list = []
    if u1_ok: sym_list.append('U(1)  [particle-number]')
    if z2_ok: sym_list.append('Z₂   [parity]')
    if clifford_frac > 0.8: sym_list.append('Clifford group (dominant)')
    report['detected_symmetries'] = sym_list
    print(f"         Symmetries: {sym_list if sym_list else ['none detected']}")
    print(f"         Clifford fraction: {clifford_frac:.1%}  U(1) charge: {u1_ok}  approx {approx_t}-design")
    return report

# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — INTERACTION GRAPH + KERNIGHAN-LIN PARTITIONING
# ──────────────────────────────────────────────────────────────────────────────
def build_interaction_graph(layout: list, n_qubits: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    for layer in layout:
        for gate in layer:
            qs = gate['qubits']
            if len(qs) >= 2:
                for i in range(len(qs)):
                    for j in range(i+1, len(qs)):
                        u, v = qs[i], qs[j]
                        if G.has_edge(u, v):
                            G[u][v]['weight'] += 1
                            G[u][v]['gates'].append(gate['gate'])
                        else:
                            G.add_edge(u, v, weight=1, gates=[gate['gate']])
    return G

def partition_graph(G: nx.Graph, n_qubits: int) -> tuple[set, set]:
    connected_nodes = set(G.nodes())
    isolated = [n for n in range(n_qubits) if n not in connected_nodes or G.degree(n) == 0]
    active   = [n for n in range(n_qubits) if n not in isolated]
    if len(active) >= 2:
        H = G.subgraph(active).copy()
        components = list(nx.connected_components(H))
        if len(components) > 1:
            for i in range(len(components)-1):
                u, v = next(iter(components[i])), next(iter(components[i+1]))
                H.add_edge(u, v, weight=0.001, gates=['virtual'])
        A_set, B_set = nx.algorithms.community.kernighan_lin_bisection(H, weight='weight', seed=SEED)
    else:
        half = n_qubits // 2
        A_set, B_set = set(range(half)), set(range(half, n_qubits))
    iso_A, iso_B = isolated[0::2], isolated[1::2]
    A_set, B_set = A_set | set(iso_A), B_set | set(iso_B)
    while abs(len(A_set) - len(B_set)) > 2:
        if len(A_set) > len(B_set):
            node = next(iter(A_set)); A_set.remove(node); B_set.add(node)
        else:
            node = next(iter(B_set)); B_set.remove(node); A_set.add(node)
    return A_set, B_set

def compute_cut_stats(G: nx.Graph, A_set: set, B_set: set) -> dict:
    cut_edges, inner_A, inner_B = [], [], []
    cut_weight = 0
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1)
        if (u in A_set) == (v in A_set):
            (inner_A if u in A_set else inner_B).append((u, v, d))
        else:
            cut_edges.append((u, v, d))
            cut_weight += w
    total_w = sum(d.get('weight',1) for _,_,d in G.edges(data=True))
    return {
        'n_cut_edges': len(cut_edges), 'cut_weight': int(cut_weight),
        'total_edge_weight': int(total_w), 'cut_fraction': round(cut_weight / max(total_w, 1), 4),
        'inner_A_edges': len(inner_A), 'inner_B_edges': len(inner_B),
        'cut_edges': cut_edges, 'inner_A': inner_A, 'inner_B': inner_B,
    }

# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — GATE CLASSIFICATION + QPD ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def classify_gates(layout: list, A_set: set, B_set: set) -> dict:
    local_A, local_B, cross = [], [], []
    for layer_idx, layer in enumerate(layout):
        for gate in layer:
            qs = set(gate['qubits'])
            entry = {**gate, 'layer': layer_idx}
            if qs.issubset(A_set): local_A.append(entry)
            elif qs.issubset(B_set): local_B.append(entry)
            else: cross.append(entry)
    return {'local_A': local_A, 'local_B': local_B, 'cross': cross}

def compute_qpd_overhead(cross_gates: list, sym_reduction: bool) -> dict:
    total_gamma_generic, total_gamma_sym, total_schmidt_rank = 1.0, 1.0, 0
    per_gate = []
    for g in cross_gates:
        name = g['gate'].upper()
        sr   = SCHMIDT_RANK.get(name, 4)
        gam  = QPD_GAMMA.get(name, 9.0)
        gam_s= QPD_GAMMA_SYM.get(name, gam)
        terms= QPD_DECOMP.get(name, [])
        total_gamma_generic *= gam
        total_gamma_sym     *= gam_s
        total_schmidt_rank  += sr
        per_gate.append({'gate': name, 'layer': g['layer'], 'qubits': g['qubits'],
                         'schmidt_rank': sr, 'gamma_generic': gam, 'gamma_sym': gam_s, 'qpd_terms': terms})
    eps = 0.05
    return {
        'n_cut_gates': len(cross_gates), 'per_gate': per_gate,
        'total_gamma_generic': total_gamma_generic, 'total_gamma_sym': total_gamma_sym,
        'gamma_reduction_ratio': round(total_gamma_sym / max(total_gamma_generic,1e-30), 6),
        'total_schmidt_rank': total_schmidt_rank,
        'shot_overhead_generic': f'{total_gamma_generic / eps**2:.2e}',
        'shot_overhead_sym': f'{total_gamma_sym / eps**2:.2e}', 'epsilon': eps,
    }

# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 2b — SCHUR-WEYL SECTOR ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def schur_weyl_analysis(n_qubits: int, depth: int, n_cut_gates: int) -> dict:
    d, n = depth, n_qubits
    sec = []
    sec.append({'label': '(n)', 'name': 'trivial / fully symmetric', 'dim_Sn': 1,
                'xi': 999.0, 'weight': 1.0, 'keep': True, 'role': 'dominant — local product reconstruction'})
    xi1 = irrep_xi(1, d)
    w1  = (n-1) * math.exp(-d / xi1)
    sec.append({'label': '(n−1,1)', 'name': 'standard / defining representation', 'dim_Sn': n - 1,
                'xi': round(xi1, 3), 'weight': round(min(w1, 1.0), 6), 'keep': True,
                'role': 'leading correction — CG coupling across cut'})
    xi2, dim2 = irrep_xi(2, d), n*(n-3)//2
    w2 = dim2 * math.exp(-d / xi2)
    sec.append({'label': '(n−2,2)', 'name': 'adjoint / second exterior power', 'dim_Sn': dim2,
                'xi': round(xi2, 3), 'weight': round(min(w2, 1.0), 6), 'keep': (w2 > 0.01),
                'role': f'truncate if depth ≥ 2ξ₂ (≈ {2*xi2:.0f})'})
    xi3, dim3 = irrep_xi(2, d), (n-1)*(n-2)//2
    w3 = dim3 * math.exp(-d / xi3)
    sec.append({'label': '(n−2,1,1)', 'name': 'hook representation', 'dim_Sn': dim3,
                'xi': round(xi3, 3), 'weight': round(min(w3, 1.0), 6), 'keep': False,
                'role': f'drop (ε-small for d ≥ {math.ceil(3*xi3)})'})
    trunc_err = sum(s['weight'] for s in sec if not s['keep'])
    kept_secs = [s for s in sec if s['keep']]
    n_cg_coeffs = sum(s['dim_Sn'] * s['dim_Sn'] for s in kept_secs)
    return {'sectors': sec, 'truncation_error': round(min(trunc_err, 1.0), 6),
            'n_sectors_kept': len(kept_secs), 'n_cg_coefficients': int(n_cg_coeffs),
            'depth': d, 'xi_defining': round(xi1, 3)}

# ──────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — SUBCIRCUIT CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────
def build_subcircuit_layout(layout: list, node_set: set, name: str, gate_class: dict) -> tuple:
    sorted_nodes = sorted(node_set)
    remap = {q: i for i, q in enumerate(sorted_nodes)}
    sub_layout = []
    for layer_idx, layer in enumerate(layout):
        sub_layer = []
        for gate_idx, gate in enumerate(layer):
            qs = gate['qubits']
            entry = {**gate, 'source_layer': layer_idx, 'source_gate_index': gate_idx}
            if all(q in node_set for q in qs):
                entry['qubits'] = [remap[q] for q in qs]
                entry['source'] = 'local'
                sub_layer.append(entry)
            elif any(q in node_set for q in qs):
                # Cross-partition gates are omitted from the local subcircuit.
                # They are handled only as cut-gate statistics / overhead terms.
                continue
        if sub_layer:
            sub_layout.append(sub_layer)
    return sub_layout, sorted_nodes, remap

def compile_subcircuit(sub_layout: list, n_sub: int):
    clean = []
    valid_gates = {'I','X','Y','Z','H','S','T','RX','RY','RZ','CNOT','CY','CZ','CH','SWAP','ISWAP',
                   'CRX','CRY','CRZ','CU1','CU3','CPHASE','U1','U2','U3','PHASE','SX','R',
                   'CCX','CSWAP','C3X','C3Z'}
    for layer in sub_layout:
        cl = [{k:v for k,v in g.items() if k in ('gate','qubits','params')}
              for g in layer if g['gate'].upper() in valid_gates]
        if cl: clean.append(cl)
    try:
        return compile_quantum_circuit(clean, num_qubits=n_sub)
    except Exception as e:
        print(f"         [warn] subcircuit compile issue: {e}")
        from qiskit import QuantumCircuit
        return QuantumCircuit(n_sub), []

# ──────────────────────────────────────────────────────────────────────────────
#  CLASSICAL AGGREGATION MAP
# ──────────────────────────────────────────────────────────────────────────────
def build_cg_aggregation(schur: dict, qpd_info: dict, sym_report: dict) -> dict:
    n, nA, nB = N_QUBITS, N_QUBITS // 2, N_QUBITS - N_QUBITS // 2
    has_u1 = sym_report['u1_conserved_charge']
    sectors = []
    if has_u1:
        for k in range(n+1):
            dim_full = math.comb(n, k)
            sub_pairs = [(kA, k-kA) for kA in range(min(nA,k)+1) if 0 <= k-kA <= nB]
            wk = 1.0 / max(dim_full, 1)
            sectors.append({'charge_k': k, 'dim_full': dim_full, 'sub_pairs': sub_pairs, 'weight': round(wk, 8)})
    else:
        for s in schur['sectors']:
            if s['keep']:
                sectors.append({'irrep': s['label'], 'dim_Sn': s['dim_Sn'], 'weight': s['weight'], 'cg_terms': s['dim_Sn']**2})
    return {
        'reconstruction_type': 'U(1) charge-sector CG' if has_u1 else 'Schur-Weyl irrep CG',
        'n_A': nA, 'n_B': nB, 'n_sectors': min(len(sectors), 20), 'sectors_shown': sectors[:6],
        'total_cg_coefficients': int(schur['n_cg_coefficients']),
        'aggregation_formula': "p̃(x) = Σ_k  Σ_{kA+kB=k}  w_k · p^A_{kA}(xA) · p^B_{kB}(xB)" if has_u1 else
                               "p̃(x) = Σ_λ  w_λ · Σ_{xA,xB→x}  p^A_λ(xA)·p^B_λ(xB)·C_λ(xA,xB)",
        'error_bound': round(schur['truncation_error'], 6),
    }

# ──────────────────────────────────────────────────────────────────────────────
#  SAVING UTILITIES & PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def _gate_type_code(gate_name: str) -> int:
    n = gate_name.upper()
    if n in SINGLE: return 1
    if n in TWO: return 2
    if n in MULTI: return 3
    return 0

GATE_CMAP = ListedColormap(['#f5f5f5', '#74b9ff', '#fd79a8', '#55efc4'])
GATE_NORM = BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5], GATE_CMAP.N)

def save_circuit_heatmap(layout, n_qubits, filepath, title='Full RQC'):
    fig, ax = plt.subplots(figsize=(max(len(layout)*0.8, 14), max(n_qubits*0.18, 14)))
    mat = np.zeros((n_qubits, len(layout)))
    for li, layer in enumerate(layout):
        for gate in layer:
            code = _gate_type_code(gate['gate'])
            for q in gate['qubits']:
                if mat[q, li] == 0: mat[q, li] = code
    im = ax.imshow(mat, aspect='auto', cmap=GATE_CMAP, norm=GATE_NORM, interpolation='nearest')
    ax.set_xlabel('Layer'); ax.set_ylabel('Qubit'); ax.set_title(title, fontweight='bold')
    ax.set_xticks(range(len(layout))); ax.set_xticklabels([str(i) for i in range(len(layout))], fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    fig.colorbar(im, ax=ax, ticks=[0,1,2,3], fraction=0.02, pad=0.02).ax.set_yticklabels(['idle','1q','2q','multi'])
    plt.tight_layout(); fig.savefig(filepath, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"         Saved: {filepath.name}")

def save_circuit_diagram_qiskit(qc, filepath, title=''):
    try:
        bound = qc.assign_parameters({p: float(i) * 0.31 for i, p in enumerate(qc.parameters)}, inplace=False) if qc.parameters else qc
        style = {'name':'clifford','displaycolor':{'h':('#4472C4','#FFF'),'cx':('#E67E22','#FFF'),'ccx':('#8E44AD','#FFF'),
                 'swap':('#27AE60','#FFF'),'rz':('#C0392B','#FFF'),'rx':('#2980B9','#FFF'),'ry':('#16A085','#FFF'),
                 'x':('#E74C3C','#FFF'),'z':('#2C3E50','#FFF')},'backgroundcolor':'#FAFAFA','linecolor':'#2C3E50',
                 'textcolor':'#2C3E50','gatetextcolor':'#FFF','subfontsize':6,'fontsize':10}
        fold_val = max(20, min(60, bound.size() // max(bound.num_qubits, 1) + 5))
        fig = bound.draw(output='mpl', fold=fold_val, style=style, plot_barriers=False, idle_wires=False)
        if title: fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#FAFAFA'); plt.close(fig)
        print(f"         Saved: {filepath.name}")
    except Exception as e:
        print(f"         [warn] Qiskit MPL draw failed ({e}), using fallback")
        _save_circuit_diagram_fallback(qc, filepath, title)

def _save_circuit_diagram_fallback(qc, filepath, title=''):
    ops = [(inst.operation.name.upper(), [qc.find_bit(q).index for q in inst.qubits]) for inst in qc.data]
    n_q = qc.num_qubits
    col_end = [0]*n_q; placed = []
    for gn, qs in ops:
        col = max(col_end[q] for q in qs)
        placed.append((col, gn, qs))
        for q in qs: col_end[q] = col + 1
    n_cols = max(col_end) if col_end else 1
    COL_W, ROW_H, PAD_L, PAD_R, PAD_T, PAD_B = 0.80, 0.55, 1.20, 0.40, 0.40, 0.40
    fig_w = PAD_L + n_cols*COL_W + PAD_R; fig_h = PAD_T + n_q*ROW_H + PAD_B
    fig, ax = plt.subplots(figsize=(max(fig_w,8), max(fig_h,4)))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')
    ax.set_facecolor('#FAFAFA'); fig.patch.set_facecolor('#FAFAFA')
    def qy(q): return fig_h - PAD_T - (q+0.5)*ROW_H
    def cx(c): return PAD_L + (c+0.5)*COL_W
    wire_end = PAD_L + n_cols*COL_W + 0.1
    for q in range(n_q):
        y = qy(q)
        ax.plot([PAD_L-0.9, wire_end], [y,y], color='#2C3E50', lw=1.0, zorder=1)
        ax.text(PAD_L-1.0, y, f'q{q}', ha='right', va='center', fontsize=8, color='#2C3E50', fontfamily='monospace')
    GATE_COLORS = {'H':'#4472C4','X':'#E74C3C','Y':'#E67E22','Z':'#2C3E50','S':'#8E44AD','T':'#16A085',
                   'RX':'#2980B9','RY':'#16A085','RZ':'#C0392B','CNOT':'#E67E22','CX':'#E67E22','CZ':'#27AE60',
                   'SWAP':'#27AE60','CCX':'#8E44AD','I':'#BDC3C7'}
    BOX_H, BOX_W = ROW_H*0.62, COL_W*0.72
    for col, gn, qs in placed:
        x, color = cx(col), GATE_COLORS.get(gn, '#999')
        if len(qs)==1:
            y=qy(qs[0]); rect=mpatches.FancyBboxPatch((x-BOX_W/2,y-BOX_H/2),BOX_W,BOX_H,boxstyle='round,pad=0.02',facecolor=color,edgecolor='white',lw=1.0,zorder=3)
            ax.add_patch(rect); ax.text(x,y,gn[:3],ha='center',va='center',fontsize=6.5,color='white',fontweight='bold',zorder=4)
        elif len(qs)==2:
            yc,yt=qy(qs[0]),qy(qs[1]); ax.plot([x,x],[min(yc,yt),max(yc,yt)],color=color,lw=2.0,zorder=2)
            ax.scatter([x],[yc],s=60,color=color,zorder=4,edgecolors='white',lw=0.5)
            if gn in ('CNOT','CX'):
                r=BOX_H*0.42; ax.add_patch(plt.Circle((x,yt),r,color=color,fill=False,lw=2.0,zorder=3))
                ax.plot([x-r,x+r],[yt,yt],color=color,lw=2.0,zorder=4); ax.plot([x,x],[yt-r,yt+r],color=color,lw=2.0,zorder=4)
            elif gn=='SWAP':
                r=BOX_H*0.28; ax.plot([x-r,x+r],[yt-r,yt+r],color=color,lw=2.0,zorder=4); ax.plot([x-r,x+r],[yt+r,yt-r],color=color,lw=2.0,zorder=4)
                ax.plot([x-r,x+r],[yc-r,yc+r],color=color,lw=2.0,zorder=4); ax.plot([x-r,x+r],[yc+r,yc-r],color=color,lw=2.0,zorder=4)
            else:
                rect=mpatches.FancyBboxPatch((x-BOX_W/2,yt-BOX_H/2),BOX_W,BOX_H,boxstyle='round,pad=0.02',facecolor=color,edgecolor='white',lw=1.0,zorder=3)
                ax.add_patch(rect); ax.text(x,yt,gn[1:4] if gn.startswith('C') else gn[:3],ha='center',va='center',fontsize=6,color='white',fontweight='bold',zorder=4)
        else:
            ys=[qy(q) for q in qs]; y_lo,y_hi=min(ys),max(ys)
            rect=mpatches.FancyBboxPatch((x-BOX_W/2,y_lo-BOX_H/2),BOX_W,(y_hi-y_lo)+BOX_H,boxstyle='round,pad=0.02',facecolor=color,edgecolor='white',lw=1.0,zorder=3,alpha=0.85)
            ax.add_patch(rect); ax.text(x,(y_lo+y_hi)/2,gn[:4],ha='center',va='center',fontsize=6,color='white',fontweight='bold',zorder=4)
    if title: ax.set_title(title, fontsize=10, fontweight='bold', pad=6, color='#2C3E50')
    plt.tight_layout(pad=0.3); fig.savefig(filepath, dpi=160, bbox_inches='tight', facecolor='#FAFAFA'); plt.close(fig)
    print(f"         Saved: {filepath.name}")

def save_circuit_heatmap_partition(layout, n_sub, sorted_nodes, filepath, title):
    from qiskit import QuantumCircuit as _QC
    gate_map = {'I':lambda qc,qs,p:qc.id(qs[0]),'X':lambda qc,qs,p:qc.x(qs[0]),'Y':lambda qc,qs,p:qc.y(qs[0]),
                'Z':lambda qc,qs,p:qc.z(qs[0]),'H':lambda qc,qs,p:qc.h(qs[0]),'S':lambda qc,qs,p:qc.s(qs[0]),
                'T':lambda qc,qs,p:qc.t(qs[0]),'RX':lambda qc,qs,p:qc.rx(p[0] if p else 0.0,qs[0]),
                'RY':lambda qc,qs,p:qc.ry(p[0] if p else 0.0,qs[0]),'RZ':lambda qc,qs,p:qc.rz(p[0] if p else 0.0,qs[0]),
                'CNOT':lambda qc,qs,p:qc.cx(*qs),'CX':lambda qc,qs,p:qc.cx(*qs),'CZ':lambda qc,qs,p:qc.cz(*qs),
                'SWAP':lambda qc,qs,p:qc.swap(*qs),'CCX':lambda qc,qs,p:qc.ccx(*qs)}
    qc_vis = _QC(n_sub)
    for layer in layout:
        for gate in layer:
            fn = gate_map.get(gate['gate'].upper())
            if fn:
                try: fn(qc_vis, gate['qubits'], [float(x) for x in gate.get('params',[])])
                except: pass
    _save_circuit_diagram_fallback(qc_vis, filepath, title)

def save_interaction_graph_dot(G, A_set, B_set, cut_stats, filepath):
    lines = ['strict graph interaction_graph {',
             '    graph [layout=sfdp, overlap=scale, splines=true, fontname="Helvetica", bgcolor="white"]',
             '    node [shape=circle, fontname="Helvetica", fontsize=9, style=filled, fixedsize=true, width=0.30]',
             '    edge [fontname="Helvetica", fontsize=7]', '']
    for q in sorted(A_set): lines.append(f'    {q} [fillcolor="#4472C4", fontcolor="white", label="q{q}"]')
    lines.append('')
    for q in sorted(B_set): lines.append(f'    {q} [fillcolor="#ED7D31", fontcolor="white", label="q{q}"]')
    lines.append('')
    for u,v,d in cut_stats['inner_A']:
        w=d.get('weight',1); label=','.join(set(d.get('gates',[])))[:20]
        lines.append(f'    {u} -- {v} [color="#4472C4", penwidth={1+w*0.4:.1f}, label="{label}"]')
    for u,v,d in cut_stats['inner_B']:
        w=d.get('weight',1); label=','.join(set(d.get('gates',[])))[:20]
        lines.append(f'    {u} -- {v} [color="#ED7D31", penwidth={1+w*0.4:.1f}, label="{label}"]')
    for u,v,d in cut_stats['cut_edges']:
        w=d.get('weight',1); label=','.join(set(d.get('gates',[])))[:20]
        lines.append(f'    {u} -- {v} [color="#C0392B", style=dashed, penwidth={1+w*0.5:.1f}, label="{label}"]')
    lines.append('}')
    filepath.write_text('\n'.join(lines)); print(f"         Saved: {filepath.name}")

def save_interaction_graph_png(G, A_set, B_set, cut_stats, filepath):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
    ax = axes[0]
    A_sorted, B_sorted = sorted(A_set), sorted(B_set)
    pos = {q:(0.0, i/max(len(A_sorted)-1,1)) for i,q in enumerate(A_sorted)}
    pos.update({q:(1.0, i/max(len(B_sorted)-1,1)) for i,q in enumerate(B_sorted)})
    nx.draw_networkx_nodes(G, pos, nodelist=A_sorted, node_color='#4472C4', node_size=500, alpha=0.92, ax=ax)
    nx.draw_networkx_labels(G, pos, {n:f'q{n}' for n in A_sorted}, font_size=8, font_color='white', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=B_sorted, node_color='#ED7D31', node_size=500, alpha=0.92, ax=ax)
    nx.draw_networkx_labels(G, pos, {n:f'q{n}' for n in B_sorted}, font_size=8, font_color='white', ax=ax)
    inner_A = [(u,v) for u,v,_ in cut_stats['inner_A']]
    nx.draw_networkx_edges(G, pos, edgelist=inner_A, edge_color='#4472C4',
                           width=[G[u][v].get('weight',1)*0.5+0.6 for u,v in inner_A], alpha=0.65, ax=ax,
                           arrows=True, connectionstyle='arc3,rad=0.35', arrowstyle='-', arrowsize=1)
    inner_B = [(u,v) for u,v,_ in cut_stats['inner_B']]
    nx.draw_networkx_edges(G, pos, edgelist=inner_B, edge_color='#ED7D31',
                           width=[G[u][v].get('weight',1)*0.5+0.6 for u,v in inner_B], alpha=0.65, ax=ax,
                           arrows=True, connectionstyle='arc3,rad=-0.35', arrowstyle='-', arrowsize=1)
    cut_list = [(u,v) for u,v,_ in cut_stats['cut_edges']]
    nx.draw_networkx_edges(G, pos, edgelist=cut_list, edge_color='#C0392B',
                           width=[G[u][v].get('weight',1)*0.6+0.8 for u,v in cut_list], style='dashed', alpha=0.85, ax=ax)
    ax.axvline(x=0.5, color='#BDC3C7', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(0.0,-0.08,'Partition A',ha='center',fontsize=11,fontweight='bold',color='#4472C4',transform=ax.transData)
    ax.text(1.0,-0.08,'Partition B',ha='center',fontsize=11,fontweight='bold',color='#ED7D31',transform=ax.transData)
    ax.legend(handles=[mpatches.Patch(color='#4472C4',label=f'A ({len(A_set)}q)'),
                       mpatches.Patch(color='#ED7D31',label=f'B ({len(B_set)}q)'),
                       mpatches.Patch(color='#C0392B',label=f'Cut ({len(cut_list)})')], fontsize=10, loc='upper left')
    ax.set_title('Qubit Interaction Graph — Bipartite Layout (KL Bisection)', fontsize=13, fontweight='bold'); ax.axis('off')
    ax2 = axes[1]
    cut_w = [G[u][v].get('weight',1) for u,v,_ in cut_stats['cut_edges']]
    inner_w = [G[u][v].get('weight',1) for u,v,_ in cut_stats['inner_A']] + [G[u][v].get('weight',1) for u,v,_ in cut_stats['inner_B']]
    bins = np.arange(0.5, max(max(cut_w+inner_w, default=1)+2, 3), 1)
    ax2.hist(inner_w, bins=bins, color='#74b9ff', alpha=0.75, label='Inner', edgecolor='white')
    ax2.hist(cut_w, bins=bins, color='#C0392B', alpha=0.75, label='Cut', edgecolor='white')
    ax2.set_xlabel('Edge weight'); ax2.set_ylabel('Count'); ax2.set_title('Edge weight distribution'); ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Bipartite Interaction Graph — Cut: {cut_stats["n_cut_edges"]} edges ({cut_stats["cut_fraction"]:.1%} weight)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); fig.savefig(filepath, dpi=130, bbox_inches='tight'); plt.close(fig); print(f"         Saved: {filepath.name}")

def save_cut_analysis(gate_class, qpd_info, filepath):
    cross = gate_class['cross']; counts = Counter(g['gate'].upper() for g in cross)
    gates, cnts = list(counts.keys()), list(counts.values())
    gammas, gsyms = [QPD_GAMMA.get(g,9.0) for g in gates], [QPD_GAMMA_SYM.get(g,9.0) for g in gates]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax=axes[0]; bars=ax.bar(range(len(gates)),cnts,color='#E67E22',edgecolor='white',linewidth=0.5)
    ax.set_xticks(range(len(gates))); ax.set_xticklabels(gates,rotation=45,ha='right',fontsize=9)
    ax.set_ylabel('Count'); ax.set_title('Cross-partition gates by type')
    for b,c in zip(bars,cnts): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,str(c),ha='center',va='bottom',fontsize=8)
    ax=axes[1]; x=np.arange(len(gates)); w=0.35
    ax.bar(x-w/2,gammas,w,label='Generic γ',color='#C0392B',alpha=0.85); ax.bar(x+w/2,gsyms,w,label='Sym-reduced γ',color='#27AE60',alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(gates,rotation=45,ha='right',fontsize=9); ax.set_ylabel('QPD overhead γ'); ax.set_title('QPD overhead per gate type'); ax.legend(fontsize=9); ax.set_yscale('log')
    ax=axes[2]; all_gg=[QPD_GAMMA.get(g['gate'].upper(),9.0) for g in cross]; all_gs=[QPD_GAMMA_SYM.get(g['gate'].upper(),9.0) for g in cross]
    cg=np.cumprod(all_gg[:30]) if all_gg else [1.0]; cs=np.cumprod(all_gs[:30]) if all_gs else [1.0]
    idx=np.arange(1,len(cg)+1); ax.semilogy(idx,cg,'o-',color='#C0392B',label='Generic Π γ',lw=1.5,ms=4); ax.semilogy(idx,cs,'s-',color='#27AE60',label='Sym Π γ',lw=1.5,ms=4)
    ax.set_xlabel('# cut gates'); ax.set_ylabel('Cumulative overhead'); ax.set_title('Sampling overhead growth'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('Phase 3 — QPD Gate-Cutting Analysis', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); fig.savefig(filepath, dpi=130, bbox_inches='tight'); plt.close(fig); print(f"         Saved: {filepath.name}")

def save_qpd_terms_table(qpd_info, filepath):
    seen = {}
    for g in qpd_info['per_gate']:
        if g['gate'] not in seen and g['gate'] in QPD_DECOMP: seen[g['gate']] = g
    if not seen: return
    fig, axes = plt.subplots(1, len(seen), figsize=(5*len(seen), 4))
    if len(seen)==1: axes=[axes]
    for ax, (gn, gi) in zip(axes, seen.items()):
        terms = QPD_DECOMP.get(gn, [])
        tbl_data = [['Coeff','Side A','Side B']] + [[str(round(t[0],3)),t[1],t[2]] for t in terms]
        ax.axis('off'); tbl=ax.table(cellText=tbl_data[1:],colLabels=tbl_data[0],loc='center',cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1,1.6)
        for j in range(3): tbl[(0,j)].set_facecolor('#2C3E50'); tbl[(0,j)].set_text_props(color='white',fontweight='bold')
        for i in range(1,len(tbl_data)):
            for j in range(3): tbl[(i,j)].set_facecolor('#ECF0F1' if i%2==0 else 'white')
        ax.set_title(f'{gn} (γ={gi["gamma_generic"]} → γ_sym={gi["gamma_sym"]})', fontsize=11, pad=12)
    fig.suptitle('Phase 3 — QPD Quasi-Probability Decompositions', fontsize=13, fontweight='bold')
    plt.tight_layout(); fig.savefig(filepath, dpi=130, bbox_inches='tight'); plt.close(fig); print(f"         Saved: {filepath.name}")

def save_schur_weyl_plot(schur, filepath):
    secs=schur['sectors']; labels=[s['label'] for s in secs]; weights=[s['weight'] for s in secs]; keeps=[s['keep'] for s in secs]
    colors=['#2ECC71' if k else '#E74C3C' for k in keeps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax=axes[0]; bars=ax.bar(range(len(labels)),weights,color=colors,edgecolor='white',linewidth=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=20,ha='right',fontsize=10)
    ax.set_ylabel('Estimated sector weight'); ax.set_title(f'Schur-Weyl irrep sector weights (depth={DEPTH})')
    for b,w in zip(bars,weights): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.002,f'{w:.4f}',ha='center',va='bottom',fontsize=8)
    ax.legend(handles=[mpatches.Patch(color='#2ECC71',label='Kept'),mpatches.Patch(color='#E74C3C',label='Truncated')],fontsize=9)
    ax2=axes[1]; depths=np.arange(1,3*DEPTH+1); errs=[]
    for d in depths:
        err=sum(s['dim_Sn']*math.exp(-d/max(irrep_xi(len(s['label'].split('-'))-1,int(d)),0.1)) for s in secs if not s['keep'])
        errs.append(min(err,1.0))
    ax2.semilogy(depths,errs,'o-',color='#8E44AD',lw=2,ms=4); ax2.axvline(DEPTH,color='#E74C3C',ls='--',lw=1.5,label=f'Depth={DEPTH}')
    ax2.axhline(0.05,color='gray',ls=':',lw=1.0,label='ε=0.05'); ax2.set_xlabel('Depth'); ax2.set_ylabel('Truncation error ε')
    ax2.set_title('Error decay vs depth'); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.suptitle('Phase 2 — Schur-Weyl Decomposition', fontsize=13, fontweight='bold')
    plt.tight_layout(); fig.savefig(filepath, dpi=130, bbox_inches='tight'); plt.close(fig); print(f"         Saved: {filepath.name}")

def save_cg_aggregation_schematic(cg_info, filepath):
    fig, ax = plt.subplots(figsize=(12, 7)); ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis('off')
    ax.add_patch(mpatches.FancyBboxPatch((0.3,4.5),2.8,2.8,boxstyle='round,pad=0.1',facecolor='#D6EAF8',edgecolor='#2980B9',lw=2))
    ax.text(1.7,6.5,'QPU A',ha='center',fontsize=14,fontweight='bold',color='#1A5276')
    ax.text(1.7,5.9,f'qubits 0–{N_QUBITS//2-1}',ha='center',fontsize=9,color='#1A5276')
    ax.text(1.7,5.4,'Subcircuit A + QPD stubs',ha='center',fontsize=9,color='#1A5276')
    ax.add_patch(mpatches.FancyBboxPatch((6.9,4.5),2.8,2.8,boxstyle='round,pad=0.1',facecolor='#FDEBD0',edgecolor='#E67E22',lw=2))
    ax.text(8.3,6.5,'QPU B',ha='center',fontsize=14,fontweight='bold',color='#784212')
    ax.text(8.3,5.9,f'qubits {N_QUBITS//2}–{N_QUBITS-1}',ha='center',fontsize=9,color='#784212')
    ax.text(8.3,5.4,'Subcircuit B + QPD stubs',ha='center',fontsize=9,color='#784212')
    ax.annotate('',xy=(6.8,6.0),xytext=(3.2,6.0),arrowprops=dict(arrowstyle='<->',color='#555',lw=1.5,ls='dashed'))
    ax.text(5.0,6.2,'shared classical seed',ha='center',fontsize=8,color='#555')
    ax.annotate('',xy=(2.8,3.5),xytext=(1.7,4.4),arrowprops=dict(arrowstyle='->',color='#2980B9',lw=2))
    ax.annotate('',xy=(7.2,3.5),xytext=(8.3,4.4),arrowprops=dict(arrowstyle='->',color='#E67E22',lw=2))
    ax.add_patch(mpatches.FancyBboxPatch((2.8,1.5),4.4,2.2,boxstyle='round,pad=0.1',facecolor='#D5F5E3',edgecolor='#1E8449',lw=2))
    ax.text(5.0,3.3,'Classical CG Aggregation',ha='center',fontsize=12,fontweight='bold',color='#145A32')
    ax.text(5.0,2.85,cg_info['aggregation_formula'],ha='center',fontsize=8,color='#145A32',style='italic')
    ax.text(5.0,2.4,f"type: {cg_info['reconstruction_type']}",ha='center',fontsize=8,color='#145A32')
    ax.text(5.0,2.0,f"CG coeffs: {cg_info['total_cg_coefficients']}  ε ≤ {cg_info['error_bound']}",ha='center',fontsize=8,color='#555')
    ax.annotate('',xy=(5.0,1.1),xytext=(5.0,1.4),arrowprops=dict(arrowstyle='->',color='#1E8449',lw=2))
    ax.add_patch(mpatches.FancyBboxPatch((3.3,0.4),3.4,0.65,boxstyle='round,pad=0.1',facecolor='#F8F9FA',edgecolor='#555',lw=1.5))
    ax.text(5.0,0.72,'Approximate full distribution p̃(x)',ha='center',fontsize=10,fontweight='bold')
    ax.set_title('Phase 4 — Classical CG Aggregation Map', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout(); fig.savefig(filepath, dpi=130, bbox_inches='tight'); plt.close(fig); print(f"         Saved: {filepath.name}")

def export_qasm(qc, filepath):
    import qiskit.qasm2 as qasm2
    try:
        bound = qc.assign_parameters({p: float(i) * 0.31 for i, p in enumerate(qc.parameters)}, inplace=False) if qc.parameters else qc
        filepath.write_text(qasm2.dumps(bound))
    except Exception as e:
        filepath.write_text(f"// QASM export failed: {e}\n")
    print(f"         Saved: {filepath.name}")

# ──────────────────────────────────────────────────────────────────────────────
#  ENTANGLEMENT ENTROPY HELPER
# ──────────────────────────────────────────────────────────────────────────────
def _entanglement_entropy(statevector: np.ndarray, A_qubits: list, n_qubits: int) -> float:
    n  = n_qubits
    nA = len(A_qubits)
    nB = n - nA
    B_qubits = [q for q in range(n) if q not in A_qubits]
    order    = A_qubits + B_qubits
    sv_tensor    = statevector.reshape([2] * n)
    sv_reordered = np.transpose(sv_tensor, order).reshape(2**nA, 2**nB)
    s = np.linalg.svd(sv_reordered, compute_uv=False)
    probs = s**2
    probs = probs[probs > 1e-14]
    entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))
    return round(max(entropy, 0.0), 6)

# ──────────────────────────────────────────────────────────────────────────────
#  PARTITIONING ALGORITHMS (BASELINES)
# ──────────────────────────────────────────────────────────────────────────────
def _naive_bisection(n_qubits): return set(range(n_qubits//2)), set(range(n_qubits//2, n_qubits))

def _spectral_bisection(G, n_qubits):
    if G.number_of_nodes()<2: return _naive_bisection(n_qubits)
    try:
        L=nx.laplacian_matrix(G,weight='weight').toarray().astype(float)
        eigvals,eigvecs=np.linalg.eigh(L); fiedler=eigvecs[:,1]; nodes=list(G.nodes())
        A={nodes[i] for i,v in enumerate(fiedler) if v>=0}; B=set(G.nodes())-A
        while abs(len(A)-len(B))>2:
            if len(A)>len(B): n=next(iter(A)); A.remove(n); B.add(n)
            else: n=next(iter(B)); B.remove(n); A.add(n)
        iso=[q for q in range(n_qubits) if q not in G.nodes()]
        for i,q in enumerate(iso): (A if i%2==0 else B).add(q)
        return A, B
    except: return _naive_bisection(n_qubits)

def _louvain_bisection(G, n_qubits):
    if G.number_of_nodes()<2: return _naive_bisection(n_qubits)
    comms=None
    try: comms=list(nx.algorithms.community.louvain_communities(G,weight='weight',seed=SEED))
    except AttributeError: pass
    if comms is None:
        try:
            import community as cl; part=cl.best_partition(G,weight='weight',random_state=SEED)
            cm={}; [cm.setdefault(c,set()).add(n) for n,c in part.items()]; comms=list(cm.values())
        except ImportError: pass
    if comms is None: return _spectral_bisection(G, n_qubits)
    comms=sorted(comms,key=len,reverse=True); A=set(comms[0]); B=set()
    for c in comms[1:]: (A if len(A)<=len(B) else B).update(c)
    while abs(len(A)-len(B))>2:
        if len(A)>len(B): n=next(iter(A)); A.remove(n); B.add(n)
        else: n=next(iter(B)); B.remove(n); A.add(n)
    iso=[q for q in range(n_qubits) if q not in A and q not in B]
    for i,q in enumerate(iso): (A if i%2==0 else B).add(q)
    return A, B

def _girvan_newman_bisection(G, n_qubits):
    if G.number_of_nodes()<2: return _naive_bisection(n_qubits)
    if G.number_of_nodes()>200 or G.number_of_edges()>1000: return _spectral_bisection(G, n_qubits)
    try:
        H=G.copy()
        for u,v,d in H.edges(data=True): H[u][v]['distance']=1.0/max(d.get('weight',1),1e-9)
        comps=next(nx.algorithms.community.girvan_newman(H))
        A=set(comps[0]); B=set(comps[1]) if len(comps)>1 else set()
        for c in comps[2:]: (A if len(A)<=len(B) else B).update(c)
        while abs(len(A)-len(B))>2:
            if len(A)>len(B): n=next(iter(A)); A.remove(n); B.add(n)
            else: n=next(iter(B)); B.remove(n); A.add(n)
        iso=[q for q in range(n_qubits) if q not in A and q not in B]
        for i,q in enumerate(iso): (A if i%2==0 else B).add(q)
        return A, B
    except: return _spectral_bisection(G, n_qubits)

def _metis_bisection(G, n_qubits):
    try: import pymetis
    except ImportError: return {'available':False,'note':'pymetis not installed'}
    if G.number_of_nodes()<2: return _naive_bisection(n_qubits)
    try:
        nodes=sorted(G.nodes()); idx={v:i for i,v in enumerate(nodes)}
        adj,ew=[],[]
        for v in nodes:
            nbrs=sorted(G.neighbors(v)); adj.append([idx[u] for u in nbrs])
            ew.append([max(1,round(G[v][u].get('weight',1))) for u in nbrs])
        _,mem=pymetis.part_graph(2,adjacency=adj,eweights=ew)
        A={nodes[i] for i,p in enumerate(mem) if p==0}; B={nodes[i] for i,p in enumerate(mem) if p==1}
        iso=[q for q in range(n_qubits) if q not in G.nodes()]
        for i,q in enumerate(iso): (A if i%2==0 else B).add(q)
        while abs(len(A)-len(B))>2:
            if len(A)>len(B): n=next(iter(A)); A.remove(n); B.add(n)
            else: n=next(iter(B)); B.remove(n); A.add(n)
        return A, B
    except: return _spectral_bisection(G, n_qubits)

def _qdislib_dag_partition(layout, n_qubits):
    ec,ed=defaultdict(int),defaultdict(list)
    for li,layer in enumerate(layout):
        for g in layer:
            qs=g['qubits']
            if len(qs)>=2:
                for i in range(len(qs)):
                    for j in range(i+1,len(qs)):
                        u,v=min(qs[i],qs[j]),max(qs[i],qs[j]); ec[(u,v)]+=1; ed[(u,v)].append(li)
    Gd=nx.Graph(); Gd.add_nodes_from(range(n_qubits))
    for (u,v),cnt in ec.items():
        ds=ed[(u,v)]; span=(max(ds)-min(ds)+1) if ds else 1
        Gd.add_edge(u,v,weight=cnt+0.5*span)
    cent={q:sum(Gd[q][nb].get('weight',1) for nb in Gd.neighbors(q)) for q in range(n_qubits)}
    seed=min(range(n_qubits),key=lambda q:cent[q]); A={seed}; frontier=[]; import heapq
    for nb in Gd.neighbors(seed): heapq.heappush(frontier,(-Gd[seed][nb].get('weight',1),nb))
    vis={seed}
    while len(A)<n_qubits//2 and frontier:
        nw,node=heapq.heappop(frontier)
        if node in vis: continue
        vis.add(node); A.add(node)
        for nb in Gd.neighbors(node):
            if nb not in vis: heapq.heappush(frontier,(-Gd[node][nb].get('weight',1),nb))
    B=set(range(n_qubits))-A
    while abs(len(A)-len(B))>2:
        if len(A)>len(B): n=next(iter(A)); A.remove(n); B.add(n)
        else: n=next(iter(B)); B.remove(n); A.add(n)
    return A, B

def _try_qdislib(layout, n_qubits):
    try:
        import qdislib as _q
        gl=[(g['gate'].upper(),g['qubits']) for l in layout for g in l]
        res=_q.partition_circuit(gl,n_qubits=n_qubits,n_partitions=2) if hasattr(_q,'partition_circuit') else _q.CircuitPartitioner(n_partitions=2).partition(gl,n_qubits=n_qubits)
        if hasattr(res,'partitions') and len(res.partitions)>=2:
            A,B=set(res.partitions[0]),set(res.partitions[1])
            iso=[q for q in range(n_qubits) if q not in A and q not in B]
            for i,q in enumerate(iso): (A if i%2==0 else B).add(q)
            return {'available':True,'A_set':sorted(A),'B_set':sorted(B),'n_A':len(A),'n_B':len(B),'note':'qdislib package'}
    except ImportError: pass
    except Exception as e: return {'available':True,'error':str(e)}
    try:
        A,B=_qdislib_dag_partition(layout,n_qubits)
        return {'available':True,'A_set':sorted(A),'B_set':sorted(B),'n_A':len(A),'n_B':len(B),'note':'built-in DAG fallback'}
    except Exception as e: return {'available':True,'error':str(e)}

# ──────────────────────────────────────────────────────────────────────────────
#  SIMULATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _deterministic_param_values(gate: dict, n_params: int, layer_idx: int = 0, gate_idx: int = 0) -> list[float]:
    """Create stable numeric parameters from gate identity so full/sub circuits
    receive the same angles whenever the logical gate is the same."""
    if n_params <= 0:
        return []
    raw_params = gate.get('params', None)
    if isinstance(raw_params, (list, tuple)) and len(raw_params) >= n_params:
        try:
            vals = [float(raw_params[i]) for i in range(n_params)]
            return vals
        except Exception:
            pass

    qubits = gate.get('qubits', [])
    qubit_sig = ','.join(str(q) for q in qubits)
    gate_name = str(gate.get('gate', 'UNKNOWN')).upper()
    src_layer = gate.get('source_layer', layer_idx)
    src_gate_index = gate.get('source_gate_index', gate_idx)
    base = f"{SEED}|{gate_name}|{src_layer}|{src_gate_index}|{qubit_sig}"

    vals = []
    for p_idx in range(n_params):
        digest = hashlib.sha256(f"{base}|{p_idx}".encode('utf-8')).digest()
        raw = int.from_bytes(digest[:8], 'big', signed=False) / 2**64
        angle = (raw * 2.0 * np.pi) - np.pi
        vals.append(float(angle))
    return vals


def _append_gate_to_qc(qc, gate: dict):
    """Append a supported gate to a QuantumCircuit using deterministic numeric parameters."""
    name = str(gate.get('gate', '')).upper()
    qs = list(gate.get('qubits', []))
    if not qs:
        return

    layer_idx = int(gate.get('source_layer', gate.get('layer', 0)) or 0)
    gate_idx = int(gate.get('source_gate_index', gate.get('gate_index', 0)) or 0)

    def pvals(n):
        return _deterministic_param_values(gate, n, layer_idx=layer_idx, gate_idx=gate_idx)

    try:
        if name in {'I', 'ID', 'IDENTITY'}:
            qc.id(qs[0])
        elif name == 'X':
            qc.x(qs[0])
        elif name == 'Y':
            qc.y(qs[0])
        elif name == 'Z':
            qc.z(qs[0])
        elif name == 'H':
            qc.h(qs[0])
        elif name == 'S':
            qc.s(qs[0])
        elif name == 'SDG':
            qc.sdg(qs[0])
        elif name == 'T':
            qc.t(qs[0])
        elif name == 'TDG':
            qc.tdg(qs[0])
        elif name == 'SX':
            qc.sx(qs[0])
        elif name == 'SXDG':
            qc.sxdg(qs[0])
        elif name == 'RX':
            qc.rx(pvals(1)[0], qs[0])
        elif name == 'RY':
            qc.ry(pvals(1)[0], qs[0])
        elif name == 'RZ':
            qc.rz(pvals(1)[0], qs[0])
        elif name in {'PHASE', 'P', 'U1'}:
            qc.p(pvals(1)[0], qs[0])
        elif name == 'U2':
            vals = pvals(2)
            qc.u(np.pi / 2, vals[0], vals[1], qs[0])
        elif name == 'U3':
            vals = pvals(3)
            qc.u(vals[0], vals[1], vals[2], qs[0])
        elif name == 'R':
            vals = pvals(2)
            qc.r(vals[0], vals[1], qs[0])
        elif name in {'CX', 'CNOT'}:
            qc.cx(qs[0], qs[1])
        elif name == 'CY':
            qc.cy(qs[0], qs[1])
        elif name == 'CZ':
            qc.cz(qs[0], qs[1])
        elif name == 'CH':
            qc.ch(qs[0], qs[1])
        elif name == 'SWAP':
            qc.swap(qs[0], qs[1])
        elif name == 'ISWAP':
            qc.iswap(qs[0], qs[1])
        elif name == 'CRX':
            qc.crx(pvals(1)[0], qs[0], qs[1])
        elif name == 'CRY':
            qc.cry(pvals(1)[0], qs[0], qs[1])
        elif name == 'CRZ':
            qc.crz(pvals(1)[0], qs[0], qs[1])
        elif name in {'CU1', 'CPHASE'}:
            qc.cp(pvals(1)[0], qs[0], qs[1])
        elif name == 'CU3':
            vals = pvals(3)
            qc.cu(vals[0], vals[1], vals[2], 0.0, qs[0], qs[1])
        elif name == 'CCX':
            qc.ccx(qs[0], qs[1], qs[2])
        elif name == 'CSWAP':
            qc.cswap(qs[0], qs[1], qs[2])
        elif name == 'C3X':
            qc.mcx(qs[:-1], qs[-1])
        elif name == 'C3Z':
            qc.mcp(np.pi, qs[:-1], qs[-1])
        else:
            # Ignore unsupported gates instead of failing the whole simulation path.
            return
    except Exception:
        # Fallback: unsupported instruction in the active qiskit version.
        return


def _materialize_layout_circuit(layout: list, n_qubits: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    for layer_idx, layer in enumerate(layout):
        for gate_idx, gate in enumerate(layer):
            gate = {**gate, 'source_layer': gate.get('source_layer', layer_idx), 'source_gate_index': gate.get('source_gate_index', gate_idx)}
            _append_gate_to_qc(qc, gate)
    return qc


def _get_statevector(qc):
    try:
        from qiskit.quantum_info import Statevector
        return np.array(Statevector(qc).data)
    except: pass
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        sim=AerSimulator(method='statevector'); qsv=qc.copy(); qsv.save_statevector()
        return np.array(sim.run(transpile(qsv,sim)).result().get_statevector())
    except: return None

def _prob_from_statevector(sv):
    p=np.abs(sv)**2; p/=p.sum(); return p

def _build_recon_index(sorted_nodes, n_qubits):
    n_full=2**n_qubits; all_idx=np.arange(n_full,dtype=np.int64); sub_idx=np.zeros(n_full,dtype=np.int64)
    for k,q in enumerate(sorted_nodes): sub_idx+=((all_idx>>q)&1)<<k
    return sub_idx

def _simulate_subcircuit(A_set, B_set, layout, n_qubits, side):
    node_set = A_set if side == 'A' else B_set
    gc = classify_gates(layout, A_set, B_set)
    sub_layout, sorted_nodes, _ = build_subcircuit_layout(layout, node_set, side, gc)
    qc = _materialize_layout_circuit(sub_layout, len(node_set))
    sv = _get_statevector(qc)
    return (_prob_from_statevector(sv), sorted_nodes) if sv is not None else (None, sorted_nodes)

def _compute_distribution_metrics(p_ideal, p_recon):
    eps = 1e-15
    p_i = np.clip(p_ideal, eps, None)
    p_r = np.clip(p_recon, eps, None)
    tvd = float(0.5 * np.sum(np.abs(p_ideal - p_recon)))
    fid = float(np.sum(np.sqrt(np.clip(p_ideal, 0, None) * np.clip(p_recon, 0, None))))
    mask = p_ideal > eps
    kl = float(np.sum(p_ideal[mask] * np.log(p_ideal[mask] / p_r[mask])))
    hell = float(np.sqrt(0.5 * np.sum((np.sqrt(p_ideal) - np.sqrt(p_recon))**2)))
    m = 0.5 * (p_i + p_r)
    js = float(0.5 * (np.sum(p_i * np.log(p_i / m)) + np.sum(p_r * np.log(p_r / m))))
    js = max(js, 0.0)
    ce = float(-np.sum(p_ideal * np.log(p_r)))
    return {'tvd': tvd, 'fidelity': fid, 'kl_divergence': kl,
            'hellinger': hell, 'js_divergence': js, 'cross_entropy': ce}


def _compute_unified_scores(results_dict):
    weights = {'tvd': 0.25, 'fidelity': 0.25, 'kl_divergence': 0.15,
               'hellinger': 0.10, 'js_divergence': 0.15, 'cross_entropy': 0.10}
    direction = {'tvd': False, 'fidelity': True, 'kl_divergence': False,
                 'hellinger': False, 'js_divergence': False, 'cross_entropy': False}
    valid = {k: v for k, v in results_dict.items() if v.get('error') is None and 'tvd' in v}
    if len(valid) < 2:
        for k in valid:
            valid[k]['unified_score'] = 1.0
            valid[k]['rank'] = 1
        return valid

    metrics = list(weights.keys())
    raw = {met: [valid[m][met] for m in valid] for met in metrics}
    for met in metrics:
        vals = np.array(raw[met], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        normed = np.full_like(vals, 0.5) if vmax - vmin < 1e-12 else (vals - vmin) / (vmax - vmin)
        if not direction[met]:
            normed = 1.0 - normed
        for i, m in enumerate(valid):
            valid[m].setdefault('_norm', {})[met] = float(normed[i])

    for m in valid:
        score = sum(weights[met] * valid[m]['_norm'][met] for met in metrics)
        valid[m]['unified_score'] = round(float(score), 6)
        del valid[m]['_norm']

    ranked = sorted(valid.items(), key=lambda kv: kv[1]['unified_score'], reverse=True)
    for rank, (m, data) in enumerate(ranked, start=1):
        data['rank'] = rank
    return valid


def _format_ranked_labels(labels, data_map, use_rank=True):
    out = []
    for lbl in labels:
        rank = data_map.get(lbl, {}).get('rank')
        out.append(f'{rank}. {lbl}' if use_rank and rank is not None else lbl)
    return out

def _rank_partition_methods_for_benchmark(chart):
    """Return labels sorted best-to-worst using a proxy partition score."""
    if not chart:
        return []

    labels = list(chart.keys())
    metrics = ['n_cut_edges', 'n_cross_gates', 'gamma_sym', 'entanglement_entropy']
    raw = {}
    for met in metrics:
        vals = []
        for lbl in labels:
            v = chart[lbl].get(met)
            if v is None:
                v = float('inf') if met == 'entanglement_entropy' else 0.0
            if met == 'gamma_sym':
                v = math.log10(max(float(v), 1e-12))
            vals.append(float(v))
        raw[met] = np.array(vals, dtype=float)

    # Lower is better for all proxy terms.
    normed = {}
    for met, vals in raw.items():
        finite = np.isfinite(vals)
        if not finite.any():
            normed[met] = np.zeros_like(vals)
            continue
        v = vals.copy()
        finite_vals = v[finite]
        vmin, vmax = finite_vals.min(), finite_vals.max()
        if vmax - vmin < 1e-12:
            nvals = np.full_like(v, 0.5)
        else:
            nvals = (v - vmin) / (vmax - vmin)
        nvals[~finite] = 1.0
        normed[met] = nvals

    proxy = (
        0.30 * normed['n_cut_edges'] +
        0.30 * normed['n_cross_gates'] +
        0.25 * normed['gamma_sym'] +
        0.15 * normed['entanglement_entropy']
    )
    order = [labels[i] for i in np.argsort(proxy)]  # best first (lowest proxy)
    return order

# ──────────────────────────────────────────────────────────────────────────────
#  NEW: ADVANCED mosaic METHODS
# ──────────────────────────────────────────────────────────────────────────────

def estimate_edge_entanglement(layout: list) -> dict:
    """Estimate pairwise entanglement pressure for each qubit pair in the layout."""
    weights = defaultdict(float)
    for layer_idx, layer in enumerate(layout):
        depth_factor = 1.0 + 0.05 * layer_idx
        for gate in layer:
            qs = gate.get('qubits', [])
            if len(qs) < 2:
                continue
            name = gate['gate'].upper()
            sr = max(int(SCHMIDT_RANK.get(name, len(qs))), 2)
            gate_weight = depth_factor * math.log(sr)
            if name in PARTICLE_PRESERVING:
                gate_weight *= 0.8
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    u, v = sorted((qs[i], qs[j]))
                    weights[(u, v)] += gate_weight
    return weights


def _partition_cost_surrogate(A_set: set, B_set: set, G: nx.Graph, layout: list, n_qubits: int) -> float:
    cut_weight = 0.0
    cross_gates = 0
    gamma_log = 0.0
    for u, v, d in G.edges(data=True):
        if (u in A_set) != (v in A_set):
            w = float(d.get('weight', 1.0))
            cut_weight += w
    gc = classify_gates(layout, A_set, B_set)
    cross_gates = len(gc['cross'])
    for g in gc['cross']:
        gamma_log += math.log(max(QPD_GAMMA_SYM.get(g['gate'].upper(), 3.0), 1.0))
    balance = abs(len(A_set) - len(B_set)) / max(n_qubits, 1)
    return float(0.55 * math.log1p(cut_weight) + 0.25 * cross_gates + 0.10 * gamma_log + 0.10 * balance)


def compute_partition_cost(A_set: set, B_set: set, G: nx.Graph, layout: list, n_qubits: int,
                           sv_full=None, p_full=None) -> tuple[float, dict]:
    """Physics-aligned partition cost used by the advanced mosaic variants.

    When the full statevector is available, the cost combines entanglement entropy,
    mutual information, QPD overhead, cross-cut weight, and balance.  When the full
    statevector cannot be computed, it falls back to a structural surrogate.
    """
    A_sorted = sorted(A_set)
    B_sorted = sorted(B_set)
    gc = classify_gates(layout, A_set, B_set)

    cut_weight = 0.0
    for u, v, d in G.edges(data=True):
        if (u in A_set) != (v in A_set):
            cut_weight += float(d.get('weight', 1.0))

    gamma_log = 0.0
    for g in gc['cross']:
        gamma_log += math.log(max(QPD_GAMMA_SYM.get(g['gate'].upper(), 3.0), 1.0))

    balance = abs(len(A_set) - len(B_set)) / max(n_qubits, 1)
    cross_gate_frac = len(gc['cross']) / max(sum(len(layer) for layer in layout), 1)

    detail = {
        'cut_weight': float(cut_weight),
        'n_cross_gates': len(gc['cross']),
        'gamma_log_sym': float(gamma_log),
        'balance_penalty': float(balance),
        'cross_gate_fraction': float(cross_gate_frac),
        'entanglement_entropy': None,
        'mutual_information': None,
        'mode': 'surrogate',
    }

    if sv_full is None:
        score = _partition_cost_surrogate(A_set, B_set, G, layout, n_qubits)
        detail['score'] = float(score)
        return float(score), detail

    if p_full is None:
        p_full = _prob_from_statevector(sv_full)

    ent = _entanglement_entropy(sv_full, A_sorted, n_qubits)
    idxA = _build_recon_index(A_sorted, n_qubits)
    idxB = _build_recon_index(B_sorted, n_qubits)
    dimA = 1 << len(A_sorted)
    dimB = 1 << len(B_sorted)
    pA = np.bincount(idxA, weights=p_full, minlength=dimA).astype(float)
    pB = np.bincount(idxB, weights=p_full, minlength=dimB).astype(float)
    if pA.sum() > 0:
        pA /= pA.sum()
    if pB.sum() > 0:
        pB /= pB.sum()
    eps = 1e-15
    pA_nz = pA[pA > eps]
    pB_nz = pB[pB > eps]
    pF_nz = p_full[p_full > eps]
    HA = float(-np.sum(pA_nz * np.log(pA_nz + eps))) if pA_nz.size else 0.0
    HB = float(-np.sum(pB_nz * np.log(pB_nz + eps))) if pB_nz.size else 0.0
    H = float(-np.sum(pF_nz * np.log(pF_nz + eps))) if pF_nz.size else 0.0
    mi = max(HA + HB - H, 0.0)

    # Lower is better. Weight the physically meaningful terms highest.
    score = (
        1.00 * ent +
        0.70 * mi +
        0.45 * gamma_log +
        0.20 * math.log1p(cut_weight) +
        0.12 * cross_gate_frac +
        0.08 * balance
    )

    detail.update({
        'score': float(score),
        'entanglement_entropy': float(ent),
        'mutual_information': float(mi),
        'mode': 'physics',
    })
    return float(score), detail


def optimize_partition_sa(G: nx.Graph, layout: list, n_qubits: int, qc_full=None,
                          start_partition=None, iters: int = 180, boundary_focus: bool = False,
                          seed: int = SEED) -> tuple[set, set]:
    """Simulated annealing refinement for mosaic partitions.

    The optimizer uses the physics-aligned partition cost when a full statevector
    is available, and a structural surrogate otherwise.
    """
    rng = random.Random(seed)
    if start_partition is None:
        A, B = partition_graph(G, n_qubits)
    else:
        A, B = set(start_partition[0]), set(start_partition[1])

    sv_full = None
    p_full = None
    if qc_full is not None:
        sv_full = _get_statevector(qc_full)
        if sv_full is not None:
            p_full = _prob_from_statevector(sv_full)

    def _cost(partA, partB):
        return compute_partition_cost(partA, partB, G, layout, n_qubits, sv_full=sv_full, p_full=p_full)[0]

    current_score = _cost(A, B)
    best_A, best_B = set(A), set(B)
    best_score = current_score
    temp0 = max(1.0, abs(current_score) + 1.0)

    for t in range(max(1, iters)):
        T = temp0 * (0.975 ** t)

        if boundary_focus:
            boundary_A = [q for q in A if any(nb in B for nb in G.neighbors(q))]
            boundary_B = [q for q in B if any(nb in A for nb in G.neighbors(q))]
            use_swap = bool(boundary_A and boundary_B and rng.random() < 0.72)
        else:
            boundary_A = [q for q in A if any(nb in B for nb in G.neighbors(q))]
            boundary_B = [q for q in B if any(nb in A for nb in G.neighbors(q))]
            use_swap = bool(boundary_A and boundary_B and rng.random() < 0.60)

        A_new, B_new = set(A), set(B)
        if use_swap:
            qa = rng.choice(boundary_A)
            qb = rng.choice(boundary_B)
            A_new.remove(qa); A_new.add(qb)
            B_new.remove(qb); B_new.add(qa)
        else:
            move_from_A = (rng.random() < 0.5 and len(A) > 1) or len(B) <= 1
            if move_from_A and len(A) > 1:
                q = rng.choice(tuple(A))
                A_new.remove(q); B_new.add(q)
            elif len(B) > 1:
                q = rng.choice(tuple(B))
                B_new.remove(q); A_new.add(q)
            else:
                continue

        if abs(len(A_new) - len(B_new)) > 2:
            continue

        new_score = _cost(A_new, B_new)
        accept = False
        if new_score <= current_score:
            accept = True
        else:
            delta = new_score - current_score
            accept = rng.random() < math.exp(-delta / max(T, 1e-12))

        if accept:
            A, B = A_new, B_new
            current_score = new_score
            if new_score < best_score:
                best_A, best_B = set(A), set(B)
                best_score = new_score

    # A brief greedy cleanup pass over boundary swaps.
    improved = True
    cleanup_steps = 0
    while improved and cleanup_steps < 12:
        improved = False
        cleanup_steps += 1
        boundary_A = [q for q in best_A if any(nb in best_B for nb in G.neighbors(q))]
        boundary_B = [q for q in best_B if any(nb in best_A for nb in G.neighbors(q))]
        candidates = []
        for qa in boundary_A[:6]:
            for qb in boundary_B[:6]:
                candidates.append((qa, qb))
        rng.shuffle(candidates)
        for qa, qb in candidates[:24]:
            A_new, B_new = set(best_A), set(best_B)
            A_new.remove(qa); A_new.add(qb)
            B_new.remove(qb); B_new.add(qa)
            if abs(len(A_new) - len(B_new)) > 2:
                continue
            score = _cost(A_new, B_new)
            if score + 1e-9 < best_score:
                best_A, best_B = A_new, B_new
                best_score = score
                improved = True
                break

    return best_A, best_B


def mosaic_plus_plus(G: nx.Graph, layout: list, n_qubits: int, qc_full=None) -> tuple[set, set]:
    """Flagship mosaic variant: entanglement-weighted graph + annealed refinement."""
    H = G.copy()
    ent_weights = estimate_edge_entanglement(layout)
    for (u, v), w in ent_weights.items():
        if H.has_edge(u, v):
            H[u][v]['weight'] = max(float(w), 0.1)
        else:
            H.add_edge(u, v, weight=max(float(w), 0.1), gates=['virtual'])

    start = partition_graph(H, n_qubits)
    return optimize_partition_sa(H, layout, n_qubits, qc_full=qc_full, start_partition=start,
                                 iters=220, boundary_focus=False, seed=SEED)


def _mosaic_syment_static(G: nx.Graph, layout: list, n_qubits: int) -> tuple[set, set]:
    """
    mosaic-SymEnt (Static): entanglement-aware edge weighting before partitioning.
    This keeps the method purely structural, while still being more informative than
    a raw gate-count or plain QPD-cost heuristic.
    """
    H = G.copy()
    ent_weights = estimate_edge_entanglement(layout)
    for (u, v), w in ent_weights.items():
        if H.has_edge(u, v):
            H[u][v]['weight'] = max(float(w), 0.1)
        else:
            H.add_edge(u, v, weight=max(float(w), 0.1), gates=['virtual'])

    init = partition_graph(H, n_qubits)
    return optimize_partition_sa(H, layout, n_qubits, qc_full=None, start_partition=init,
                                 iters=120, boundary_focus=False, seed=SEED)


def _mosaic_adaptive(G: nx.Graph, layout: list, n_qubits: int, qc_full, outdir: Path) -> tuple[set, set]:
    """
    mosaic-Adapt (Closed-Loop): boundary-focused annealing guided by the same
    physics-aligned cost, with the full circuit used when available.
    """
    if n_qubits > 24:
        print("         [Adapt] Using structural refinement only (n_qubits > 24)")
        return optimize_partition_sa(G, layout, n_qubits, qc_full=None,
                                      start_partition=partition_graph(G, n_qubits),
                                      iters=160, boundary_focus=True, seed=SEED)

    # For moderate size circuits, allow the full statevector to guide the search.
    H = G.copy()
    ent_weights = estimate_edge_entanglement(layout)
    for (u, v), w in ent_weights.items():
        if H.has_edge(u, v):
            H[u][v]['weight'] = max(float(w), 0.1)
        else:
            H.add_edge(u, v, weight=max(float(w), 0.1), gates=['virtual'])

    start = partition_graph(H, n_qubits)
    return optimize_partition_sa(H, layout, n_qubits, qc_full=qc_full, start_partition=start,
                                 iters=260, boundary_focus=True, seed=SEED)

# ──────────────────────────────────────────────────────────────────────────────
#  PLOTTING: INDIVIDUAL METRICS & UNIFIED SCORE
# ──────────────────────────────────────────────────────────────────────────────

def _save_individual_metric_plots(results, outdir):
    import numpy as np
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # Collect metric keys
    metric_keys = list(results[next(iter(results))].keys())

    for met_key in metric_keys:
        names = []
        values = []

        for method, res in results.items():
            v = res.get(met_key, None)
            if v is None or np.isnan(v) or np.isinf(v):
                continue
            names.append(method)
            values.append(v)

        if len(values) == 0:
            continue

        # ---- SAFE FIGURE SIZE ----
        n = len(values)
        width = 6
        height = max(3, min(0.5 * n + 2, 12))  # clamp between 3 and 12

        fig, ax = plt.subplots(figsize=(width, height))

        # ---- SORT (important for ranking) ----
        order = np.argsort(values)
        values_sorted = [values[i] for i in order]
        names_sorted = [names[i] for i in order]

        ax.barh(names_sorted, values_sorted)

        ax.set_title(f"{met_key} (ranked)")
        ax.set_xlabel(met_key)

        # ---- REMOVE TIGHT_LAYOUT BUG ----
        try:
            fig.tight_layout()
        except Exception:
            pass

        # ---- SAFE SAVE ----
        save_path = outdir / f"sim_{met_key}.png"

        try:
            fig.savefig(save_path, dpi=120)
        except Exception:
            # fallback if layout still breaks
            fig.set_size_inches(6, 4)
            fig.savefig(save_path, dpi=100)

        plt.close(fig)

def _save_unified_score_plot(results, outdir):
    valid = {k: v for k, v in results.items() if v.get('error') is None and 'unified_score' in v}
    if not valid:
        return

    labels = sorted(valid.keys(), key=lambda k: (valid[k].get('rank', 10**9), -valid[k].get('unified_score', -1.0), k))
    display_labels = _format_ranked_labels(labels, valid, use_rank=True)
    scores = [valid[l]['unified_score'] for l in labels]
    n = len(labels)
    pal = ['#BDC3C7','#F39C12','#4472C4','#8E44AD','#27AE60','#E74C3C','#16A085','#D35400']
    cols = [pal[i % len(pal)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))
    bars = ax.bar(range(n), scores, color=cols, edgecolor='white', lw=0.8)
    ax.set_xticks(range(n)); ax.set_xticklabels(display_labels, fontsize=8, rotation=18, ha='right')
    ax.set_title('Unified Quality Score (ranked best-to-worst)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score (higher=better, max=1.0)', fontsize=10); ax.grid(axis='y', alpha=0.3)
    best_idx = int(np.argmax(scores))
    bars[best_idx].set_edgecolor('#27AE60'); bars[best_idx].set_linewidth(2.5)

    for i, lbl in enumerate(labels):
        if 'mosaic' in lbl.lower() or 'kl' in lbl.lower():
            bars[i].set_ls('--'); bars[i].set_edgecolor('#2C3E50'); bars[i].set_lw(1.8)

    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(abs(v) * 0.02, 0.002),
                f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    fig.savefig(outdir / 'sim_unified_score.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('         Saved: sim_unified_score.png')
# ──────────────────────────────────────────────────────────────────────────────
#  run_and_compare_methods — PRODUCT-MARGINAL / SEPARABILITY COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def run_and_compare_methods(method_partitions, layout, n_qubits, qc_full, outdir):
    print('[SimCompare] Running product-marginal / separability comparison across methods …')
    results = {}
    if n_qubits > 24:
        print(f'         Skipping simulation: n_qubits={n_qubits} > 24 (memory limit)')
        return results

    print('         Simulating full circuit …')
    sv_full = _get_statevector(_materialize_layout_circuit(layout, n_qubits))
    if sv_full is None:
        print('         [warn] Full SV failed')
        return results
    p_ideal = _prob_from_statevector(sv_full)

    lock = threading.Lock()
    sim_data = {}

    def _run_pair(lbl, A, B):
        try:
            pA, sA = _simulate_subcircuit(A, B, layout, n_qubits, 'A')
            pB, sB = _simulate_subcircuit(A, B, layout, n_qubits, 'B')
            with lock:
                sim_data[lbl] = {'A': (pA, sA), 'B': (pB, sB), 'n_A': len(A), 'n_B': len(B)}
        except Exception as e:
            with lock:
                sim_data[lbl] = {'error': str(e)}

    max_w = min(len(method_partitions), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futs = {pool.submit(_run_pair, lbl, A, B): lbl for lbl, (A, B) in method_partitions.items()}
        for fut in as_completed(futs):
            lbl = futs[fut]
            try:
                fut.result()
                print(f'         Simulated: {lbl.replace(chr(10), " "):35s} ✓')
            except Exception as e:
                print(f'         [warn] Sim failed {lbl}: {e}')

    for lbl, data in sim_data.items():
        clean = lbl.replace('\n', ' ')
        if 'error' in data:
            results[clean] = {'error': data['error'], 'n_A': len(method_partitions[lbl][0]), 'n_B': len(method_partitions[lbl][1])}
            continue

        pA, sA = data['A']
        pB, sB = data['B']
        if pA is None or pB is None:
            results[clean] = {'error': 'SV None', 'n_A': data['n_A'], 'n_B': data['n_B']}
            continue

        try:
            idxA = _build_recon_index(sA, n_qubits)
            idxB = _build_recon_index(sB, n_qubits)
            p_rec = pA[idxA] * pB[idxB]
            p_rec_sum = float(p_rec.sum())
            if p_rec_sum <= 0:
                results[clean] = {'error': 'reconstruction sum is zero', 'n_A': data['n_A'], 'n_B': data['n_B']}
                continue
            p_rec /= p_rec_sum

            mets = _compute_distribution_metrics(p_ideal, p_rec)
            results[clean] = {**mets, 'n_A': data['n_A'], 'n_B': data['n_B'], 'error': None}
            print(f'         {clean:35s} TVD={mets["tvd"]:.4f} F={mets["fidelity"]:.4f} KL={mets["kl_divergence"]:.4f}')
        except Exception as e:
            results[clean] = {'error': str(e), 'n_A': data['n_A'], 'n_B': data['n_B']}

    results = _compute_unified_scores(results)
    _save_individual_metric_plots(results, outdir)
    _save_unified_score_plot(results, outdir)
    (outdir / 'simulation_comparison.json').write_text(json.dumps(results, indent=2))
    print('         Saved: simulation_comparison.json')
    return results

# ──────────────────────────────────────────────────────────────────────────────
#  VALIDATION & SUBCIRCUIT SAVING
# ──────────────────────────────────────────────────────────────────────────────
def _save_method_subcircuits(method_key, A_set, B_set, layout, n_qubits, methods_dir):
    safe=method_key.replace('\n','_').replace(' ','_').replace('/','_').replace('(','').replace(')','')
    mdir=methods_dir/safe; mdir.mkdir(parents=True,exist_ok=True)
    gc=classify_gates(layout,A_set,B_set); res={'method':safe,'A':{},'B':{}}
    def _gh(gl):
        c=Counter(g['gate'].upper() for g in gl)
        return {'single':sum(v for k,v in c.items() if k in SINGLE),'two':sum(v for k,v in c.items() if k in TWO),'multi':sum(v for k,v in c.items() if k in MULTI)}
    for pn,ns in [('A',A_set),('B',B_set)]:
        sl,sn,rm=build_subcircuit_layout(layout,ns,pn,gc); nsub=len(ns)
        qc,par=compile_subcircuit(sl,nsub)
        lg=gc.get(f'local_{pn}',[]); cg=gc['cross']
        stubs=[g for l in sl for g in l if g.get('source','').startswith('qpd_stub')]
        hl,hs=_gh(lg),_gh(stubs)
        res[pn]={'n_qubits':qc.num_qubits,'n_gates':qc.size(),'depth':qc.depth(),'n_params':len(par),
                 'n_local_gates':len(lg),'n_cross_gates':len(cg),'n_stub_gates':len(stubs),
                 'single_q_local':hl['single'],'two_q_local':hl['two'],'multi_q_local':hl['multi'],'single_q_stub':hs['single']}
        with open(mdir/f'circuit_{pn}_layout.pkl','wb') as f: pickle.dump({'layout':sl,'n_qubits':nsub,'sorted_nodes':sn,'remap':rm,'partition':sorted(ns),'method':safe},f)
        export_qasm(qc, mdir/f'circuit_{pn}.qasm')
        save_circuit_heatmap_partition(sl,nsub,sn,mdir/f'circuit_{pn}_heatmap.png',f'{safe} — {pn} ({nsub}q)')
        save_circuit_diagram_qiskit(qc,mdir/f'circuit_{pn}_diagram.png',f'{safe} — {pn} ({nsub}q {qc.size()}g d{qc.depth()})')
    return res

def validate_and_compare_partition(G, layout, n_qubits, A_mosaic, B_mosaic, qc_full, outdir, methods_dir):
    print('[Validate] Benchmarking partition quality …')

    # Baseline mosaic from the original KL-style partitioner, plus the advanced mosaic++.
    A_base, B_base = partition_graph(G, n_qubits)
    all_parts = {
        'mosaic (KL, baseline)': (A_base, B_base),
        'mosaic++ (physics-SA)': (A_mosaic, B_mosaic),
        'Naive (half-split)': _naive_bisection(n_qubits),
        'Spectral (Fiedler)': _spectral_bisection(G, n_qubits),
        'mosaic-SymEnt (static)': _mosaic_syment_static(G, layout, n_qubits)
    }
    try:
        all_parts['Louvain (modularity)'] = _louvain_bisection(G, n_qubits)
    except Exception as e:
        print(f"         [warn] Louvain failed: {e}")
    try:
        all_parts['Girvan-Newman (betweenness)'] = _girvan_newman_bisection(G, n_qubits)
    except Exception as e:
        print(f"         [warn] GN failed: {e}")
    met = _metis_bisection(G, n_qubits)
    if isinstance(met, tuple):
        all_parts['METIS (multilevel)'] = met
    qd = _try_qdislib(layout, n_qubits)
    if qd.get('available') and 'A_set' in qd:
        all_parts['qdislib (DAG cut)'] = (set(qd['A_set']), set(qd['B_set']))

    print('[Adaptive] Running mosaic-Adapt closed-loop optimization …')
    try:
        A_adapt, B_adapt = _mosaic_adaptive(G, layout, n_qubits, qc_full, outdir)
        all_parts['mosaic-Adapt (closed-loop)'] = (A_adapt, B_adapt)
    except Exception as e:
        print(f"         [warn] mosaic-Adapt failed: {e}")

    methods = {}
    for lbl, (A, B) in all_parts.items():
        cs = compute_cut_stats(G, A, B)
        gc = classify_gates(layout, A, B)
        qpd = compute_qpd_overhead(gc['cross'], sym_reduction=True)
        bal = min(len(A), len(B)) / max(len(A), len(B), 1)
        methods[lbl] = {
            'n_A': len(A), 'n_B': len(B), 'balance_ratio': round(bal, 4),
            'n_cut_edges': cs['n_cut_edges'], 'cut_weight': cs['cut_weight'], 'cut_fraction': cs['cut_fraction'],
            'n_cross_gates': qpd['n_cut_gates'], 'gamma_generic': round(qpd['total_gamma_generic'], 4),
            'gamma_sym': round(qpd['total_gamma_sym'], 4), 'shot_overhead': qpd['shot_overhead_sym'],
            'A_set': sorted(A), 'B_set': sorted(B)
        }

    sv = None
    if n_qubits <= 20:
        try:
            sv = _get_statevector(_materialize_layout_circuit(layout, n_qubits))
        except Exception as e:
            print(f'         [warn] SV failed: {e}')

    for lbl, m in methods.items():
        if 'error' in m:
            m['entanglement_entropy'] = None
            continue
        if sv is not None:
            m['entanglement_entropy'] = _entanglement_entropy(sv, m['A_set'], n_qubits)
        else:
            m['entanglement_entropy'] = None
        print(f'         {lbl.replace(chr(10)," "):32s} cut={m["n_cut_edges"]:2d} γ={m["gamma_sym"]:.2e} S={m["entanglement_entropy"]}')

    sub_stats = []
    if methods_dir:
        print('[SubCircuits] Building/saving subcircuits (multithreaded) …')
        lock = threading.Lock()

        def _save_thr(lbl, A, B):
            try:
                st = _save_method_subcircuits(lbl, A, B, layout, n_qubits, methods_dir)
                with lock:
                    sub_stats.append(st)
                print(f'         {st["method"]:40s} A:{st["A"].get("n_gates",0):4d}g d={st["A"].get("depth",0):3d} B:{st["B"].get("n_gates",0):4d}g d={st["B"].get("depth",0):3d}')
            except Exception as e:
                print(f'         [warn] Save failed {lbl}: {e}')

        with ThreadPoolExecutor(max_workers=min(len(all_parts), os.cpu_count() or 4)) as pool:
            futs = [pool.submit(_save_thr, lbl, A, B) for lbl, (A, B) in all_parts.items()]
            for f in as_completed(futs):
                f.result()

    chart = {k: v for k, v in methods.items() if 'error' not in v}
    if chart:
        # Assign a ranked proxy score directly so the bars and labels are sorted.
        order = _rank_partition_methods_for_benchmark(chart)
        for r, lbl in enumerate(order, start=1):
            chart[lbl]['rank'] = r
        labels = order
        n_m = len(labels)
        pal = ['#BDC3C7', '#F39C12', '#4472C4', '#8E44AD', '#27AE60', '#E74C3C', '#16A085', '#D35400']
        cols = [pal[i % len(pal)] for i in range(n_m)]
        mets = {
            'Cut edges': [chart[m]['n_cut_edges'] for m in labels],
            'Cross gates': [chart[m]['n_cross_gates'] for m in labels],
            'γ sym (log10)': [math.log10(max(chart[m]['gamma_sym'], 1e-9)) for m in labels],
            'Ent S': [chart[m].get('entanglement_entropy') or 0.0 for m in labels]
        }
        fig, axes = plt.subplots(1, len(mets), figsize=(max(5 * len(mets), 20), 6))
        if len(mets) == 1:
            axes = [axes]
        display_labels = [f"{chart[lbl]['rank']}. {lbl}" for lbl in labels]
        for ax, (mn, vals) in zip(axes, mets.items()):
            bars = ax.bar(range(n_m), vals, color=cols, edgecolor='white', lw=0.8)
            ax.set_xticks(range(n_m)); ax.set_xticklabels(display_labels, fontsize=8, rotation=15, ha='right')
            ax.set_title(mn, fontsize=10, fontweight='bold'); ax.grid(axis='y', alpha=0.3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(abs(v) * 0.02, 0.01), f'{v:.2f}',
                        ha='center', va='bottom', fontsize=7.5)
            bars[int(np.argmin(vals))].set_edgecolor('#27AE60'); bars[int(np.argmin(vals))].set_linewidth(2.5)
        fig.suptitle(f'Partition Benchmark — {n_qubits}-qubit (ranked best-to-worst)', fontsize=10, fontweight='bold')
        plt.tight_layout(); fig.savefig(outdir / 'partition_benchmark.png', dpi=130, bbox_inches='tight'); plt.close(fig)
        print('         Saved: partition_benchmark.png')
        (outdir / 'partition_benchmark.json').write_text(json.dumps({k.replace('\n', ' '): v for k, v in methods.items()}, indent=2, default=str))

    run_and_compare_methods(all_parts, layout, n_qubits, qc_full, outdir)
    return methods

# ──────────────────────────────────────────────────────────────────────────────
#  SUMMARY & MAIN
# ──────────────────────────────────────────────────────────────────────────────
def save_summary(outdir, sym, part, qpd, schur, cg, A, B, filepath):
    lines=['='*72,' EVAL (mosaic) — SUMMARY REPORT','='*72,
           f' Circuit: {N_QUBITS}q | depth {DEPTH} | {sym["total_gates"]} gates | {len(sym["gate_type_counts"])} types',
           f' Partition: A={len(A)}q B={len(B)}q','',
           '── PHASE 1: SYMMETRY ──',f' Detected: {sym["detected_symmetries"]}',f' U(1): {sym["u1_conserved_charge"]}',
           f' Clifford: {sym["clifford_fraction"]:.1%}',f' t-design: {sym["approx_t_design_order"]}',f' DLA est: {sym["dla_dimension_estimate"]}','',
           '── PHASE 2: PARTITION ──',f' Edges: {part["n_interaction_edges"]}',f' Cut weight: {part["cut_weight"]} ({part["cut_fraction"]:.1%})',
           f' Local A:{part["n_local_A"]} B:{part["n_local_B"]} Cross:{part["n_cross"]}','',
           '── SCHUR-WEYL ──',f' Kept: {schur["n_sectors_kept"]}',f' Trunc ε: {schur["truncation_error"]:.6f}',f' CG coeffs: {schur["n_cg_coefficients"]}']
    for s in schur['sectors']:
        lines.append(f'   {"✓" if s["keep"] else "✗"} {s["label"]:15s} dim={s["dim_Sn"]:7d} ξ={s["xi"]:6.2f} w={s["weight"]:.6f}')
    lines+=['','── PHASE 3: QPD ──',f' Cuts: {qpd["n_cut_gates"]}',f' Schmidt: {qpd["total_schmidt_rank"]}',
            f' γ gen: {qpd["total_gamma_generic"]:.3e}',f' γ sym: {qpd["total_gamma_sym"]:.3e}',f' Ratio: {qpd["gamma_reduction_ratio"]:.6f}',
            f' Shots (ε=0.05): gen={qpd["shot_overhead_generic"]} sym={qpd["shot_overhead_sym"]}','',
            '── PHASE 4: CG ──',f' Type: {cg["reconstruction_type"]}',f' Formula: {cg["aggregation_formula"]}',f' ε bound: {cg["error_bound"]}','',
            '── OUTPUT ──',f' Dir: {outdir}']
    filepath.write_text('\n'.join(lines))
    for l in lines: print(' ',l)

def main():
    print('='*70); print(' mosaic — Evaluation'); print(f' {N_QUBITS}-qubit RQC → 2×{N_QUBITS//2}-qubit subcircuits'); print('='*70)
    dirs={'full':OUTDIR/'full_circuit','A':OUTDIR/'partition_A','B':OUTDIR/'partition_B','anal':OUTDIR/'analysis','meth':OUTDIR/'methods'}
    for d in dirs.values(): d.mkdir(parents=True,exist_ok=True)
    layout, qc_full, params = generate_circuit(N_QUBITS, DEPTH, MAX_Q_PER_LAYER)
    sym = detect_symmetry(layout, N_QUBITS)
    print('[Phase 2] Building interaction graph + advanced mosaic++ …')
    G = build_interaction_graph(layout, N_QUBITS)
    A_base, B_base = partition_graph(G, N_QUBITS)
    A_set, B_set = mosaic_plus_plus(G, layout, N_QUBITS, qc_full=qc_full)
    cut_stats = compute_cut_stats(G, A_set, B_set)
    gate_class = classify_gates(layout, A_set, B_set)
    print(f'         Local A:{len(gate_class["local_A"])} B:{len(gate_class["local_B"])} Cross:{len(gate_class["cross"])}')
    part_rep = {'n_A':len(A_set),'n_B':len(B_set),'partition_A_qubits':sorted(A_set),'partition_B_qubits':sorted(B_set),
                'baseline_A_qubits':sorted(A_base),'baseline_B_qubits':sorted(B_base),
                'n_interaction_edges':G.number_of_edges(),'n_cut_edges':cut_stats['n_cut_edges'],'cut_weight':cut_stats['cut_weight'],
                'cut_fraction':cut_stats['cut_fraction'],'n_local_A':len(gate_class['local_A']),'n_local_B':len(gate_class['local_B']),'n_cross':len(gate_class['cross'])}
    print('[Phase 3] QPD gate-cutting analysis …')
    qpd_info = compute_qpd_overhead(gate_class['cross'], sym_reduction=True)
    print(f'         γ_total generic={qpd_info["total_gamma_generic"]:.3e} sym={qpd_info["total_gamma_sym"]:.3e}')
    schur = schur_weyl_analysis(N_QUBITS, DEPTH, len(gate_class['cross']))
    cg_info = build_cg_aggregation(schur, qpd_info, sym)
    print('[Build] Constructing subcircuit layouts …')
    slA,snA,rmA = build_subcircuit_layout(layout, A_set, 'A', gate_class)
    slB,snB,rmB = build_subcircuit_layout(layout, B_set, 'B', gate_class)
    qcA,pA = compile_subcircuit(slA, len(A_set))
    qcB,pB = compile_subcircuit(slB, len(B_set))
    print(f'         A: {qcA.num_qubits}q {qcA.size()}g {len(pA)}p | B: {qcB.num_qubits}q {qcB.size()}g {len(pB)}p')
    print('[Save] Writing output files …')
    with open(dirs['full']/'circuit_layout.pkl','wb') as f: pickle.dump({'layout':layout,'n_qubits':N_QUBITS,'depth':DEPTH,'seed':SEED},f)
    export_qasm(qc_full, dirs['full']/'circuit_full.qasm')
    save_circuit_heatmap(layout, N_QUBITS, dirs['full']/'circuit_heatmap.png', f'Full {N_QUBITS}-qubit RQC')
    save_circuit_diagram_qiskit(qc_full, dirs['full']/'circuit_full_diagram.png', f'Full {N_QUBITS}-qubit RQC')
    for pn,sl,sn,rm,qc,par in [('A',slA,snA,rmA,qcA,pA),('B',slB,snB,rmB,qcB,pB)]:
        with open(dirs[pn]/f'circuit_{pn}_layout.pkl','wb') as f: pickle.dump({'layout':sl,'n_qubits':len(sn),'sorted_nodes':sn,'remap':rm,'partition':sorted(A_set if pn=='A' else B_set)},f)
        export_qasm(qc, dirs[pn]/f'circuit_{pn}.qasm')
        save_circuit_heatmap_partition(sl,len(sn),sn,dirs[pn]/f'circuit_{pn}_diagram.png',f'Partition {pn} ({len(sn)}q)')
        save_circuit_diagram_qiskit(qc,dirs[pn]/f'circuit_{pn}_qiskit.png',f'Partition {pn} ({len(sn)}q)')
    save_interaction_graph_dot(G,A_set,B_set,cut_stats,dirs['anal']/'interaction_graph.dot')
    save_interaction_graph_png(G,A_set,B_set,cut_stats,dirs['anal']/'interaction_graph.png')
    save_cut_analysis(gate_class,qpd_info,dirs['anal']/'cut_analysis.png')
    save_qpd_terms_table(qpd_info,dirs['anal']/'qpd_terms.png')
    save_schur_weyl_plot(schur,dirs['anal']/'schur_weyl_sectors.png')
    save_cg_aggregation_schematic(cg_info,dirs['anal']/'cg_aggregation.png')
    validate_and_compare_partition(G, layout, N_QUBITS, A_set, B_set, qc_full, dirs['anal'], dirs['meth'])
    for nm,rep in [('symmetry_report.json',sym),('partitioning_report.json',part_rep),('qpd_report.json',{k:v for k,v in qpd_info.items() if k!='per_gate'}),('schur_weyl_report.json',schur),('cg_aggregation_report.json',cg_info)]:
        (dirs['anal']/nm).write_text(json.dumps(rep,indent=2))
    save_summary(OUTDIR, sym, part_rep, qpd_info, schur, cg_info, A_set, B_set, OUTDIR/'mosaic_summary.txt')
    print(f'\nAll artefacts saved to: {OUTDIR.resolve()}')

if __name__ == '__main__':
    main()
