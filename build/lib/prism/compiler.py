"""
prism.compiler
==============
Translate PRISM's layout format into a Qiskit ``QuantumCircuit``.

A *layout* is ``list[list[dict]]`` — a list of layers, each layer a list of
gate dicts ``{'gate': str, 'qubits': [int, ...], 'params': [float, ...]}``.

Two compilation paths are provided:

``compile_circuit``      faithful translation; with ``use_numeric_params=True``
                         the numeric angles in each gate dict are inserted
                         directly so the *same* logical gate yields the *same*
                         unitary in the full circuit and in any subcircuit
                         (this exactness is what makes cut reconstruction
                         meaningful).  With ``use_numeric_params=False`` it
                         creates ``Parameter`` symbols instead.

``materialize_layout``   like ``compile_circuit(..., numeric)`` but any gate
                         missing explicit params is given a *deterministic*
                         angle hashed from its identity, so full/sub circuits
                         still agree even for layouts authored without angles.
"""
from __future__ import annotations

import hashlib
import numpy as np

PARAM_SEED = 42  # salt for deterministic angle hashing


def _param_objects(raw_params, use_numeric_params, param_list, counter):
    if use_numeric_params:
        return [float(p) for p in raw_params], counter
    from qiskit.circuit import Parameter
    objs = []
    for _ in raw_params:
        p = Parameter(f'theta_{counter}')
        objs.append(p)
        param_list.append(p)
        counter += 1
    return objs, counter


def compile_circuit(layout, num_qubits=None, use_numeric_params=True):
    """Compile a layout into a ``(QuantumCircuit, param_list)`` pair.

    Parameters
    ----------
    layout : list[list[dict]]
    num_qubits : int, optional
        Inferred from the highest qubit index when omitted.
    use_numeric_params : bool, default True
        Insert numeric angles directly (cut-exact).  When False, build
        symbolic ``Parameter`` objects in traversal order.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (
        U1Gate, U2Gate, U3Gate, PhaseGate, RGate, CRXGate, CRYGate, CRZGate,
        CU1Gate, CU3Gate, CPhaseGate, iSwapGate,
    )
    if num_qubits is None:
        num_qubits = max(q for layer in layout for g in layer for q in g['qubits']) + 1

    qc = QuantumCircuit(num_qubits)
    param_list: list = []
    counter = 0

    for layer in layout:
        for gate in layer:
            name = gate['gate'].upper()
            qubits = gate['qubits']
            raw = gate.get('params', []) or []
            p, counter = _param_objects(raw, use_numeric_params, param_list, counter)

            if name == 'I':
                qc.id(qubits[0])
            elif name == 'X':
                qc.x(qubits[0])
            elif name == 'Y':
                qc.y(qubits[0])
            elif name == 'Z':
                qc.z(qubits[0])
            elif name == 'H':
                qc.h(qubits[0])
            elif name == 'S':
                qc.s(qubits[0])
            elif name == 'SDG':
                qc.sdg(qubits[0])
            elif name == 'T':
                qc.t(qubits[0])
            elif name == 'TDG':
                qc.tdg(qubits[0])
            elif name == 'SX':
                qc.sx(qubits[0])
            elif name == 'SXDG':
                qc.sxdg(qubits[0])
            elif name == 'RX':
                qc.rx(p[0] if p else 1.0, qubits[0])
            elif name == 'RY':
                qc.ry(p[0] if p else 1.0, qubits[0])
            elif name == 'RZ':
                qc.rz(p[0] if p else 1.0, qubits[0])
            elif name == 'U1':
                qc.append(U1Gate(p[0] if p else 0.5), [qubits[0]])
            elif name == 'U2':
                qc.append(U2Gate(*(p if len(p) == 2 else [0, 3.14159])), [qubits[0]])
            elif name == 'U3':
                qc.append(U3Gate(*(p if len(p) == 3 else [1.0, 0.0, 0.0])), [qubits[0]])
            elif name in ('PHASE', 'P'):
                qc.append(PhaseGate(p[0] if p else 0.5), [qubits[0]])
            elif name == 'R':
                qc.append(RGate(*(p if len(p) == 2 else [0.5, 0.5])), [qubits[0]])
            elif name in ('CNOT', 'CX'):
                qc.cx(*qubits)
            elif name == 'CY':
                qc.cy(*qubits)
            elif name == 'CZ':
                qc.cz(*qubits)
            elif name == 'CH':
                qc.ch(*qubits)
            elif name == 'CRX':
                qc.append(CRXGate(p[0] if p else 1.0), qubits)
            elif name == 'CRY':
                qc.append(CRYGate(p[0] if p else 1.0), qubits)
            elif name == 'CRZ':
                qc.append(CRZGate(p[0] if p else 1.0), qubits)
            elif name == 'CU1':
                qc.append(CU1Gate(p[0] if p else 0.5), qubits)
            elif name == 'CU3':
                qc.append(CU3Gate(*(p if len(p) == 3 else [1.0, 0.0, 0.0])), qubits)
            elif name == 'CPHASE':
                qc.append(CPhaseGate(p[0] if p else 0.5), qubits)
            elif name == 'SWAP':
                qc.swap(*qubits)
            elif name == 'ISWAP':
                qc.append(iSwapGate(), qubits)
            elif name == 'CCX':
                qc.ccx(*qubits)
            elif name == 'CSWAP':
                qc.cswap(*qubits)
            elif name == 'C3X':
                if len(qubits) >= 4:
                    qc.mcx(qubits[:3], qubits[3])
                else:
                    qc.ccx(qubits[0], qubits[1], qubits[2])
            elif name == 'C3Z':
                if len(qubits) >= 4:
                    qc.h(qubits[3]); qc.mcx(qubits[:3], qubits[3]); qc.h(qubits[3])
                else:
                    qc.ccz(qubits[0], qubits[1], qubits[2])
            else:
                raise ValueError(f"Unsupported gate: {name}")

    return qc, param_list


# Legacy name kept so older scripts importing ``compile_quantum_circuit`` work.
compile_quantum_circuit = compile_circuit


# ---------------------------------------------------------------------------
#  Deterministic-angle materialisation (used for cut reconstruction)
# ---------------------------------------------------------------------------
def deterministic_param_values(gate, n_params, layer_idx=0, gate_idx=0):
    """Stable numeric params hashed from gate identity (or its own params)."""
    if n_params <= 0:
        return []
    raw = gate.get('params')
    if isinstance(raw, (list, tuple)) and len(raw) >= n_params:
        try:
            return [float(raw[i]) for i in range(n_params)]
        except Exception:
            pass
    qsig = ','.join(str(q) for q in gate.get('qubits', []))
    name = str(gate.get('gate', 'UNKNOWN')).upper()
    sl = gate.get('source_layer', layer_idx)
    sg = gate.get('source_gate_index', gate_idx)
    base = f"{PARAM_SEED}|{name}|{sl}|{sg}|{qsig}"
    vals = []
    for i in range(n_params):
        digest = hashlib.sha256(f"{base}|{i}".encode()).digest()
        raw_f = int.from_bytes(digest[:8], 'big') / 2 ** 64
        vals.append(float(raw_f * 2.0 * np.pi - np.pi))
    return vals


def append_gate(qc, gate):
    """Append one gate to ``qc`` using deterministic numeric parameters.

    Unsupported instructions are silently skipped (rather than aborting an
    entire simulation), which is what subcircuit construction needs.
    """
    name = str(gate.get('gate', '')).upper()
    qs = list(gate.get('qubits', []))
    if not qs:
        return
    li = int(gate.get('source_layer', gate.get('layer', 0)) or 0)
    gi = int(gate.get('source_gate_index', gate.get('gate_index', 0)) or 0)

    def pv(n):
        return deterministic_param_values(gate, n, layer_idx=li, gate_idx=gi)

    try:
        if name in ('I', 'ID', 'IDENTITY'):
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
            qc.rx(pv(1)[0], qs[0])
        elif name == 'RY':
            qc.ry(pv(1)[0], qs[0])
        elif name == 'RZ':
            qc.rz(pv(1)[0], qs[0])
        elif name in ('PHASE', 'P', 'U1'):
            qc.p(pv(1)[0], qs[0])
        elif name == 'U2':
            v = pv(2); qc.u(np.pi / 2, v[0], v[1], qs[0])
        elif name == 'U3':
            v = pv(3); qc.u(v[0], v[1], v[2], qs[0])
        elif name == 'R':
            v = pv(2); qc.r(v[0], v[1], qs[0])
        elif name in ('CX', 'CNOT'):
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
            qc.crx(pv(1)[0], qs[0], qs[1])
        elif name == 'CRY':
            qc.cry(pv(1)[0], qs[0], qs[1])
        elif name == 'CRZ':
            qc.crz(pv(1)[0], qs[0], qs[1])
        elif name in ('CU1', 'CPHASE'):
            qc.cp(pv(1)[0], qs[0], qs[1])
        elif name == 'CU3':
            v = pv(3); qc.cu(v[0], v[1], v[2], 0.0, qs[0], qs[1])
        elif name == 'CCX':
            qc.ccx(qs[0], qs[1], qs[2])
        elif name == 'CSWAP':
            qc.cswap(qs[0], qs[1], qs[2])
        elif name == 'C3X':
            qc.mcx(qs[:-1], qs[-1])
        elif name == 'C3Z':
            qc.mcp(np.pi, qs[:-1], qs[-1])
        # else: silently skip unsupported gate
    except Exception:
        return


def materialize_layout(layout, num_qubits):
    """Build a circuit from a layout using deterministic angles for any gate
    that lacks explicit numeric params. Always safe for reconstruction."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(num_qubits)
    for li, layer in enumerate(layout):
        for gi, gate in enumerate(layer):
            g = {**gate,
                 'source_layer': gate.get('source_layer', li),
                 'source_gate_index': gate.get('source_gate_index', gi)}
            append_gate(qc, g)
    return qc


__all__ = [
    'compile_circuit', 'compile_quantum_circuit',
    'append_gate', 'materialize_layout', 'deterministic_param_values',
]
