from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    U1Gate, U2Gate, U3Gate, PhaseGate, RGate, CRXGate,
    CRYGate, CRZGate, CU1Gate, CU3Gate, CPhaseGate, iSwapGate, SwapGate
)


def compile_quantum_circuit(layout, num_qubits=None, use_numeric_params=False):
    """
    Compile a layout-format gate list into a Qiskit QuantumCircuit.

    Parameters
    ----------
    layout : list of layers; each layer is a list of dicts:
             {'gate': str, 'qubits': [int, ...], 'params': [float, ...]}
    num_qubits : int, optional
    use_numeric_params : bool
        If True, the numeric values in `gate['params']` are inserted directly
        into the circuit (no `Parameter` objects are created). This guarantees
        that the *same* gate produces the *same* unitary in the full circuit
        and in any subcircuit, which is essential for cutting validation.
        If False (default), keeps the original behaviour of creating
        Parameter('theta_k') symbols in traversal order.

    Returns
    -------
    qc : QuantumCircuit
    param_list : list[Parameter]   (empty when use_numeric_params=True)
    """
    if num_qubits is None:
        num_qubits = max(q for layer in layout for gate in layer for q in gate['qubits']) + 1

    qc = QuantumCircuit(num_qubits)
    param_list = []
    param_count = 0

    for layer in layout:
        for gate in layer:
            name = gate['gate'].upper()
            qubits = gate['qubits']
            raw_params = gate.get('params', [])

            if use_numeric_params:
                # Use the numeric values directly — no Parameter objects created.
                param_objs = [float(p) for p in raw_params]
            else:
                param_objs = []
                for _ in raw_params:
                    p = Parameter(f'theta_{param_count}')
                    param_objs.append(p)
                    param_list.append(p)
                    param_count += 1

            # --- Single-qubit gates ---
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
            elif name == 'T':
                qc.t(qubits[0])
            elif name == 'RX':
                qc.rx(param_objs[0] if param_objs else 1.0, qubits[0])
            elif name == 'RY':
                qc.ry(param_objs[0] if param_objs else 1.0, qubits[0])
            elif name == 'RZ':
                qc.rz(param_objs[0] if param_objs else 1.0, qubits[0])
            elif name == 'U1':
                qc.append(U1Gate(param_objs[0] if param_objs else 0.5), [qubits[0]])
            elif name == 'U2':
                qc.append(U2Gate(*(param_objs if len(param_objs) == 2 else [0, 3.14])), [qubits[0]])
            elif name == 'U3':
                qc.append(U3Gate(*(param_objs if len(param_objs) == 3 else [1.0, 0.0, 0.0])), [qubits[0]])
            elif name == 'PHASE':
                qc.append(PhaseGate(param_objs[0] if param_objs else 0.5), [qubits[0]])
            elif name == 'SX':
                qc.sx(qubits[0])
            elif name == 'R':
                qc.append(RGate(*(param_objs if len(param_objs) == 2 else [0.5, 0.5])), [qubits[0]])

            # --- Two-qubit gates ---
            elif name == 'CNOT' or name == 'CX':
                qc.cx(*qubits)
            elif name == 'CY':
                qc.cy(*qubits)
            elif name == 'CZ':
                qc.cz(*qubits)
            elif name == 'CH':
                qc.ch(*qubits)
            elif name == 'CRX':
                qc.append(CRXGate(param_objs[0] if param_objs else 1.0), qubits)
            elif name == 'CRY':
                qc.append(CRYGate(param_objs[0] if param_objs else 1.0), qubits)
            elif name == 'CRZ':
                qc.append(CRZGate(param_objs[0] if param_objs else 1.0), qubits)
            elif name == 'CU1':
                qc.append(CU1Gate(param_objs[0] if param_objs else 0.5), qubits)
            elif name == 'CU3':
                qc.append(CU3Gate(*(param_objs if len(param_objs) == 3 else [1.0, 0.0, 0.0])), qubits)
            elif name == 'CPHASE':
                qc.append(CPhaseGate(param_objs[0] if param_objs else 0.5), qubits)
            elif name == 'SWAP':
                qc.swap(*qubits)
            elif name == 'ISWAP':
                qc.append(iSwapGate(), qubits)

            # --- Three-qubit gates ---
            elif name == 'CCX':
                qc.ccx(*qubits)
            elif name == 'CSWAP':
                qc.cswap(*qubits)

            # --- Four-qubit gates (3 controls, 1 target) ---
            elif name == 'C3X':
                # Triple-controlled X: 3 controls + 1 target
                if len(qubits) >= 4:
                    qc.mcx(qubits[:3], qubits[3])
                else:
                    # Fallback: treat as Toffoli with available qubits
                    qc.ccx(qubits[0], qubits[1], qubits[2])
            elif name == 'C3Z':
                # Triple-controlled Z: 3 controls + 1 target
                if len(qubits) >= 4:
                    # Build C3Z = H_target · C3X · H_target
                    qc.h(qubits[3])
                    qc.mcx(qubits[:3], qubits[3])
                    qc.h(qubits[3])
                else:
                    qc.ccz(qubits[0], qubits[1], qubits[2])

            else:
                raise ValueError(f"Unsupported gate: {name}")

    return qc, param_list