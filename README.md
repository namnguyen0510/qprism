# qPRISM: Multi-Objective Symmetry-Aware Irrep-truncated Circuit Partitioning

`qprism` is Python package for **partitioning quantum circuits for distributed
(multi-QPU) execution**. It implements the **PRISM** method family — a single-shot, light-cone-tempered heuristic solver
(**PRISM-LCT**) plus a four-rung ablation ladder — and benchmarks them against six classical baselines on
ten circuit families with exact statevector reconstruction.

```python
import prism

layout   = prism.make_layout('QAOA', n=10, seed=19)      # a circuit (one of 10 families)
G        = prism.build_interaction_graph(layout, 10)      # qubit interaction graph
qc, _    = prism.compile_circuit(layout, num_qubits=10)   # Qiskit circuit

A, B     = prism.partition('PRISM-LCT', G, layout, 10, qc_full=qc, seed=19)   # the main method
results, fails = prism.run_all_partition_methods(G, layout, 10, qc_full=qc)   # all 11 methods
```

## Highlights

- **Eleven partition methods** behind one uniform API and registry: six baselines
  (Naive, Spectral, Louvain, Girvan–Newman, METIS, qdislib) and the five-rung PRISM ladder
  (PRISM-KL → OE → MI → BF → **LCT**).
- **Multi-objective, symmetry-aware cost** (paper eq. 5): entanglement entropy `S(A)`, classical mutual
  information `I(A:B)`, symmetry-reduced QPD overhead `log γ_sym`, weighted cut `W_cut`, and balance `Δ`.
- **Application layer** (`prism.qml`): QAOA-MaxCut with eleven classical optimizers, trainable QNNs
  (angle, data re-uploading, and amplitude encoding), and a bridge to partition **any Qiskit circuit**.
- **Benchmark driver + CLI**: `prism-benchmark` runs all methods on all families × qubit counts × seeds.
- **Noisy execution** (`prism.noise`): exact density-matrix simulation with a depolarizing + readout model,
  and noisy fragment reconstruction — to show when cutting beats running the monolith.
- **Exact distributed execution** (`prism.distributed`): verified non-local CNOT by gate teleportation
  (1 ebit + 2 cbits), so a partition can be run *exactly* across QPUs — the entanglement-assisted foil to
  quasi-probability cutting.
- **QNN application suite** (`prism.qml`): Iris/Wine/Breast-Cancer/Digits/synthetic loaders, four entangling
  topologies (`ring`/`linear`/`blocks`/`full`), and distributed-deployment helpers (cut a trained QNN across
  QPUs and measure preserved accuracy / prediction agreement).

## Installation

```bash
pip install -e .                 # core (numpy, networkx, qiskit, qiskit-aer)
pip install -e ".[all]"          # + pymetis, python-louvain, scikit-learn, scipy, matplotlib, pandas, seaborn
pip install -e ".[dev]"          # + pytest, jupyter, nbconvert
```

Optional backends degrade gracefully: METIS falls back to spectral bisection if `pymetis` is missing;
Louvain falls back to spectral if `python-louvain`/`networkx` community detection is unavailable; the
`qml` module needs `scikit-learn`/`scipy`.

## Package layout

```
src/prism/
├── gates.py        gate taxonomy: arity, Schmidt rank, QPD γ (generic & symmetry-reduced), classes
├── compiler.py     layout → Qiskit circuit (cut-exact numeric params; deterministic materialisation)
├── circuits.py     10 circuit families (QFT, QPE, VQE-HEA/UCC, QAOA, QNN, MERA, MPS, OTOC, RQC)
├── graph.py        interaction graph, light-cone augmentation, gate classification, subcircuits
├── symmetry.py     U(1)/Z2 detection, Schur–Weyl sectors, QPD overhead, CG aggregation
├── simulate.py     statevector, product reconstruction, distribution metrics, Q-Score
├── partition/
│   ├── cost.py     multi-objective partition cost (C1–C5) + structural surrogate
│   ├── baselines.py  Naive, Spectral, Louvain, Girvan–Newman, METIS, qdislib (+ real-package adapter)
│   ├── ladder.py     PRISM-KL/OE/MI/BF + shared simulated-annealing engine
│   ├── lct.py        PRISM-LCT: light-cone + parallel tempering + tabu + consensus + polish
│   ├── kway.py       recursive k-way partition + fragment statistics
│   └── __init__.py   registry, dispatcher, run_all_partition_methods
├── benchmark.py    evaluate_partition, benchmark_instance, run_benchmark, summarise_benchmark
├── noise.py        depolarizing+readout model, exact noisy probabilities, noisy reconstruction
├── distributed.py  non-local CNOT/CZ by gate teleportation, exact distributed circuits, ebit accounting
├── qml.py          QAOA-MaxCut, QNNs (datasets, entangler topologies, distributed deployment),
│                   optimizer registry, Qiskit→layout bridge
└── cli.py          `prism-benchmark` entry point
```

## Partition methods

| Name | Kind | Idea |
|---|---|---|
| `Naive` | baseline | half-split |
| `Spectral` | baseline | Fiedler (2nd Laplacian eigenvector) bisection |
| `Louvain` | baseline | modularity communities merged to a bisection |
| `Girvan-Newman` | baseline | edge-betweenness hierarchical split |
| `METIS` | baseline | multilevel k-way bisection (pymetis) |
| `qdislib` | baseline | **DAG gate-cut** (the method in the qdislib library) |
| `PRISM-KL` | ladder | Kernighan–Lin on the gate-count graph |
| `PRISM-OE` | ladder | + operator-entanglement edge weighting + structural SA |
| `PRISM-MI` | ladder | + entanglement-entropy & mutual-information cost (SV-aware SA) |
| `PRISM-BF` | ladder | + boundary-focused move proposals |
| `PRISM-LCT` | **main** | + light-cone graph + parallel tempering + tabu + consensus + polish |

## Benchmark

```bash
# all methods on all families × qubit counts × seeds
prism-benchmark --families QAOA QNN MPS --qubits 6 8 10 --seeds 19 23 --out results/bench
# or, from a source checkout without installing:
python scripts/run_benchmark.py --families QAOA --qubits 8 10 --methods all
```

```python
from prism import run_benchmark, summarise_benchmark
df = run_benchmark(families=['QAOA', 'QNN'], qubits=[6, 8, 10], seeds=[19, 23])
print(summarise_benchmark(df))      # mean Q-Score, rank, runtime, win-count per method
```

## Notebooks

Under `notebooks/` (each ships executed at the laptop-friendly `SCALE='small'`; flip to `'full'` or set
`RECOMPUTE=True` to regenerate). Heavy cells use a **cache-or-compute** pattern: they load a cached CSV/JSON
if present, otherwise compute from scratch.

1. **`01_qaoa_maxcut_optimizers.ipynb`** — partitioned QAOA/MaxCut across all eleven classical optimizers
   *and* all eleven partition methods; recovered approximation ratio and cut-expectation error per method.
2. **`02_qnn_classification.ipynb`** — trainable QNNs on **Iris** (4 features → 4 qubits, angle) and
   **Digits** (all 64 pixels → 8 qubits, data re-uploading; plus a 6-qubit amplitude-encoded variant),
   **no feature reduction**; partition reconstruction + prediction-preservation across all methods.
3. **`03_prism_insights.ipynb`** — eight insight experiments: ablation ladder, light-cone value,
   symmetry-reduced QPD overhead, scaling, the deception dial (when PRISM is necessary), restart robustness,
   an honest count-blind limitation, and a "when does PRISM win?" feature model.
4. **`04_prism_in_practice.ipynb`** — headline case study: split one circuit across two QPUs, with fragment
   circuit diagrams, the interaction graph + cut, an ideal-vs-reconstructed distribution overlay, a
   per-method scorecard, and a cross-seed robustness check.
5. **`05_benchmark_scorecard.ipynb`** — the comprehensive results dashboard across all nine families:
   Q-Score leaderboard, family-stratified bars, per-metric box-plots, rank CDF + win-rate, and a
   family × method heatmap.
6. **`06_distributed_execution_payoff.ipynb`** — the resource value proposition: ebits/cbits per cut,
   QPD overhead γ and shot budget, maximum fragment width ("run a wide circuit on two small QPUs"), and
   savings versus the naive cut.
7. **`07_noise_resilience.ipynb`** — exact density-matrix noise sweep showing when partition+reconstruct
   beats the monolithic circuit as gate noise rises, and where the reconstruction floor caps the gain.
8. **`08_multi_qpu_kway.ipynb`** — multi-QPU (k-way) cutting: reconstruction quality, fragment width, and
   cut-gate growth as a circuit is split across 2, 3, 4 … QPUs, per method.
9. **`09_optimality_gap.ipynb`** — brute-force *every* balanced cut for small circuits, then measure each
   method's optimality gap and its percentile in the full cut landscape.
10. **`10_distributed_cnot_qaoa.ipynb`** — implement + verify the non-local CNOT (gate teleportation), build
    an exact distributed QAOA, and compare the ebit/teleport paradigm against PRISM cutting across all
    methods (ebits vs reconstruction-fidelity trade-off).
11. **`11_distributed_qnn_deployment.ipynb`** — train QNN classifiers on several datasets and *deploy them
    across QPUs*: reconstruction fidelity, prediction agreement, accuracy, and k-way splits — with the
    instructive distribution-vs-decision caveat.
12. **`12_qnn_ansatz_encoding_study.ipynb`** — which QNN designs are partition-friendly: entangler topology
    (`ring`/`linear`/`blocks`/`full`) × encoding (`angle`/re-uploading/`amplitude`) vs partitionability,
    entanglement, and the expressibility ↔ partitionability trade-off — concrete design guidance.

## Testing

```bash
pytest -q          # 25 tests: API surface, all families, all methods, reconstruction, qml, benchmark
```

## Notes & scope

- Exact statevector reconstruction is used throughout, so experiments target `n ≤ ~20` qubits
  (laptop-friendly defaults stay `≤ 14`). The structural surrogate cost extends to larger `n`.
- The framework is defined for bipartition (`k=2`), matching the paper.
- `prism.qml.qiskit_to_layout` converts arbitrary bound Qiskit circuits into PRISM layouts, so any circuit
  (including library ansätze and amplitude-encoded feature maps) can be partitioned by every method.

## License

MIT — see `LICENSE`.
