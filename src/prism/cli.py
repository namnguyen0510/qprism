"""
prism.cli
=========
Command-line entry point for the PRISM benchmark.

    prism-benchmark --families QAOA QNN MPS --qubits 6 8 10 --seeds 19 23 \
                    --methods all --out results/bench

or, without installation::

    python -m prism.cli --families QAOA --qubits 8 --seeds 19
"""
from __future__ import annotations

import argparse
import sys

from .benchmark import (run_benchmark, summarise_benchmark,
                        DEFAULT_FAMILIES, DEFAULT_QUBITS, DEFAULT_SEEDS)
from .partition import METHOD_ORDER
from .circuits import CIRCUIT_FAMILIES


def build_parser():
    p = argparse.ArgumentParser(
        prog='prism-benchmark',
        description='Benchmark all circuit-partition methods on all circuit families.')
    p.add_argument('--families', nargs='+', default=DEFAULT_FAMILIES,
                   help=f'circuit families (default: {DEFAULT_FAMILIES}); '
                        f'available: {list(CIRCUIT_FAMILIES) + ["RQC"]}')
    p.add_argument('--qubits', nargs='+', type=int, default=DEFAULT_QUBITS,
                   help=f'qubit counts (default: {DEFAULT_QUBITS})')
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                   help=f'random seeds (default: {DEFAULT_SEEDS})')
    p.add_argument('--methods', nargs='+', default=None,
                   help=f'subset of methods, or "all" (default: all). '
                        f'Available: {METHOD_ORDER}')
    p.add_argument('--out', default=None, help='output path prefix for results CSV')
    p.add_argument('--no-sim', action='store_true',
                   help='skip statevector reconstruction (structural metrics only)')
    p.add_argument('--quiet', action='store_true', help='suppress progress output')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    methods = None
    if args.methods and args.methods != ['all']:
        methods = args.methods

    if not args.quiet:
        print(f'PRISM benchmark | families={args.families} qubits={args.qubits} '
              f'seeds={args.seeds} methods={"all" if methods is None else methods}')

    df = run_benchmark(families=args.families, qubits=args.qubits, seeds=args.seeds,
                       methods=methods, simulate=not args.no_sim, out=args.out,
                       verbose=not args.quiet, as_dataframe=True)

    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame) and not df.empty:
            print('\n=== per-method summary (mean over all instances) ===')
            summary = summarise_benchmark(df)
            with pd.option_context('display.width', 140, 'display.max_columns', 20):
                print(summary.round(4).to_string())
    except ImportError:
        print(f'\n{len(df)} rows collected (install pandas for the summary table).')

    if args.out:
        print(f'\nresults written with prefix: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
