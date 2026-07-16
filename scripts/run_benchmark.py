#!/usr/bin/env python3
"""
run_benchmark.py
================
Runnable benchmark script for all partition methods on all circuit families.

Works without installing the package (it puts ``src`` on the path). Examples::

    python scripts/run_benchmark.py                         # default grid
    python scripts/run_benchmark.py --families QAOA QNN --qubits 6 8 10
    python scripts/run_benchmark.py --methods all --out results/bench

If the package is installed (``pip install -e .``), the console command
``prism-benchmark`` does the same thing.
"""
import os
import sys

# Make the package importable from a source checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from prism.cli import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
