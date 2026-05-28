#!/usr/bin/env python3
"""
Convenience runner for the main experiments.

This script assumes raw datasets have been placed under data/:

    data/wdbc.data
    data/bank-full.csv
    data/bank-additional-full.csv

It runs the main reproducibility commands used in the README.
For faster testing, run individual scripts instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([
        PYTHON,
        "wdbc_conformal_2x2.py",
        "--data-path",
        "data/wdbc.data",
        "--outdir",
        "out_wdbc",
        "--overwrite",
    ])

    run([
        PYTHON,
        "wdbc_synthetic_shift_experiment.py",
        "--data-path",
        "data/wdbc.data",
        "--outdir",
        "out_wdbc_synth",
    ])

    run([
        PYTHON,
        "bank_conformal_experiments.py",
        "--data-path",
        "data/bank-full.csv",
        "--outdir",
        "out_bank",
    ])

    run([
        PYTHON,
        "bank_conformal_experiments.py",
        "--data-path",
        "data/bank-additional-full.csv",
        "--outdir",
        "out_bank_additional",
    ])

    run([
        PYTHON,
        "run_robustness_scan.py",
        "--data-path",
        "data/bank-full.csv",
        "--outdir",
        "out_bank_robustness",
    ])

    run([
        PYTHON,
        "twm_bank_experiment.py",
        "--data-path",
        "data/bank-full.csv",
        "--dataset-name",
        "orig_bank",
        "--outdir",
        "out_twm_bank",
    ])


if __name__ == "__main__":
    main()