#!/usr/bin/env python3
"""
Wrapper that runs the 3x10 robustness scan reported for Table 4.

It calls evaluate_single_split() from bank_conformal_experiments.py over
train_frac in {0.55, 0.60, 0.65} x seed values 0..9, with alpha=0.10 and
model_names=["LR", "HGB"]. Only HGB changes with seed; LR-based methods are
otherwise deterministic except for APS smoothing.

Usage example:
    python run_robustness_scan.py \
      --data-path /path/to/bank-full.csv \
      --outdir /path/to/out_bank_robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import bank_conformal_experiments as bce

TRAIN_FRACS = [0.55, 0.60, 0.65]
SEEDS = list(range(10))
CAL_FRAC = 0.20
ALPHA = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3x10 robustness scan for Original Bank.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to bank-full.csv")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out_bank_robustness"),
        help="Directory to save raw and aggregated scan outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = bce.load_bank_csv(args.data_path)
    all_rows = []

    original_build = bce.build_classifier

    for train_frac in TRAIN_FRACS:
        for seed in SEEDS:
            def seeded_build(name: str):
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression

                name = name.upper()
                if name == "HGB":
                    return HistGradientBoostingClassifier(
                        random_state=seed,
                        max_iter=200,
                        learning_rate=0.08,
                    )
                if name == "LR":
                    return LogisticRegression(max_iter=2000)
                raise ValueError(f"Unsupported model: {name}")

            bce.build_classifier = seeded_build
            try:
                outputs = bce.evaluate_single_split(
                    df=df,
                    include_duration=False,
                    model_names=["LR", "HGB"],
                    alphas=[ALPHA],
                    seed=seed,
                    train_frac=train_frac,
                    cal_frac=CAL_FRAC,
                )
            finally:
                bce.build_classifier = original_build

            single_df = outputs[0]
            single_df["train_frac"] = train_frac
            single_df["seed"] = seed
            all_rows.append(single_df)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(args.outdir / "robustness_scan_raw.csv", index=False)

    agg = (
        raw.groupby(["model", "method"], as_index=False)
        .agg(
            cov_mean=("coverage", "mean"),
            cov_std=("coverage", "std"),
            cov1_mean=("coverage_1", "mean"),
            cov1_std=("coverage_1", "std"),
            cov0_mean=("coverage_0", "mean"),
            cov0_std=("coverage_0", "std"),
            size_mean=("avg_size", "mean"),
            size_std=("avg_size", "std"),
        )
    )
    agg.to_csv(args.outdir / "robustness_scan_summary.csv", index=False)

    print(agg.to_string(index=False))
    print(f"\nRaw: {args.outdir / 'robustness_scan_raw.csv'}")
    print(f"Summary: {args.outdir / 'robustness_scan_summary.csv'}")


if __name__ == "__main__":
    main()
