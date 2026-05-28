#!/usr/bin/env python3
"""
Bank Marketing conformal experiments for a chronological deployment-style study.

What this script does
---------------------
1. Loads a Bank Marketing CSV (semicolon-separated; preserves original row order).
2. Runs a single chronological split (train -> calibration -> test).
3. Evaluates:
   - SplitCP
   - Mondrian SplitCP
   - APS (allow-empty)
   - APS (+fallback, non-conformal)
   - Weighted SplitCP (classifier-based weighting)
4. Runs a rolling-window evaluation for local coverage over time.
5. Writes CSV tables and PNG figures to an output directory.

Recommended inputs
------------------
Use bank-full.csv or bank-additional-full.csv.
Do NOT use the 10% random subsets (bank.csv / bank-additional.csv) for the
chronological experiment, because they are randomly sampled and break the time-order logic.

Dependencies
------------
pandas, numpy, scikit-learn, matplotlib

Example
-------
python bank_conformal_experiments.py \
  --data-path /path/to/bank-full.csv \
  --outdir out_bank
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def finite_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal order statistic for split conformal.

    If scores are sorted ascending as S_(1) <= ... <= S_(m), return
    S_(ceil((m+1)(1-alpha))).
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    m = len(scores)
    if m == 0:
        raise ValueError("Calibration score array is empty.")
    k = int(math.ceil((m + 1) * (1.0 - alpha)))
    k = min(max(k, 1), m)
    return float(scores[k - 1])



def format_percent(x: float) -> str:
    return f"{100.0 * x:.2f}%"


# -----------------------------
# Data preparation
# -----------------------------


def load_bank_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # UCI Bank Marketing CSVs are semicolon-separated.
    df = pd.read_csv(path, sep=";")
    if "y" not in df.columns:
        raise ValueError("Expected a target column named 'y'.")

    # Preserve row order exactly as stored in the file.
    df = df.copy()
    df["y_bin"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
    return df



def get_feature_columns(df: pd.DataFrame, include_duration: bool) -> Tuple[List[str], List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in {"y", "y_bin"}]
    if not include_duration and "duration" in feature_cols:
        feature_cols.remove("duration")

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols



def make_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_cols),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_cols),
            ),
        ],
        remainder="drop",
    )



def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    cal_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or cal_frac <= 0 or train_frac + cal_frac >= 1:
        raise ValueError("Need 0 < train_frac, cal_frac and train_frac + cal_frac < 1.")

    n = len(df)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    train_df = df.iloc[:n_train].copy()
    cal_df = df.iloc[n_train : n_train + n_cal].copy()
    test_df = df.iloc[n_train + n_cal :].copy()
    return train_df, cal_df, test_df


# -----------------------------
# Base models
# -----------------------------


def build_classifier(model_name: str):
    model_name = model_name.upper()
    if model_name == "LR":
        return LogisticRegression(max_iter=2000)
    if model_name == "HGB":
        return HistGradientBoostingClassifier(random_state=42, max_iter=200, learning_rate=0.08)
    raise ValueError(f"Unsupported model: {model_name}")


# -----------------------------
# Conformal methods
# -----------------------------


@dataclass
class Metrics:
    coverage: float
    coverage_1: float
    coverage_0: float
    avg_size: float
    diagnostic: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "coverage": self.coverage,
            "coverage_1": self.coverage_1,
            "coverage_0": self.coverage_0,
            "avg_size": self.avg_size,
            "diagnostic": self.diagnostic,
        }



def summarize_sets(sets: np.ndarray, y_true: np.ndarray) -> Metrics:
    cover = sets[np.arange(len(y_true)), y_true]
    size = sets.sum(axis=1)
    return Metrics(
        coverage=float(cover.mean()),
        coverage_1=float(cover[y_true == 1].mean()),
        coverage_0=float(cover[y_true == 0].mean()),
        avg_size=float(size.mean()),
        diagnostic=float((size == 0).mean()),
    )



def splitcp_metrics(probs_cal: np.ndarray, y_cal: np.ndarray, probs_test: np.ndarray, y_test: np.ndarray, alpha: float) -> Metrics:
    probs_cal_true = probs_cal[np.arange(len(y_cal)), y_cal]
    q = finite_quantile(1.0 - probs_cal_true, alpha)
    threshold = 1.0 - q
    sets = (probs_test >= threshold).astype(int)
    return summarize_sets(sets, y_test)



def mondrian_metrics(probs_cal: np.ndarray, y_cal: np.ndarray, probs_test: np.ndarray, y_test: np.ndarray, alpha: float) -> Metrics:
    sets = np.zeros_like(probs_test, dtype=int)
    for label in [0, 1]:
        class_scores = 1.0 - probs_cal[y_cal == label, label]
        q_label = finite_quantile(class_scores, alpha)
        thr_label = 1.0 - q_label
        sets[:, label] = (probs_test[:, label] >= thr_label).astype(int)
    return summarize_sets(sets, y_test)



def aps_calibration_scores(probs: np.ndarray, y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, n_classes = probs.shape
    scores = np.empty(n, dtype=float)
    for i in range(n):
        p = probs[i]
        order = np.argsort(-p)
        rank_map = {label: rank for rank, label in enumerate(order)}
        y = int(y_true[i])
        rank = rank_map[y]
        higher = order[:rank]
        scores[i] = float(p[higher].sum() + rng.random() * p[y])
    return scores



def aps_prediction_sets(probs_test: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    n, n_classes = probs_test.shape
    sets = np.zeros((n, n_classes), dtype=int)
    for i in range(n):
        p = probs_test[i]
        order = np.argsort(-p)
        rank_map = {label: rank for rank, label in enumerate(order)}
        for y in range(n_classes):
            rank = rank_map[y]
            higher = order[:rank]
            score = float(p[higher].sum() + rng.random() * p[y])
            if score <= q:
                sets[i, y] = 1
    return sets



def aps_metrics(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    seed: int,
) -> Tuple[Metrics, Metrics]:
    rng_cal = np.random.default_rng(seed)
    cal_scores = aps_calibration_scores(probs_cal, y_cal, rng_cal)
    q = finite_quantile(cal_scores, alpha)

    rng_test = np.random.default_rng(seed + 1)
    raw_sets = aps_prediction_sets(probs_test, q, rng_test)
    raw_metrics = summarize_sets(raw_sets, y_test)

    fb_sets = raw_sets.copy()
    empty = fb_sets.sum(axis=1) == 0
    if empty.any():
        fb_sets[empty, np.argmax(probs_test[empty], axis=1)] = 1
    fallback_metrics = summarize_sets(fb_sets, y_test)
    # For APS(+fallback), the diagnostic should be the fallback trigger rate.
    fallback_metrics = Metrics(
        coverage=fallback_metrics.coverage,
        coverage_1=fallback_metrics.coverage_1,
        coverage_0=fallback_metrics.coverage_0,
        avg_size=fallback_metrics.avg_size,
        diagnostic=float(empty.mean()),
    )
    return raw_metrics, fallback_metrics


# -----------------------------
# Classifier-based weighting
# -----------------------------


def estimate_domain_weights(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    eval_source_df: pd.DataFrame,
    eval_target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate density-ratio-like weights via a domain classifier.

    source = earlier data (train + calibration)
    target = later data (test)
    """
    x_dom = pd.concat([source_df[list(feature_cols)], target_df[list(feature_cols)]], axis=0)
    y_dom = np.concatenate(
        [np.zeros(len(source_df), dtype=int), np.ones(len(target_df), dtype=int)]
    )

    dom_pipeline = Pipeline(
        steps=[
            ("pre", make_preprocessor(numeric_cols, categorical_cols)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    dom_pipeline.fit(x_dom, y_dom)

    p_source = np.clip(dom_pipeline.predict_proba(eval_source_df[list(feature_cols)])[:, 1], 1e-4, 1 - 1e-4)
    p_target = np.clip(dom_pipeline.predict_proba(eval_target_df[list(feature_cols)])[:, 1], 1e-4, 1 - 1e-4)

    prior_ratio = len(source_df) / len(target_df)
    w_source = (p_source / (1.0 - p_source)) * prior_ratio
    w_target = (p_target / (1.0 - p_target)) * prior_ratio
    return w_source, w_target



def weighted_threshold(cal_scores: np.ndarray, w_cal: np.ndarray, w_test_scalar: float, alpha: float) -> float:
    order = np.argsort(cal_scores)
    s = cal_scores[order]
    w = w_cal[order]
    total_weight = float(w.sum() + w_test_scalar)
    cum = np.cumsum(w) / total_weight
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    if idx >= len(s):
        return math.inf
    return float(s[idx])



def weighted_splitcp_metrics(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    w_cal: np.ndarray,
    w_test: np.ndarray,
    alpha: float,
) -> Metrics:
    probs_cal_true = probs_cal[np.arange(len(y_cal)), y_cal]
    cal_scores = 1.0 - probs_cal_true
    q_test = np.array([weighted_threshold(cal_scores, w_cal, wt, alpha) for wt in w_test])
    thresholds = 1.0 - q_test
    sets = (probs_test >= thresholds[:, None]).astype(int)
    return summarize_sets(sets, y_test)


# -----------------------------
# Evaluation helpers
# -----------------------------


def evaluate_single_split(
    df: pd.DataFrame,
    include_duration: bool,
    model_names: Sequence[str],
    alphas: Sequence[float],
    seed: int,
    train_frac: float,
    cal_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(df, include_duration)
    train_df, cal_df, test_df = chronological_split(df, train_frac=train_frac, cal_frac=cal_frac)

    source_df = pd.concat([train_df, cal_df], axis=0)
    w_cal, w_test = estimate_domain_weights(
        source_df=source_df,
        target_df=test_df,
        eval_source_df=cal_df,
        eval_target_df=test_df,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    pre = make_preprocessor(numeric_cols, categorical_cols)
    x_train = pre.fit_transform(train_df[feature_cols])
    x_cal = pre.transform(cal_df[feature_cols])
    x_test = pre.transform(test_df[feature_cols])
    y_train = train_df["y_bin"].to_numpy()
    y_cal = cal_df["y_bin"].to_numpy()
    y_test = test_df["y_bin"].to_numpy()

    rows: List[Dict[str, float | str]] = []
    auc_rows: List[Dict[str, float | str]] = []

    setting_name = "with_duration" if include_duration else "without_duration"

    for model_name in model_names:
        clf = build_classifier(model_name)
        clf.fit(x_train, y_train)
        probs_cal = clf.predict_proba(x_cal)
        probs_test = clf.predict_proba(x_test)
        auc_rows.append(
            {
                "setting": setting_name,
                "model": model_name,
                "test_auc": float(roc_auc_score(y_test, probs_test[:, 1])),
            }
        )

        for alpha in alphas:
            rows.append(
                {
                    "setting": setting_name,
                    "model": model_name,
                    "alpha": alpha,
                    "method": "SplitCP",
                    **splitcp_metrics(probs_cal, y_cal, probs_test, y_test, alpha).as_dict(),
                }
            )
            rows.append(
                {
                    "setting": setting_name,
                    "model": model_name,
                    "alpha": alpha,
                    "method": "Mondrian",
                    **mondrian_metrics(probs_cal, y_cal, probs_test, y_test, alpha).as_dict(),
                }
            )
            aps_raw, aps_fb = aps_metrics(
                probs_cal=probs_cal,
                y_cal=y_cal,
                probs_test=probs_test,
                y_test=y_test,
                alpha=alpha,
                seed=seed,
            )
            rows.append(
                {
                    "setting": setting_name,
                    "model": model_name,
                    "alpha": alpha,
                    "method": "APS (allow-empty)",
                    **aps_raw.as_dict(),
                }
            )
            rows.append(
                {
                    "setting": setting_name,
                    "model": model_name,
                    "alpha": alpha,
                    "method": "APS (+fallback)",
                    **aps_fb.as_dict(),
                }
            )
            rows.append(
                {
                    "setting": setting_name,
                    "model": model_name,
                    "alpha": alpha,
                    "method": "Weighted SplitCP",
                    **weighted_splitcp_metrics(
                        probs_cal=probs_cal,
                        y_cal=y_cal,
                        probs_test=probs_test,
                        y_test=y_test,
                        w_cal=w_cal,
                        w_test=w_test,
                        alpha=alpha,
                    ).as_dict(),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(auc_rows)



def rolling_windows(
    n: int,
    train_frac: float,
    cal_frac: float,
    test_frac: float,
    step_frac: float,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Return windows as (train_start, train_end, cal_start, cal_end, test_start, test_end)."""
    train_len = int(n * train_frac)
    cal_len = int(n * cal_frac)
    test_len = int(n * test_frac)
    step_len = max(1, int(n * step_frac))

    total_len = train_len + cal_len + test_len
    if total_len >= n:
        raise ValueError("Rolling window fractions are too large for the dataset length.")

    windows = []
    start = 0
    while start + total_len <= n:
        train_start = start
        train_end = start + train_len
        cal_start = train_end
        cal_end = cal_start + cal_len
        test_start = cal_end
        test_end = test_start + test_len
        windows.append((train_start, train_end, cal_start, cal_end, test_start, test_end))
        start += step_len
    return windows



def evaluate_rolling_windows(
    df: pd.DataFrame,
    include_duration: bool,
    model_names: Sequence[str],
    alphas: Sequence[float],
    seed: int,
    rolling_train_frac: float,
    rolling_cal_frac: float,
    rolling_test_frac: float,
    rolling_step_frac: float,
) -> pd.DataFrame:
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(df, include_duration)
    windows = rolling_windows(
        n=len(df),
        train_frac=rolling_train_frac,
        cal_frac=rolling_cal_frac,
        test_frac=rolling_test_frac,
        step_frac=rolling_step_frac,
    )

    rows: List[Dict[str, float | str | int]] = []
    setting_name = "with_duration" if include_duration else "without_duration"

    for window_id, (tr0, tr1, ca0, ca1, te0, te1) in enumerate(windows, start=1):
        train_df = df.iloc[tr0:tr1].copy()
        cal_df = df.iloc[ca0:ca1].copy()
        test_df = df.iloc[te0:te1].copy()
        source_df = pd.concat([train_df, cal_df], axis=0)

        w_cal, w_test = estimate_domain_weights(
            source_df=source_df,
            target_df=test_df,
            eval_source_df=cal_df,
            eval_target_df=test_df,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        pre = make_preprocessor(numeric_cols, categorical_cols)
        x_train = pre.fit_transform(train_df[feature_cols])
        x_cal = pre.transform(cal_df[feature_cols])
        x_test = pre.transform(test_df[feature_cols])
        y_train = train_df["y_bin"].to_numpy()
        y_cal = cal_df["y_bin"].to_numpy()
        y_test = test_df["y_bin"].to_numpy()

        for model_name in model_names:
            clf = build_classifier(model_name)
            clf.fit(x_train, y_train)
            probs_cal = clf.predict_proba(x_cal)
            probs_test = clf.predict_proba(x_test)

            for alpha in alphas:
                methods = {
                    "SplitCP": splitcp_metrics(probs_cal, y_cal, probs_test, y_test, alpha),
                    "Mondrian": mondrian_metrics(probs_cal, y_cal, probs_test, y_test, alpha),
                    "Weighted SplitCP": weighted_splitcp_metrics(
                        probs_cal=probs_cal,
                        y_cal=y_cal,
                        probs_test=probs_test,
                        y_test=y_test,
                        w_cal=w_cal,
                        w_test=w_test,
                        alpha=alpha,
                    ),
                }
                aps_raw, aps_fb = aps_metrics(
                    probs_cal=probs_cal,
                    y_cal=y_cal,
                    probs_test=probs_test,
                    y_test=y_test,
                    alpha=alpha,
                    seed=seed + window_id,
                )
                methods["APS (allow-empty)"] = aps_raw
                methods["APS (+fallback)"] = aps_fb

                for method_name, metric in methods.items():
                    rows.append(
                        {
                            "setting": setting_name,
                            "window_id": window_id,
                            "model": model_name,
                            "alpha": alpha,
                            "method": method_name,
                            "test_start_row": te0,
                            "test_end_row": te1,
                            **metric.as_dict(),
                        }
                    )

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------


def plot_local_coverage(rolling_df: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)
    targets = sorted(rolling_df["alpha"].unique())
    models = sorted(rolling_df["model"].unique())

    for model in models:
        for alpha in targets:
            subset = rolling_df[(rolling_df["model"] == model) & (rolling_df["alpha"] == alpha)]
            if subset.empty:
                continue

            plt.figure(figsize=(8, 4.8))
            for method in ["SplitCP", "Mondrian", "APS (allow-empty)", "Weighted SplitCP"]:
                m = subset[subset["method"] == method].sort_values("window_id")
                if m.empty:
                    continue
                plt.plot(m["window_id"], m["coverage"], marker="o", label=method)

            plt.axhline(1.0 - alpha, linestyle="--", linewidth=1.2, label="target 1-alpha")
            plt.xlabel("Rolling window")
            plt.ylabel("Local coverage")
            plt.title(f"Local coverage over time - {model} - alpha={alpha:.2f}")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(outdir / f"local_coverage_{model}_alpha_{alpha:.2f}.png", dpi=180)
            plt.close()



def plot_set_size(rolling_df: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)
    targets = sorted(rolling_df["alpha"].unique())
    models = sorted(rolling_df["model"].unique())

    for model in models:
        for alpha in targets:
            subset = rolling_df[(rolling_df["model"] == model) & (rolling_df["alpha"] == alpha)]
            if subset.empty:
                continue

            plt.figure(figsize=(8, 4.8))
            for method in ["SplitCP", "Mondrian", "APS (allow-empty)", "Weighted SplitCP"]:
                m = subset[subset["method"] == method].sort_values("window_id")
                if m.empty:
                    continue
                plt.plot(m["window_id"], m["avg_size"], marker="o", label=method)

            plt.xlabel("Rolling window")
            plt.ylabel("Average set size")
            plt.title(f"Set size over time - {model} - alpha={alpha:.2f}")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(outdir / f"set_size_{model}_alpha_{alpha:.2f}.png", dpi=180)
            plt.close()


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bank Marketing conformal experiments")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to bank-full.csv or bank-additional-full.csv")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--models", nargs="+", default=["LR", "HGB"], help="Base models to run: LR HGB")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20], help="Miscoverage levels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--rolling-train-frac", type=float, default=0.50)
    parser.add_argument("--rolling-cal-frac", type=float, default=0.20)
    parser.add_argument("--rolling-test-frac", type=float, default=0.10)
    parser.add_argument("--rolling-step-frac", type=float, default=0.05)
    parser.add_argument(
        "--rolling-include-duration",
        action="store_true",
        help="Also run rolling windows with duration included. By default rolling windows use duration excluded only.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    ensure_dir(args.outdir / "figures")

    df = load_bank_csv(args.data_path)

    # Save a compact dataset summary.
    summary = {
        "n_rows": int(len(df)),
        "n_columns_including_target": int(df.shape[1] - 1),  # minus y_bin helper column
        "positive_rate_yes": float(df["y_bin"].mean()),
        "negative_rate_no": float(1.0 - df["y_bin"].mean()),
        "has_duration": bool("duration" in df.columns),
    }
    pd.DataFrame([summary]).to_csv(args.outdir / "dataset_summary.csv", index=False)

    # Single chronological split: both settings.
    all_single = []
    all_auc = []
    for include_duration in [False, True]:
        single_df, auc_df = evaluate_single_split(
            df=df,
            include_duration=include_duration,
            model_names=args.models,
            alphas=args.alphas,
            seed=args.seed,
            train_frac=args.train_frac,
            cal_frac=args.cal_frac,
        )
        all_single.append(single_df)
        all_auc.append(auc_df)

    single_out = pd.concat(all_single, axis=0, ignore_index=True)
    auc_out = pd.concat(all_auc, axis=0, ignore_index=True)
    single_out.to_csv(args.outdir / "single_split_results.csv", index=False)
    auc_out.to_csv(args.outdir / "single_split_auc.csv", index=False)

    # Add two derived metrics that are convenient for the report.
    single_aug = single_out.copy()
    single_aug["target_coverage"] = 1.0 - single_aug["alpha"]
    single_aug["coverage_gap"] = single_aug["coverage"] - single_aug["target_coverage"]
    single_aug["class_disparity_gap"] = (single_aug["coverage_1"] - single_aug["coverage_0"]).abs()
    single_aug.to_csv(args.outdir / "single_split_results_with_gaps.csv", index=False)

    # Rolling windows: duration excluded by default.
    rolling_dfs = []
    rolling_dfs.append(
        evaluate_rolling_windows(
            df=df,
            include_duration=False,
            model_names=args.models,
            alphas=args.alphas,
            seed=args.seed,
            rolling_train_frac=args.rolling_train_frac,
            rolling_cal_frac=args.rolling_cal_frac,
            rolling_test_frac=args.rolling_test_frac,
            rolling_step_frac=args.rolling_step_frac,
        )
    )
    if args.rolling_include_duration:
        rolling_dfs.append(
            evaluate_rolling_windows(
                df=df,
                include_duration=True,
                model_names=args.models,
                alphas=args.alphas,
                seed=args.seed,
                rolling_train_frac=args.rolling_train_frac,
                rolling_cal_frac=args.rolling_cal_frac,
                rolling_test_frac=args.rolling_test_frac,
                rolling_step_frac=args.rolling_step_frac,
            )
        )

    rolling_out = pd.concat(rolling_dfs, axis=0, ignore_index=True)
    rolling_out.to_csv(args.outdir / "rolling_window_results.csv", index=False)

    rolling_aug = rolling_out.copy()
    rolling_aug["target_coverage"] = 1.0 - rolling_aug["alpha"]
    rolling_aug["coverage_gap"] = rolling_aug["coverage"] - rolling_aug["target_coverage"]
    rolling_aug["class_disparity_gap"] = (rolling_aug["coverage_1"] - rolling_aug["coverage_0"]).abs()
    rolling_aug.to_csv(args.outdir / "rolling_window_results_with_gaps.csv", index=False)

    # Compact summaries for the write-up.
    rolling_summary = (
        rolling_aug.groupby(["setting", "model", "alpha", "method"], as_index=False)
        .agg(
            mean_local_coverage=("coverage", "mean"),
            min_local_coverage=("coverage", "min"),
            max_local_coverage=("coverage", "max"),
            mean_avg_size=("avg_size", "mean"),
            mean_empty_or_trigger=("diagnostic", "mean"),
            mean_coverage_gap=("coverage_gap", "mean"),
            mean_class_disparity_gap=("class_disparity_gap", "mean"),
        )
    )
    rolling_summary.to_csv(args.outdir / "rolling_window_summary.csv", index=False)

    plot_local_coverage(rolling_aug[rolling_aug["setting"] == "without_duration"], args.outdir / "figures")
    plot_set_size(rolling_aug[rolling_aug["setting"] == "without_duration"], args.outdir / "figures")

    # A tiny run log for convenience.
    with open(args.outdir / "run_log.txt", "w", encoding="utf-8") as f:
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"rows: {len(df)}\n")
        f.write(f"positive_rate_yes: {df['y_bin'].mean():.6f}\n")
        f.write(f"models: {', '.join(args.models)}\n")
        f.write(f"alphas: {args.alphas}\n")
        f.write("single split settings: without_duration, with_duration\n")
        f.write("rolling windows: without_duration")
        if args.rolling_include_duration:
            f.write(", with_duration")
        f.write("\n")

    print("Done.")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
