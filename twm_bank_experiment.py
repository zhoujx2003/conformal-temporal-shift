#!/usr/bin/env python3
r"""
Temporally Weighted Mondrian Split Conformal Prediction (TWM-SplitCP)
for Bank Marketing and Bank Additional Marketing.

Main comparisons:
- SplitCP
- Mondrian SplitCP
- Classifier-based Weighted SplitCP
- TWM-SplitCP (class-conditional + explicit temporal weighting)

Design choices for report-grade evaluation:
- Chronological train/cal/test splits (no shuffling)
- Optional exclusion of duration in the main analysis
- Rolling-window evaluation with window-level t*
- Pre-specified lambda grid (no test-label tuning)
- Effective sample size reporting for TWM-SplitCP
- Safeguard fallback to unweighted Mondrian when class-specific ESS gets too small

Usage examples (Windows paths):

python twm_bank_experiment.py ^
  --data-path "C:\Users\a9281\OneDrive\Desktop\bank-full.csv" ^
  --dataset-name orig_bank ^
  --outdir "C:\Users\a9281\OneDrive\Desktop\out_twm_bank" ^
  --models LR HGB ^
  --alphas 0.05 0.10 0.20 ^
  --lambda-grid 0 1 2 5 ^
  --report-lambda 2 ^
  --rolling-windows 5

python twm_bank_experiment.py ^
  --data-path "C:\Users\a9281\OneDrive\Desktop\bank-additional-full.csv" ^
  --dataset-name add_bank ^
  --outdir "C:\Users\a9281\OneDrive\Desktop\out_twm_bank_additional" ^
  --models LR HGB ^
  --alphas 0.05 0.10 0.20 ^
  --lambda-grid 0 1 2 5 ^
  --report-lambda 2 ^
  --rolling-windows 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


RANDOM_SEED = 42
EPS = 1e-12


@dataclass
class SplitConfig:
    train_frac: float = 0.60
    cal_frac: float = 0.20


@dataclass
class MetricRow:
    dataset: str
    eval_type: str  # single_split or rolling
    window: Optional[int]
    model: str
    method: str
    alpha: float
    lambda_: Optional[float]
    include_duration: bool
    coverage: float
    cov_1: float
    cov_0: float
    avg_size: float
    gap: float
    disparity: float
    n_test: int
    positive_rate_test: float
    auc: float
    ess_y1: Optional[float] = None
    ess_y0: Optional[float] = None
    twm_fallback_y1: Optional[bool] = None
    twm_fallback_y0: Optional[bool] = None
    t_star: Optional[float] = None


@dataclass
class RollingSummaryRow:
    dataset: str
    model: str
    method: str
    alpha: float
    lambda_: Optional[float]
    include_duration: bool
    mean_local_cov: float
    min_local_cov: float
    max_local_cov: float
    mean_cov1: float
    min_cov1: float
    mean_size: float
    mean_gap: float
    mean_disparity: float
    mean_auc: float
    mean_ess_y1: Optional[float] = None
    min_ess_y1: Optional[float] = None
    mean_ess_y0: Optional[float] = None
    min_ess_y0: Optional[float] = None


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------


def infer_csv_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    # UCI bank files are semicolon-separated and quoted.
    if first.count(";") > first.count(","):
        return ";"
    return ","


def load_bank_csv(path: str) -> pd.DataFrame:
    sep = infer_csv_sep(path)
    df = pd.read_csv(path, sep=sep)
    if "y" not in df.columns:
        raise ValueError("Expected a target column named 'y'.")
    return df


def prepare_xy(df: pd.DataFrame, include_duration: bool) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = df.copy()
    out["__time_index__"] = np.arange(len(out), dtype=float)
    y = (out["y"].astype(str).str.lower() == "yes").astype(int).to_numpy()
    time_index = out["__time_index__"].to_numpy()

    feature_cols = [c for c in out.columns if c not in ["y", "__time_index__"]]
    if not include_duration and "duration" in feature_cols:
        feature_cols.remove("duration")

    X = out[feature_cols].copy()
    return X, y, time_index


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def build_model(model_name: str) -> BaseEstimator:
    model_name = model_name.upper()
    if model_name == "LR":
        return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED)
    if model_name == "HGB":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=None,
            max_iter=300,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown model: {model_name}")


def fit_base_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
) -> Tuple[Pipeline, np.ndarray]:
    pre = build_preprocessor(X_train)
    model = build_model(model_name)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    train_proba = pipe.predict_proba(X_train)[:, 1]
    return pipe, train_proba


# -----------------------------------------------------------------------------
# Conformal helpers
# -----------------------------------------------------------------------------


def nonconformity_scores(y_true: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """s(x,y)=1-p_hat(y|x) evaluated at true labels."""
    p_true = np.where(y_true == 1, p1, 1.0 - p1)
    return 1.0 - np.clip(p_true, EPS, 1.0 - EPS)


def k_order_threshold(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    m = len(scores)
    if m == 0:
        raise ValueError("No calibration scores available.")
    k = int(math.ceil((m + 1) * (1.0 - alpha)))
    k = min(max(k, 1), m)
    return float(scores[k - 1])


def weighted_quantile_threshold(scores: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(scores) == 0:
        raise ValueError("No calibration scores available.")
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    cumw = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cumw, 1.0 - alpha, side="left")
    idx = min(idx, len(s) - 1)
    return float(s[idx])


def prediction_set_sizes_from_thresholds(
    p1_test: np.ndarray,
    q_global: Optional[float] = None,
    q_class: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    p1_test = np.asarray(p1_test, dtype=float)
    if q_class is not None:
        include_1 = (1.0 - p1_test) <= q_class[1]
        include_0 = p1_test <= q_class[0]
    elif q_global is not None:
        include_1 = (1.0 - p1_test) <= q_global
        include_0 = p1_test <= q_global
    else:
        raise ValueError("Either q_global or q_class must be provided.")
    return include_0.astype(int) + include_1.astype(int)


def coverage_from_thresholds(
    y_test: np.ndarray,
    p1_test: np.ndarray,
    q_global: Optional[float] = None,
    q_class: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1_test = np.asarray(p1_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)
    if q_class is not None:
        include_true = np.where(
            y_test == 1,
            (1.0 - p1_test) <= q_class[1],
            p1_test <= q_class[0],
        )
    elif q_global is not None:
        include_true = np.where(
            y_test == 1,
            (1.0 - p1_test) <= q_global,
            p1_test <= q_global,
        )
    else:
        raise ValueError("Either q_global or q_class must be provided.")

    set_sizes = prediction_set_sizes_from_thresholds(p1_test, q_global=q_global, q_class=q_class)
    return include_true.astype(bool), set_sizes, p1_test


def summarize_metrics(
    y_test: np.ndarray,
    covered: np.ndarray,
    set_sizes: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    y_test = np.asarray(y_test, dtype=int)
    covered = np.asarray(covered, dtype=bool)
    set_sizes = np.asarray(set_sizes, dtype=float)

    cov = float(np.mean(covered))
    mask1 = y_test == 1
    mask0 = y_test == 0
    cov1 = float(np.mean(covered[mask1])) if mask1.any() else float("nan")
    cov0 = float(np.mean(covered[mask0])) if mask0.any() else float("nan")
    avg_size = float(np.mean(set_sizes))
    gap = cov - (1.0 - alpha)
    disparity = abs(cov1 - cov0) if (mask1.any() and mask0.any()) else float("nan")
    return {
        "coverage": cov,
        "cov_1": cov1,
        "cov_0": cov0,
        "avg_size": avg_size,
        "gap": gap,
        "disparity": disparity,
    }


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    sw = float(np.sum(w))
    s2 = float(np.sum(w ** 2))
    if sw <= 0 or s2 <= 0:
        return 0.0
    return sw * sw / s2


# -----------------------------------------------------------------------------
# Weighting schemes
# -----------------------------------------------------------------------------


def domain_ratio_weights(
    X_cal: pd.DataFrame,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Classifier-based density-ratio-like weights.
    Uses only unlabeled X from calibration and test blocks.
    """
    X_dom = pd.concat([X_cal, X_test], axis=0, ignore_index=True)
    z = np.concatenate([np.zeros(len(X_cal), dtype=int), np.ones(len(X_test), dtype=int)])

    pre = build_preprocessor(X_dom)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_dom, z)

    p_test_like = pipe.predict_proba(X_cal)[:, 1]
    p_test_like = np.clip(p_test_like, 1e-4, 1 - 1e-4)
    # Approximate density ratio with prior correction.
    prior_ratio = len(X_cal) / max(len(X_test), 1)
    w = (p_test_like / (1.0 - p_test_like)) * prior_ratio
    w = np.clip(w, 1e-4, np.quantile(w, 0.99))
    w = w / np.mean(w)
    return w


def temporal_decay_weights(
    cal_times: np.ndarray,
    t_star: float,
    lam: float,
) -> np.ndarray:
    """
    Exponential time-decay weights using a normalised time gap.
    When lam=0, all weights are equal and the method reduces to Mondrian-style calibration.
    """
    cal_times = np.asarray(cal_times, dtype=float)
    if lam <= 0:
        return np.ones_like(cal_times, dtype=float)
    denom = max(float(t_star - np.min(cal_times)), 1.0)
    gap = (t_star - cal_times) / denom
    w = np.exp(-lam * gap)
    return w / np.mean(w)


# -----------------------------------------------------------------------------
# Experiment blocks
# -----------------------------------------------------------------------------


def evaluate_single_split(
    X: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    dataset_name: str,
    include_duration: bool,
    model_name: str,
    alphas: Sequence[float],
    lambda_grid: Sequence[float],
    split_cfg: SplitConfig,
    tstar_mode: str,
    min_eff_n: float,
) -> List[MetricRow]:
    n = len(X)
    n_train = int(split_cfg.train_frac * n)
    n_cal = int(split_cfg.cal_frac * n)
    idx_train = np.arange(0, n_train)
    idx_cal = np.arange(n_train, n_train + n_cal)
    idx_test = np.arange(n_train + n_cal, n)

    X_train, y_train = X.iloc[idx_train], y[idx_train]
    X_cal, y_cal, t_cal = X.iloc[idx_cal], y[idx_cal], t[idx_cal]
    X_test, y_test, t_test = X.iloc[idx_test], y[idx_test], t[idx_test]

    pipe, train_p = fit_base_model(X_train, y_train, model_name)
    p1_cal = pipe.predict_proba(X_cal)[:, 1]
    p1_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p1_test)

    scores_cal = nonconformity_scores(y_cal, p1_cal)
    tw = domain_ratio_weights(X_cal, X_test)
    if tstar_mode == "start":
        t_star = float(t_test.min())
    elif tstar_mode == "mid":
        t_star = float((t_test.min() + t_test.max()) / 2.0)
    else:
        t_star = float(t_test.max())

    rows: List[MetricRow] = []
    for alpha in alphas:
        # SplitCP
        q = k_order_threshold(scores_cal, alpha)
        covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_global=q)
        metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
        rows.append(MetricRow(dataset_name, "single_split", None, model_name, "SplitCP", alpha, None,
                              include_duration, **metrics, n_test=len(y_test),
                              positive_rate_test=float(np.mean(y_test)), auc=float(auc)))

        # Mondrian
        q_class = {
            cls: k_order_threshold(scores_cal[y_cal == cls], alpha)
            for cls in [0, 1]
        }
        covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_class=q_class)
        metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
        rows.append(MetricRow(dataset_name, "single_split", None, model_name, "Mondrian SplitCP", alpha, None,
                              include_duration, **metrics, n_test=len(y_test),
                              positive_rate_test=float(np.mean(y_test)), auc=float(auc)))

        # Classifier-weighted SplitCP
        q_w = weighted_quantile_threshold(scores_cal, tw, alpha)
        covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_global=q_w)
        metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
        rows.append(MetricRow(dataset_name, "single_split", None, model_name, "Weighted SplitCP", alpha, None,
                              include_duration, **metrics, n_test=len(y_test),
                              positive_rate_test=float(np.mean(y_test)), auc=float(auc)))

        # TWM-SplitCP sensitivity grid
        for lam in lambda_grid:
            q_class_twm = {}
            ess = {}
            fallback = {}
            for cls in [0, 1]:
                mask = y_cal == cls
                s_cls = scores_cal[mask]
                t_cls = t_cal[mask]
                w_cls = temporal_decay_weights(t_cls, t_star=t_star, lam=lam)
                ess_cls = effective_sample_size(w_cls)
                ess[cls] = ess_cls
                if ess_cls < min_eff_n:
                    q_cls = k_order_threshold(s_cls, alpha)
                    fallback[cls] = True
                else:
                    q_cls = weighted_quantile_threshold(s_cls, w_cls, alpha)
                    fallback[cls] = False
                q_class_twm[cls] = q_cls
            covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_class=q_class_twm)
            metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
            rows.append(MetricRow(dataset_name, "single_split", None, model_name, "TWM-SplitCP", alpha, float(lam),
                                  include_duration, **metrics, n_test=len(y_test),
                                  positive_rate_test=float(np.mean(y_test)), auc=float(auc),
                                  ess_y1=ess[1], ess_y0=ess[0],
                                  twm_fallback_y1=fallback[1], twm_fallback_y0=fallback[0],
                                  t_star=t_star))
    return rows


def rolling_blocks(n_total: int, split_cfg: SplitConfig, n_windows: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fixed train block + sequential calibration/test block pairs.
    Remaining data after training are split into 2*n_windows equal chronological chunks.
    Window w uses chunk 2w-1 as calibration and chunk 2w as test.
    """
    n_train = int(split_cfg.train_frac * n_total)
    remaining = np.arange(n_train, n_total)
    n_chunks = 2 * n_windows
    chunks = np.array_split(remaining, n_chunks)
    windows = []
    train_idx = np.arange(0, n_train)
    for w in range(n_windows):
        cal_idx = chunks[2 * w]
        test_idx = chunks[2 * w + 1]
        windows.append((train_idx, cal_idx, test_idx))
    return windows


def evaluate_rolling_windows(
    X: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    dataset_name: str,
    include_duration: bool,
    model_name: str,
    alphas: Sequence[float],
    lambda_grid: Sequence[float],
    split_cfg: SplitConfig,
    n_windows: int,
    tstar_mode: str,
    min_eff_n: float,
) -> List[MetricRow]:
    rows: List[MetricRow] = []
    for window_id, (idx_train, idx_cal, idx_test) in enumerate(rolling_blocks(len(X), split_cfg, n_windows), start=1):
        X_train, y_train = X.iloc[idx_train], y[idx_train]
        X_cal, y_cal, t_cal = X.iloc[idx_cal], y[idx_cal], t[idx_cal]
        X_test, y_test, t_test = X.iloc[idx_test], y[idx_test], t[idx_test]

        pipe, train_p = fit_base_model(X_train, y_train, model_name)
        p1_cal = pipe.predict_proba(X_cal)[:, 1]
        p1_test = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p1_test) if len(np.unique(y_test)) > 1 else float("nan")
        scores_cal = nonconformity_scores(y_cal, p1_cal)
        cw = domain_ratio_weights(X_cal, X_test)

        if tstar_mode == "start":
            t_star = float(t_test.min())
        elif tstar_mode == "mid":
            t_star = float((t_test.min() + t_test.max()) / 2.0)
        else:
            t_star = float(t_test.max())

        for alpha in alphas:
            # SplitCP
            q = k_order_threshold(scores_cal, alpha)
            covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_global=q)
            metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
            rows.append(MetricRow(dataset_name, "rolling", window_id, model_name, "SplitCP", alpha, None,
                                  include_duration, **metrics, n_test=len(y_test),
                                  positive_rate_test=float(np.mean(y_test)), auc=float(auc), t_star=t_star))

            # Mondrian
            q_class = {cls: k_order_threshold(scores_cal[y_cal == cls], alpha) for cls in [0, 1]}
            covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_class=q_class)
            metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
            rows.append(MetricRow(dataset_name, "rolling", window_id, model_name, "Mondrian SplitCP", alpha, None,
                                  include_duration, **metrics, n_test=len(y_test),
                                  positive_rate_test=float(np.mean(y_test)), auc=float(auc), t_star=t_star))

            # Classifier-weighted SplitCP
            q_w = weighted_quantile_threshold(scores_cal, cw, alpha)
            covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_global=q_w)
            metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
            rows.append(MetricRow(dataset_name, "rolling", window_id, model_name, "Weighted SplitCP", alpha, None,
                                  include_duration, **metrics, n_test=len(y_test),
                                  positive_rate_test=float(np.mean(y_test)), auc=float(auc), t_star=t_star))

            # TWM-SplitCP grid
            for lam in lambda_grid:
                q_class_twm = {}
                ess = {}
                fallback = {}
                for cls in [0, 1]:
                    mask = y_cal == cls
                    s_cls = scores_cal[mask]
                    t_cls = t_cal[mask]
                    w_cls = temporal_decay_weights(t_cls, t_star=t_star, lam=lam)
                    ess_cls = effective_sample_size(w_cls)
                    ess[cls] = ess_cls
                    if ess_cls < min_eff_n:
                        q_cls = k_order_threshold(s_cls, alpha)
                        fallback[cls] = True
                    else:
                        q_cls = weighted_quantile_threshold(s_cls, w_cls, alpha)
                        fallback[cls] = False
                    q_class_twm[cls] = q_cls
                covered, set_sizes, _ = coverage_from_thresholds(y_test, p1_test, q_class=q_class_twm)
                metrics = summarize_metrics(y_test, covered, set_sizes, alpha)
                rows.append(MetricRow(dataset_name, "rolling", window_id, model_name, "TWM-SplitCP", alpha, float(lam),
                                      include_duration, **metrics, n_test=len(y_test),
                                      positive_rate_test=float(np.mean(y_test)), auc=float(auc),
                                      ess_y1=ess[1], ess_y0=ess[0],
                                      twm_fallback_y1=fallback[1], twm_fallback_y0=fallback[0],
                                      t_star=t_star))
    return rows


# -----------------------------------------------------------------------------
# Summaries and plots
# -----------------------------------------------------------------------------


def rows_to_df(rows: Sequence[MetricRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def summarize_rolling(df_roll: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[RollingSummaryRow] = []
    group_cols = ["dataset", "model", "method", "alpha", "lambda_", "include_duration"]
    for keys, g in df_roll.groupby(group_cols, dropna=False):
        dataset, model, method, alpha, lam, include_duration = keys
        out_rows.append(
            RollingSummaryRow(
                dataset=dataset,
                model=model,
                method=method,
                alpha=float(alpha),
                lambda_=None if pd.isna(lam) else float(lam),
                include_duration=bool(include_duration),
                mean_local_cov=float(g["coverage"].mean()),
                min_local_cov=float(g["coverage"].min()),
                max_local_cov=float(g["coverage"].max()),
                mean_cov1=float(g["cov_1"].mean()),
                min_cov1=float(g["cov_1"].min()),
                mean_size=float(g["avg_size"].mean()),
                mean_gap=float(g["gap"].mean()),
                mean_disparity=float(g["disparity"].mean()),
                mean_auc=float(g["auc"].mean()),
                mean_ess_y1=None if g["ess_y1"].isna().all() else float(g["ess_y1"].mean()),
                min_ess_y1=None if g["ess_y1"].isna().all() else float(g["ess_y1"].min()),
                mean_ess_y0=None if g["ess_y0"].isna().all() else float(g["ess_y0"].mean()),
                min_ess_y0=None if g["ess_y0"].isna().all() else float(g["ess_y0"].min()),
            )
        )
    return pd.DataFrame([asdict(r) for r in out_rows])


def plot_local_coverage(
    df_roll: pd.DataFrame,
    outdir: Path,
    dataset_name: str,
    model_name: str,
    alpha: float,
    report_lambda: float,
):
    g = df_roll[(df_roll["dataset"] == dataset_name) & (df_roll["model"] == model_name) & (df_roll["alpha"] == alpha)]
    if g.empty:
        return
    plt.figure(figsize=(8, 5))
    for method in ["SplitCP", "Mondrian SplitCP", "Weighted SplitCP"]:
        h = g[g["method"] == method].sort_values("window")
        plt.plot(h["window"], h["coverage"], marker="o", label=method)
    h = g[(g["method"] == "TWM-SplitCP") & (g["lambda_"] == report_lambda)].sort_values("window")
    if not h.empty:
        plt.plot(h["window"], h["coverage"], marker="o", label=f"TWM-SplitCP (lambda={report_lambda:g})")
    plt.axhline(1.0 - alpha, linestyle="--", label="target 1-alpha")
    plt.xlabel("Rolling window")
    plt.ylabel("Local coverage")
    plt.title(f"Local coverage over time - {dataset_name} - {model_name} - alpha={alpha:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"local_coverage_{dataset_name}_{model_name}_alpha_{alpha:.2f}.png", dpi=200)
    plt.close()


def plot_lambda_sensitivity(
    df_roll_summary: pd.DataFrame,
    outdir: Path,
    dataset_name: str,
    model_name: str,
    alpha: float,
):
    g = df_roll_summary[
        (df_roll_summary["dataset"] == dataset_name)
        & (df_roll_summary["model"] == model_name)
        & (df_roll_summary["alpha"] == alpha)
        & (df_roll_summary["method"] == "TWM-SplitCP")
    ].sort_values("lambda_")
    if g.empty:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(g["lambda_"], g["mean_local_cov"], marker="o", label="mean local coverage")
    plt.plot(g["lambda_"], g["min_local_cov"], marker="o", label="min local coverage")
    plt.plot(g["lambda_"], g["mean_size"], marker="o", label="mean set size")
    plt.xlabel("lambda")
    plt.title(f"TWM sensitivity - {dataset_name} - {model_name} - alpha={alpha:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"twm_sensitivity_{dataset_name}_{model_name}_alpha_{alpha:.2f}.png", dpi=200)
    plt.close()


def save_metadata(args: argparse.Namespace, df_single: pd.DataFrame, df_roll: pd.DataFrame, outdir: Path) -> None:
    meta = {
        "data_path": args.data_path,
        "dataset_name": args.dataset_name,
        "outdir": str(outdir),
        "models": args.models,
        "alphas": args.alphas,
        "lambda_grid": args.lambda_grid,
        "report_lambda": args.report_lambda,
        "split": {"train_frac": args.train_frac, "cal_frac": args.cal_frac, "test_frac": 1 - args.train_frac - args.cal_frac},
        "include_duration": args.include_duration,
        "rolling_windows": args.rolling_windows,
        "tstar_mode": args.tstar_mode,
        "min_effective_n": args.min_effective_n,
        "single_rows": int(len(df_single)),
        "rolling_rows": int(len(df_roll)),
    }
    with open(outdir / "metadata_twm.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TWM-SplitCP Bank experiments.")
    p.add_argument("--data-path", required=True, help="Path to bank-full.csv or bank-additional-full.csv")
    p.add_argument("--dataset-name", default="bank", help="Short dataset tag for outputs")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--models", nargs="+", default=["LR", "HGB"], choices=["LR", "HGB"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--lambda-grid", nargs="+", type=float, default=[0.0, 1.0, 2.0, 5.0],
                   help="Pre-specified temporal sensitivity grid. lambda=0 should reduce to Mondrian-style weighting.")
    p.add_argument("--report-lambda", type=float, default=2.0,
                   help="Pre-specified lambda to use in comparison plots. This is not selected with test labels.")
    p.add_argument("--train-frac", type=float, default=0.60)
    p.add_argument("--cal-frac", type=float, default=0.20)
    p.add_argument("--rolling-windows", type=int, default=5)
    p.add_argument("--include-duration", action="store_true",
                   help="Include the leakage feature duration. Default is to exclude it.")
    p.add_argument("--tstar-mode", choices=["start", "mid", "end"], default="end",
                   help="Window-level t*: start, midpoint, or end of the test block.")
    p.add_argument("--min-effective-n", type=float, default=30.0,
                   help="Minimum class-specific effective sample size before TWM falls back to unweighted Mondrian for that class.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.train_frac + args.cal_frac >= 1.0:
        raise ValueError("train_frac + cal_frac must be < 1.")
    if args.report_lambda not in args.lambda_grid:
        raise ValueError("report_lambda must be one of the values in lambda_grid.")

    df = load_bank_csv(args.data_path)
    X, y, t = prepare_xy(df, include_duration=args.include_duration)
    split_cfg = SplitConfig(train_frac=args.train_frac, cal_frac=args.cal_frac)

    single_rows: List[MetricRow] = []
    rolling_rows: List[MetricRow] = []

    for model_name in args.models:
        single_rows.extend(
            evaluate_single_split(
                X=X,
                y=y,
                t=t,
                dataset_name=args.dataset_name,
                include_duration=args.include_duration,
                model_name=model_name,
                alphas=args.alphas,
                lambda_grid=args.lambda_grid,
                split_cfg=split_cfg,
                tstar_mode=args.tstar_mode,
                min_eff_n=args.min_effective_n,
            )
        )
        rolling_rows.extend(
            evaluate_rolling_windows(
                X=X,
                y=y,
                t=t,
                dataset_name=args.dataset_name,
                include_duration=args.include_duration,
                model_name=model_name,
                alphas=args.alphas,
                lambda_grid=args.lambda_grid,
                split_cfg=split_cfg,
                n_windows=args.rolling_windows,
                tstar_mode=args.tstar_mode,
                min_eff_n=args.min_effective_n,
            )
        )

    df_single = rows_to_df(single_rows)
    df_roll = rows_to_df(rolling_rows)
    df_roll_summary = summarize_rolling(df_roll)

    df_single.to_csv(outdir / "single_split_results_twm.csv", index=False)
    df_roll.to_csv(outdir / "rolling_window_results_twm.csv", index=False)
    df_roll_summary.to_csv(outdir / "rolling_window_summary_twm.csv", index=False)

    # Convenience extracts for report writing
    sens = df_roll_summary[df_roll_summary["method"] == "TWM-SplitCP"].copy()
    sens.to_csv(outdir / "twm_lambda_sensitivity_summary.csv", index=False)

    # Plots
    for model_name in args.models:
        for alpha in args.alphas:
            plot_local_coverage(df_roll, outdir, args.dataset_name, model_name, alpha, args.report_lambda)
            plot_lambda_sensitivity(df_roll_summary, outdir, args.dataset_name, model_name, alpha)

    save_metadata(args, df_single, df_roll, outdir)

    print("Done.")
    print(f"Saved: {outdir / 'single_split_results_twm.csv'}")
    print(f"Saved: {outdir / 'rolling_window_results_twm.csv'}")
    print(f"Saved: {outdir / 'rolling_window_summary_twm.csv'}")
    print(f"Saved: {outdir / 'twm_lambda_sensitivity_summary.csv'}")
    print("Suggested report focus:")
    print("- Compare SplitCP / Mondrian / Weighted SplitCP / TWM-SplitCP at alpha=0.10")
    print("- Highlight Cov(1), min local coverage, and average set size")
    print("- Use lambda=0 as the Mondrian-style ablation and discuss ESS safeguards")


if __name__ == "__main__":
    main()
