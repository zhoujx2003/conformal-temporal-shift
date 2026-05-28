#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
WDBC_COLUMNS = [
    'id','diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
    'concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst',
    'area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst',
    'fractal_dimension_worst']

def load_wdbc(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, header=None, names=WDBC_COLUMNS)
    y = (df['diagnosis'].astype(str).str.upper() == 'M').astype(int).to_numpy()
    X = df.drop(columns=['id','diagnosis']).astype(float)
    return X, y

def build_model(name: str, seed: int):
    if name.upper() == 'LR':
        return LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed)
    if name.upper() == 'HGB':
        return HistGradientBoostingClassifier(random_state=seed, learning_rate=0.05, max_iter=300)
    raise ValueError(name)

def finite_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float)); m=len(scores)
    k = int(math.ceil((m+1)*(1-alpha))); k = min(max(k,1),m)
    return float(scores[k-1])

def summarize_sets(sets: np.ndarray, y_true: np.ndarray) -> Dict[str,float]:
    cover = sets[np.arange(len(y_true)), y_true]; size = sets.sum(axis=1)
    m1 = y_true==1; m0 = y_true==0
    return {'coverage': float(cover.mean()), 'coverage_1': float(cover[m1].mean()), 'coverage_0': float(cover[m0].mean()), 'avg_size': float(size.mean()), 'empty_or_trigger': float((size==0).mean())}

def splitcp_metrics(probs_cal,y_cal,probs_test,y_test,alpha):
    q = finite_quantile(1.0 - probs_cal[np.arange(len(y_cal)), y_cal], alpha); thr = 1.0 - q
    return summarize_sets((probs_test >= thr).astype(int), y_test)

def mondrian_metrics(probs_cal,y_cal,probs_test,y_test,alpha):
    sets = np.zeros_like(probs_test, dtype=int)
    for label in [0,1]:
        q = finite_quantile(1.0 - probs_cal[y_cal==label, label], alpha); thr = 1.0 - q
        sets[:,label] = (probs_test[:,label] >= thr).astype(int)
    return summarize_sets(sets, y_test)

def aps_calibration_scores(probs,y_true,rng):
    n, k = probs.shape; scores = np.empty(n)
    for i in range(n):
        p = probs[i]; order = np.argsort(-p); rank_map = {label:rank for rank,label in enumerate(order)}; y=int(y_true[i]); rank=rank_map[y]; higher=order[:rank]
        scores[i] = float(p[higher].sum() + rng.random() * p[y])
    return scores

def aps_prediction_sets(probs_test,q,rng):
    n,k = probs_test.shape; sets = np.zeros((n,k), dtype=int)
    for i in range(n):
        p = probs_test[i]; order=np.argsort(-p); rank_map={label:rank for rank,label in enumerate(order)}
        for y in range(k):
            rank=rank_map[y]; higher=order[:rank]; score=float(p[higher].sum()+rng.random()*p[y])
            if score <= q: sets[i,y]=1
    return sets

def aps_metrics(probs_cal,y_cal,probs_test,y_test,alpha,seed):
    q = finite_quantile(aps_calibration_scores(probs_cal,y_cal,np.random.default_rng(seed)), alpha)
    raw_sets = aps_prediction_sets(probs_test, q, np.random.default_rng(seed+1)); raw = summarize_sets(raw_sets, y_test)
    fb_sets = raw_sets.copy(); empty = fb_sets.sum(axis=1)==0
    if empty.any(): fb_sets[empty, np.argmax(probs_test[empty], axis=1)] = 1
    fb = summarize_sets(fb_sets, y_test); fb['empty_or_trigger'] = float(empty.mean())
    return raw, fb

def true_label_scores(probs,y):
    return 1.0 - np.clip(probs[np.arange(len(y)), y], EPS, 1.0-EPS)

def score_drift(probs_cal,y_cal,probs_test,y_test):
    s_cal = true_label_scores(probs_cal,y_cal); s_test = true_label_scores(probs_test,y_test)
    out = {'score_ks_global': float(ks_2samp(s_cal,s_test).statistic), 'score_wass_global': float(wasserstein_distance(s_cal,s_test))}
    for label in [0,1]:
        cal=s_cal[y_cal==label]; test=s_test[y_test==label]
        out[f'score_ks_y{label}'] = float(ks_2samp(cal,test).statistic)
        out[f'score_wass_y{label}'] = float(wasserstein_distance(cal,test))
    return out

def correlation_summary(df):
    rows=[]
    for (model,alpha,method), g in df.groupby(['model','alpha','method']):
        for drift_col in ['score_ks_global','score_wass_global','score_ks_y1','score_wass_y1']:
            gg=g[[drift_col,'coverage_gap']].dropna(); row={'model':model,'alpha':alpha,'method':method,'drift_metric':drift_col,'n':len(gg)}
            if len(gg)>=2 and gg[drift_col].nunique()>1 and gg['coverage_gap'].nunique()>1:
                row['pearson_r']=float(np.corrcoef(gg[drift_col], gg['coverage_gap'])[0,1]); row['spearman_rho']=float(spearmanr(gg[drift_col], gg['coverage_gap']).statistic)
            else:
                row['pearson_r']=float('nan'); row['spearman_rho']=float('nan')
            rows.append(row)
    return pd.DataFrame(rows)

def synthetic_shift(X_train,y_train,X_test,y_test,delta):
    scaler=StandardScaler().fit(X_train); z_train=scaler.transform(X_train); z_test=scaler.transform(X_test)
    v = z_train[y_train==1].mean(axis=0) - z_train[y_train==0].mean(axis=0)
    norm=np.linalg.norm(v)
    if norm <= 0: return X_test.copy()
    v = v / norm; z_shift = z_test.copy(); z_shift[y_test==1] -= delta * v; z_shift[y_test==0] += delta * v
    return scaler.inverse_transform(z_shift)

def plot_metric(summary, outdir, metric, title_prefix):
    for (model,alpha), g in summary.groupby(['model','alpha']):
        plt.figure(figsize=(7,4.5))
        for method in ['SplitCP','Mondrian','APS (allow-empty)']:
            h=g[g['method']==method].sort_values('delta')
            if not h.empty: plt.plot(h['delta'], h[metric], marker='o', label=method)
        if metric == 'mean_coverage':
            plt.axhline(1.0-alpha, linestyle='--', label='target 1-alpha'); ylabel='Mean coverage'; stem=f'coverage_gap_vs_shift_{model}_alpha_{alpha:.2f}.png'
        else:
            ylabel='Mean score Wasserstein'; stem=f'score_drift_vs_shift_{model}_alpha_{alpha:.2f}.png'
        plt.xlabel('Shift severity $\\delta$'); plt.ylabel(ylabel); plt.title(f'{title_prefix} - {model} - alpha={alpha:.2f}')
        plt.legend(frameon=False); plt.tight_layout(); plt.savefig(outdir/stem, dpi=180); plt.close()

def run_experiment(data_path: Path, outdir: Path, alphas=None, models=None, deltas=None, seeds=None):
    if alphas is None: alphas=[0.10]
    if models is None: models=['LR','HGB']
    if deltas is None: deltas=[0.0,0.25,0.50,0.75,1.0,1.25]
    if seeds is None: seeds=list(range(1,31))
    X_df, y = load_wdbc(data_path); X = X_df.to_numpy(dtype=float); rows=[]
    for seed in seeds:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
        for model_name in models:
            scaler = StandardScaler().fit(X_train); Xtr=scaler.transform(X_train); Xca=scaler.transform(X_cal)
            base = build_model(model_name, seed); base.fit(Xtr,y_train); probs_cal = base.predict_proba(Xca)
            for delta in deltas:
                X_shift = synthetic_shift(X_train,y_train,X_test,y_test,delta); Xte=scaler.transform(X_shift); probs_test=base.predict_proba(Xte); auc=float(roc_auc_score(y_test, probs_test[:,1])); drift=score_drift(probs_cal,y_cal,probs_test,y_test)
                for alpha in alphas:
                    methods={'SplitCP': splitcp_metrics(probs_cal,y_cal,probs_test,y_test,alpha), 'Mondrian': mondrian_metrics(probs_cal,y_cal,probs_test,y_test,alpha)}
                    aps_raw, aps_fb = aps_metrics(probs_cal,y_cal,probs_test,y_test,alpha,seed); methods['APS (allow-empty)']=aps_raw; methods['APS (+fallback)']=aps_fb
                    for method, metrics in methods.items():
                        row={'seed':seed,'model':model_name,'delta':float(delta),'alpha':float(alpha),'method':method,'auc':auc,**drift,**metrics}
                        row['target_coverage']=1.0-alpha; row['coverage_gap']=row['coverage']-row['target_coverage']; row['class_disparity_gap']=abs(row['coverage_1']-row['coverage_0']); rows.append(row)
    raw=pd.DataFrame(rows); outdir.mkdir(parents=True, exist_ok=True); raw.to_csv(outdir/'wdbc_synthetic_shift_raw.csv', index=False)
    summary=(raw.groupby(['model','alpha','delta','method'], as_index=False).agg(mean_coverage=('coverage','mean'), sd_coverage=('coverage','std'), mean_cov1=('coverage_1','mean'), mean_cov0=('coverage_0','mean'), mean_size=('avg_size','mean'), mean_gap=('coverage_gap','mean'), mean_disparity=('class_disparity_gap','mean'), mean_auc=('auc','mean'), mean_score_ks_global=('score_ks_global','mean'), mean_score_wass_global=('score_wass_global','mean'), mean_score_ks_y1=('score_ks_y1','mean'), mean_score_wass_y1=('score_wass_y1','mean')))
    summary.to_csv(outdir/'wdbc_synthetic_shift_summary.csv', index=False)
    corr=correlation_summary(raw[raw['method'].isin(['SplitCP','Mondrian','APS (allow-empty)'])]); corr.to_csv(outdir/'wdbc_synthetic_shift_correlations.csv', index=False)
    plot_metric(summary, outdir, 'mean_coverage', 'Coverage under controlled synthetic shift')
    plot_metric(summary, outdir, 'mean_score_wass_global', 'Score-drift proxy under controlled synthetic shift')
    return raw, summary, corr

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data-path', type=Path, required=True); p.add_argument('--outdir', type=Path, required=True); p.add_argument('--alphas', nargs='+', type=float, default=[0.10]); p.add_argument('--models', nargs='+', default=['LR','HGB']); p.add_argument('--deltas', nargs='+', type=float, default=[0.0,0.25,0.50,0.75,1.0,1.25]); p.add_argument('--seeds', nargs='+', type=int, default=list(range(1,31))); return p.parse_args()

def main():
    args=parse_args(); run_experiment(args.data_path, args.outdir, args.alphas, args.models, args.deltas, args.seeds)
    print('Done.')

if __name__ == '__main__':
    main()
