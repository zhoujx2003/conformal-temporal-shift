# Conformal Prediction under Class Imbalance and Temporal Shift

This repository contains the code and public report for a final project on conformal prediction for binary tabular classification under class imbalance and temporal shift.

The project studies when standard conformal coverage guarantees behave as expected and when they become fragile because calibration data and deployment data are no longer exchangeable.

## Project summary

The experiments cover three settings:

1. **WDBC benchmark**
   A near-exchangeable baseline using the Wisconsin Diagnostic Breast Cancer dataset with random stratified splitting.

2. **WDBC synthetic shift**
   A controlled shift experiment that tests whether increasing calibration-test score mismatch leads to larger coverage loss.

3. **Bank Marketing chronological evaluation**
   A deployment-style experiment using chronological train, calibration, and test splits on UCI Bank Marketing data. This is the main empirical setting for studying temporal shift.

The main conformal methods are:

* Split Conformal Prediction
* Mondrian Split Conformal Prediction
* Adaptive Prediction Sets
* Classifier-weighted Split Conformal Prediction
* Temporally weighted Mondrian SplitCP, included as an exploratory appendix experiment

## Repository structure

```text
.
├── bank_conformal_experiments.py
├── run_robustness_scan.py
├── run_all.py
├── twm_bank_experiment.py
├── wdbc_conformal_2x2.py
├── wdbc_synthetic_shift_experiment.py
├── data/
│   └── README.md
├── reports/
│   └── final_project_report_public.pdf
├── assets/
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

Raw datasets are not included in this repository.

Place the following files in the `data/` directory before running the experiments:

```text
data/wdbc.data
data/bank-full.csv
data/bank-additional-full.csv
```

The Bank Marketing datasets and WDBC dataset can be downloaded from the UCI Machine Learning Repository.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Reproducing experiments

### WDBC baseline

```powershell
python wdbc_conformal_2x2.py --data-path data/wdbc.data --outdir out_wdbc --overwrite
```

### WDBC synthetic shift

```powershell
python wdbc_synthetic_shift_experiment.py --data-path data/wdbc.data --outdir out_wdbc_synth
```

### Original Bank chronological experiment

```powershell
python bank_conformal_experiments.py --data-path data/bank-full.csv --outdir out_bank
```

### Additional Bank robustness check

```powershell
python bank_conformal_experiments.py --data-path data/bank-additional-full.csv --outdir out_bank_additional
```

### Original Bank robustness scan

```powershell
python run_robustness_scan.py --data-path data/bank-full.csv --outdir out_bank_robustness
```

### Exploratory TWM extension

```powershell
python twm_bank_experiment.py --data-path data/bank-full.csv --dataset-name orig_bank --outdir out_twm_bank
```

## Outputs

Experiment scripts write CSV tables and PNG figures to output folders such as:

```text
out_wdbc/
out_wdbc_synth/
out_bank/
out_bank_additional/
out_bank_robustness/
out_twm_bank/
```

These generated output folders are intentionally ignored by Git.

## Report

A public version of the project report is available at:

```text
reports/final_project_report_public.pdf
```

## Notes

The weighted SplitCP experiment should be interpreted as a retrospective batch-level covariate-shift adjustment because it uses unlabelled covariates from the later test block.

The TWM experiment is included as an exploratory appendix mechanism check, not as a validated new conformal algorithm.
