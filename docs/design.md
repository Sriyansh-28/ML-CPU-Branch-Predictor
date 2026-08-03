# Design Document — ML-Based CPU Branch Predictor

This document records the architecture and engineering decisions behind the
project. It is committed so reviewers can see the reasoning, not just the code.

## 1. Objective

Demonstrate that a machine-learning branch predictor (a **perceptron**, the
software analog of the hardware perceptron predictor introduced by Jiménez &
Lin, HPCA 2001) can be benchmarked fairly against the classical **2-bit
saturating (bimodal)** and **gshare** predictors, using the metrics computer
architects actually use — prediction **accuracy** and **MPKI** (mispredictions
per 1000 instructions).

The emphasis is deliberately on **correctness, reproducibility, modularity, and
engineering quality**, not deep-learning complexity.

## 2. Predictors implemented

| Predictor          | Type       | Index / inputs                     | Role                         |
|--------------------|------------|------------------------------------|------------------------------|
| 2-bit saturating   | classical  | low bits of PC                     | required hardware baseline   |
| gshare             | classical  | `(PC ^ global_history)`            | stronger classical baseline  |
| Perceptron         | ML (Torch) | global history bits (±1) + bias    | the learned predictor        |

An MLP was intentionally **excluded** to keep the ML side interpretable and
close to real hardware predictors.

## 3. Shared predictor interface

Every predictor — classical or ML — implements the same contract:

```python
class BasePredictor:
    def predict(self, branch) -> int:   # 0/1, using only pre-resolution info
        ...
    def update(self, branch, outcome) -> None:  # counter / weight update after resolution
        ...
```

This mirrors a real pipeline (predict → resolve → update) and lets the evaluator
drive ML and hardware predictors through the **same online, single-pass, leak-
free protocol**.

## 4. Data strategy

We first search for suitable **public branch-trace datasets** (e.g. Championship
Branch Prediction / ChampSim traces). Because such traces are large, license-
encumbered, and simulator-bound — a poor fit for a self-contained, reproducible
repo — the project supports **both**:

* a **file loader** that ingests real traces in a simple `PC outcome` format, and
* a **synthetic generator** producing program-like branch behaviour (biased
  loops, nested loops, history-correlated branches, call/return, and noise),
  deterministic given a seed.

Synthetic generation is the default because it makes every reported number
exactly reproducible; the loader keeps the pipeline real-data-ready.

## 5. Feature engineering (causal)

For branch *i*, features use **only** outcomes strictly before *i*:

* global history register (last *N* outcomes),
* per-PC local history (last *M* outcomes),
* the PC hashed into a bucket index.

The train/val/test split is **chronological** (never shuffled) — reflecting
deployment, where a predictor learns from a program's past to predict its
future. A unit test asserts there is no future leakage.

## 6. Evaluation

`evaluation/evaluator.py` streams the held-out trace once, calling `predict`
then `update` for any predictor, and aggregates:

* **Accuracy**, **MPKI** (headline metric),
* precision / recall / F1 and a confusion matrix,
* a cumulative-misprediction curve.

Fairness safeguards: identical evaluation trace for all predictors, strictly
causal features, chronological split, and honest reporting of the ML model's
size/history alongside its accuracy.

## 7. Repository layout

```
src/branchpred/
  data/         trace_generator, trace_loader, feature_engineering
  baselines/    base_predictor, two_bit_predictor, gshare_predictor
  models/       perceptron, ml_predictor (BasePredictor adapter), torch_dataset
  training/     config, trainer
  evaluation/   metrics, evaluator
  visualization/plots
  utils/        seed, logging
scripts/        generate_data, train, evaluate, run_comparison
tests/          one module per component (correctness-focused)
configs/        default.yaml + experiment overrides
```

## 8. Milestones (one commit each)

1. Scaffolding, tooling, CI, utils, this design doc. *(current)*
2. Data layer — dataset search, synthetic generator, loader.
3. Classical baselines — 2-bit saturating + gshare.
4. Causal feature engineering + PyTorch dataset.
5. Perceptron model + BasePredictor adapter.
6. Config-driven trainer.
7. Metrics + online evaluator.
8. Visualization.
9. End-to-end comparison CLIs + results.
10. README overhaul with architecture diagrams + final polish.
