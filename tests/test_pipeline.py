"""Tests for the end-to-end comparison pipeline."""

from __future__ import annotations

import json

from branchpred.pipeline import format_results_table, run_comparison
from branchpred.training.config import Config


def _small_config(tmp_path) -> Config:
    cfg = Config()
    cfg.data.num_branches = 12000
    cfg.data.trace_path = str(tmp_path / "trace.csv")
    cfg.features.global_history_length = 8
    cfg.features.local_history_length = 4
    cfg.features.pc_hash_bits = 8
    cfg.model.epochs = 3
    cfg.output.checkpoint_path = str(tmp_path / "m.pt")
    cfg.output.metrics_dir = str(tmp_path / "metrics")
    cfg.output.figures_dir = str(tmp_path / "figures")
    return cfg


def test_run_comparison_produces_all_predictors(tmp_path) -> None:
    cfg = _small_config(tmp_path)
    out = run_comparison(cfg, save_artifacts=True)

    assert set(out.results) == {"2-bit", "gshare", "perceptron"}
    for res in out.results.values():
        assert 0.0 <= res.metrics.accuracy <= 1.0
        assert res.metrics.mpki >= 0.0
    # All predictors scored on the same held-out test tail.
    totals = {res.metrics.total for res in out.results.values()}
    assert len(totals) == 1
    assert out.test_start > 0


def test_artifacts_written(tmp_path) -> None:
    cfg = _small_config(tmp_path)
    run_comparison(cfg, save_artifacts=True)

    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    assert (metrics_dir / "comparison.json").exists()
    assert (metrics_dir / "comparison.md").exists()
    assert (figures_dir / "accuracy.png").exists()
    assert (figures_dir / "mpki.png").exists()
    assert (tmp_path / "m.pt").exists()

    data = json.loads((metrics_dir / "comparison.json").read_text())
    assert set(data) == {"2-bit", "gshare", "perceptron"}


def test_no_save_skips_artifacts(tmp_path) -> None:
    cfg = _small_config(tmp_path)
    run_comparison(cfg, save_artifacts=False)
    assert not (tmp_path / "metrics").exists()
    assert not (tmp_path / "m.pt").exists()


def test_results_table_is_sorted_by_accuracy(tmp_path) -> None:
    cfg = _small_config(tmp_path)
    out = run_comparison(cfg, save_artifacts=False)
    table = format_results_table(out.results)
    assert table.startswith("| Predictor |")
    # Body rows appear in descending accuracy order.
    accs = sorted((round(res.metrics.accuracy, 4) for res in out.results.values()), reverse=True)
    lines = table.splitlines()[2:]
    reported = [float(line.split("|")[2]) for line in lines]
    assert reported == accs
    assert reported == sorted(reported, reverse=True)  # descending
