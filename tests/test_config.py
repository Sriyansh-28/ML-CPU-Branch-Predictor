"""Tests for typed configuration loading, merging, and validation."""

from __future__ import annotations

import pytest

from branchpred.training.config import Config


def test_defaults_are_valid() -> None:
    cfg = Config()
    cfg.validate()  # should not raise
    assert cfg.features.feature_dim == cfg.features.global_history_length + cfg.features.local_history_length


def test_load_actual_default_yaml() -> None:
    cfg = Config.from_yaml("configs/default.yaml")
    assert cfg.data.source == "synthetic"
    assert cfg.features.global_history_length == 16
    assert cfg.model.epochs == 5


def test_partial_override_merges_over_defaults(tmp_path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text("seed: 7\nmodel:\n  epochs: 8\n")
    cfg = Config.from_yaml(p)
    assert cfg.seed == 7
    assert cfg.model.epochs == 8            # overridden
    assert cfg.model.batch_size == 256      # kept from defaults
    assert cfg.data.num_branches == 200_000  # untouched section kept


def test_experiment_yaml_loads() -> None:
    cfg = Config.from_yaml("configs/experiment_perceptron.yaml")
    assert cfg.seed == 7
    assert cfg.features.global_history_length == 32
    assert cfg.model.learning_rate == 0.005
    assert cfg.data.num_branches == 500_000


def test_unknown_key_rejected(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  bogus_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.from_yaml(p)


def test_validation_catches_bad_split(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  train_frac: 0.9\n  val_frac: 0.2\n")
    with pytest.raises(ValueError, match="test set"):
        Config.from_yaml(p)


def test_validation_catches_bad_source(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  source: magic\n")
    with pytest.raises(ValueError, match="source"):
        Config.from_yaml(p)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("configs/does_not_exist.yaml")
