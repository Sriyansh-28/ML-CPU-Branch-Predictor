"""Tests for the training pipeline: data prep, training, and checkpointing."""

from __future__ import annotations

import pytest

from branchpred.data.feature_engineering import build_features, chronological_split
from branchpred.models.torch_dataset import BranchDataset
from branchpred.training.config import Config
from branchpred.training.trainer import (
    build_model,
    get_trace,
    load_adapter_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_perceptron,
)
from branchpred.utils.seed import set_seed


def _small_config(**overrides) -> Config:
    cfg = Config()
    cfg.data.num_branches = 8000
    cfg.features.global_history_length = 8
    cfg.features.local_history_length = 4
    cfg.features.pc_hash_bits = 8
    cfg.model.epochs = 3
    cfg.model.batch_size = 128
    for k, v in overrides.items():
        setattr(cfg.model, k, v)
    return cfg


def test_get_trace_synthetic(tmp_path) -> None:
    cfg = _small_config()
    cfg.data.trace_path = str(tmp_path / "trace.csv")
    df = get_trace(cfg)
    assert len(df) == 8000
    assert (tmp_path / "trace.csv").exists()  # cached


def test_training_improves_and_early_stops() -> None:
    set_seed(0)
    cfg = _small_config(epochs=6)
    df = get_trace_in_memory(cfg)
    feats = build_features(
        df,
        cfg.features.global_history_length,
        cfg.features.local_history_length,
        cfg.features.pc_hash_bits,
    )
    tr, va, te = chronological_split(len(feats), cfg.data.train_frac, cfg.data.val_frac)
    model = build_model(cfg, feats)
    result = train_perceptron(model, BranchDataset(feats, tr), BranchDataset(feats, va), cfg)

    # Training should reach a sensible accuracy well above chance on this mix.
    assert result.best_val_acc > 0.75
    assert result.best_epoch >= 1
    assert len(result.history) >= 1


def test_checkpoint_round_trip(tmp_path) -> None:
    cfg = _small_config()
    df = get_trace_in_memory(cfg)
    feats = build_features(
        df,
        cfg.features.global_history_length,
        cfg.features.local_history_length,
        cfg.features.pc_hash_bits,
    )
    model = build_model(cfg, feats)
    path = save_checkpoint(model, tmp_path / "m.pt")
    assert path.exists()

    reloaded = load_checkpoint(path)
    assert reloaded.feature_dim == model.feature_dim
    assert reloaded.num_pc_buckets == model.num_pc_buckets
    # Weights match.
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert k1 == k2
        assert (v1 == v2).all()


def test_empty_validation_split_is_rejected() -> None:
    # A tiny trace whose validation slice rounds to zero rows must fail fast rather
    # than silently freezing epoch 1 as the "best" checkpoint.
    cfg = _small_config()
    df = get_trace_in_memory(cfg)
    feats = build_features(
        df,
        cfg.features.global_history_length,
        cfg.features.local_history_length,
        cfg.features.pc_hash_bits,
    )
    model = build_model(cfg, feats)
    empty_val = BranchDataset(feats, slice(0, 0))
    full_train = BranchDataset(feats, slice(0, len(feats)))
    with pytest.raises(ValueError, match="Validation set is empty"):
        train_perceptron(model, full_train, empty_val, cfg)


def _feats_for(cfg: Config):
    df = get_trace_in_memory(cfg)
    return build_features(
        df,
        cfg.features.global_history_length,
        cfg.features.local_history_length,
        cfg.features.pc_hash_bits,
    )


def test_checkpoint_stores_feature_spec_and_verifies_on_load(tmp_path) -> None:
    cfg = _small_config()
    feats = _feats_for(cfg)
    model = build_model(cfg, feats)
    path = save_checkpoint(model, tmp_path / "m.pt", cfg)

    # Matching config loads fine.
    adapter = load_adapter_from_checkpoint(path, cfg)
    assert adapter.model.feature_dim == model.feature_dim


def test_checkpoint_rejects_mismatched_split(tmp_path) -> None:
    # Same total feature_dim (8+4), but the global/local split differs -> columns
    # would mean different things. Must be rejected, not silently accepted.
    cfg = _small_config()
    feats = _feats_for(cfg)
    model = build_model(cfg, feats)
    path = save_checkpoint(model, tmp_path / "m.pt", cfg)

    mismatched = _small_config()
    mismatched.features.global_history_length = 6  # was 8
    mismatched.features.local_history_length = 6   # was 4  (sum still 12)
    with pytest.raises(ValueError, match="feature spec does not match"):
        load_adapter_from_checkpoint(path, mismatched)


def test_checkpoint_rejects_mismatched_pc_hash_bits(tmp_path) -> None:
    cfg = _small_config()
    feats = _feats_for(cfg)
    model = build_model(cfg, feats)
    path = save_checkpoint(model, tmp_path / "m.pt", cfg)

    mismatched = _small_config()
    mismatched.features.pc_hash_bits = 10  # was 8
    with pytest.raises(ValueError, match="feature spec does not match"):
        load_adapter_from_checkpoint(path, mismatched)


def get_trace_in_memory(cfg: Config):
    # Avoid touching disk in pure-training tests.
    from branchpred.data.trace_generator import generate_trace

    return generate_trace(
        num_branches=cfg.data.num_branches,
        pattern_mix=cfg.data.pattern_mix,
        seed=cfg.seed,
        global_history_length=cfg.features.global_history_length,
    )
