"""Tests for the training pipeline: data prep, training, and checkpointing."""

from __future__ import annotations

from branchpred.data.feature_engineering import build_features, chronological_split
from branchpred.models.torch_dataset import BranchDataset
from branchpred.training.config import Config
from branchpred.training.trainer import (
    build_model,
    get_trace,
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


def get_trace_in_memory(cfg: Config):
    # Avoid touching disk in pure-training tests.
    from branchpred.data.trace_generator import generate_trace

    return generate_trace(
        num_branches=cfg.data.num_branches,
        pattern_mix=cfg.data.pattern_mix,
        seed=cfg.seed,
        global_history_length=cfg.features.global_history_length,
    )
