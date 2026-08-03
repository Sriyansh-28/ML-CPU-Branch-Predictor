"""Typed, validated experiment configuration.

The YAML files in ``configs/`` map onto the dataclasses below. Dataclass defaults
mirror ``configs/default.yaml`` exactly, so a *partial* experiment file (e.g.
``experiment_perceptron.yaml``) is simply deep-merged on top of those defaults —
you only specify what changes. ``Config.from_yaml`` performs the merge and
validates the result, failing fast on bad values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"          # "synthetic" | "file"
    trace_path: str = "data/raw/trace.csv"
    num_branches: int = 200_000
    pattern_mix: dict[str, float] = field(
        default_factory=lambda: {
            "loop": 0.35,
            "nested_loop": 0.20,
            "correlated": 0.25,
            "function": 0.10,
            "random": 0.10,
        }
    )
    train_frac: float = 0.7
    val_frac: float = 0.1


@dataclass
class FeaturesConfig:
    global_history_length: int = 16
    local_history_length: int = 8
    pc_hash_bits: int = 12

    @property
    def feature_dim(self) -> int:
        return self.global_history_length + self.local_history_length


@dataclass
class TwoBitConfig:
    table_bits: int = 14


@dataclass
class GshareConfig:
    table_bits: int = 14
    history_bits: int = 14


@dataclass
class BaselinesConfig:
    two_bit: TwoBitConfig = field(default_factory=TwoBitConfig)
    gshare: GshareConfig = field(default_factory=GshareConfig)


@dataclass
class ModelConfig:
    learning_rate: float = 0.01
    epochs: int = 5
    batch_size: int = 256
    patience: int = 3
    online_update: bool = True


@dataclass
class OutputConfig:
    checkpoint_path: str = "results/metrics/perceptron.pt"
    metrics_dir: str = "results/metrics"
    figures_dir: str = "results/figures"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a (possibly partial) YAML file and merge it over the defaults."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
        cfg = _merge_into_dataclass(cls, data)
        cfg.validate()
        return cfg

    # -- validation ------------------------------------------------------------
    def validate(self) -> None:
        if self.data.source not in ("synthetic", "file"):
            raise ValueError("data.source must be 'synthetic' or 'file'.")
        if self.data.num_branches <= 0:
            raise ValueError("data.num_branches must be positive.")
        if not (0 < self.data.train_frac < 1):
            raise ValueError("data.train_frac must be in (0, 1).")
        if not (0 <= self.data.val_frac < 1):
            raise ValueError("data.val_frac must be in [0, 1).")
        if self.data.train_frac + self.data.val_frac >= 1:
            raise ValueError("train_frac + val_frac must leave a test set (< 1).")
        if self.features.feature_dim <= 0:
            raise ValueError("features must have positive total history length.")
        if self.model.epochs <= 0:
            raise ValueError("model.epochs must be positive.")
        if self.model.batch_size <= 0:
            raise ValueError("model.batch_size must be positive.")
        if self.model.learning_rate <= 0:
            raise ValueError("model.learning_rate must be positive.")


def _merge_into_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build ``cls`` from its defaults, overlaying ``data``."""
    kwargs: dict[str, Any] = {}
    field_names = {f.name for f in fields(cls)}
    # Resolve string annotations (from __future__ import annotations) to real types.
    hints = get_type_hints(cls)
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")

    for name in field_names:
        if name not in data:
            continue
        value = data[name]
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _merge_into_dataclass(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
