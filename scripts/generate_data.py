#!/usr/bin/env python
"""Generate (or load) a branch trace and report basic statistics.

Usage:
    python scripts/generate_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make branchpred importable when run without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from branchpred.training.config import Config  # noqa: E402
from branchpred.training.trainer import get_trace  # noqa: E402
from branchpred.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)

    df = get_trace(config)
    print(f"Trace source     : {config.data.source}")
    print(f"Branches         : {len(df):,}")
    print(f"Distinct PCs     : {df['pc'].nunique():,}")
    print(f"Taken rate       : {df['outcome'].mean():.3f}")
    if config.data.source == "synthetic":
        print(f"Cached to        : {config.data.trace_path}")


if __name__ == "__main__":
    main()
