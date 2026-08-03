"""Load and save branch traces in the project's canonical schema.

The whole pipeline is built around one simple, source-agnostic representation:

    a table of ``(pc, outcome)`` rows in *dynamic program order*,

where ``pc`` is the branch instruction address (a non-negative integer) and
``outcome`` is ``1`` for *taken* and ``0`` for *not-taken*. The row order is the
order in which the branches were executed — it carries the temporal information
every predictor depends on, so it is never shuffled here.

Two on-disk formats are accepted, which lets the same code consume both our
synthetic traces and real decoded CBP/ChampSim traces (see docs/datasets.md):

* **CSV** (``.csv``) with a ``pc,outcome`` header.
* **Whitespace text** (``.txt`` / ``.trace``) with one ``<pc> <outcome>`` pair
  per line. ``pc`` may be decimal or ``0x``-prefixed hex; blank lines and lines
  beginning with ``#`` are ignored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Canonical column names, in order.
COLUMNS = ["pc", "outcome"]


def _parse_pc(token: str) -> int:
    """Parse a PC token that may be decimal or ``0x``-prefixed hexadecimal."""
    token = token.strip()
    return int(token, 16) if token.lower().startswith("0x") else int(token)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a trace DataFrame to the canonical schema."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Trace is missing required column(s): {missing}")

    df = df[COLUMNS].copy()
    df["pc"] = df["pc"].astype("int64")
    df["outcome"] = df["outcome"].astype("int8")

    if len(df) == 0:
        raise ValueError("Trace is empty.")
    if (df["pc"] < 0).any():
        raise ValueError("Trace contains negative PC values.")
    bad = ~df["outcome"].isin([0, 1])
    if bad.any():
        raise ValueError(
            f"Trace 'outcome' must be 0 or 1; found other values in {int(bad.sum())} row(s)."
        )
    return df.reset_index(drop=True)


def _load_text(path: Path) -> pd.DataFrame:
    """Load a whitespace ``<pc> <outcome>`` trace file."""
    pcs: list[int] = []
    outcomes: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<pc> <outcome>', got: {line!r}"
                )
            pcs.append(_parse_pc(parts[0]))
            outcomes.append(int(parts[1]))
    return pd.DataFrame({"pc": pcs, "outcome": outcomes})


def load_trace(path: str | Path) -> pd.DataFrame:
    """Load a branch trace into the canonical ``(pc, outcome)`` DataFrame.

    Parameters
    ----------
    path:
        Path to a ``.csv`` or whitespace ``.txt`` / ``.trace`` file.

    Returns
    -------
    pandas.DataFrame
        Columns ``[pc, outcome]`` with a fresh ``RangeIndex`` representing
        dynamic execution order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = _load_text(path)

    return _validate(df)


def save_trace(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a canonical trace DataFrame to CSV, creating parent dirs as needed.

    Returns the resolved output path.
    """
    df = _validate(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
