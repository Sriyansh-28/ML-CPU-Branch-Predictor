"""Tests for the canonical trace loader/saver.

Covers the round-trip through CSV, parsing of the whitespace ``PC outcome`` text
format (including hex PCs and comments), and the schema-validation guarantees the
rest of the pipeline relies on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from branchpred.data.trace_loader import load_trace, save_trace


def test_csv_round_trip(tmp_path) -> None:
    df = pd.DataFrame({"pc": [0x400000, 0x400004, 0x400000], "outcome": [1, 0, 1]})
    path = save_trace(df, tmp_path / "trace.csv")
    loaded = load_trace(path)
    pd.testing.assert_frame_equal(loaded, df.astype({"pc": "int64", "outcome": "int8"}))


def test_text_format_decimal(tmp_path) -> None:
    p = tmp_path / "trace.txt"
    p.write_text("4194304 1\n4194308 0\n4194304 1\n")
    df = load_trace(p)
    assert list(df["pc"]) == [4194304, 4194308, 4194304]
    assert list(df["outcome"]) == [1, 0, 1]


def test_text_format_hex_and_comments(tmp_path) -> None:
    p = tmp_path / "trace.trace"
    p.write_text("# a real decoded trace\n0x400000 1\n\n0x400004 0\n")
    df = load_trace(p)
    assert list(df["pc"]) == [0x400000, 0x400004]
    assert list(df["outcome"]) == [1, 0]


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trace(tmp_path / "nope.csv")


def test_bad_outcome_rejected(tmp_path) -> None:
    df = pd.DataFrame({"pc": [1, 2], "outcome": [1, 2]})
    with pytest.raises(ValueError):
        save_trace(df, tmp_path / "bad.csv")


def test_negative_pc_rejected(tmp_path) -> None:
    df = pd.DataFrame({"pc": [-1, 2], "outcome": [1, 0]})
    with pytest.raises(ValueError):
        save_trace(df, tmp_path / "bad.csv")


def test_missing_column_rejected(tmp_path) -> None:
    p = tmp_path / "bad.csv"
    pd.DataFrame({"pc": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_trace(p)


def test_malformed_text_line_rejected(tmp_path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("0x400000 1 extra\n")
    with pytest.raises(ValueError):
        load_trace(p)


def test_save_creates_parent_dirs(tmp_path) -> None:
    df = pd.DataFrame({"pc": [1], "outcome": [1]})
    out = save_trace(df, tmp_path / "nested" / "dir" / "trace.csv")
    assert out.exists()
