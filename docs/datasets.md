# Branch-Trace Datasets — Research Notes & Decision

Per the project requirement, publicly available CPU branch-trace datasets were
investigated **before** committing to synthetic data. This note records what
exists, why it is hard to use directly, and the resulting design decision.

## Public datasets that exist

| Source | Contents | Access | Format |
|--------|----------|--------|--------|
| **Championship Branch Prediction — CBP2016 / CBP5** | 440+ traces from Samsung (long/short × mobile/server workloads) | Download link via `cbp_2016` Google group / NCSU velocity server subscription | Compressed **binary** trace, replayed by the CBP simulator |
| **CBP3 (JWAC-2)** | Trace suite (> 2.1 GB total) | `cbp3` mailing-list subscription | CBP binary trace format |
| **ChampSim traces** | Traces used by the ChampSim simulator; CBP5 traces linked from its repo | Public links, but large | ChampSim binary (`.champsimtrace.xz`) |
| **CBP2025 (6th championship)** | Latest simulator + traces | Published on the CBP2025 website | CBP binary trace format |

References:
- CBP2025 — https://ericrotenberg.wordpress.ncsu.edu/cbp2025/
- CBP framework/kit — https://jilp.org/cbp2016/framework.html
- ChampSim — https://github.com/ChampSim/ChampSim
- CBP-16 simulation & Python tooling — https://github.com/craymichael/CBP-16-Simulation

## Why they are not used as the default in this repo

1. **Access-gated** — traces require mailing-list / Google-group subscription
   rather than a direct, scriptable download, so a fresh clone cannot fetch them
   reproducibly.
2. **Size** — suites run to multiple gigabytes; they cannot be committed and are
   impractical for a portfolio CI pipeline.
3. **Binary format** — traces are compressed binary streams meant to be *replayed
   by a specific simulator* (CBP/ChampSim). Extracting a plain
   `(PC, taken/not-taken)` sequence requires standing up that simulator toolchain
   — out of scope for a self-contained project.

## Decision: support both, default to synthetic

The pipeline is built around a **canonical trace schema** — a table of
`(pc, outcome)` rows in program (dynamic) order — and supports two sources:

* **`file`** — `trace_loader.load_trace()` ingests any trace already in the
  simple `PC outcome` text/CSV form. This is exactly the record you get when you
  decode a CBP/ChampSim trace with their tools, so **real datasets are fully
  supported**: drop the decoded file in `data/raw/` and point the config at it.
* **`synthetic`** (default) — `trace_generator.generate_trace()` produces a
  realistic, program-like branch stream (biased loops, nested loops, history-
  correlated branches, call/return, and noise) that is **deterministic given a
  seed**, so every reported number in this repo is exactly reproducible with no
  external downloads.

This keeps the project reproducible out of the box while remaining
real-data-ready by design.
