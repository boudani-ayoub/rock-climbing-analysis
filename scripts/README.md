# Planned Utilities

This folder is reserved for small reproducibility tools after the first hardware
acceptance run. Planned utilities are:

- `benchmark.py`: run a fixed clip, collect per-stage timing, and write a machine-
  readable report with environment and model hashes.
- `validate_models.py`: print RKNN input/output contracts and run a small set of
  known frames through the decoders.
- `compare_runs.py`: compare contact events, paths, coverage, and scores between
  two versioned reports.

Do not duplicate production tracking or scoring logic in these scripts. They
should call the eventual package API after the single-file pipeline is validated
and split into modules.
