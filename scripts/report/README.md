# Report Scripts

This directory hosts stage-oriented experiment/report drivers.

## Conventions

- Script naming:
  - `run_stageXX_*.py` for one stage entrypoint.
  - `bench_*.py` for framework-side micro/rollout benchmark helpers.
- Output naming:
  - default output directory: `reports/<stage_name>_<timestamp>`
  - deterministic runs should pass explicit `--out-dir`.
- Shared helpers:
  - use `common_report.py` for `die/read_csv/write_csv/safe_float/split_ints`.
  - avoid duplicating the same utility functions in new scripts.

## Quality Guardrails

- Validate all required paths early and fail fast with actionable messages.
- Keep outputs machine-readable first (`.csv/.json`), then add `.md` summary.
- Keep a single source of truth per metric table; avoid hand-edited report files.
- For simulated/proxy experiments, explicitly mark assumptions in summary markdown.

## Next Cleanup Candidates

- Migrate older stage scripts still carrying local copies of CSV/parse helpers to `common_report.py`.
- Unify CLI flags for common knobs (`--num-rounds`, `--num-gpus`, `--out-dir`) where possible.
- Add lightweight regression checks that compare key output CSV schemas across runs.

