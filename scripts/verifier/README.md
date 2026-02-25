# Verifier Scripts

Reward/evaluation helpers used by stage 5.x training loops.

## Scripts

- `extraction_verifier.py`: JSON extraction verifier
  - parse validity
  - schema checks
  - field-level matching
  - reward modes: `binary`, `weighted`, `decomposed`
- `sql_verifier.py`: SQL verifier over SQLite
  - execute predicted SQL safely (SELECT/WITH only)
  - compare result sets against gold query outputs

## Intended Use

- Run standalone for reward diagnostics.
- Integrate into Best-of-N / DPO / GRPO training loops as reward backends.

