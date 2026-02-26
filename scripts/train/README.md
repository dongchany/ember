# Train Scripts

Minimal training/baseline utilities for stage 5.x.

## Current Entrypoints

- `train_min_lora_adapter.py`: small PEFT SFT adapter trainer.
- `run_stage52_best_of_n_extraction.py`: Best-of-N baseline with reward metrics.
- `run_stage52_build_dpo_pairs.py`: build chosen/rejected pairs from candidate outputs.
- `run_stage52_dpo_min.py`: minimal DPO loop on pair data (LoRA policy).
- `run_stage52_validate_dataset.py`: dataset validator (schema/type/leakage/split-overlap checks).
- `run_stage52_snapshot_dataset.py`: freeze train/val/test + schema with SHA256 manifest for reproducibility.

## Notes

- These scripts are intentionally lightweight (few dependencies and explicit outputs).
- Smoke datasets under `reports/*_smoke_*` are for pipeline validation only.
- For paper-quality results, replace smoke data with real extraction datasets.
- External dataset generation template: `docs/stage52_external_dataset_generation_template.md`.
