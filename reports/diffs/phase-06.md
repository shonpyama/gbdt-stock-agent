# Phase 06 Diff Summary

- strengthened Colab file sync safety:
  - replaced second-level mtime compare with nanosecond compare.
  - added content-hash fallback for metadata files (`state/*.json`, `metrics.json`, `report.md`, logs) when mtime/size match.
- enabled durable checkpoint continuation:
  - `run` now supports `--run-id`, `--allow-conf-mismatch-resume`, `--checkpoint-drive-path`.
  - orchestrator now writes stage checkpoint sync records to `artifacts/runs/<run_id>/logs/checkpoint_sync.log`.
  - stage completion now triggers optional drive sync via `GBDT_CHECKPOINT_DRIVE_PATH`.
- state durability improvements:
  - save per-run state snapshots under `state/runs/<run_id>.json`.
  - migration bundle now includes `state/runs/<run_id>.json`.
- Colab notebook updated for safer resume default (`force_unlock=True` in stage runner).
- README updated with deterministic crash-recovery command sequence.
