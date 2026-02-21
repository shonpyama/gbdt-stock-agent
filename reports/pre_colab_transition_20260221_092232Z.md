# Pre-Colab Transition Report

- target: `colab`
- run_id: `20260221_092119Z_021cc3de_c507b2ff`
- status: `success`
- current_stage: `stage_80_report_ready`
- generated_at: `2026-02-21T09:22:32.887406+00:00`

## Local Validation Summary

- leakage_passed: `True`
- chosen_model: `gbdt`
- sharpe: `nan`
- max_drawdown: `nan`
- total_return: `nan`

## Errors

- error_count: `0`
## Git Diff Summary

```
M conf/default.yaml
 M conf/universe_custom.yaml
 M src/gbdt_agent/models/gbdt.py
?? conf/smoke_local.yaml
?? reports/pre_colab_transition_20260221_091601Z.md
?? requirements.txt
```

## Recent Commits

```
4beb145 phase-06 add transition artifacts and local-vs-colab comparison report
7cc3fe2 phase-05 add phase diff and review records
ee1d021 phase-04 add staged colab notebook and review operation scripts
2fba9a3 phase-03 implement cli, migration bundle, and colab drive sync commands
958d55f phase-02 add local e2e and unit tests for pipeline, migration, and sync
841b862 phase-01 migrate core gbdt pipeline and stage orchestrator
a6d4acb phase-00 bootstrap repository and base configs
```

## Artifact Manifest

```json
{
  "run_id": "20260221_092119Z_021cc3de_c507b2ff",
  "stage": "stage_80_report_ready",
  "generated_at": "2026-02-21T09:21:37.200279+00:00",
  "github_lightweight": [
    "conf/",
    "src/",
    "notebooks/",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/metrics.json",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/report.md",
    "reports/"
  ],
  "drive_heavy": [
    "data/raw/",
    "data/cache_http/",
    "artifacts/runs/",
    "state/",
    "logs/"
  ],
  "run_files": [
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/artifact_manifest.json",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/backtest.parquet",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/backtest_positions.parquet",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/config.yaml",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/env.json",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/gpu_info.txt",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/leakage.json",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/logs/run.log",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/metrics.json",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/model_ckpt/gbdt_model.pkl",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/model_ckpt/gbdt_model.pkl.meta.json",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/pip_freeze.txt",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/predictions.parquet",
    "artifacts/runs/20260221_092119Z_021cc3de_c507b2ff/report.md"
  ]
}
```

## Migration Risks

- Colabランタイム再起動時に未同期データが失われる可能性。
- FMP APIレート制限到達時にデータ更新ステージが遅延する可能性。
- GPUランタイム差異で学習再現性が揺らぐ可能性。

## Rollback Plan

1. `python -m gbdt_agent.cli migrate pack --run-id <run_id> --out <zip>` でローカル状態を固定。
2. Colabで失敗時は `python -m gbdt_agent.cli colab restore --drive-path <path>` で復元。
3. `python -m gbdt_agent.cli run --conf conf/default.yaml --resume --force-stage <stage>` で再開。

## Approval Gate

Colab実行は**ユーザー明示承認後のみ**開始可能です。
