# Pre-Colab Transition Report

- target: `colab`
- run_id: `local-smoke`
- status: `success`
- current_stage: `stage_80_report_ready`
- generated_at: `2026-02-21T07:51:25.862320+00:00`

## Local Validation Summary

- leakage_passed: `True`
- chosen_model: `gbdt`
- sharpe: `0.9`
- max_drawdown: `-0.12`
- total_return: `0.08`

## Errors

- error_count: `0`
## Git Diff Summary

```
(clean)
```

## Recent Commits

```
7cc3fe2 phase-05 add phase diff and review records
ee1d021 phase-04 add staged colab notebook and review operation scripts
2fba9a3 phase-03 implement cli, migration bundle, and colab drive sync commands
958d55f phase-02 add local e2e and unit tests for pipeline, migration, and sync
841b862 phase-01 migrate core gbdt pipeline and stage orchestrator
a6d4acb phase-00 bootstrap repository and base configs
```

## Artifact Manifest

```json
{}
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
