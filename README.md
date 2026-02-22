# gbdt-stock-agent

GBDTベースの株式予測エージェントです。ローカル実行を起点に、Google Colab(GPU)へスムーズに移行できるように設計しています。

## 主要要件
- タスク: S&P500 PIT を対象とした 20営業日先リターン予測
- モデル: LightGBM 主体 (CPU/GPUフォールバック)
- ステージ: `stage_00` 〜 `stage_80` の段階実行・再開
- 保存: GitHub(軽量) + Google Drive(重量)
- 移行: `transition-report` の承認前は Colab 実行不可

## クイックスタート
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export FMP_API_KEY="..."
# or put key in /content/.env_fmp (or path in FMP_API_KEY_FILE)
python -m gbdt_agent.cli preflight --conf conf/default.yaml
python -m gbdt_agent.cli run --conf conf/default.yaml --resume
```

## CLI
- `python -m gbdt_agent.cli preflight --conf <path>`
- `python -m gbdt_agent.cli run --conf <path> --resume --stop-after-stage <stage>`
- `python -m gbdt_agent.cli run --conf <path> --resume --run-id <id> --allow-conf-mismatch-resume`
- `python -m gbdt_agent.cli run --conf <path> --resume --checkpoint-drive-path <drive_path>`
- `python -m gbdt_agent.cli report --run-id <id>`
- `python -m gbdt_agent.cli transition-report --run-id <id> --target colab`
- `python -m gbdt_agent.cli migrate pack --run-id <id> --out <zip>`
- `python -m gbdt_agent.cli migrate restore --archive <zip>`
- `python -m gbdt_agent.cli colab restore --drive-path <path> --mode quick --run-id <id>`
- `python -m gbdt_agent.cli colab sync --drive-path <path> --mode quick --run-id <id>`
- `python -m gbdt_agent.cli ops-status --max-age-hours 72 --require-gpu`
- `python -m gbdt_agent.cli ops-snapshot --max-age-hours 72 --require-gpu`
- `python -m gbdt_agent.cli ops-gate --policy conf/ops_policy.yaml`
- `python -m gbdt_agent.cli gpu-usage --lookback-hours 24`

## ディレクトリ
- `src/gbdt_agent`: 実装本体
- `conf/`: 設定
- `artifacts/runs/<run_id>/`: 実行成果物
- `state/last_run_state.json`: 再開状態
- `reports/`: 差分/レビュー/移行前報告

## 運用
- 健全性チェック: `python -m gbdt_agent.cli ops-status --max-age-hours 72 --require-gpu`
- スナップショット保存: `python -m gbdt_agent.cli ops-snapshot --max-age-hours 72 --require-gpu`
- 一括運用（preflight→run→report→snapshot→sync）: `scripts/ops_autopilot.sh`
- モデル安定性検証（本番3期間）: `python scripts/model_stability_prod.py --base-conf conf/default.yaml --promote-default`
- 特徴量安定性検証（本番3期間）: `python scripts/feature_stability_prod.py --base-conf conf/default.yaml --promote-default`
- 実務 readiness 判定: `python scripts/prod_readiness_check.py --model-results reports/model_stability_prod_results.json --feature-results reports/feature_stability_prod_results.json --strict`
- 候補分割で並列実行（multi-agent）する場合:
  - 例1: `python scripts/model_stability_prod.py --only-candidates baseline_auto --out-json reports/model_stability_agent1.json --out-md reports/model_stability_agent1.md`
  - 例2: `python scripts/model_stability_prod.py --only-candidates compact_31 --out-json reports/model_stability_agent2.json --out-md reports/model_stability_agent2.md`
  - 例3: `python scripts/feature_stability_prod.py --only-candidates baseline_current --out-json reports/feature_stability_agent1.json --out-md reports/feature_stability_agent1.md`
- マージ（モデル）: `python scripts/stability_merge.py --inputs reports/model_stability_agent1.json,reports/model_stability_agent2.json --out-json reports/model_stability_merged.json --out-md reports/model_stability_merged.md`
- 採用反映（モデル）: `python scripts/promote_stability_model.py --results reports/model_stability_merged.json`
- マージ（特徴量）: `python scripts/stability_merge.py --inputs reports/feature_stability_agent1.json,reports/feature_stability_agent2.json --out-json reports/feature_stability_merged.json --out-md reports/feature_stability_merged.md`
- 採用反映（特徴量）: `python scripts/promote_stability_feature.py --results reports/feature_stability_merged.json`
- ローカル並列（worktree分離で同時実行）:
  - モデル: `python scripts/stability_multiagent.py --mode model --end-dates 2025-12-31,2026-01-31,2026-02-21 --promote-default`
  - 特徴量: `python scripts/stability_multiagent.py --mode feature --end-dates 2025-12-31,2026-01-31,2026-02-21 --promote-default`
- 安定性検証込み一括運用:
  - `RUN_MODEL_STABILITY=1 RUN_FEATURE_STABILITY=1 scripts/ops_autopilot.sh`
- readinessまで一括判定:
  - `RUN_MODEL_STABILITY=1 RUN_FEATURE_STABILITY=1 RUN_READINESS_CHECK=1 scripts/ops_autopilot.sh`
- 運用ゲート（品質/鮮度の合否判定）: `python -m gbdt_agent.cli ops-gate --policy conf/ops_policy.yaml`
- GPU実稼働率の確認（run単位/直近窓集計）: `python -m gbdt_agent.cli gpu-usage --run-id <id>` / `python -m gbdt_agent.cli gpu-usage --lookback-hours 24`
- 閾値は `conf/ops_policy.yaml` で調整できます。ゲートNG時は `reports/ops_incident_*.md` と `logs/ops/` に記録されます。
- `colab sync --mode quick` は `state/reports/logs` と `artifacts/runs/<run_id>` を中心に同期し、再開に必要な情報を残しつつ時間を短縮します。
- HTTPキャッシュ (`data/cache_http`) は Drive 側へリンクされるため、FMP API の同一リクエストは再DLを抑制できます。
- `colab sync` は `reports/` を含めて同期するため、差分・レビュー・運用記録もDriveへ保存されます。

## 注意
- APIキーは `FMP_API_KEY` を優先し、未設定時は `/content/.env_fmp`（または `FMP_API_KEY_FILE` 指定ファイル）を参照します。
- ログはキー値をマスクします。
- LightGBM は `models.gbdt.prefer_gpu` (既定: `true`) でGPUを自動利用し、利用不可時はCPUへ自動フォールバックします。
- 取得並列度は `data.fetch_workers` で調整できます（単一T4 + FMP標準レートでは `8` 前後が目安）。
- 前処理GPU化を試す場合は `GBDT_ENABLE_CUDF_PANDAS=1` を付けて実行します（`cudf.pandas` 未導入時は自動で通常pandasへフォールバック）。
- GPUサンプリングは既定で有効です（既定: `sample_interval_sec=1.0`, `active_util_threshold=5.0`, `active_mem_threshold_mib=128.0`）。`runtime.gpu_sampling.*` で調整できます。
- `conf/default.yaml` では `data.include_news: true` のため、`news.parquet` を取得できる場合はニュース由来特徴量（`news_*`, `mkt_news_*`）を自動で利用します。

## Colab切断時の確実再開
1. まず復元: `python -m gbdt_agent.cli colab restore --drive-path /content/drive/MyDrive/gbdt-stock-agent --mode quick`
2. 直近runを確認: `cat state/last_run_state.json`（`run_id` と `stage` を見る）
3. 同じrunを指定して再開:
   - `python -m gbdt_agent.cli run --conf conf/default.yaml --resume --run-id <run_id> --allow-conf-mismatch-resume --force-unlock --checkpoint-drive-path /content/drive/MyDrive/gbdt-stock-agent`
4. 局面終了時に明示同期:
   - `python -m gbdt_agent.cli colab sync --drive-path /content/drive/MyDrive/gbdt-stock-agent --mode quick --run-id <run_id>`

補足:
- stage完了ごとに `checkpoint_sync.log` へ同期結果（mode/件数/skip理由）を記録します。
- `artifacts/runs/<run_id>/auto_review.json` に自動レビュー（rank_ic整合・Sharpe整合の簡易診断）を保存します。
- `state/runs/<run_id>.json` も保存されるため、`last_run_state.json` が古くても run_id 指定で再開できます。
