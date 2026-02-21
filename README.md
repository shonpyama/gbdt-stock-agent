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
- `python -m gbdt_agent.cli report --run-id <id>`
- `python -m gbdt_agent.cli transition-report --run-id <id> --target colab`
- `python -m gbdt_agent.cli migrate pack --run-id <id> --out <zip>`
- `python -m gbdt_agent.cli migrate restore --archive <zip>`
- `python -m gbdt_agent.cli colab restore --drive-path <path>`
- `python -m gbdt_agent.cli colab sync --drive-path <path>`
- `python -m gbdt_agent.cli ops-status --max-age-hours 72 --require-gpu`
- `python -m gbdt_agent.cli ops-snapshot --max-age-hours 72 --require-gpu`
- `python -m gbdt_agent.cli ops-gate --policy conf/ops_policy.yaml`

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
- 運用ゲート（品質/鮮度の合否判定）: `python -m gbdt_agent.cli ops-gate --policy conf/ops_policy.yaml`
- 閾値は `conf/ops_policy.yaml` で調整できます。ゲートNG時は `reports/ops_incident_*.md` と `logs/ops/` に記録されます。
- `colab sync` は `reports/` を含めて同期するため、差分・レビュー・運用記録もDriveへ保存されます。

## 注意
- APIキーは `FMP_API_KEY` を優先し、未設定時は `/content/.env_fmp`（または `FMP_API_KEY_FILE` 指定ファイル）を参照します。
- ログはキー値をマスクします。
- LightGBM は `models.gbdt.prefer_gpu` (既定: `true`) でGPUを自動利用し、利用不可時はCPUへ自動フォールバックします。
