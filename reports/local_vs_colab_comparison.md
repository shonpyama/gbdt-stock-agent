# Local vs Colab Comparison

- generated_at: 2026-02-21
- branch: `codex/phase-00-bootstrap`
- pull_request: https://github.com/shonpyama/gbdt-stock-agent/pull/1

## Local

- 実装検証: `pytest -q` => `14 passed`
- CLI検証:
  - `python -m gbdt_agent.cli --help`
  - `python -m gbdt_agent.cli migrate pack --run-id local-smoke --out /tmp/local-smoke-bundle.zip`
  - `python -m gbdt_agent.cli migrate restore --archive /tmp/local-smoke-bundle.zip`
  - `python -m gbdt_agent.cli transition-report --run-id local-smoke --target colab`
- 注意: preflightは現在ネットワーク制限下で接続失敗するが、エラーはキーを含まず安全に返却。

## Colab

- 実行方式: `/content` ローカル実行 + 10分Drive同期
- ノートブック: `notebooks/gbdt_stock_agent_colab.ipynb`
- 未実施項目: 実際のColab GPUランタイム上での stage_00->stage_80 完走
- 移行ゲート: `reports/pre_colab_transition_*.md` を提出し、明示承認後に実行

## Gap / Next

1. Colab GPUランタイムで本番設定 run を1回完走
2. Colab側レビュー（Claude Agent Teams glm_only）結果を `reports/reviews/phase-colab.md` に保存
3. 同期後のDrive成果物manifestを更新
