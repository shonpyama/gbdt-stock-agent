from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_suggestions(
    *,
    validation: Optional[Dict[str, Any]],
    leakage: Optional[Dict[str, Any]],
    backtest_summary: Optional[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    if validation and validation.get("missing_ratio_overall") is not None:
        if float(validation.get("missing_ratio_overall", 0.0)) > 0.02:
            out.append("欠損率が高めのため、銘柄除外または補完ルールの強化を検討。")
    if leakage and not leakage.get("passed", False):
        out.append("リークゲート不合格。特徴量公開タイミングとラベル窓の再監査を優先。")
    if backtest_summary:
        sharpe = backtest_summary.get("sharpe")
        if sharpe is not None:
            try:
                if float(sharpe) < 0.5:
                    out.append("Sharpeが低いため、特徴量拡充とコスト係数の感度分析を実施。")
            except Exception:
                pass
        mdd = backtest_summary.get("max_drawdown")
        if mdd is not None:
            try:
                if float(mdd) < -0.25:
                    out.append("最大DDが大きいため、ポジション上限とリバランス頻度の再調整を推奨。")
            except Exception:
                pass
    if not out:
        out.append("次回はvalidation閾値とコスト係数を固定した上で、特徴量セットのみ差分比較を推奨。")
    return out


def render_report_md(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    conf_hash: str,
    dataset_id: Optional[str],
    feature_store_id: Optional[str],
    code_hash: Optional[str],
    universe_info: Optional[Dict[str, Any]],
    split_info: Optional[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    leakage: Optional[Dict[str, Any]],
    training_info: Optional[Dict[str, Any]],
    model_metrics: Optional[Dict[str, Any]],
    chosen_model: Optional[str],
    backtest_summary: Optional[Dict[str, Any]],
    status: str,
    errors: Optional[Any],
    historical_errors: Optional[Any] = None,
) -> str:
    run = cfg.get("run", {}) or {}
    data = cfg.get("data", {}) or {}
    models = cfg.get("models", {}) or {}
    split = cfg.get("split", {}) or {}

    lines = []
    lines.append(f"# GBDT Stock Agent Report")
    lines.append("")
    lines.append(f"- generated_at: `{_utc_now()}`")
    lines.append(f"- status: `{status}`")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- conf_hash: `{conf_hash[:12]}`")
    if dataset_id:
        lines.append(f"- dataset_id: `{dataset_id}`")
    if feature_store_id:
        lines.append(f"- feature_store_id: `{feature_store_id}`")
    if code_hash:
        lines.append(f"- code_hash: `{code_hash[:12]}`")
    lines.append(f"- seed: `{run.get('seed')}`")
    lines.append("")

    lines.append("## Data Spec")
    lines.append("")
    lines.append(f"- start_date: `{data.get('start_date')}`")
    lines.append(f"- end_date: `{data.get('end_date') or ''}`")
    lines.append(f"- end_date_effective: `{(universe_info or {}).get('end_date_effective','')}`")
    lines.append(f"- adjusted_flag: `{data.get('adjusted_flag')}`")
    lines.append(f"- endpoints_version: `{json.dumps(data.get('endpoints_version', []))}`")
    if universe_info:
        lines.append(f"- universe_provider_used: `{universe_info.get('provider_used')}`")
        lines.append(f"- universe_symbols_count: `{universe_info.get('symbols_count')}`")
        lines.append(f"- universe_fallback_used: `{universe_info.get('fallback_used')}`")
        if universe_info.get("fallback_used"):
            lines.append(f"- universe_fallback_reason: `{universe_info.get('fallback_reason')}`")
        lines.append(f"- end_date_truncated: `{universe_info.get('end_date_truncated')}`")
    lines.append(f"- timezone_assumption: `{run.get('timezone_assumption')}`")
    lines.append("")

    lines.append("## Split Settings")
    lines.append("")
    lines.append(f"- walkforward: `{json.dumps((split.get('walkforward') or {}))}`")
    lines.append(f"- purge: `{split.get('purge')}`")
    lines.append(f"- embargo: `{split.get('embargo')}`")
    if split_info:
        lines.append(f"- train_window: `{json.dumps(split_info.get('train_window'))}`")
        lines.append(f"- val_window: `{json.dumps(split_info.get('val_window'))}`")
        lines.append(f"- test_window: `{json.dumps(split_info.get('test_window'))}`")
    lines.append("")

    lines.append("## Validation Gate")
    lines.append("")
    if validation:
        lines.append(f"- passed: `{validation.get('passed')}`")
        if validation.get("fail_reasons"):
            lines.append(f"- fail_reasons: `{json.dumps(validation.get('fail_reasons'))}`")
        lines.append(f"- missing_ratio_overall: `{validation.get('missing_ratio_overall')}`")
        lines.append(f"- ohlc_anomaly_ratio: `{validation.get('ohlc_anomaly_ratio')}`")
        lines.append(f"- symbols_excluded_min_obs: `{validation.get('symbols_excluded_min_obs')}`")
    else:
        lines.append("- (no validation recorded)")
    lines.append("")

    lines.append("## Leakage Gate")
    lines.append("")
    if leakage:
        lines.append(f"- passed: `{leakage.get('passed')}`")
        lines.append(f"- critical: `{json.dumps(leakage.get('critical', {}))}`")
        if leakage.get("messages"):
            lines.append(f"- messages: `{json.dumps(leakage.get('messages'))}`")
        if leakage.get("suspicious_feature_corr"):
            lines.append(f"- suspicious_feature_corr(top): `{json.dumps(leakage.get('suspicious_feature_corr')[:10])}`")
    else:
        lines.append("- (no leakage audit recorded)")
    lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append(f"- models_enabled: `{json.dumps({k:v.get('enabled', True) for k,v in models.items() if isinstance(v, dict)})}`")
    if model_metrics:
        for m, mm in model_metrics.items():
            lines.append(f"### {m}")
            lines.append("")
            for split_name in ["val", "test"]:
                if split_name in mm:
                    rank_ic = mm[split_name].get("rank_ic", {})
                    ic = mm[split_name].get("ic", {})
                    lines.append(f"- {split_name}.rank_ic_mean: `{rank_ic.get('mean', mm[split_name].get('mean'))}`")
                    lines.append(f"- {split_name}.rank_ic_std: `{rank_ic.get('std', mm[split_name].get('std'))}`")
                    lines.append(f"- {split_name}.ic_mean: `{ic.get('mean')}`")
                    lines.append(f"- {split_name}.ic_std: `{ic.get('std')}`")
                    lines.append(f"- {split_name}.n_days: `{rank_ic.get('n', mm[split_name].get('n'))}`")
            lines.append("")
    else:
        lines.append("- (no model metrics recorded)")
        lines.append("")

    lines.append("## Training Runtime")
    lines.append("")
    gbdt_train = (training_info or {}).get("gbdt") if isinstance(training_info, dict) else None
    if isinstance(gbdt_train, dict):
        for k in ["framework", "accelerator", "gpu_attempted", "prefer_gpu", "val_mse"]:
            if k in gbdt_train:
                lines.append(f"- {k}: `{gbdt_train.get(k)}`")
    else:
        lines.append("- (no training runtime recorded)")
    lines.append("")

    lines.append("## Backtest")
    lines.append("")

    lines.append("## Improvement Suggestions")
    lines.append("")
    for s in _auto_suggestions(validation=validation, leakage=leakage, backtest_summary=backtest_summary):
        lines.append(f"- {s}")
    lines.append("")
    if chosen_model:
        lines.append(f"- chosen_model: `{chosen_model}`")
    if backtest_summary:
        for k in ["sharpe", "max_drawdown", "total_return", "avg_turnover", "avg_cost", "days"]:
            if k in backtest_summary:
                lines.append(f"- {k}: `{backtest_summary.get(k)}`")
    else:
        lines.append("- (no backtest recorded)")
    lines.append("")

    lines.append("## Repro")
    lines.append("")
    lines.append("CLI (Colab/Drive):")
    lines.append("")
    lines.append("```bash")
    lines.append("export PYTHONPATH=/content/gbdt-stock-agent/src")
    lines.append("cd /content/gbdt-stock-agent")
    lines.append("python -m gbdt_agent.cli preflight --conf conf/default.yaml")
    lines.append("python -m gbdt_agent.cli run --conf conf/default.yaml --resume")
    lines.append("python -m gbdt_agent.cli report --run-id " + run_id)
    lines.append("python -m gbdt_agent.cli transition-report --run-id " + run_id + " --target colab")
    lines.append("```")
    lines.append("")

    errs = errors if isinstance(errors, list) else []
    hist_errs = historical_errors if isinstance(historical_errors, list) else []
    if status.upper() == "SUCCESS" and errs and not hist_errs:
        # Backward compatibility for older metrics that stored only `errors`.
        hist_errs = list(errs)
        errs = []
    if errs or hist_errs:
        lines.append("## Errors")
        lines.append("")
        lines.append(f"- active_error_count: `{len(errs)}`")
        lines.append(f"- historical_error_count: `{len(hist_errs)}`")
        if errs:
            lines.append("```json")
            lines.append(json.dumps(errs, indent=2, ensure_ascii=True))
            lines.append("```")

    return "\n".join(lines).strip() + "\n"


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
