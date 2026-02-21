from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent import orchestrator as rexp
from gbdt_agent.data_store import UniverseResult
from gbdt_agent.features import FeatureBuildResult
from gbdt_agent.logging_utils import EnvSnapshot


def _make_conf(project_dir: Path) -> dict:
    return {
        "project": {"project_dir": str(project_dir)},
        "run": {
            "seed": 7,
            "log_level": "INFO",
            "timezone_assumption": "US/Eastern; events_effective=next_trading_day",
        },
        "universe": {
            "name": "synthetic10",
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2024-08-31",
            "adjusted_flag": True,
            "endpoints_version": ["prices_eod", "splits", "dividends"],
            "include_news": False,
        },
        "fmp": {
            "base_url": "https://financialmodelingprep.com/stable",
            "timeout": 30,
            "rate_limit": {"max_calls_per_minute": 300},
            "retry": {"max_attempts": 3, "backoff_base_seconds": 0.1},
            "endpoints": {},
        },
        "validation": {
            "thresholds": {
                "missing_ratio_max": 0.1,
                "anomaly_row_ratio_max": 0.1,
                "duplicates_ratio_max": 0.0,
                "min_obs_per_symbol": 40,
                "min_symbols_after_filter": 5,
                "adjclose_missing_ratio_max": 0.1,
                "price_jump_threshold": 0.5,
                "volume_zero_streak_len": 5,
                "volume_zero_streak_ratio_max": 0.5,
            }
        },
        "features": {"lookbacks": [1, 5, 20], "event_safe_shift_days": 1},
        "labels": {"horizon_trading_days": 1, "target_col": "future_return_1d"},
        "split": {
            "walkforward": {"train": 40, "val": 20, "test": 20, "step": 20},
            "purge": 1,
            "embargo": 1,
            "leakage": {"corr_threshold": 0.995},
        },
        "models": {
            "gbdt": {"enabled": True, "framework": "sklearn", "params": {"n_estimators": 20, "max_depth": 2}},
        },
        "backtest": {
            "long_short": False,
            "topn": 3,
            "max_names": 3,
            "single_name_cap": 0.5,
            "rebalance": "daily",
            "notional": 1_000_000.0,
        },
        "cost_model": {"commission_bps": 1.0, "slippage": {"base_bps": 1.0, "k_adv": 0.0, "k_vol": 0.0}},
    }


def _install_fakes(monkeypatch, project_dir: Path) -> None:
    def fake_from_env(*args, **kwargs):
        return object()

    def fake_capture_env_snapshot(run_dir: Path) -> EnvSnapshot:
        run_dir.mkdir(parents=True, exist_ok=True)
        env = run_dir / "env.json"
        pipf = run_dir / "pip_freeze.txt"
        gpu = run_dir / "gpu_info.txt"
        env.write_text(json.dumps({"python": "test"}))
        pipf.write_text("pytest")
        gpu.write_text("no gpu")
        return EnvSnapshot(env_json_path=str(env), pip_freeze_path=str(pipf), gpu_info_path=str(gpu))

    def fake_update_data(cfg, paths, fmp, force=False):
        symbols = [f"S{i:02d}" for i in range(10)]
        dates = pd.bdate_range(cfg["data"]["start_date"], cfg["data"]["end_date"]).date
        dataset_id = "synth_dataset"
        raw_dir = paths.raw_dir / dataset_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        p_rows = []
        for j, s in enumerate(symbols):
            px = 100.0 + j
            for i, d in enumerate(dates):
                px = px * (1.0 + 0.0005 + ((i + j) % 7) * 0.0001)
                p_rows.append(
                    {
                        "date": d,
                        "symbol": s,
                        "open": px * 0.999,
                        "high": px * 1.003,
                        "low": px * 0.997,
                        "close": px,
                        "adj_close": px,
                        "volume": 100000 + (i % 20) * 1000,
                    }
                )
        prices = pd.DataFrame(p_rows)
        mem = pd.DataFrame([{"date": d, "symbol": s, "is_member": True} for d in dates for s in symbols])
        prices.to_parquet(raw_dir / "prices.parquet", index=False)
        mem.to_parquet(raw_dir / "universe_membership.parquet", index=False)
        pd.DataFrame(columns=["date", "symbol"]).to_parquet(raw_dir / "splits.parquet", index=False)
        pd.DataFrame(columns=["date", "symbol"]).to_parquet(raw_dir / "dividends.parquet", index=False)
        pd.DataFrame(columns=["date", "symbol"]).to_parquet(raw_dir / "earnings.parquet", index=False)
        (raw_dir / "dataset_spec.json").write_text(json.dumps({"dataset_id": dataset_id}))

        uni = UniverseResult(
            provider_used="synthetic",
            fallback_used=False,
            fallback_reason="",
            end_date_truncated=False,
            start_date=str(dates[0]),
            end_date_effective=str(dates[-1]),
            membership_long=mem,
            symbols_union=symbols,
        )
        return dataset_id, uni, str(dates[-1])

    monkeypatch.setattr(rexp.FMPClient, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(rexp, "capture_env_snapshot", fake_capture_env_snapshot)
    monkeypatch.setattr(rexp, "update_data", fake_update_data)


def test_pipeline_runs_end_to_end_with_resume(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "qp_proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    conf = _make_conf(project_dir)
    conf_path = project_dir / "conf.yaml"
    conf_path.write_text(yaml.safe_dump(conf, sort_keys=False))
    _install_fakes(monkeypatch, project_dir)

    run_id1 = rexp.run_pipeline(
        project_dir=project_dir,
        conf_path=conf_path,
        resume=False,
        force_unlock=True,
        stop_after_stage="stage_30_features_ready",
    )
    run_id2 = rexp.run_pipeline(
        project_dir=project_dir,
        conf_path=conf_path,
        resume=True,
        force_unlock=True,
    )
    assert run_id1 == run_id2
    run_dir = project_dir / "artifacts" / "runs" / run_id2
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.parquet").exists()
    assert (run_dir / "backtest.parquet").exists()
    assert (run_dir / "report.md").exists()
    state = json.loads((project_dir / "state" / "last_run_state.json").read_text())
    assert state["stage"] == "stage_80_report_ready"


def test_pipeline_fails_on_deliberate_leak(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "qp_proj_leak"
    project_dir.mkdir(parents=True, exist_ok=True)
    conf = _make_conf(project_dir)
    conf_path = project_dir / "conf.yaml"
    conf_path.write_text(yaml.safe_dump(conf, sort_keys=False))
    _install_fakes(monkeypatch, project_dir)

    def fake_build_feature_store(
        *,
        dataset_id,
        prices,
        universe_membership,
        earnings,
        adjusted_flag,
        lookbacks,
        event_safe_shift_days,
        out_dir,
    ):
        px = prices.copy()
        px["date"] = pd.to_datetime(px["date"]).dt.date
        px = px.sort_values(["symbol", "date"])
        px["leak_feature"] = px.groupby("symbol")["close"].shift(-1) / px["close"] - 1.0
        feats = px.rename(columns={"date": "decision_date"})[["decision_date", "symbol", "leak_feature"]].dropna().copy()
        feats["feature_available_date"] = (pd.to_datetime(feats["decision_date"]) - pd.Timedelta(days=1)).dt.date
        feats["feature_version"] = "leak"
        fs_id = "leak_store"
        fs_dir = out_dir / fs_id
        fs_dir.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(fs_dir / "features.parquet", index=False)
        (fs_dir / "feature_spec.json").write_text(json.dumps({"leak": True}))
        return FeatureBuildResult(feature_store_id=fs_id, features=feats, spec={"leak": True})

    monkeypatch.setattr(rexp, "build_feature_store", fake_build_feature_store)

    with pytest.raises(rexp.GateFailed) as ei:
        rexp.run_pipeline(project_dir=project_dir, conf_path=conf_path, resume=False, force_unlock=True)
    assert ei.value.stage == "stage_40_split_leakcheck_failed"
    state = json.loads((project_dir / "state" / "last_run_state.json").read_text())
    assert state["stage"] == "stage_40_split_leakcheck_failed"


def test_resume_archives_old_errors_in_metrics(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "qp_proj_resume_errors"
    project_dir.mkdir(parents=True, exist_ok=True)
    conf = _make_conf(project_dir)
    conf_path = project_dir / "conf.yaml"
    conf_path.write_text(yaml.safe_dump(conf, sort_keys=False))
    _install_fakes(monkeypatch, project_dir)

    original_train_models = rexp.train_models
    calls = {"n": 0}

    def flaky_train_models(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic training failure")
        return original_train_models(*args, **kwargs)

    monkeypatch.setattr(rexp, "train_models", flaky_train_models)

    with pytest.raises(RuntimeError):
        rexp.run_pipeline(project_dir=project_dir, conf_path=conf_path, resume=False, force_unlock=True)

    run_id = json.loads((project_dir / "state" / "last_run_state.json").read_text())["run_id"]
    rexp.run_pipeline(project_dir=project_dir, conf_path=conf_path, resume=True, force_unlock=True)

    metrics = json.loads((project_dir / "artifacts" / "runs" / run_id / "metrics.json").read_text())
    assert metrics["status"] == "success"
    assert metrics.get("errors") == []
    assert len(metrics.get("historical_errors") or []) >= 1
