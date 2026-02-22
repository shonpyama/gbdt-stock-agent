# Production Readiness Check

- generated_at: `2026-02-22T01:30:20.952524+00:00`
- readiness_score: `100.0`
- readiness_100: `True`

## Checks

- model_stability: `True`
- feature_stability: `True`
- ops_gate: `True`
- default_conf_alignment: `True`

## Summary

- model_selected: `{"name": "compact_31", "params": {"n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 20}, "periods": 3, "gate_pass_periods": 3, "all_periods_gate_pass": true, "score_mean": 34.02030525452115, "score_min": 29.095079030378983, "rank_ic_mean": 0.08973324555357094, "sharpe_mean": 17.063309547907604, "total_return_mean": 66.20169121648195, "max_drawdown_mean": -0.13343505755514698}`
- feature_selected: `{"name": "baseline_current", "lookbacks": [1, 5, 20, 60, 120], "event_shift": 1, "periods": 3, "gate_pass_periods": 3, "all_periods_gate_pass": true, "score_mean": 21.096182615939554, "score_min": 16.719322884428575, "rank_ic_mean": 0.05301766087253341, "sharpe_mean": 11.660525338141118, "total_return_mean": 33.647578782847866, "max_drawdown_mean": -0.2352365084190463}`
- ops_gate_violations: `[]`
