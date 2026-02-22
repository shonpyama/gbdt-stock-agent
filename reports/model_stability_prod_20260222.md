# Model Stability (Production) - 2026-02-22

- base_conf: `/content/gbdt-stock-agent/conf/default.yaml`
- policy: `/content/gbdt-stock-agent/conf/ops_policy.yaml`
- end_dates: `["2025-12-31", "2026-01-31", "2026-02-21"]`

## Per-Period

| end_date | model | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |
|---|---|---:|---:|---:|---:|---:|---|---|
| 2025-12-31 | baseline_auto | 33.83397310987668 | 0.06879430244337993 | 18.090404057057754 | 84.61972807328837 | -0.09020361386565823 | pass | 20260222_001458Z_03081a3d_cfbce2bd |
| 2025-12-31 | compact_31 | 36.48291836659223 | 0.11501104096559271 | 16.55565252048701 | 62.66869233646352 | -0.07965604767161938 | pass | 20260222_002508Z_0f119b28_cfbce2bd |
| 2026-01-31 | baseline_auto | 33.83397310987668 | 0.06879430244337993 | 18.090404057057754 | 84.61972807328837 | -0.09020361386565823 | pass | 20260222_002652Z_0baeed6b_cfbce2bd |
| 2026-01-31 | compact_31 | 36.48291836659223 | 0.11501104096559271 | 16.55565252048701 | 62.66869233646352 | -0.07965604767161938 | pass | 20260222_003658Z_d5657a4a_cfbce2bd |
| 2026-02-21 | baseline_auto | 28.777353203715016 | 0.03577394847537635 | 18.15014936663848 | 70.88484963678033 | -0.20998558378383492 | pass | 20260222_003841Z_13c043d8_cfbce2bd |
| 2026-02-21 | compact_31 | 29.095079030378983 | 0.03917765472952744 | 18.0786236027488 | 73.26768897651878 | -0.2409930773222022 | pass | 20260222_004034Z_44013df1_cfbce2bd |

## Aggregate

| rank | model | all_periods_gate_pass | score_mean | score_min | rank_ic_mean | sharpe_mean | total_return_mean | max_drawdown_mean |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | compact_31 | True | 34.02030525452115 | 29.095079030378983 | 0.08973324555357094 | 17.063309547907604 | 66.20169121648195 | -0.13343505755514698 |
| 2 | baseline_auto | True | 32.148433141156126 | 28.777353203715016 | 0.057787517787378735 | 18.11031916025133 | 80.04143526111902 | -0.13013093717171711 |

## Selected

- name: `compact_31`
- params: `{"n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 20}`
- all_periods_gate_pass: `True`
- score_mean: `34.02030525452115`
- score_min: `29.095079030378983`
