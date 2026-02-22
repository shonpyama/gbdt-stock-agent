# Feature Stability (Production) - 2026-02-22

- base_conf: `/content/gbdt-stock-agent/conf/default.yaml`
- policy: `/content/gbdt-stock-agent/conf/ops_policy.yaml`
- end_dates: `["2026-02-21"]`

## Per-Period

| end_date | feature_set | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |
|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-02-21 | baseline_current | 29.095079030378983 | 0.03917765472952744 | 18.0786236027488 | 73.26768897651878 | -0.2409930773222022 | pass | 20260222_010126Z_44013df1_cfbce2bd |
| 2026-02-21 | lb_1_5_20_60_120 | 29.849902078961513 | 0.04250327633964105 | 18.122059423476838 | 78.74253653835335 | -0.2335223042213207 | pass | 20260222_010319Z_4ecef6ab_cfbce2bd |
| 2026-02-21 | lb_1_5_20_60_120_shift2 | 29.849902078961513 | 0.04250327633964105 | 18.122059423476838 | 78.74253653835335 | -0.2335223042213207 | pass | 20260222_010510Z_7bbb0df1_cfbce2bd |

## Aggregate

| rank | feature_set | all_periods_gate_pass | score_mean | score_min | rank_ic_mean | sharpe_mean | total_return_mean | max_drawdown_mean |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | lb_1_5_20_60_120 | True | 29.849902078961513 | 29.849902078961513 | 0.04250327633964105 | 18.122059423476838 | 78.74253653835335 | -0.2335223042213207 |
| 2 | lb_1_5_20_60_120_shift2 | True | 29.849902078961513 | 29.849902078961513 | 0.04250327633964105 | 18.122059423476838 | 78.74253653835335 | -0.2335223042213207 |
| 3 | baseline_current | True | 29.095079030378983 | 29.095079030378983 | 0.03917765472952744 | 18.0786236027488 | 73.26768897651878 | -0.2409930773222022 |

## Selected

- name: `lb_1_5_20_60_120`
- lookbacks: `[1, 5, 20, 60, 120]`
- event_shift: `1`
- all_periods_gate_pass: `True`
- score_mean: `29.849902078961513`
- score_min: `29.849902078961513`
