# Model Search (Production) - 2026-02-22

- base_conf: `/content/gbdt-stock-agent/conf/default.yaml`
- policy: `/content/gbdt-stock-agent/conf/ops_policy.yaml`
- selected_end_date: `2026-02-21`
- trials: `3`

| rank | name | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | ops_gate | run_id |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | compact_31 | 29.095079030378983 | 0.03917765472952744 | 18.0786236027488 | 73.26768897651878 | -0.2409930773222022 | pass | 20260222_000726Z_44013df1_cfbce2bd |
| 2 | baseline_auto | 28.777360512592622 | 0.035774009382689737 | 18.15014936663848 | 70.88484963678033 | -0.20998558378383492 | pass | 20260222_000535Z_13c043d8_cfbce2bd |
| 3 | regularized_63 | 26.437501405542633 | 0.037409970041942596 | 16.91497910640704 | 52.135704485457204 | -0.2391137878629458 | pass | 20260222_000913Z_4b496bda_cfbce2bd |

## Selected

- name: `compact_31`
- run_id: `20260222_000726Z_44013df1_cfbce2bd`
- score: `29.095079030378983`
- params: `{"n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 20}`
