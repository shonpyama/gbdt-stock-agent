# Feature Search (Production) - 2026-02-21

| candidate | lookbacks | run_id | sharpe | total_return | max_drawdown | rank_ic_test_mean | gate |
|---|---|---|---:|---:|---:|---:|---|
| ext1_winner | [1,3,5,10,20,60,120] | 20260221_165044Z_47f27df1_9192a824 | 18.1053 | 70.6309 | -0.2128 | 0.035764 | pass |
| baseline_prev | [1,5,20,60,120] | 20260221_164744Z_f9b5b57b_9192a824 | 17.3311 | 66.6752 | -0.2724 | 0.043351 | pass |
| ext2_rejected | [1,5,20,60,120,252] | 20260221_165350Z_8ce6dab8_9192a824 | 10.7906 | 18.4734 | -0.4520 | 0.035678 | fail (max_drawdown) |
| ext3_tested | [1,2,3,5,10,20,60,120] | 20260221_165958Z_30a5685c_9192a824 | 17.6017 | 56.8177 | -0.2450 | 0.034303 | pass |

Selected default: `[1,3,5,10,20,60,120]`
