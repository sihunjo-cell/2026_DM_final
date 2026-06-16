# Parameter Rationale

This document explains why the main configuration values exist. They are engineering and diagnostic choices, not certainty labels or probability outputs.
주요 파라미터는 재현성과 진단 안정성을 위한 설정이며 확정 판정 기준이 아니다.

## prep.min_n=10
- Minimum observation length for stable session summary statistics.
- It is not a suspicious-session criterion.

## prep.clock_gap_reset_min=1.1
- Clock-gap reset reflecting the roughly 60-second viewer snapshot cadence.
- Prevents collection gaps from being misread as one long zero-chat run.

## minute_state.viewer_bin_n=10
- Decile bins make expected chat and unique chatter comparable across viewer scale.
- The bin is descriptive and not a labeling rule.

## minute_state.rolling_windows=[5, 10]
- Rolling windows capture short persistence in the 0-10 minute range seen during EDA.
- Final ranking uses rolling evidence as continuous rank evidence, not a hard threshold.

## minute_cluster.features
- `log_viewer`: viewer scale after log transform.
- `chat_deficit`: chat response deficit relative to viewer scale.
- `unique_deficit`: unique chatter deficit relative to viewer scale.
- `rolling_chat_deficit_5m`: short persistence of chat deficit.
- `log_zero_run_len`: clock-gap-aware zero-chat persistence.
- `rolling_zero_rate_5m`: recent zero-chat concentration.
- Configured features: log_viewer, chat_deficit, unique_deficit, rolling_chat_deficit_5m, log_zero_run_len, rolling_zero_rate_5m

## RobustScaler
- Session scaler: RobustScaler. Minute scaler: RobustScaler.
- Median/IQR scaling reduces the influence of heavy-tailed viewer and chat values.

## KMeans K candidates
- Session K candidates: 2..6.
- Minute K candidates: 2..8.
- K is selected by selection_score, not fixed by assumption.
- selection_score combines silhouette, Calinski-Harabasz, Davies-Bouldin, size balance, profile separation, and stability in the correct direction.

## m2_scan.n_perm=200
- Shuffled-null diagnostic count; with n_perm=200, the minimum empirical resolution is 1/(200+1).
- The resulting empirical_p is diagnostic evidence, not a calibrated probability.
- If a tighter null estimate is required, increase n_perm and document the sensitivity plan.

## m2_scan.max_scan_n=500
- Computational cap for long-session scan search.
- Any pruning behavior is documented in m2_scan.csv note fields.

## expected-response baseline
- GroupKFold by run_id reduces leakage across broadcasts from the same run.
- max_train_rows_per_fold=200000 keeps training cost bounded.
- This estimates expected chat and unique chatter response; it is not a classifier.

## interval anomaly
- RobustScaler setting: RobustScaler.
- IsolationForest, ECOD, and LOF are auxiliary directional evidence sources.
- They are not final label sources.

## synthetic_sanity.enabled=False
- When disabled, m2_synth.csv reports status=not_run and leaves recovered_rate blank.
- Blank recovered_rate must not be interpreted as 0% recovery.

## Interpretation limits
- cluster_number, minute_cluster, rra_q, empirical_p, family_consensus_score, and review_order are review-priority or diagnostic values.
- No output column is a final decision field.
