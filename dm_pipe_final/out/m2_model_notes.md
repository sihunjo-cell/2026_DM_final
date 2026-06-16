# Method 2 모델 노트

expected-response baseline은 classifier가 아니며 GroupKFold by run_id 기반 no-label baseline이다.
minute KMeans와 session KMeans는 behavior state 요약이며 final label source가 아니다.
interval anomaly support는 보조 evidence이고 short spike를 과도하게 상위로 올리면 diagnostic으로 격리한다.
최종 review_order는 family-level equal-weight consensus + family RRA이며 rra_q는 family_rra_q와 같다.
raw evidence 기반 값은 raw_rra_p/raw_rra_q로 보존한다.
scan_interval_rank, empirical_p_rank, scan_strength_rank는 같은 scan family의 내부 근거이며 final RRA에 각각 독립 evidence로 들어가지 않는다.
hard threshold나 확정 label을 쓰지 않고 rank consensus로 처리한다.
synthetic_sanity.enabled=false이면 m2_synth.csv는 status=not_run으로 남기고 recovered_rate를 비워 둔다.
stale synthetic_intervals.csv만 있고 현재 m2_scores/m2_scan에 synthetic session_key가 없으면 status=not_run_stale_input으로 기록한다.
qc_zero_session_review.csv는 manual QC appendix이며 Method2 ranking 입력이 아니다.
