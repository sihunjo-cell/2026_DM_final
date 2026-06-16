# Method 2 최종 파이프라인

목표: 확정 탐지가 아니라 viewer-chat mismatch 기반 수동 검토 우선순위를 만든다.
`cluster_number`, `minute_cluster`, `rra_q`, `empirical_p`, `review_order`는 정답 라벨이나 확률이 아니다.

final review_order는 raw evidence RRA가 아니라 `equal_weight_family_consensus_plus_family_rra` 설정에 따라 정렬한다.
legacy threshold grid 산출물은 appendix diagnostic이며 final ranking을 직접 바꾸지 않는다.
Family evidence list: scan_family_rank, persistence_family_rank, expected_response_family_rank, minute_state_family_rank, interval_anomaly_family_rank, reason_support_family_rank.
scan_interval_rank, empirical_p_rank, scan_strength_rank는 같은 scan family의 내부 근거이며 final RRA에 각각 독립 evidence로 들어가지 않는다.
persistence_family는 top interval duration과 state dwell evidence를 hard threshold 없이 rank/percentile 기반으로 반영한다.
reason_support_rank는 continuous support 기반이며 설명 문구는 세션별 상위 근거를 순위 기반으로 선택한다.
empirical_p는 shuffled-null 대비 scan statistic 근거이며 확률이 아니다.
rra_q, family_rra_q, family_consensus_score는 확률이 아니다.

all-zero chat sessions are preserved in qc_zero_session_review.csv for manual QC.
They are not used as positive labels or confirmed cases.
They are excluded from behavior modeling to avoid mixing WebSocket collection failures with behavioral mismatch.

synthetic sanity는 실제 ground-truth 성능 평가가 아니다.
status=not_run이면 recovered_rate를 보고하지 않으며 0% recovery로 해석하지 않는다.
status=ok일 때만 recovery summary를 내부 sanity check로 표시한다.

모든 operational conclusion은 raw WebSocket/chat QC와 수동 검토가 필요하다.
