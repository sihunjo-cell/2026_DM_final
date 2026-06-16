# Method 2 평가/진단 계획

현재 실제 ground-truth label이 없으므로 accuracy, precision, recall, F1, AUC를 실제 성능처럼 보고하지 않는다.
사용 가능한 검증은 no-label sanity check, shuffled-null diagnostic, stability diagnostic, handoff 재현성 검증이다.
m2_review.csv의 review_order는 family-level equal-weight consensus + family RRA 기반 수동 검토 우선순위이지 정답 라벨이 아니다.
top10 short interval count는 WARNING으로 보고하며 duration <= 1 같은 hard cutoff로 세션을 자동 제외하지 않는다.
synthetic sanity는 합성 세션이 같은 Method2 scoring/scan/review pipeline을 통과하고 status=ok일 때만 recovery summary를 sanity check로 제시한다.
status=not_run 또는 not_run_stale_input이면 recovered_rate는 보고하지 않는다.
y_syn은 실제 viewbot label이 아니다.
all-zero chat sessions are preserved in qc_zero_session_review.csv for manual QC.
They are not used as positive labels or confirmed cases.
They are excluded from behavior modeling to avoid mixing WebSocket collection failures with behavioral mismatch.
