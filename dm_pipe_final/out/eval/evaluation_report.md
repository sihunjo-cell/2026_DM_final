# 평가 안정성 리포트

본 평가는 실제 viewbot label에 대한 성능평가가 아니라, label-free review ranking pipeline의 robustness와 synthetic mismatch recovery를 확인하기 위한 것이다.

## 1. 감독학습 성능지표를 쓰지 않는 이유
현재 프로젝트에는 실제 최종 판정 라벨이 없으므로 accuracy, precision, recall, ROC-AUC, PR-AUC를 실제 성능처럼 계산하지 않는다.
이 폴더의 산출물은 label-free review ranking의 안정성과 synthetic mismatch 회수 여부를 확인하는 진단 자료다.
요약 문구의 not real viewbot performance는 synthetic 결과가 실제 viewbot 성능평가가 아니라는 제한을 명시하기 위한 표현이다.

## 2. 설정 스냅샷
- topk 기준: [20, 50, 100, 200]
- family 목록: scan_family_rank, persistence_family_rank, expected_response_family_rank, minute_state_family_rank, interval_anomaly_family_rank, reason_support_family_rank
- minute signal 목록: chat_deficit, unique_deficit, rolling_chat_deficit_5m, zero_run_len, rolling_zero_rate_5m, cluster_mismatch_rank
- synthetic 입력: minutes=data/synthetic/synthetic_many_minutes.csv, intervals=data/synthetic/synthetic_many_intervals.csv
- synthetic 중간 CSV 보존 여부: False

## 3. Family 제거 민감도
각 family를 하나씩 제거한 뒤 남은 family strength로 후보 순서를 다시 계산한다. 최소 top100 overlap은 0.540이다.
overlap이 높으면 최종 review_order가 특정 family 하나에만 임의로 의존하지 않는다는 근거가 된다. 낮은 값은 핵심 근거 family를 식별하는 민감도 신호로만 해석한다.

## 4. 집계 방식 민감도
consensus-only, RRA-only, consensus-plus-RRA, median, trimmed mean, family-exclusion 변형을 비교한다. 최소 top100 overlap은 0.610이다.
RRA는 순위 집계 근거이며, 주 해석은 consensus-first이고 RRA는 보조 근거로 사용한다.

## 5. 근거 균형과 동점 점검
Evidence balance는 review_order 구간별로 요약한다. tie audit 행 수는 6개다.
상위 구간에서 strong-family 수가 높으면 여러 근거 family가 함께 지지한다는 의미다. 단일 family 지배가 있으면 제한사항으로 보고한다.

## 6. Minute signal 민감도
signal 제거 실험의 최소 top100 session overlap은 0.740이고, weight 변형의 최소 top100 session overlap은 0.590이다.
equal weight는 라벨 없는 상태에서 학습 가중치를 임의로 만들지 않기 위한 보수적 설계다. overlap 안정성은 minute score가 하나의 signal 또는 하나의 가중치 설정에 과도하게 의존하지 않는지 확인한다.

## 7. Synthetic mismatch interval 회수
Synthetic interval recovery의 median IoU는 0.967이다. 이 값은 synthetic mismatch recovery이며 not real viewbot performance이다.
연속 mismatch scenario는 interval localization sanity check로 해석한다. intermittent zero control은 별도 negative-control-like diagnostic으로 보고한다.

## 8. 해석 제한
이 진단은 supervised class correctness를 증명하지 않는다. evidence-family 제거, 집계 방식 변경, minute-signal 제거, synthetic interval localization에 대해 review 우선순위가 얼마나 안정적인지 확인한다.

## 9. 권장 해석
최종 review_order는 label-free mismatch pipeline이 만든 수동 검토 우선순위로 사용한다. 이 eval 폴더는 robustness 근거와 민감도 한계를 기록하는 appendix로 해석한다.

## 10. 상위 후보 vs 나머지 프로파일 대비
top_session_profile.csv는 상위 review 세션과 나머지 세션의 minute-signal 프로파일을 표준화 평균차(Cohen d)로 비교한다. 표준화 평균차 절댓값이 가장 큰 신호는 rolling_chat_deficit_5m_median(higher_in_top, d=1.584)이다.
이는 상위 후보가 어떤 신호 때문에 위로 올라갔는지 설명하는 자료이며 확률이나 판정 라벨이 아니다.

## 11. label-free 평가 스코어카드
eval_scorecard.csv는 robustness 최소 overlap과 synthetic positive/negative-control localization을 한 표로 모은다. synthetic positive median IoU는 0.978, negative-control median IoU는 0.775이다.
모든 값은 robustness 또는 synthetic sanity 진단이며 supervised 성능지표가 아니다.
