# CHZZK Viewbot 후보 탐지 파이프라인

산업공학과 〈데이터마이닝〉 프로젝트. CHZZK(네이버) 라이브스트림에서 **viewer–chat mismatch가 지속적으로 나타나는 방송 구간**을 찾아, 사람이 직접 review할 방송 수를 줄여주는 것이 목표다.

> **핵심 전제**: viewbot 여부의 최종 판단은 사람이 한다. 이 파이프라인은 "정답 라벨"을 만드는 것이 아니라, **수동 검토 우선순위(review ranking)** 를 만든다. 모든 점수는 viewbot 확률이 아니라 *검토 근거(evidence)* 다.

---

## 1. 레포 구조 (한눈에 보기)

```
DM_project-1-/
├── README.md                  ← (이 문서) 설치 · 실행 · 파이프라인 설명 전부
├── live_review_pipeline.py    ← 메인 실행 진입점 (수집 → 점수화 → 리포트)
├── before_run.py              ← 최초 1회: 로컬 MySQL DB/유저 생성
│
├── chzzk-crawler/             ── [Stage 1] 데이터 수집
│   ├── .env                   ← DB/ API 키 설정 (직접 만들어야 함, git 제외)
│   ├── build_pool.py          ← 상위 라이브 100명 풀 → 무작위 타깃 선정
│   ├── collectors/            ← viewer 스냅샷 + 채팅 WebSocket 수집기
│   ├── pipeline/              ← 수집 관리(manager) + 분단위 집계(aggregate)
│   ├── core/                  ← DB 엔진/ORM 모델
│   ├── configs/settings.py    ← .env 로딩
│   └── scripts/               ← run_pilot(수집) · export_csv(내보내기) · setup(DB 초기화)
│
├── dm_pipe_final/             ── [Stage 2] 점수화 파이프라인 (= 제출 PDF의 방법론)
│   ├── cfg.yml                ← 파이프라인 설정 (시간대/하이퍼파라미터)
│   ├── run.py                 ← 점수화 오케스트레이터 (분→세션→6개 evidence→ranking)
│   ├── src/                   ← 단계별 모듈 (아래 4절 참고)
│   ├── data/features/         ← 입력 feature 엑셀 (Run_*_Features.xlsx)
│   ├── out/                   ← 산출물 (m2_review.csv, plots/ 등)
│   └── tests/                 ← 단위 테스트
│
└── my_job/                    ── EDA 노트북 · 중간발표 자료 (분석 기록 보관)
```

수집(Stage 1)과 점수화(Stage 2)는 독립적이다. `live_review_pipeline.py`가 둘을 이어준다: **새로 수집 → feature 내보내기 → 점수화 → 우선순위 리포트 출력.**

---

## 2. 설치 (무엇을 깔아야 하는가)

### 2.1 필수 소프트웨어
| 항목 | 버전 | 비고 |
| :-- | :-- | :-- |
| Python | 3.11+ | |
| MySQL | 8.0 | 로컬에 설치되어 실행 중이어야 함 |

### 2.2 Python 패키지
```powershell
# 수집기 의존성
pip install -r chzzk-crawler/requirements.txt
# 점수화 파이프라인 추가 의존성
pip install scikit-learn scipy matplotlib openpyxl pyyaml
```

### 2.3 CHZZK Open API 키 발급
라이브 방송 목록 조회에 필요. https://chzzk.naver.com/creator/api 에서 `Client ID` / `Client Secret`을 발급받는다.

---

## 3. 실행 방법 (어디서 무엇을 바꾸는가)

### 단계 0 — 로컬 MySQL DB 생성 (최초 1회)
```powershell
$env:MYSQL_ROOT_PASSWORD="본인_MySQL_root_비밀번호"
python before_run.py
```
→ `chzzk_dm` DB와 `chzzk_user` 유저를 만든다. (비밀번호는 코드에 하드코딩하지 말고 환경변수로 전달)

### 단계 1 — `chzzk-crawler/.env` 작성
`chzzk-crawler/.env.example`을 복사해 `.env`로 만들고 값을 채운다.
```dotenv
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=chzzk_dm
DB_USER=chzzk_user            # before_run.py 가 만든 유저
DB_PASSWORD=ChzzkCrawler2026! # before_run.py 의 NEW_PASSWORD 와 동일하게
CHZZK_CLIENT_ID=...           # 2.3에서 발급
CHZZK_CLIENT_SECRET=...
```
> ⚠️ `.env`의 `DB_USER`/`DB_PASSWORD`는 `before_run.py`에서 만든 값과 **반드시 일치**해야 한다. 비밀번호 앞뒤 공백 주의.

### 단계 2 — `dm_pipe_final/cfg.yml`에서 시간대 확인
수집한 데이터의 시간대(KST 보정·유효 수집 윈도우)를 맞춘다.
```yaml
time:
  shift_hours: 9                # DB가 UTC면 9, 이미 KST면 0
  valid_windows: ["15:00-17:00", "17:00-19:00", "20:00-22:00", "23:00-01:00"]
```
> **결과는 `valid_windows` 안에서 수집했을 때만 나온다.** 이 시간대(KST)가 곧 정상반응 baseline의 모집단이라, 지정 외 시간대 데이터는 baseline을 오염시키므로 받지 않는다.

### 단계 3 — 실행
```powershell
# (A) 새로 수집부터 점수화까지 한 번에 (기본 duration=900초)
python live_review_pipeline.py --duration 900

# (B) 이미 수집된 run_id를 재점수화 (수집 생략)
python live_review_pipeline.py --skip-collect --run-id 48
```
주요 옵션: `--duration`(수집 초), `--run-id`(특정 run 사용), `--skip-collect`(수집 건너뜀), `--refresh-targets`(타깃 풀 재추첨), `--keep-temp`(임시 산출물 보존, 디버깅용).

> **시간대 가드:** (A)는 **수집 시작 전** 현재 시각(+duration 구간)이 `valid_windows` 안인지 먼저 검사한다. 벗어나면 **수집조차 하지 않고** OFF-WINDOW 안내만 출력한다 (불필요한 수집 방지). (B)는 이미 수집된 데이터의 KST 시각으로 동일 검사를 한다.

실행이 끝나면 콘솔에 **LIVE REVIEW SUMMARY**가 출력된다 — live 세션들의 review 우선순위(rank, top_pct, level, family_score, reason)와 검토 불가 세션(QC 사유). 시간대를 벗어나면 그 대신 **OFF-WINDOW 안내**가 나온다.

---

## 4. 점수화 파이프라인 (Stage 2 상세 = PDF 방법론)

> 문제의식: viewbot은 **minute mismatch가 "연속적으로"** 일어나는 것. 띄엄띄엄 일어난 mismatch는 정상 방송으로 본다. 따라서 1분 단위 mismatch를 정의하고, 그것이 세션 안에서 **지속 구간**으로 뭉치는지를 찾는다.

### 4.1 흐름
```
load → prep(분단위 전처리) → minute_state(롤링/zero-run 등) → minute_cluster(행동상태)
  → m2_baseline(HistGBM 조건부 기대치) → m2_scan(minute mismatch score + 구간 탐색)
  → m2_interval(이상치) → m2_review(6개 evidence family 결합 → 최종 ranking)
```
`run.py`가 위 모듈(`src/`)을 순서대로 호출한다.

### 4.2 mismatch의 두 기준선
- **bin baseline**: viewer 10분위별 chat median과의 log gap = `chat_deficit` (예전 main 방법론).
- **모델 baseline (main)**: `HistGradientBoostingRegressor`로 `[log_viewer, viewer_bin, minute_idx_norm, hour, category one-hot]`에서 조건부 log-chat median을 예측 → `model_chat_deficit`. **mismatch 본판단은 항상 이 조건부 median 기준.**

### 4.3 minute mismatch score (왜 이 분이 mismatch인가)
6개 evidence의 percentile rank를 산술평균:
`chat_deficit`, `unique_deficit`, `rolling_chat_deficit_5m`, `zero_run_len`, `rolling_zero_rate_5m`, `minute KMeans 상태`.
→ score가 높을수록 검토 우선. (q90/q95 선은 분포 참고용 grid일 뿐, viewbot 판정선이 아님)

### 4.4 interval scan (지속 구간 찾기)
분별 z-score를 만들고, 같은 세션 내 모든 연속 구간 조합에 대해 `S(I)=Σz/√|I|`를 계산해 최대 구간 `I*`를 찾는다. shuffled-null 대비로 그 구간이 우연 이상으로 몰려 있는지(`empirical_p`)를 진단한다. → `m2_scan.csv`.

### 4.5 evidence family → 최종 review ranking
집계에 들어가는 **evidence family는 서로 다른 5개 축**이다. Reason-support는 이들을 다시 센 것이라 **집계에서 제외**하고 "왜 상위인가"를 사람에게 보여주는 **설명용**으로만 쓴다.

| Family | 축(독립 관점) | 질문 | 산출 | 집계 |
| :-- | :-- | :-- | :-- | :-- |
| Scan | 시간적 집중도 | mismatch가 특정 구간에 몰렸나 | `empirical_p`, `observed_scan_z` | ✅ |
| Persistence | 지속/길이 | 얼마나 오래 지속되나 | interval duration, max/total run | ✅ |
| Expected-response | 크기(모델 기준) | GBM 조건부 기대 대비 부족한가 | model chat/unique deficit | ✅ |
| Minute-state | 비지도 행동상태 확증 | 비지도 cluster도 mismatch 상태로 보나 | mismatch cluster 비율 | ✅ |
| Interval-anomaly | profile 이질성 | top interval이 다른 구간 대비 이질적인가 | IsolationForest/LOF/ECOD | ✅ |
| Reason-support | (설명) | 왜 상위 후보인지 근거 빈도 | rolling_chat_deficit 등 | ❌ 설명용 |

최종 정렬: `eligible_review=True` → **`family_consensus_score` 높은 순** → 보조로 `family_rra_q` → `family_rra_p` → persistence/scan rank → session_key.

- **`family_consensus_score` (주 기준)** = **5개 evidence family strength의 단순 평균**. 어떤 family가 결측이면(=그 관점의 근거 없음) strength를 **0으로 채운 뒤 평균** ([m2_review.py](dm_pipe_final/src/m2_review.py)) → "한 family만 강하고 나머지 결측"인 세션이 무임승차하지 못하게 하여 **여러 축에서 고르게 높은 세션**을 위로 올린다 (PDF 7p "고른 신호" 목표).
- **`family_rra_q` (보조 기준)** = Robust Rank Aggregation. consensus가 못 잡는 **소수 축에 신호가 몰린(outlier형) 세션**을 robust하게 끌어올리는 백업. 결측 rank를 worst로 채워 consensus와 결측 규칙이 일치한다. 보조 기준으로만 사용.

### 4.6 evidence 독립성 설계 (정직한 해석 + 근거)
- 근본은 **하나의 신호 — "조건부 기대 대비 chat/unique 부족(deficit) + zero-chat 침묵이 *지속적으로* 나타나는가"**. 그래서 "독립 증거 6개"라고 말하면 방어가 안 된다. **대신 그 신호를 서로 겹치지 않는 5개 축(집중도·지속성·크기·비지도 확증·profile 이질성)으로만 요약**해 consensus를 구성했다.
- **Reason-support를 집계에서 뺀 근거는 프로젝트 스펙 자체**: PDF 6p가 "reason support의 count는 confidence·lift·정답 라벨이 *아니라 설명 근거의 빈도*"라고 명시 → 증거가 아니라 설명이므로 집계 대상이 아니다. (코드에선 여전히 계산·기록되어 `dominant_reason`으로 보여준다.) 이로써 RRA의 독립성 가정 위반(같은 신호를 6번째로 다시 센 것)이 사라진다.
- 5개 축의 **방법 다양성**: Expected-response=지도학습(GBM) 크기, Minute-state=비지도(KMeans) 확증, Interval-anomaly=구간 outlier 탐지 — 서로 다른 추정 방식이라 단순 중복이 아니다. (Minute-state는 deficit feature 기반이라 *약한* 상관은 남으나, 지도/비지도 교차확증 역할.)
- **설계 근거(표준 기법):** 누수 없는 조건부 기대치 = **run_id GroupKFold OOF + quantile(0.5) 회귀**(cross-fitting), 지속 구간 = **scan statistic `Σz/√n`**, 우연성 = **세션내 permutation null**, 결합 = **percentile-rank 평균 + RRA**(Kolde 2012) + **BH-FDR**. 모두 임의 규칙이 아니라 통계 표준 기법.
- **모든 점수는 viewbot 확률이 아니라 검토 근거.** 최종 판단은 사람이 한다.

---

## 5. 주요 산출물 (`dm_pipe_final/out/`)
| 파일 | 내용 |
| :-- | :-- |
| `m2_review.csv` | **최종 review 우선순위** (검토 가능 세션) |
| `m2_review_all.csv` | 전체 세션 + 검토 불가 QC 사유 |
| `m2_scan.csv` | 세션별 top mismatch 구간 요약 |
| `base_pred.csv` | HistGBM 조건부 기대 chat/unique median |
| `m2_reason.csv` | 후보별 설명 근거 |
| `int_scores.csv` | interval 이상치 점수 |
| `plots/` | EDA·진단 그림 (01_data_quality ~ 26_synthetic_recovery) |

---

## 6. 데이터 사전

### 6.1 Minute Features (`Run_*_Features.xlsx`) — 분 단위 집계
| 필드 | 설명 |
| :-- | :-- |
| `run_id` | 수집 세션 고유 ID |
| `broad_no` | 방송 고유 식별자 |
| `minute_ts` | 데이터 시점 (YYYY-MM-DD HH:MM:00) |
| `user_id` | 스트리머 ID (해시) |
| `category_id` | 게임/콘텐츠 카테고리 ID |
| `viewer_count_last` | 해당 분 종료 시점 시청자 수 |
| `chat_count` | 해당 분 총 채팅 수 |
| `unique_chatters` | 채팅 보낸 고유 유저 수 |
| `avg_msg_len` | 메시지당 평균 글자 수 |
| `repeat_msg_ratio` | 반복/유사 메시지 비율 (스팸 지표) |
| `new_chatter_ratio` | 첫 채팅 유저 비율 |
| `chat_per_viewer` | 참여도 (chat / viewer) |
| `delta_viewer_1m` | 전분 대비 시청자 변동 |
| `delta_chat_1m` | 전분 대비 채팅 변동 |

### 6.2 Raw Chat (`Run_*_Chats.csv`) — 개별 채팅 원본
`chat_id`, `run_id`, `event_ts`, `broad_no`, `user_id`(해시), `user_nick`, `message_raw`, `message_clean`, `message_hash`, `raw_json`, `created_at`.

---

## 7. 시스템 아키텍처 (수집 Stage 1)

두 단계 무작위 샘플링으로 특정 스트리머 편향을 피한다: 매 수집 윈도우마다 **상위 ~100명 풀**을 만들고 그중 **무작위 N명**을 추첨해 모니터링한다.

```
build_pool.py (Top 100)
      │  무작위 추첨
      ▼
setup.py (--load-csv top30_targets.csv) ── DB에 타깃 등록
      │
      ▼
run_pilot.py → CrawlManager(manager.py)
      ├─ Live API 스냅샷(60초)  → live_snapshots
      └─ 채팅 WebSocket(×N)     → chat_messages_raw
                                      │  aggregate.py
                                      ▼
                                 minute_features  → export_csv.py → Run_*_Features.xlsx
```
| 수집 항목 | 방법 | 주기 | 저장 |
| :-- | :-- | :-- | :-- |
| Viewer 스냅샷 | CHZZK Open API v1 | 60초 | `live_snapshots` |
| 채팅 메시지 | 네이버 게임 채팅 WebSocket | 실시간(3초 flush) | `chat_messages_raw` |

운영 배포(EC2 cron) 예시는 `chzzk-crawler/new_crontab.txt` 참고.

---

## 8. 알려진 이슈 / 주의

### 방법론상 의도적 한계 (해석 시 반드시 인지)
- **bin baseline은 보조 수단.** `chat_deficit`(viewer 10분위 median 기준)은 **전체 코퍼스를 viewer 분위로만** 나눠 구하므로 카테고리/시간대/스트리머를 섞는다. 그래서 **본판단 기준은 항상 카테고리·시간 조건화된 모델 baseline(`model_chat_deficit`)** 이고, bin은 `baseline_agree_*`의 거친 교차확인용으로만 쓴다.
- **`empirical_p`는 "지속성(구간 집중도)" 진단값**이지 세션 이상도/viewbot 확률이 아니다. 균일하게 높은 세션은 `empirical_p`가 크게 나올 수 있으며, 크기(magnitude)는 expected-response·minute score가 따로 본다.
- **결과는 reference 구성에 의존.** 모든 percentile rank가 전역 기준(PDF의 "전체기준 ranking")이라, `data/features`에 번들된 reference 집합이 바뀌면 동일 세션의 `review_order`도 바뀐다. 실행 간 절대 비교는 피하고 우선순위로만 사용. (HistGBM은 run_id OOF라 live는 held-out → 이 부분은 누수 없음)

### 운영/정리
- **`chzzk-crawler/.env`의 DB 계정**: 현재 `root`로 직접 접속하도록 되어 있고 비밀번호 앞에 공백이 있을 수 있음. `before_run.py`가 만든 `chzzk_user`로 맞추는 것을 권장.
- **`before_run.py`의 root 비밀번호**: 이제 환경변수 `MYSQL_ROOT_PASSWORD`로 받는다(하드코딩 제거).
- **`src/minute_ml.py`**: 현재 파이프라인에서 import되지 않는 미사용 모듈(정리 후보).
- **대용량 zip**(`dm_pipe_final/*.zip`): 제출/분석 보관용 데이터 아카이브. 코드가 아니며 `.gitignore` 처리됨.
