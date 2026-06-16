from pathlib import Path

from .detect import FEATS as DETECTOR_FEATS
from .model import FEATS as MODEL_FEATS, MODEL_SCORE_COLS


FINAL_SENTENCE = (
    "This project does not have platform-internal labels that confirm actual viewbot use. "
    "It uses observable viewer count and chat activity to score one-minute viewer-chat mismatch states, "
    "groups contiguous high-scoring minutes into episode candidates, and ranks sessions for human review. "
    "The final outputs are review aids, not ground-truth labels and not probability."
)


SCORE_EXPLANATION = [
    "## Score Interpretation",
    "",
    "- `minute_mismatch_score` is not probability.",
    "- Each signal is converted to a percentile rank before aggregation.",
    "- The score does not average raw values directly; rank conversion puts differently scaled signals on a common scale.",
    "- There is no real target label, so weights are not learned from ground truth.",
    "- Equal-weight rank aggregation is a transparent baseline review index, not the final model.",
]

EPISODE_EXPLANATION = [
    "## Episode Definition",
    "",
    "- An episode is a time-contiguous interval of high viewer-chat mismatch score inside one session.",
    "- An episode is not a confirmed viewbot-active interval.",
    "- An episode candidate is an interval for a reviewer to inspect.",
    "- `min_duration` and `threshold_q` are calibration candidates, not answer cutoffs.",
]

MODEL_IO_EXPLANATION = [
    "## Model Input / Output Separation",
    "",
    "- `session_summary_processed.csv` contains session-level behavior features and legacy baseline scores for review and modeling handoff.",
    "- `session_review_summary.csv` is a rule-based review ranking output.",
    "- Ranking outputs must not be fed back as model input because that would leak the review target into the feature matrix.",
    "- `overall_session_review_rank_score` and `overall_session_review_rank` must not be used as model features.",
    "- Use `X_core_cols.txt` for recommended model input columns.",
    "- Use `X_no_leak_cols.txt` for columns to exclude from model input.",
]


def _get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _join(values):
    return ", ".join(str(v) for v in values)


def _nrows(df):
    return 0 if df is None else len(df)


def _value_counts(df, col):
    if df is None or col not in df.columns:
        return "not available"
    vc = df[col].value_counts(dropna=False).sort_index()
    return ", ".join(f"{k}:{v}" for k, v in vc.items())


def _write(path, lines):
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_task_docs(out, cfg=None):
    out = Path(out)
    cfg = cfg or {}
    if not _get(cfg, "docs", "write_task_definition", default=True):
        return

    _write(out / "task_definition.md", [
        "# Task Definition",
        "",
        FINAL_SENTENCE,
        "",
        "## Primary Task",
        "- Primary task: minute-level viewer-chat mismatch episode detection.",
        "- Input unit: `(run_id, broad_no, minute_ts)`.",
        "- Episode unit: contiguous mismatch minutes inside the same `(run_id, broad_no)` session.",
        "- Session unit: `(run_id, broad_no)`.",
        "- Output policy: review candidate ranking.",
        "",
        "## Non-Goals",
        "- true viewbot confirmation.",
        "- confirming actual viewbot usage without platform internal logs.",
        "- using KMeans clusters as ground-truth labels.",
        "- using `cluster_number`, `minute_cluster`, or `y_syn` as actual labels.",
        "",
        *SCORE_EXPLANATION,
        "",
        *EPISODE_EXPLANATION,
        "",
        *MODEL_IO_EXPLANATION,
        "",
        "## Outputs",
        "- `minute_all.csv`, `minute_model.csv`: raw minute rows without state features, labels, or scores.",
        "- `minute_all_feat.csv`, `minute_model_feat.csv`: minute rows with state features.",
        "- `minute_scores.csv`: minute mismatch score and rank for review.",
        "- `episode_candidates_all_thresholds.csv`: episode candidate intervals under calibration thresholds.",
        "- `threshold_calibration.csv`: sensitivity table for threshold and duration candidates.",
        "- `session_review_candidates.csv`: session-level review candidates by calibration setting.",
        "- `session_review_summary.csv`: session-level review ranking output.",
    ])

    _write(out / "data_dictionary.md", [
        "# Data Dictionary",
        "",
        "## Raw Minute Columns",
        "- `source_file`: source workbook file name.",
        "- `run_id`, `broad_no`, `session_key`: session identifiers.",
        "- `user_id`, `category_id`: broadcast metadata when available.",
        "- `minute_ts`: minute timestamp after configured timezone shift.",
        "- `viewer_count_last`: last observed viewer count in the minute.",
        "- `chat_count`: number of chat messages in the minute.",
        "- `unique_chatters`: number of unique chatters in the minute.",
        "- `avg_msg_len`: average message length; zero-chat structural missing values are filled as 0.",
        "- `repeat_msg_ratio`: repeated-message ratio in the minute.",
        "- `new_chatter_ratio`: new chatter ratio in the minute.",
        "- `chat_per_viewer`: chat count divided by viewer count when viewer count is positive.",
        "- `delta_viewer_1m`, `delta_chat_1m`: within-session one-minute changes.",
        "",
        "## Minute State Features",
        "- `minute_idx`: time-order index inside `(run_id, broad_no)`.",
        "- `log_viewer`, `log_chat`, `log_unique`: log1p transforms for skewed counts.",
        "- `viewer_chat_gap`: `log_viewer - log_chat`; high values mean weaker chat response for scale.",
        "- `viewer_unique_gap`: `log_viewer - log_unique`; high values mean fewer unique chatters for scale.",
        "- `zero_chat`: whether chat_count is zero.",
        "- `zero_run_len`: consecutive zero-chat length inside the same session.",
        "- `viewer_bin`: scale bin for comparing minutes with similar viewer count.",
        "- `expected_log_chat_bin`, `expected_log_unique_bin`: median response within viewer_bin.",
        "- `chat_deficit`, `unique_deficit`: expected response minus observed response.",
        "- `rolling_*`: current-and-past rolling summaries; no future minutes are used.",
        "",
        "## Behavior Cluster Features",
        "- `cluster_number`: session-level behavior cluster id; categorical feature, not target label.",
        "- `session_behavior_cluster`: alias for cluster_number; use only with one-hot encoding if needed.",
        "- `minute_cluster`: minute-level behavior cluster id, not a label.",
        "- `cluster_mismatch_rank`: cluster-level mismatch indicator rank; interpretation aid only.",
        "",
        *SCORE_EXPLANATION,
        "",
        *EPISODE_EXPLANATION,
        "",
        "## Session Review Outputs",
        "- `episode_duration_ratio`: total candidate duration divided by observed session length.",
        "- `max_episode_score`: strongest candidate interval score in the session.",
        "- `p95_minute_mismatch_score`: session-level tail of minute mismatch scores.",
        "- `session_review_rank_score`: equal average of percentile ranks for one calibration setting.",
        "- `session_review_rank`: percentile rank of session_review_rank_score.",
        "- `review_stability`: share of calibration settings where the session appears as a review candidate.",
        "- `overall_session_review_rank_score`, `overall_session_review_rank`: review output only, not model input.",
        "- `session_bucket`: `no_episode`, `review_candidate`, or `recommended_setting_candidate`; not a label.",
        "",
        "## Synthetic Sanity-Check Fields",
        "- `y_syn`: artificial injection label used only for sanity checks and legacy auxiliary scores.",
        "- `y_syn` is not ground-truth label.",
    ])

    _write(out / "modeling_handoff.md", [
        "# Modeling Handoff",
        "",
        FINAL_SENTENCE,
        "",
        *MODEL_IO_EXPLANATION,
        "",
        "## Recommended Input Columns",
        "- Use `X_core_cols.txt` as the session-level starting feature list.",
        "- `cluster_number` may be used as a categorical behavior feature, not as a target.",
        "- `session_behavior_cluster` is an alias and should be one-hot encoded only if used.",
        "",
        "## Forbidden As Model Inputs",
        "- Do not use `overall_session_review_rank_score` or `overall_session_review_rank` as model features.",
        "- Do not use `session_review_rank_score`, `session_review_rank`, or `session_bucket` as model features.",
        "- Do not use `minute_mismatch_score`, `minute_mismatch_rank`, `episode_count`, or `episode_duration_ratio` as model features.",
        "- Do not use `minute_cluster` as a target label.",
        "- Do not use `y_syn` as an actual label.",
        "",
        "## Completed",
        "- Raw minute CSVs are separated from feature minute CSVs.",
        "- Minor off-window boundary rows are trimmed when below the configured tolerance.",
        "- Larger wrong-window files still drop at file level.",
        "- Minute-level mismatch scoring, episode candidates, and session review ranking are generated.",
        "",
        "## Modeling Owner Should Do",
        "- Treat this as review ranking or episode detection, not a true binary classifier.",
        "- Inspect `threshold_calibration.csv` before choosing any presentation setting.",
        "- Use `synthetic_intervals.csv`, when present, only for episode-recovery sanity checks.",
        "- Report ranking-oriented behavior, not only accuracy.",
        "",
        "## Modeling Owner Must Not Do",
        "- Do not create a definitive `viewbot_probability` column.",
        "- Do not create `true_viewbot_label` or `bot_detected` columns.",
        "- Do not claim confirmed viewbot usage from behavior clusters or review scores.",
        "",
        "## Key Files",
        "- `minute_all.csv`, `minute_model.csv`: raw minute rows.",
        "- `minute_all_feat.csv`, `minute_model_feat.csv`: minute state-feature rows.",
        "- `session_summary_processed.csv`: session behavior features, cluster_number, and legacy baselines.",
        "- `session_review_summary.csv`: rule-based review ranking output.",
        "- `X_core_cols.txt`, `X_no_leak_cols.txt`: model input and no-leak column lists.",
    ])

    _write(out / "pipeline_definition.md", [
        "# Pipeline Definition",
        "",
        "## Revised Pipeline",
        "`load_features -> prep_minute -> split raw minute outputs -> add_minute_state_features -> split feature minute outputs -> add_minute_clusters -> build_episode_calibration -> make_session -> legacy session-level baselines -> merge review outputs for handoff`.",
        "",
        "## Load Policy",
        "- `drop_off_window` stays true.",
        "- `off_window_max_rate` is a data-cleaning tolerance, not a viewbot threshold.",
        "- Files with small boundary overruns are kept after trimming only the off-window rows.",
        "- Files with larger off-window rates are wrong-window collections and are dropped.",
        "",
        "## Raw Versus Feature Minute Files",
        "- `minute_all.csv`, `minute_model.csv`, and `qc_zero.csv` contain raw minute rows only.",
        "- `minute_all_feat.csv`, `minute_model_feat.csv`, and `qc_zero_feat.csv` contain state features.",
        "",
        *SCORE_EXPLANATION,
        "",
        *EPISODE_EXPLANATION,
        "",
        *MODEL_IO_EXPLANATION,
        "",
        "## Legacy Baselines",
        "Existing session-level cluster, detector, synthetic, and model scores remain as legacy session-level baselines. They should be interpreted as auxiliary review signals.",
    ])

    _write(out / "final_task_slide_plan.md", [
        "# Final Task Slide Plan",
        "",
        "1. Feedback and task reset",
        "2. Input / Output definition",
        "3. Minute state features",
        "4. 07_ms.png: minute score distribution",
        "5. 08_mc.png: minute behavior cluster",
        "6. 09_sc.png: session cluster vs review score",
        "7. 10_cal.png: threshold calibration",
        "8. 11_ep.png: episode examples",
        "9. 12_rank.png: session review ranking",
        "10. 13_pipe.png: final pipeline",
        "11. Legacy baseline as appendix",
        "12. Limits and handoff",
    ])


def write_score_doc(out, cfg=None, session_df=None, syn_train=None, model_scores=None, session_model=None):
    if session_df is None and session_model is not None:
        session_df = session_model

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_session = _nrows(session_df)
    n_train = _nrows(syn_train)
    syn_counts = _value_counts(syn_train, "y_syn")
    score_cols_present = []
    if model_scores is not None:
        score_cols_present = [c for c in MODEL_SCORE_COLS if c in model_scores.columns]

    lines = [
        "Legacy Session-Level Score Document",
        "===================================",
        "These outputs are legacy session-level baselines, not the primary task output.",
        "The primary task is minute-level viewer-chat mismatch episode detection.",
        "",
        f"session score input rows: {n_session}",
        f"synthetic train rows: {n_train}",
        f"synthetic y_syn distribution: {syn_counts}",
        "",
        "Unsupervised detector features:",
        f"- {_join(DETECTOR_FEATS)}",
        "Model score features:",
        f"- {_join(MODEL_FEATS)}",
        "",
        "Current model score columns:",
        f"- {_join(score_cols_present) if score_cols_present else 'not available'}",
        "",
        "Important interpretation rules:",
        "- `cluster_number` and `session_behavior_cluster` are behavior cluster ids, not labels.",
        "- `y_syn` is an artificial mismatch injection label for synthetic sanity checks only.",
        "- Do not interpret any score as confirmed viewbot usage.",
        "- Do not feed review ranking outputs back into model input features.",
        "- Use review candidate, review ranking, mismatch episode, and legacy session-level baseline wording.",
    ]
    (out / "score.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_task_docs(out, cfg)
