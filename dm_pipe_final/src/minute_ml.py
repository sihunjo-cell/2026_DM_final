from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


FEATURES = [
    "log_viewer",
    "chat_deficit",
    "unique_deficit",
    "rolling_chat_deficit_5m",
    "log_zero_run_len",
    "rolling_zero_rate_5m",
]

OUT_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "minute_ts",
    "if_score",
    "if_rank",
    "if_dir_score",
    "lof_score",
    "lof_rank",
    "lof_dir_score",
    "ocsvm_score",
    "ocsvm_rank",
    "ocsvm_dir_score",
    "directional_weight",
]


def _enabled(value):
    return str(value).lower() not in {"false", "0", "no", "none", "off", "disabled"}


def _pct_rank(s):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    return x.rank(method="average", pct=True)


def _make_x(df):
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def _empty(out, note):
    out = Path(out)
    cols = OUT_COLS
    pd.DataFrame(columns=cols).to_csv(out / "ml_scores.csv", index=False, encoding="utf-8-sig")
    (out / "ml_info.txt").write_text(note.rstrip() + "\n", encoding="utf-8")
    return pd.DataFrame(columns=cols)


def build_ml_scores(minute_df, out, cfg):
    """Build optional detector scores for method 2 diagnostics.

    Detector scores are candidate signals only. They are not true performance
    metrics and are not merged into the rule-based method 2 score.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ml_cfg = cfg.get("minute_ml", {})
    if not _enabled(ml_cfg.get("enabled", "optional")):
        return _empty(out, "minute_ml disabled; detector scores skipped.")

    df = minute_df.copy()
    if df.empty:
        return _empty(out, "minute_ml skipped because minute data is empty.")

    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    if "session_key" not in df.columns:
        df["session_key"] = df["run_id"].astype(str) + "_" + df["broad_no"].astype(str)

    score = df[["session_key", "run_id", "broad_no", "minute_ts"]].copy()
    for col in ["if_score", "if_rank", "if_dir_score", "lof_score", "lof_rank", "lof_dir_score", "ocsvm_score", "ocsvm_rank", "ocsvm_dir_score"]:
        score[col] = np.nan

    x = _make_x(df)
    xs = RobustScaler().fit_transform(x)
    direction_cols = [
        "chat_deficit",
        "unique_deficit",
        "rolling_chat_deficit_5m",
        "rolling_zero_rate_5m",
    ]
    direction = pd.concat([_pct_rank(df[c]) for c in direction_cols], axis=1).mean(axis=1, skipna=True)
    score["directional_weight"] = direction

    notes = [
        "Minute ML detector diagnostics.",
        "Detector scores are auxiliary candidate signals, not true performance.",
        "No AUC-ROC, accuracy, or label metric is computed.",
    ]
    detectors = [str(x).lower() for x in ml_cfg.get("detectors", ["iforest", "lof_optional", "ocsvm_optional"])]
    seed = int(ml_cfg.get("seed", 42))

    try:
        if_model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )
        if_model.fit(xs)
        raw = -if_model.score_samples(xs)
        score["if_score"] = raw
        score["if_rank"] = _pct_rank(score["if_score"])
        score["if_dir_score"] = score["if_rank"] * score["directional_weight"]
        notes.append("IsolationForest completed.")
    except Exception as exc:
        notes.append(f"IsolationForest skipped after failure: {exc}")

    if any("lof" in d for d in detectors):
        if len(df) > int(ml_cfg.get("lof_max_rows", 50000)):
            notes.append("LOF skipped because row count exceeds lof_max_rows.")
        elif len(df) <= 40:
            notes.append("LOF skipped because there are too few rows.")
        else:
            try:
                n_neighbors = min(35, max(2, len(df) - 1))
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
                lab = lof.fit_predict(xs)
                raw = -lof.negative_outlier_factor_
                score["lof_score"] = raw
                score["lof_rank"] = _pct_rank(score["lof_score"])
                score["lof_dir_score"] = score["lof_rank"] * score["directional_weight"]
                notes.append(f"LOF completed with n_neighbors={n_neighbors}; labels are not used as labels.")
            except Exception as exc:
                notes.append(f"LOF skipped after failure: {exc}")

    if any("ocsvm" in d for d in detectors):
        if not bool(ml_cfg.get("run_ocsvm", False)):
            notes.append("OneClassSVM skipped by default because it can be slow.")
        else:
            try:
                max_rows = int(ml_cfg.get("ocsvm_max_rows", 50000))
                if len(df) > max_rows:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(np.arange(len(df)), size=max_rows, replace=False)
                    fit_x = xs[idx]
                    notes.append(f"OneClassSVM fit on sample of {max_rows} rows.")
                else:
                    fit_x = xs
                svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
                svm.fit(fit_x)
                raw = -svm.score_samples(xs)
                score["ocsvm_score"] = raw
                score["ocsvm_rank"] = _pct_rank(score["ocsvm_score"])
                score["ocsvm_dir_score"] = score["ocsvm_rank"] * score["directional_weight"]
                notes.append("OneClassSVM completed.")
            except Exception as exc:
                notes.append(f"OneClassSVM skipped after failure: {exc}")

    for col in OUT_COLS:
        if col not in score.columns:
            score[col] = np.nan
    score[OUT_COLS].to_csv(out / "ml_scores.csv", index=False, encoding="utf-8-sig")
    (out / "ml_info.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return score[OUT_COLS]

