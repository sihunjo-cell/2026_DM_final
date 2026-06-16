from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


KEY = ["run_id", "broad_no"]
ID_COLS = ["session_key", "run_id", "broad_no", "minute_ts"]
BASE_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "minute_ts",
    "base_log_chat_q50",
    "base_log_unique_q50",
    "model_chat_deficit",
    "model_unique_deficit",
    "baseline_agree_chat",
    "baseline_agree_unique",
]


def _write_csv(df, path, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if columns is not None:
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        out = out[columns]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def _ensure_keys(df):
    out = df.copy()
    if "session_key" not in out.columns:
        out["session_key"] = out["run_id"].astype(str) + "_" + out["broad_no"].astype(str)
    out["minute_ts"] = pd.to_datetime(out.get("minute_ts"), errors="coerce")
    if "minute_idx" not in out.columns:
        out = out.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)
        out["minute_idx"] = out.groupby(KEY).cumcount() + 1
    return out


def _minute_idx_norm(df):
    idx = pd.to_numeric(df["minute_idx"], errors="coerce")
    n = df.groupby("session_key")["minute_idx"].transform("max")
    den = pd.to_numeric(n, errors="coerce").sub(1).replace(0, np.nan)
    return idx.sub(1).div(den).fillna(0)


def _feature_frame(df):
    feat = pd.DataFrame(index=df.index)
    for col in ["log_viewer", "viewer_bin"]:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            feat[col] = np.nan
    feat["minute_idx_norm"] = _minute_idx_norm(df)
    if "minute_ts" in df.columns:
        feat["hour"] = pd.to_datetime(df["minute_ts"], errors="coerce").dt.hour
    else:
        feat["hour"] = np.nan
    if "category_id" in df.columns:
        cat = df["category_id"].astype("string").fillna("UNKNOWN_CAT")
        top = cat.value_counts().head(50).index
        cat = cat.where(cat.isin(top), "OTHER_CAT")
        dummies = pd.get_dummies(cat, prefix="category", dtype=float)
        feat = pd.concat([feat, dummies], axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat.fillna(feat.median(numeric_only=True)).fillna(0)


def _make_model():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.5,
            random_state=42,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            loss="quantile",
            alpha=0.5,
            random_state=42,
        )


def _folds(df):
    groups = df["run_id"] if "run_id" in df.columns else pd.Series(0, index=df.index)
    n_groups = int(pd.Series(groups).nunique(dropna=True))
    if n_groups >= 2:
        n_splits = min(5, n_groups)
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, groups=groups)), f"GroupKFold by run_id, n_splits={n_splits}"
    n_splits = min(5, len(df))
    if n_splits >= 2:
        splitter = KFold(n_splits=n_splits, shuffle=False)
        return list(splitter.split(df)), f"chronological KFold fallback, n_splits={n_splits}"
    return [], "descriptive median fallback because fewer than two rows were available"


def _oof_predict(df, x, target_col, cfg=None):
    cfg = cfg or {}
    base_cfg = cfg.get("m2_expected_response", {})
    max_train_rows = int(base_cfg.get("max_train_rows_per_fold", 200000))
    seed = int(base_cfg.get("seed", 42))
    y = pd.to_numeric(df.get(target_col), errors="coerce")
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    folds, note = _folds(df)
    rng = np.random.default_rng(seed + (17 if target_col == "log_unique" else 0))
    sampled_any = False
    for train_idx, test_idx in folds:
        train_mask = y.iloc[train_idx].notna()
        usable_train = np.asarray(train_idx)[train_mask.to_numpy()]
        if len(usable_train) < 10:
            continue
        if max_train_rows > 0 and len(usable_train) > max_train_rows:
            usable_train = rng.choice(usable_train, size=max_train_rows, replace=False)
            sampled_any = True
        model = _make_model()
        model.fit(x.iloc[usable_train], y.iloc[usable_train])
        pred.iloc[test_idx] = model.predict(x.iloc[test_idx])
    if sampled_any:
        note = note + f"; fold별 최대 {max_train_rows}개 row로 학습 표본을 제한했다"
    if pred.isna().any():
        fill = y.median()
        if pd.isna(fill):
            fill = 0.0
        pred = pred.fillna(float(fill))
        note = note + "; median-filled rows are descriptive fallback"
    return pred, note


def _agreement(model_deficit, bin_deficit):
    model = pd.to_numeric(model_deficit, errors="coerce")
    rule = pd.to_numeric(bin_deficit, errors="coerce")
    return (model.gt(0) & rule.gt(0)).astype(int)


def build_expected_response(minute_df, out, cfg=None):
    """Estimate normal chat and unique response without same-row fit/predict leakage."""
    out = Path(out)
    if minute_df is None or minute_df.empty:
        empty = _write_csv(pd.DataFrame(columns=BASE_COLS), out / "base_pred.csv", BASE_COLS)
        (out / "base_pred_info.txt").write_text("기대 반응 baseline은 입력 데이터가 비어 있어 실행하지 않았다.\n", encoding="utf-8")
        return empty

    df = _ensure_keys(minute_df)
    x = _feature_frame(df)
    chat_pred, chat_note = _oof_predict(df, x, "log_chat", cfg)
    unique_pred, unique_note = _oof_predict(df, x, "log_unique", cfg)

    actual_chat = pd.to_numeric(df.get("log_chat"), errors="coerce").fillna(0)
    actual_unique = pd.to_numeric(df.get("log_unique"), errors="coerce").fillna(0)
    result = df.copy()
    result["base_log_chat_q50"] = chat_pred
    result["base_log_unique_q50"] = unique_pred
    result["model_chat_deficit"] = result["base_log_chat_q50"] - actual_chat
    result["model_unique_deficit"] = result["base_log_unique_q50"] - actual_unique
    result["baseline_agree_chat"] = _agreement(result["model_chat_deficit"], result.get("chat_deficit"))
    result["baseline_agree_unique"] = _agreement(result["model_unique_deficit"], result.get("unique_deficit"))

    info = [
        "# expected-response baseline 근거",
        "",
        "목적: viewer scale, category, hour, session minute position을 고려했을 때 기대되는 chat/unique 수준을 추정한다.",
        "이 절차는 classifier가 아니며 정답 라벨을 사용하지 않는다.",
        "알고리즘: HistGradientBoostingRegressor(loss='quantile', quantile=0.5, random_state=42).",
        "fallback 모델: GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=42); 주 estimator 사용이 불가능할 때만 사용한다.",
        "validation: GroupKFold by run_id를 우선 사용하여 같은 run_id가 train/test에 동시에 들어가는 leakage를 줄인다.",
        f"chat fit: {chat_note}.",
        f"unique fit: {unique_note}.",
        f"사용 feature: {', '.join(x.columns[:80])}" + (" ..." if len(x.columns) > 80 else ""),
        "출력: base_log_chat_q50, base_log_unique_q50, model_chat_deficit, model_unique_deficit.",
        "한계: 실제 정답 라벨이 없는 baseline이다. deficit은 수동 검토 근거이며 확률이 아니다.",
    ]
    (out / "base_pred_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    return _write_csv(result[BASE_COLS], out / "base_pred.csv", BASE_COLS)
