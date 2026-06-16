from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


INT_COLS = [
    "session_key",
    "run_id",
    "broad_no",
    "int_if_score",
    "int_if_rank",
    "int_if_dir_score",
    "int_ecod_score",
    "int_ecod_rank",
    "int_ecod_dir_score",
    "int_lof_score",
    "int_lof_rank",
    "int_lof_dir_score",
    "directional_weight",
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


def _read(out, name):
    path = Path(out) / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _pct_rank(s):
    if s is None:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return x.rank(method="average", pct=True)


def _feature_cols(scan):
    cols = [
        "top_interval_duration",
        "interval_mean_rank",
        "interval_p95_rank",
        "interval_chat_deficit_mean",
        "interval_unique_deficit_mean",
        "interval_model_chat_deficit_mean",
        "interval_model_unique_deficit_mean",
        "interval_zero_rate",
        "interval_max_zero_run",
        "interval_mismatch_state_rate",
    ]
    for col in cols:
        if col not in scan.columns:
            scan[col] = np.nan
    return cols


def build_interval_anomaly(out, cfg=None):
    out = Path(out)
    cfg = cfg or {}
    scan = _read(out, "m2_scan.csv")
    if scan.empty:
        return _write_csv(pd.DataFrame(columns=INT_COLS), out / "int_scores.csv", INT_COLS)

    result = scan[["session_key", "run_id", "broad_no"]].copy()
    for col in [
        "int_if_score",
        "int_if_rank",
        "int_if_dir_score",
        "int_ecod_score",
        "int_ecod_rank",
        "int_ecod_dir_score",
        "int_lof_score",
        "int_lof_rank",
        "int_lof_dir_score",
    ]:
        result[col] = np.nan

    direction_cols = [
        "interval_chat_deficit_mean",
        "interval_unique_deficit_mean",
        "interval_zero_rate",
        "interval_mismatch_state_rate",
    ]
    result["directional_weight"] = pd.concat([_pct_rank(scan.get(col)) for col in direction_cols], axis=1).mean(axis=1, skipna=True)

    features = _feature_cols(scan)
    x = scan[features].replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    scaler_name = str(cfg.get("m2_interval_anomaly", {}).get("scaler", "RobustScaler"))
    if scaler_name == "RobustScaler":
        x_model = RobustScaler().fit_transform(x)
    else:
        x_model = x.to_numpy(dtype=float)
    if len(x) >= 5:
        if_cfg = cfg.get("m2_interval_anomaly", {}).get("iforest", {})
        iso = IsolationForest(
            n_estimators=int(if_cfg.get("n_estimators", 200)),
            max_samples=if_cfg.get("max_samples", "auto"),
            contamination=if_cfg.get("contamination", "auto"),
            random_state=int(if_cfg.get("random_state", 42)),
        )
        result["int_if_score"] = -iso.fit(x_model).decision_function(x_model)
        result["int_if_rank"] = _pct_rank(result["int_if_score"])
        result["int_if_dir_score"] = result["int_if_rank"] * result["directional_weight"]

        try:
            from pyod.models.ecod import ECOD

            ecod = ECOD()
            ecod.fit(x_model)
            result["int_ecod_score"] = ecod.decision_scores_
            result["int_ecod_rank"] = _pct_rank(result["int_ecod_score"])
            result["int_ecod_dir_score"] = result["int_ecod_rank"] * result["directional_weight"]
        except Exception:
            pass

        if len(x) >= 25:
            try:
                lof = LocalOutlierFactor(n_neighbors=min(20, len(x) - 1))
                lof.fit_predict(x_model)
                result["int_lof_score"] = -lof.negative_outlier_factor_
                result["int_lof_rank"] = _pct_rank(result["int_lof_score"])
                result["int_lof_dir_score"] = result["int_lof_rank"] * result["directional_weight"]
            except Exception:
                pass

    notes = [
        "# interval anomaly support 근거",
        "",
        "목적: adaptive scan으로 찾은 top interval의 profile이 전체 interval 분포에서 이질적인지 보는 보조 evidence이다.",
        "final label source가 아니며 anomaly score는 label/probability가 아니다.",
        f"사용 feature: {', '.join(features)}",
        "preprocessing: median imputation 후 all-missing column은 0으로 채운다.",
        f"scaler: {scaler_name}; heavy-tail interval feature의 extreme value 영향을 줄이기 위해 사용한다.",
        f"IsolationForest: n_estimators={cfg.get('m2_interval_anomaly', {}).get('iforest', {}).get('n_estimators', 200)}, max_samples={cfg.get('m2_interval_anomaly', {}).get('iforest', {}).get('max_samples', 'auto')}, contamination={cfg.get('m2_interval_anomaly', {}).get('iforest', {}).get('contamination', 'auto')}, random_state={cfg.get('m2_interval_anomaly', {}).get('iforest', {}).get('random_state', 42)}",
        "LOF: n_neighbors=min(20, n_sessions - 1), n_sessions >= 25일 때 scaled matrix에 적용한다.",
        "ECOD: pyod가 사용 가능할 때 scaled matrix에 적용한다.",
        "주의: 이 support가 short spike를 과도하게 밀어올리면 final RRA에서 제거하거나 appendix diagnostic으로 격리한다.",
    ]
    (out / "interval_anomaly.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return _write_csv(result, out / "int_scores.csv", INT_COLS)
