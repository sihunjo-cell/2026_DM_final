import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


FEATS = ["log_viewer", "log_chat", "log_unique", "zero_rate", "gap_med", "gap_max", "log_zrun"]
MODEL_SCORE_COLS = ["ae_score", "svm_syn_score", "xgb_syn_score", "lgb_syn_score", "pu_score"]


def make_x(df):
    x = df[FEATS].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def safe_metric(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def ae_scores(real, seed):
    """Unsupervised autoencoder score trained on all eligible sessions.

    This avoids using an arbitrary anomaly-score cutoff to define a normal set.
    The output remains an individual reconstruction-error score, not a label.
    """
    if len(real) < 5:
        return pd.Series(np.nan, index=real.index)

    scaler = RobustScaler()
    x_real = scaler.fit_transform(make_x(real))

    hidden = max(2, min(8, x_real.shape[1] // 2 + 1))
    ae = MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", max_iter=300, random_state=seed)
    ae.fit(x_real, x_real)
    rec = ae.predict(x_real)
    return pd.Series(((x_real - rec) ** 2).mean(axis=1), index=real.index)


def pu_scores(real, label_file, seed):
    path = Path(label_file)
    if not path.exists():
        return pd.Series(np.nan, index=real.index), "manual_label 없음"

    lab = pd.read_csv(path)
    if not {"session_key", "label"}.issubset(lab.columns):
        return pd.Series(np.nan, index=real.index), "manual_label 형식 오류"

    lab["session_key"] = lab["session_key"].astype(str)
    lab["label"] = pd.to_numeric(lab["label"], errors="coerce")
    pos_key = set(lab.loc[lab["label"].eq(1), "session_key"])
    data = real.copy()
    data["pos"] = data["session_key"].isin(pos_key).astype(int)

    pos = data[data["pos"].eq(1)]
    unl = data[data["pos"].eq(0)]
    if len(pos) < 2 or len(unl) < 3:
        return pd.Series(np.nan, index=real.index), "positive label 부족"

    rng = np.random.default_rng(seed)
    out = np.zeros(len(real))

    for _ in range(30):
        n_unl = min(len(unl), max(len(pos) * 3, 10))
        neg = unl.sample(n=n_unl, replace=len(unl) < n_unl, random_state=int(rng.integers(1_000_000)))
        tr = pd.concat([pos, neg], ignore_index=True)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        clf = make_pipeline(RobustScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
        clf.fit(make_x(tr), y)
        out += clf.predict_proba(make_x(real))[:, 1]

    return pd.Series(out / 30, index=real.index), "ok"


def add_model_scores(train, real, out, cfg):
    """Run AE, synthetic SVM/XGBoost/LightGBM, and PU scores individually."""
    out = Path(out)
    seed = int(cfg["model"]["seed"])
    result = real[["session_key"]].copy()
    metrics = []

    result["ae_score"] = ae_scores(real, seed).values

    y_ok = False
    if not train.empty and "y_syn" in train.columns and train["y_syn"].nunique() == 2:
        y_cnt = train["y_syn"].astype(int).value_counts()
        y_ok = bool(y_cnt.min() >= 2 and len(train) >= 8)

    if not y_ok:
        for c in ["svm_syn_score", "xgb_syn_score", "lgb_syn_score"]:
            result[c] = np.nan
        metrics.append({"model": "synthetic_supervised", "note": "synthetic train 부족"})
    else:
        x = make_x(train)
        y = train["y_syn"].astype(int)
        idx_tr, idx_te = train_test_split(
            np.arange(len(train)),
            test_size=float(cfg["model"]["test_size"]),
            random_state=seed,
            stratify=y,
        )
        x_real = make_x(real)

        models = {
            "svm_syn": make_pipeline(
                RobustScaler(),
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
            "xgb_syn": XGBClassifier(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.07,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            ),
            "lgb_syn": LGBMClassifier(
                n_estimators=60,
                learning_rate=0.07,
                num_leaves=15,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                verbose=-1,
                n_jobs=1,
            ),
        }

        for name, model in models.items():
            model.fit(x.iloc[idx_tr], y.iloc[idx_tr])
            p = model.predict_proba(x.iloc[idx_te])[:, 1]
            auc, ap = safe_metric(y.iloc[idx_te], p)
            metrics.append({"model": name, "auc": auc, "ap": ap})
            result[name + "_score"] = model.predict_proba(x_real)[:, 1]

    pu, status = pu_scores(real, cfg["path"]["label_file"], seed)
    result["pu_score"] = pu.values
    metrics.append({"model": "pu_bag", "note": status})

    for c in MODEL_SCORE_COLS:
        if c in result.columns:
            result[c + "_rank"] = result[c].rank(method="average", pct=True)

    result.to_csv(out / "scores_model.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(out / "model_metrics.csv", index=False, encoding="utf-8-sig")
    return result
