import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


FEATS = ["log_viewer", "zero_rate", "gap_med", "gap_max", "log_unique", "log_zrun"]
SCORE_COLS = ["if_score", "lof_score", "ocsvm_score"]
LAB_COLS = ["if_lab", "lof_lab", "ocsvm_lab"]


def make_x(df):
    x = df[FEATS].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0)


def add_detectors(df, out, cfg):
    """Run each unsupervised detector separately and save individual scores.

    No final ensemble score is created here. Rank columns are only for plotting
    and scale-free inspection of each algorithm's own output.
    """
    out = Path(out)
    s = df.copy()

    if len(s) < 5:
        for c in SCORE_COLS + LAB_COLS + [f"{c}_rank" for c in SCORE_COLS]:
            s[c] = np.nan
        s.to_csv(out / "scores_unsup.csv", index=False, encoding="utf-8-sig")
        return s

    xs = RobustScaler().fit_transform(make_x(s))
    contam = float(cfg["detector"]["contam"])
    seed = int(cfg["detector"]["seed"])

    iso = IsolationForest(n_estimators=200, contamination=contam, random_state=seed)
    s["if_lab"] = iso.fit_predict(xs)
    s["if_score"] = -iso.decision_function(xs)

    lof = LocalOutlierFactor(n_neighbors=min(20, len(s) - 1), contamination=contam)
    s["lof_lab"] = lof.fit_predict(xs)
    s["lof_score"] = -lof.negative_outlier_factor_

    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=contam)
    s["ocsvm_lab"] = oc.fit_predict(xs)
    s["ocsvm_score"] = -oc.decision_function(xs).ravel()

    for c in SCORE_COLS:
        s[c + "_rank"] = s[c].rank(method="average", pct=True)

    s.to_csv(out / "scores_unsup.csv", index=False, encoding="utf-8-sig")
    return s
