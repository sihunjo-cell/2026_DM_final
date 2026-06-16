import numpy as np
import pandas as pd


def good_rank(s, higher_is_better=True):
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=vals.index, dtype=float)
    return vals.rank(
        ascending=True if higher_is_better else False,
        method="average",
        pct=True,
    )
