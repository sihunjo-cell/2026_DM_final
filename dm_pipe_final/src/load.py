from pathlib import Path
import re
import numpy as np
import pandas as pd


def run_id_from_name(name):
    m = re.search(r"Run_(\d+)_Features", name)
    return int(m.group(1)) if m else None


def _min_of_day(ts):
    ts = pd.to_datetime(ts, errors="coerce").dt.floor("min")
    return ts.dt.hour * 60 + ts.dt.minute


def _parse_time(t):
    h, m = str(t).split(":")[:2]
    return int(h) * 60 + int(m)


def _parse_window(w):
    a, b = str(w).split("-")
    return _parse_time(a), _parse_time(b), str(w)


def _in_window(mins, start, end, tol):
    start = start - tol
    end = end + tol
    if start < 0:
        return (mins >= start + 1440) | (mins <= end)
    if end >= 1440:
        return (mins >= start) | (mins <= end - 1440)
    if start <= end:
        return (mins >= start) & (mins <= end)
    return (mins >= start) | (mins <= end)


def window_mask(ts, cfg):
    ts = pd.to_datetime(ts, errors="coerce")
    windows = cfg.get("time", {}).get(
        "valid_windows",
        ["14:00-16:00", "17:00-19:00", "20:00-22:00", "23:00-01:00"],
    )
    tol = int(cfg.get("time", {}).get("tolerance_min", 1))

    if ts.empty:
        return {
            "best_mask": pd.Series(False, index=ts.index, dtype=bool),
            "time_ok": 0,
            "time_window": "no_valid_time",
            "off_window_rows": 0,
            "off_window_rate": np.nan,
        }

    valid = ts.notna()
    if not valid.any():
        return {
            "best_mask": pd.Series(False, index=ts.index, dtype=bool),
            "time_ok": 0,
            "time_window": "no_valid_time",
            "off_window_rows": int(len(ts)),
            "off_window_rate": 1.0 if len(ts) else np.nan,
        }

    mins = _min_of_day(ts.loc[valid])
    best_name = "off_window"
    best_ok = pd.Series(False, index=ts.index, dtype=bool)
    for w in windows:
        start, end, name = _parse_window(w)
        ok = pd.Series(False, index=ts.index, dtype=bool)
        ok.loc[valid] = _in_window(mins, start, end, tol).to_numpy()
        if ok.mean() > best_ok.mean():
            best_name, best_ok = name, ok

    off = int((~best_ok).sum())
    return {
        "best_mask": best_ok,
        "time_ok": int(off == 0),
        "time_window": best_name,
        "off_window_rows": off,
        "off_window_rate": float(off / len(ts)),
    }


def window_check(ts, cfg):
    out = window_mask(ts, cfg).copy()
    out.pop("best_mask", None)
    return out


def load_features(cfg):
    in_dir = Path(cfg["path"]["in_dir"])
    files = sorted(in_dir.glob(cfg["path"]["pattern"]))
    if not files:
        raise FileNotFoundError(f"no files: {in_dir / cfg['path']['pattern']}")

    shift_h = int(cfg.get("time", {}).get("shift_hours", 9))
    drop_runs = set(int(x) for x in cfg.get("time", {}).get("drop_runs", []))
    drop_off = bool(cfg.get("time", {}).get("drop_off_window", True))
    off_window_max = float(cfg.get("time", {}).get("off_window_max_rate", 0.0))
    trim_off_window = bool(cfg.get("time", {}).get("trim_off_window_rows", False))
    frames, audit = [], []

    for file in files:
        df = pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        df["source_file"] = file.name

        if "run_id" not in df.columns or df["run_id"].isna().all():
            df["run_id"] = run_id_from_name(file.name)

        df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
        df["minute_ts"] = pd.to_datetime(df.get("minute_ts"), errors="coerce")
        raw_start, raw_end = df["minute_ts"].min(), df["minute_ts"].max()
        df["minute_ts"] = df["minute_ts"] + pd.Timedelta(hours=shift_h)

        rows_raw = len(df)
        df = df.dropna(subset=["run_id", "minute_ts"]).copy()
        df["run_id"] = df["run_id"].astype(int)
        run_id = run_id_from_name(file.name)
        if run_id is None and len(df):
            run_id = int(df["run_id"].iloc[0])

        wm = window_mask(df["minute_ts"], cfg)
        best_mask = wm.pop("best_mask")
        reason = "ok"
        use = len(df) > 0
        rows_after_basic_clean = len(df)
        rows_trimmed_off_window = 0
        off_window_policy = "keep_all"

        if run_id in drop_runs:
            use = False
            reason = "dropped_by_run_id"
            off_window_policy = "drop_file"
        elif rows_after_basic_clean == 0:
            use = False
            reason = "no_valid_rows"
            off_window_policy = "drop_file"
        elif not drop_off:
            reason = "ok_no_window_filter"
            off_window_policy = "keep_all"
        elif int(wm["off_window_rows"]) == 0:
            reason = "ok"
            off_window_policy = "keep_all"
        elif float(wm["off_window_rate"]) <= off_window_max and trim_off_window:
            rows_trimmed_off_window = int(wm["off_window_rows"])
            df = df.loc[best_mask].copy()
            use = len(df) > 0
            reason = "trim_off_window" if use else "no_valid_rows"
            off_window_policy = "trim_rows" if use else "drop_file"
        elif float(wm["off_window_rate"]) > off_window_max:
            use = False
            reason = "drop_off_window_major"
            off_window_policy = "drop_file"
        else:
            use = False
            reason = "drop_off_window_minor_trim_disabled"
            off_window_policy = "drop_file"

        rows_used_after_trim = len(df) if use else 0

        audit.append({
            "file": file.name,
            "use": int(use),
            "rows_raw": rows_raw,
            "rows_after_basic_clean": rows_after_basic_clean,
            "rows_used": rows_used_after_trim,
            "rows_trimmed_off_window": rows_trimmed_off_window,
            "rows_used_after_trim": rows_used_after_trim,
            "off_window_policy": off_window_policy,
            "run_id": run_id,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "kst_start": df["minute_ts"].min() if len(df) else pd.NaT,
            "kst_end": df["minute_ts"].max() if len(df) else pd.NaT,
            "shift_hours": shift_h,
            **wm,
            "reason": reason,
        })
        if use:
            frames.append(df)

    file_audit = pd.DataFrame(audit)
    if not frames:
        out = Path(cfg["path"]["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        file_audit.to_csv(out / "file_audit.csv", index=False, encoding="utf-8-sig")
        raise ValueError("no usable feature rows; see out/file_audit.csv")

    raw = pd.concat(frames, ignore_index=True)
    used = (
        raw.groupby("run_id")
           .agg(rows=("run_id", "size"), start=("minute_ts", "min"), end=("minute_ts", "max"), files=("source_file", "nunique"))
           .reset_index()
           .sort_values("run_id")
    )
    checks = used.apply(lambda r: pd.Series(window_check(pd.Series([r["start"], r["end"]]), cfg)), axis=1)
    used = pd.concat([used, checks], axis=1)
    return raw, file_audit, used
