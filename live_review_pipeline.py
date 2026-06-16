import argparse
import copy
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
CHZZK = ROOT / "chzzk-crawler"
DM = ROOT / "dm_pipe_final"
BASE_FEATURES = DM / "data" / "features"
TMP_ROOT = ROOT / "tmp"


def fail(msg):
    raise SystemExit(f"\n[ERROR] {msg}\n")


def run(cmd, cwd):
    print(f"\n$ {' '.join(str(x) for x in cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        fail(f"command failed: {' '.join(str(x) for x in cmd)}")


def db():
    load_dotenv(CHZZK / ".env", override=True)
    sys.path.insert(0, str(CHZZK))
    from core.db import engine

    return engine


def sql_df(query):
    return pd.read_sql(query, db())


def latest_run_id():
    df = sql_df("SELECT MAX(run_id) AS run_id FROM crawl_runs")
    val = df.iloc[0]["run_id"]
    return None if pd.isna(val) else int(val)


def run_status(run_id):
    df = sql_df(f"SELECT run_id,status,started_at,ended_at FROM crawl_runs WHERE run_id={int(run_id)}")
    return {} if df.empty else df.iloc[0].to_dict()


def ensure_tables():
    run([sys.executable, "scripts/setup.py", "--init"], CHZZK)


def ensure_targets(refresh=False):
    target_csv = CHZZK / "top30_targets.csv"
    if refresh or not target_csv.exists():
        run([sys.executable, "build_pool.py"], CHZZK)
    if not target_csv.exists():
        fail("top30_targets.csv was not created")
    run([sys.executable, "scripts/setup.py", "--load-csv", "top30_targets.csv"], CHZZK)


def collect(duration, window):
    before = latest_run_id()
    run([sys.executable, "scripts/run_pilot.py", "--window", window, "--duration", str(duration)], CHZZK)
    after = latest_run_id()
    if after is None or after == before:
        fail("collector finished but no new run_id was created")
    return after


def export_features(run_id):
    export_dir = CHZZK / "exports"
    export_dir.mkdir(exist_ok=True)
    feature_file = export_dir / f"Run_{run_id}_Features.xlsx"
    if not feature_file.exists():
        run(
            [
                sys.executable,
                "scripts/export_csv.py",
                "--table",
                "minute_features",
                "--run_id",
                str(run_id),
                "--out",
                str(feature_file),
                "--excel",
            ],
            CHZZK,
        )
    if not feature_file.exists():
        fail(f"feature export missing: {feature_file}")
    df = pd.read_excel(feature_file)
    if df.empty:
        fail(f"feature export has no rows: {feature_file}")
    return feature_file


def window_for(ts):
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    if ts.empty:
        return None
    start = ts.min().floor("h")
    end = (ts.max() + pd.Timedelta(minutes=1)).ceil("h")
    if start == end:
        end = start + pd.Timedelta(hours=1)
    return f"{start:%H:%M}-{end:%H:%M}"


def live_in_valid_window(final_ts, cfg):
    """True if final_ts falls in a cfg valid_window, using dm_pipe's own window_mask
    (same off_window_rate threshold the model uses to keep/drop a file)."""
    sys.path.insert(0, str(DM))
    from src.load import window_mask

    ts = pd.to_datetime(final_ts, errors="coerce").dropna()
    if ts.empty:
        return False, "no_valid_time"
    wm = window_mask(ts, cfg)
    off_rate = wm.get("off_window_rate")
    off_rate = 1.0 if off_rate is None or pd.isna(off_rate) else float(off_rate)
    off_max = float(cfg.get("time", {}).get("off_window_max_rate", 0.0))
    return off_rate <= off_max, wm.get("time_window")


def normalize_live_feature(src, dst, cfg):
    df = pd.read_excel(src)
    ts = pd.to_datetime(df["minute_ts"], errors="coerce")
    shift_h = int(cfg.get("time", {}).get("shift_hours", 0))
    now = datetime.now()

    raw_gap = abs((ts.dropna().median().to_pydatetime() - now).total_seconds()) if ts.notna().any() else 999999
    shifted_gap = (
        abs(((ts.dropna().median() + pd.Timedelta(hours=shift_h)).to_pydatetime() - now).total_seconds())
        if ts.notna().any()
        else 999999
    )
    live_already_local = shift_h and raw_gap < shifted_gap
    if live_already_local:
        df["minute_ts"] = ts - pd.Timedelta(hours=shift_h)
        final_ts = pd.to_datetime(df["minute_ts"], errors="coerce") + pd.Timedelta(hours=shift_h)
        print("[INFO] live feature timestamps look local/KST; temp copy shifted before dm_pipe shift.")
    else:
        final_ts = ts + pd.Timedelta(hours=shift_h)

    live_window = window_for(final_ts)
    in_window, matched_window = live_in_valid_window(final_ts, cfg)
    if in_window:
        print(f"[INFO] live window {live_window} is inside cfg valid_windows ({matched_window}).")
    else:
        print(f"[INFO] live window {live_window} is outside cfg valid_windows; this run will not be scored.")

    df.to_excel(dst, index=False)
    return {"live_window": live_window, "in_window": in_window, "matched_window": matched_window}


def hardlink_feature(src, dst):
    try:
        os.link(src, dst)
    except OSError as exc:
        fail(
            "baseline feature hardlink failed. "
            "tmp and dm_pipe_final/data/features must be on the same filesystem, "
            f"and hardlinks must be supported. src={src} dst={dst} error={exc}"
        )


def make_temp_inputs(run_id, live_feature):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = TMP_ROOT / f"live_review_{stamp}"
    feat_dir = work / "features"
    out_dir = work / "out"
    feat_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    cfg_path = DM / "cfg.yml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["path"]["in_dir"] = str(feat_dir.resolve())
    cfg["path"]["out_dir"] = str(out_dir.resolve())
    cfg.setdefault("eval_robustness", {})["enabled"] = False

    conflict = BASE_FEATURES / f"Run_{run_id}_Features.xlsx"
    if conflict.exists():
        fail(f"run_id conflict with baseline feature file: {conflict}")

    linked = 0
    for file in sorted(BASE_FEATURES.glob("Run_*_Features.xlsx")):
        hardlink_feature(file, feat_dir / file.name)
        linked += 1
    print(f"[INFO] hardlinked {linked} baseline feature files without duplicating file data.")

    live_status = normalize_live_feature(live_feature, feat_dir / live_feature.name, cfg)

    temp_cfg = work / "cfg.yml"
    with open(temp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return work, temp_cfg, out_dir, live_status


def run_model(temp_cfg):
    run([sys.executable, str(DM / "run.py"), str(temp_cfg), "--archive-mode", "none"], ROOT)


def cleanup_temp_work(work):
    work = Path(work).resolve()
    root = TMP_ROOT.resolve()
    if root not in work.parents or not work.name.startswith("live_review_"):
        fail(f"refusing to remove unexpected temp work dir: {work}")
    try:
        shutil.rmtree(work)
        print(f"[INFO] removed temp work dir after successful report: {work}")
    except OSError as exc:
        print(f"[WARN] could not remove temp work dir: {work} ({exc})")


def level(top_pct):
    if top_pct <= 1:
        return "최우선 리뷰"
    if top_pct <= 5:
        return "우선 리뷰"
    if top_pct <= 10:
        return "리뷰 권장"
    if top_pct <= 25:
        return "관찰 후보"
    return "낮은 우선순위"


def print_report(out_dir, run_id):
    review = pd.read_csv(out_dir / "m2_review.csv", encoding="utf-8-sig")
    review_all = pd.read_csv(out_dir / "m2_review_all.csv", encoding="utf-8-sig")
    live_all = review_all[pd.to_numeric(review_all["run_id"], errors="coerce").eq(run_id)].copy()
    live = review[pd.to_numeric(review["run_id"], errors="coerce").eq(run_id)].copy()

    total_eligible = len(review)
    reference_eligible = total_eligible - len(live)
    print("\nLIVE REVIEW SUMMARY")
    print(f"run_id: {run_id}")
    print(f"reference eligible sessions: {reference_eligible}")
    print(f"live sessions in review_all: {len(live_all)}")
    print(f"reviewable live sessions: {len(live)}")
    print(f"not reviewable live sessions: {len(live_all) - len(live)}")

    if not live.empty:
        live["review_order"] = pd.to_numeric(live["review_order"], errors="coerce")
        live["top_pct"] = live["review_order"] / max(total_eligible, 1) * 100
        live = live.sort_values("review_order")
        print("\n[PRIORITY]")
        print("rank | broad_no | top_pct | level | family_score | duration | reason")
        for _, r in live.iterrows():
            reason = r.get("dominant_reason") or r.get("reason_set") or "reason_pending"
            print(
                f"{int(r['review_order']):>4} | {r['broad_no']} | "
                f"{float(r['top_pct']):>6.2f}% | {level(float(r['top_pct']))} | "
                f"{float(r.get('family_consensus_score', 0) or 0):.4f} | "
                f"{r.get('top_interval_duration', '')} | {reason}"
            )
    else:
        print("\n[PRIORITY]\nNo live sessions were eligible for final review.")

    blocked = live_all[~live_all["session_key"].astype(str).isin(set(live["session_key"].astype(str)))].copy()
    if not blocked.empty:
        print("\n[NOT REVIEWABLE]")
        for _, r in blocked.iterrows():
            print(f"{r.get('broad_no')} | {r.get('review_qc_reason', 'unknown_qc_reason')}")


def report_off_window(run_id, live_status, temp_cfg):
    with open(temp_cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    valid = cfg.get("time", {}).get("valid_windows", [])
    print("\nLIVE REVIEW SUMMARY")
    print(f"run_id: {run_id}")
    print(f"collected window (KST): {live_status.get('live_window')}")
    print("\n[OFF-WINDOW]")
    print("지정되지 않은 시간대에 수집되었습니다.")
    print(f"cfg.yml의 valid_windows에 지정된 시간대만 결과를 냅니다: {valid}")
    print("→ 이 run은 review 결과를 생성하지 않습니다. 위 시간대 중 하나에서 수집한 뒤 다시 실행하세요.")


def load_dm_cfg():
    with open(DM / "cfg.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def current_window_status(cfg, duration):
    """Pre-collection check: is [now, now+duration] inside a cfg valid_window?
    now() is local KST and the model maps rows back to that same KST time-of-day,
    so we compare the wall clock directly (no shift)."""
    start = pd.Timestamp(datetime.now()).floor("min")
    end = (start + pd.Timedelta(seconds=max(int(duration), 60))).ceil("min")
    span = pd.Series(pd.date_range(start, end, freq="min"))
    in_window, matched = live_in_valid_window(span, cfg)
    return in_window, matched, window_for(span)


def report_off_window_precollect(cfg, window_label):
    valid = cfg.get("time", {}).get("valid_windows", [])
    print("\nLIVE REVIEW SUMMARY")
    print(f"current window (KST): {window_label}")
    print("\n[OFF-WINDOW]")
    print("현재 시간대가 cfg.yml의 valid_windows에 없어 수집을 시작하지 않습니다.")
    print(f"cfg.yml에 지정된 시간대에만 수집·결과가 생성됩니다: {valid}")
    print("→ 위 시간대 중 하나에서 다시 실행하세요. (기존 run 재점수화는 --skip-collect/--run-id 로 가능)")


def active_running_run():
    try:
        df = sql_df("SELECT run_id,status FROM crawl_runs WHERE status='running' ORDER BY run_id DESC LIMIT 1")
        if not df.empty:
            return int(df.iloc[0]["run_id"])
    except Exception:
        return None
    return None


def main():
    p = argparse.ArgumentParser(description="Collect CHZZK live data and rank live sessions with existing Method2 review pipeline.")
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--window", default="live_review")
    p.add_argument("--run-id", type=int, help="Use an existing collected run_id instead of collecting a new one.")
    p.add_argument("--skip-collect", action="store_true", help="Use --run-id or latest run without starting the collector.")
    p.add_argument("--refresh-targets", action="store_true")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary model outputs for debugging.")
    args = p.parse_args()

    ensure_tables()

    if args.skip_collect or args.run_id:
        run_id = args.run_id or latest_run_id()
        if run_id is None:
            fail("no existing run_id found")
        st = run_status(run_id)
        if str(st.get("status")) == "running":
            fail(f"run_id={run_id} is still running. Wait for run_pilot.py to finish, then rerun this command.")
    else:
        running = active_running_run()
        if running:
            fail(f"collector is already running as run_id={running}. Wait, then run: python live_review_pipeline.py --skip-collect --run-id {running}")
        cfg = load_dm_cfg()
        in_window, _matched, window_label = current_window_status(cfg, args.duration)
        if not in_window:
            report_off_window_precollect(cfg, window_label)
            return
        ensure_targets(refresh=args.refresh_targets)
        run_id = collect(args.duration, args.window)

    feature = export_features(run_id)
    work, temp_cfg, out_dir, live_status = make_temp_inputs(run_id, feature)
    print(f"[INFO] temp work dir: {work}")
    if live_status["in_window"]:
        run_model(temp_cfg)
        print_report(out_dir, run_id)
    else:
        report_off_window(run_id, live_status, temp_cfg)
    if args.keep_temp:
        print(f"\n[INFO] temp output kept: {out_dir}")
    else:
        cleanup_temp_work(work)


if __name__ == "__main__":
    main()
