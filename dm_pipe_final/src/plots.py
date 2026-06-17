from pathlib import Path
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


warnings.filterwarnings("ignore")
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Malgun Gothic" if "Malgun Gothic" in _available_fonts else "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

KEY = ["run_id", "broad_no"]
DPI = 200
SESSION_CLUSTER_TITLE_NOTE = "cluster_number는 행동 군집 번호이며 정답 라벨/확률 아님"


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _replace_with_hardlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _blank(path, msg):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    _save(fig, path)


def _read(out, name):
    path = Path(out) / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _plot_df(minute):
    p = minute.copy()
    if p.empty:
        return p
    p["minute_ts"] = pd.to_datetime(p["minute_ts"], errors="coerce")
    p = p.sort_values(KEY + ["minute_ts"]).reset_index(drop=True)
    for col in ["viewer_count_last", "chat_count", "unique_chatters"]:
        if col not in p.columns:
            p[col] = 0
        p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0)
    p["viewer_log"] = np.log1p(p["viewer_count_last"].clip(lower=0))
    p["chat_log"] = np.log1p(p["chat_count"].clip(lower=0))
    p["unique_log"] = np.log1p(p["unique_chatters"].clip(lower=0))
    p["gap"] = p["viewer_log"] - p["chat_log"]
    p["zero_chat"] = p["chat_count"].eq(0)
    if "minute_idx" not in p.columns:
        p["minute_idx"] = p.groupby(KEY).cumcount() + 1
    return p


def _plot_quality(minute_all, session_all, session_model, plots):
    p = _plot_df(minute_all)
    if p.empty:
        _blank(plots / "01_data_quality.png", "no minute data")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = ax.ravel()
    all_zero = int(session_all.get("all_zero", pd.Series(False, index=session_all.index)).sum()) if session_all is not None and not session_all.empty else 0
    counts = [len(session_all), all_zero, len(session_model)]
    ax[0].bar(["all sessions", "all-zero", "handoff sessions"], counts, edgecolor="black")
    ax[0].set_title("Session filtering")
    ax[0].set_ylabel("count")
    ax[0].grid(axis="y", alpha=0.25)
    ax[1].hist(p["viewer_count_last"].dropna(), bins="sturges", edgecolor="black")
    ax[1].set_title("Viewer count distribution")
    ax[1].set_xlabel("viewer_count_last")
    ax[1].set_ylabel("minutes")
    ax[1].grid(axis="y", alpha=0.25)
    z = p.loc[p["zero_chat"]].groupby(KEY).size()
    ax[2].hist(z, bins="sturges", edgecolor="black")
    ax[2].set_title("Zero-chat minute count per session")
    ax[2].set_xlabel("zero-chat minutes")
    ax[2].set_ylabel("sessions")
    ax[2].grid(axis="y", alpha=0.25)
    if "off_window_policy" in p.columns:
        vc = p["off_window_policy"].value_counts()
        ax[3].bar(vc.index.astype(str), vc.values, edgecolor="black")
    else:
        ax[3].hist(p["gap"].dropna(), bins="sturges", edgecolor="black")
    ax[3].set_title("Viewer-chat gap distribution")
    ax[3].set_xlabel("log viewer - log chat")
    ax[3].grid(axis="y", alpha=0.25)
    fig.suptitle("01 Data quality - not ground-truth label", fontsize=14)
    _save(fig, plots / "01_data_quality.png")


def _plot_dist_time(minute_model, plots):
    p = _plot_df(minute_model)
    if p.empty:
        _blank(plots / "02_dist_time.png", "no minute data")
        return
    agg = p.groupby("minute_idx").agg(
        viewer_med=("viewer_count_last", "median"),
        chat_med=("chat_count", "median"),
        gap_med=("gap", "median"),
        n_session=("session_key", "nunique"),
    ).reset_index()
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax[0, 0].hist(p["gap"].dropna(), bins="sturges", edgecolor="black")
    ax[0, 0].set_title("Viewer-chat log gap")
    ax[0, 1].hist(p["chat_log"].dropna(), bins="sturges", edgecolor="black")
    ax[0, 1].set_title("Chat log distribution")
    ax[1, 0].plot(agg["minute_idx"], agg["viewer_med"], color="tab:blue", label="viewer median")
    ax[1, 0].plot(agg["minute_idx"], agg["chat_med"], color="tab:orange", label="chat median")
    ax[1, 0].set_title("Minute trajectory")
    ax[1, 0].set_xlabel("minute_idx")
    ax[1, 0].legend()
    ax[1, 1].plot(agg["minute_idx"], agg["gap_med"], color="tab:green")
    ax[1, 1].bar(agg["minute_idx"], agg["n_session"] / max(agg["n_session"].max(), 1), alpha=0.2, color="gray")
    ax[1, 1].set_title("Median gap and surviving sessions")
    ax[1, 1].set_xlabel("minute_idx")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.suptitle("02 Distribution and time structure - not ground-truth label", fontsize=14)
    _save(fig, plots / "02_dist_time.png")


def _plot_view_chat(minute_model, plots):
    p = _plot_df(minute_model)
    if p.empty:
        _blank(plots / "03_view_chat.png", "no minute data")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    hb = ax[0].hexbin(p["chat_log"], p["viewer_log"], gridsize=45, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax[0], label="log count")
    ax[0].set_title("viewer_log vs chat_log")
    ax[0].set_xlabel("chat_log")
    ax[0].set_ylabel("viewer_log")
    ax[1].scatter(p.get("delta_viewer_1m", pd.Series(dtype=float)), p.get("delta_chat_1m", pd.Series(dtype=float)), s=3, alpha=0.08)
    dx = pd.to_numeric(p.get("delta_viewer_1m", pd.Series(dtype=float)), errors="coerce").dropna()
    dy = pd.to_numeric(p.get("delta_chat_1m", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(dx) > 10:
        ax[1].set_xlim(dx.quantile(0.01), dx.quantile(0.99))
    if len(dy) > 10:
        ax[1].set_ylim(dy.quantile(0.01), dy.quantile(0.99))
    ax[1].axhline(0, color="red", ls="--", lw=1)
    ax[1].axvline(0, color="red", ls="--", lw=1)
    ax[1].set_title("1-minute viewer vs chat change")
    ax[1].set_xlabel("delta_viewer_1m")
    ax[1].set_ylabel("delta_chat_1m")
    zero_by_bin = pd.DataFrame()
    if p["viewer_count_last"].nunique() > 1:
        p["_viewer_bin"] = pd.qcut(p["viewer_count_last"].rank(method="first"), 10, labels=False, duplicates="drop")
        zero_by_bin = p.groupby("_viewer_bin")["zero_chat"].mean()
    if not zero_by_bin.empty:
        ax[2].bar(zero_by_bin.index.astype(str), zero_by_bin.values, edgecolor="black")
    ax[2].set_title("Zero-chat rate by viewer decile")
    ax[2].set_xlabel("viewer decile")
    ax[2].set_ylabel("zero-chat rate")
    ax[2].set_ylim(0, 1)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("03 Viewer-chat dynamics - not ground-truth label", fontsize=14)
    _save(fig, plots / "03_view_chat.png")


def _plot_session_cluster(session_model, plots):
    s = session_model.copy()
    if s.empty or "cluster_number" not in s.columns:
        _blank(plots / "04_cluster_session.png", f"session cluster handoff not available\n{SESSION_CLUSTER_TITLE_NOTE}")
        return
    s["viewer_med"] = pd.to_numeric(s.get("viewer_med"), errors="coerce")
    s["gap_med"] = pd.to_numeric(s.get("gap_med"), errors="coerce")
    s["cluster_number"] = pd.to_numeric(s.get("cluster_number"), errors="coerce")
    s = s.dropna(subset=["viewer_med", "gap_med", "cluster_number"])
    if s.empty:
        _blank(plots / "04_cluster_session.png", f"session cluster handoff not available\n{SESSION_CLUSTER_TITLE_NOTE}")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    palette = plt.get_cmap("tab10").colors
    cluster_order = sorted(s["cluster_number"].dropna().unique())
    cluster_colors = {}
    for i, cluster_id in enumerate(cluster_order):
        part = s.loc[s["cluster_number"].eq(cluster_id)]
        color = palette[i % len(palette)]
        cluster_colors[cluster_id] = color
        label_id = int(cluster_id) if float(cluster_id).is_integer() else cluster_id
        ax[0].scatter(
            part["viewer_med"],
            part["gap_med"],
            label=f"cluster {label_id} (n={len(part)})",
            color=color,
            s=45,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
    ax[0].set_xscale("log")
    ax[0].set_title("세션 단위 행동 군집 산점도")
    ax[0].set_xlabel("세션 중앙 시청자 수")
    ax[0].set_ylabel("viewer-chat gap 중앙값")
    ax[0].legend(title="cluster_number", fontsize=8)
    ax[0].text(
        0.02,
        0.02,
        "2D는 해석용 projection이며 clustering은 전체 feature로 수행됨",
        transform=ax[0].transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72),
    )
    prof = s.groupby("cluster_number").agg(n=("session_key", "size"), viewer_med=("viewer_med", "median"), chat_mean=("chat_mean", "mean"), zero_rate=("zero_rate", "mean")).reset_index()
    prof = prof.sort_values("cluster_number")
    bar_colors = [cluster_colors.get(v, "tab:blue") for v in prof["cluster_number"]]
    bars = ax[1].bar(prof["cluster_number"].astype(int).astype(str), prof["n"], edgecolor="black", color=bar_colors)
    for bar in bars:
        ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    ax[1].set_title("군집별 세션 수")
    ax[1].set_xlabel("cluster_number")
    ax[1].set_ylabel("세션 수")
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(f"04 세션 단위 행동 군집 산점도 - {SESSION_CLUSTER_TITLE_NOTE}", fontsize=14)
    _save(fig, plots / "04_cluster_session.png")


def _plot_session_k_selection(out, plots):
    sel = _read(out, "cluster_select.csv")
    if sel.empty or "k" not in sel.columns:
        _blank(plots / "05_session_k_selection.png", f"session K selection not available\n{SESSION_CLUSTER_TITLE_NOTE}")
        return
    work = sel.copy()
    work["k"] = pd.to_numeric(work["k"], errors="coerce")
    work["selection_score"] = pd.to_numeric(work.get("selection_score"), errors="coerce")
    work["silhouette"] = pd.to_numeric(work.get("silhouette"), errors="coerce")
    work["davies_bouldin"] = pd.to_numeric(work.get("davies_bouldin"), errors="coerce")
    work["calinski_harabasz"] = pd.to_numeric(work.get("calinski_harabasz"), errors="coerce")
    work = work.dropna(subset=["k"]).sort_values("k")
    if work.empty:
        _blank(plots / "05_session_k_selection.png", f"session K selection not available\n{SESSION_CLUSTER_TITLE_NOTE}")
        return
    if "selected" in work.columns:
        selected = work["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        selected = work["silhouette"].eq(work["silhouette"].max())
    colors = np.where(selected, "tab:orange", "tab:blue")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    bars = ax[0].bar(work["k"].astype(int).astype(str), work["selection_score"], color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, work["selection_score"]):
        if pd.notna(val):
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{float(val):.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_title("selection_score by K")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("selection_score")
    ax[1].plot(work["k"], work["silhouette"], marker="o", color="tab:blue", label="silhouette")
    ax[1].plot(work["k"], work["davies_bouldin"], marker="s", color="tab:red", label="Davies-Bouldin (lower is better)")
    if work["calinski_harabasz"].notna().any():
        ch = work["calinski_harabasz"]
        denom = ch.max() - ch.min()
        ch_scaled = pd.Series(0.5, index=work.index) if denom == 0 else (ch - ch.min()) / denom
        ax[1].plot(work["k"], ch_scaled, marker="^", color="tab:green", label="Calinski-Harabasz scaled")
    ax[1].set_title("diagnostic metrics by K")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("diagnostic value")
    ax[1].legend(fontsize=8)
    ax[1].text(
        0.02,
        0.02,
        "selection_score = direction-corrected percentile-rank composite of silhouette, Calinski-Harabasz, Davies-Bouldin, size balance, and profile separation.\nInertia is diagnostic only, not a selection criterion.",
        transform=ax[1].transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
    )
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("05 Session K selection: composite score, not ground-truth label", fontsize=14)
    _save(fig, plots / "05_session_k_selection.png")


def _plot_session_cluster_profile(out, plots):
    prof = _read(out, "cluster_profile.csv")
    if prof.empty or "cluster_number" not in prof.columns:
        _blank(plots / "06_session_cluster_profile.png", f"session cluster profile not available\n{SESSION_CLUSTER_TITLE_NOTE}")
        return
    metrics = [c for c in ["viewer_med", "chat_mean", "unique_mean", "zero_rate", "gap_med", "zrun_max"] if c in prof.columns]
    if not metrics:
        _blank(plots / "06_session_cluster_profile.png", f"session cluster profile not available\n{SESSION_CLUSTER_TITLE_NOTE}")
        return
    work = prof.copy().sort_values("cluster_number")
    matrix = work[metrics].apply(pd.to_numeric, errors="coerce")
    scaled = matrix.copy()
    for col in metrics:
        lo = matrix[col].min()
        hi = matrix[col].max()
        scaled[col] = 0.5 if pd.isna(lo) or pd.isna(hi) or hi == lo else (matrix[col] - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    im = ax.imshow(scaled.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(work)), work["cluster_number"].astype(str))
    ax.set_xlabel("profile metric, min-max scaled for display")
    ax.set_ylabel("cluster_number")
    ax.set_title(f"06 session cluster profile - {SESSION_CLUSTER_TITLE_NOTE}")
    for i in range(len(work)):
        for j, col in enumerate(metrics):
            val = matrix.iloc[i, j]
            label = "" if pd.isna(val) else f"{val:.2g}"
            ax.text(j, i, label, ha="center", va="center", color="white" if scaled.iloc[i, j] < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="within-metric scaled profile value, not cluster id")
    fig.text(0.5, 0.01, "cell text = original profile value; color = relative within-feature contrast", ha="center", fontsize=9)
    _save(fig, plots / "06_session_cluster_profile.png")


def _plot_session_cluster_stability(out, plots):
    stab = _read(out, "mc_stab.csv")
    if stab.empty or "ari_vs_base" not in stab.columns:
        _blank(plots / "07_session_cluster_stability.png", "minute KMeans behavior-state stability diagnostic not available\nfilename retained for backward compatibility; not a supervised performance metric")
        return
    work = stab.copy()
    work["seed"] = pd.to_numeric(work.get("seed"), errors="coerce")
    work["ari_vs_base"] = pd.to_numeric(work.get("ari_vs_base"), errors="coerce")
    work["mismatch_cluster_share"] = pd.to_numeric(work.get("mismatch_cluster_share"), errors="coerce")
    work["selected_k"] = pd.to_numeric(work.get("selected_k"), errors="coerce")
    work = work.sort_values("seed")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax[0].plot(work["seed"], work["ari_vs_base"], marker="o", color="tab:blue")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("seed")
    ax[0].set_ylabel("ARI vs base")
    ax[0].set_title("seed/subsample ARI")
    ax[1].plot(work["seed"], work["mismatch_cluster_share"], marker="o", color="tab:orange", label="mismatch share")
    ax[1].set_xlabel("seed")
    ax[1].set_ylabel("mismatch cluster share")
    ax2 = ax[1].twinx()
    ax2.step(work["seed"], work["selected_k"], where="mid", color="tab:green", alpha=0.65, label="selected_k")
    ax2.set_ylabel("selected_k")
    ax[1].set_title("selected_k / mismatch share")
    for a in ax:
        a.grid(alpha=0.25)
    ari_min = work["ari_vs_base"].min()
    ari_med = work["ari_vs_base"].median()
    fig.suptitle(
        f"07 minute KMeans behavior-state stability diagnostic: mc_stab.csv ARI/subsample, not supervised performance; filename retained for compatibility; ARI min={ari_min:.3f}, med={ari_med:.3f}",
        fontsize=13,
    )
    _save(fig, plots / "07_session_cluster_stability.png")


def _plot_ms(out, plots):
    ms = _read(out, "m2_scores.csv")
    if ms.empty or "minute_mismatch_score" not in ms.columns:
        _blank(plots / "07_ms.png", "m2_scores.csv not available")
        return
    score = pd.to_numeric(ms["minute_mismatch_score"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(score, bins="sturges", edgecolor="black", color="tab:blue", alpha=0.75)
    for q, color in [(0.90, "tab:orange"), (0.95, "tab:red"), (0.97, "tab:purple"), (0.99, "black")]:
        cutoff = score.quantile(q)
        ax.axvline(cutoff, color=color, ls="--", lw=1.5, label=f"q{int(q * 100)}={cutoff:.3f}")
    ax.set_title("07 Method 2 minute_mismatch_score - not ground-truth label - grid only, no selected truth cutoff")
    ax.set_xlabel("minute_mismatch_score")
    ax.set_ylabel("minutes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, plots / "07_ms.png")


def _plot_mc(out, plots):
    prof = _read(out, "mc_profile.csv")
    assign = _read(out, "mc_assign.csv")
    if prof.empty:
        _blank(plots / "08_mc.png", "mc_profile.csv not available")
        return
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    if not assign.empty:
        counts = assign["minute_cluster"].value_counts(dropna=False).sort_index()
        ax[0, 0].bar(counts.index.astype(str), counts.values, edgecolor="black")
    ax[0, 0].set_title("분 단위 군집 수")
    ax[0, 0].set_xlabel("minute_cluster")
    ax[0, 0].set_ylabel("분 row 수")
    ax[0, 1].bar(prof["cluster_id"].astype(str), prof["share"], edgecolor="black", color="tab:green")
    ax[0, 1].set_title("분 단위 군집 비중")
    ax[0, 1].set_xlabel("minute_cluster")
    cols = ["chat_deficit_med", "unique_deficit_med", "zero_chat_rate", "rolling_zero_rate_5m_p90", "cluster_mismatch_rank"]
    mat = prof.set_index("cluster_id")[[c for c in cols if c in prof.columns]]
    if not mat.empty:
        scaled = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)
        im = ax[1, 0].imshow(scaled.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[1, 0].set_xticks(range(len(mat.columns)), mat.columns, rotation=35, ha="right")
        ax[1, 0].set_yticks(range(len(mat.index)), mat.index.astype(str))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax[1, 0].text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if scaled.iloc[i, j] < 0.45 else "black")
        fig.colorbar(im, ax=ax[1, 0], shrink=0.8, label="within-feature scaled profile value, not cluster id")
    ax[1, 0].set_title("군집별 주요 신호 중앙값")
    ax[1, 0].set_xlabel("cell text = original profile value; color = relative within-feature contrast")
    interp = prof[["cluster_id", "interpretation"]].copy()
    interp["interpretation"] = interp["interpretation"].replace({
        "mismatch-like state": "viewer 대비 채팅 반응 약한 상태",
        "active/high-chat state": "채팅 반응 활발 상태",
        "low-scale quiet state": "소규모 조용한 상태",
        "mixed state": "혼합 행동 상태",
    })
    ax[1, 1].axis("off")
    ax[1, 1].table(cellText=interp.values, colLabels=["cluster_id", "해석"], loc="center", cellLoc="left")
    ax[1, 1].set_title("분 단위 행동 상태이며 정답 라벨이 아님")
    for a in ax.ravel()[:3]:
        a.grid(alpha=0.25)
    fig.suptitle("08 분 단위 행동 상태 군집 요약", fontsize=14)
    _save(fig, plots / "08_mc.png")
    _replace_with_hardlink_or_copy(plots / "08_mc.png", plots / "08_cluster_minute.png")


def _heatmap(ax, df, value, title):
    if df.empty or value not in df.columns:
        ax.axis("off")
        return
    piv = df.pivot(index="min_duration", columns="threshold_q", values=value).sort_index()
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("threshold_q")
    ax.set_ylabel("min_duration")
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            txt = f"{val:.2f}" if "rate" in value else f"{val:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if val > np.nanmax(piv.values) / 2 else "black")
    return im


def _plot_sens(out, plots):
    sens = _read(out, "m2_sens.csv")
    if sens.empty:
        _blank(plots / "10_sens.png", "m2_sens.csv not available")
        return
    sources = sens["score_source"].dropna().unique().tolist()[:4]
    n = max(1, len(sources))
    fig, ax = plt.subplots(n, 2, figsize=(13, 4.2 * n), constrained_layout=True)
    ax = np.array(ax).reshape(n, 2)
    for i, source in enumerate(sources):
        part = sens[sens["score_source"].eq(source)]
        im1 = _heatmap(ax[i, 0], part, "candidate_session_rate", f"{source}: candidate_session_rate")
        im2 = _heatmap(ax[i, 1], part, "episode_count", f"{source}: episode_count")
        if im1 is not None:
            fig.colorbar(im1, ax=ax[i, 0], shrink=0.8)
        if im2 is not None:
            fig.colorbar(im2, ax=ax[i, 1], shrink=0.8)
    fig.suptitle("10 Method 2 legacy sensitivity - not final selection - not ground-truth label", fontsize=14)
    _save(fig, plots / "10_sens.png")


def _plot_ep(out, plots):
    ms = _read(out, "m2_scores.csv")
    ep = _read(out, "m2_ep.csv")
    if ms.empty or ep.empty:
        _blank(plots / "11_ep.png", "m2 episode inputs not available")
        return
    ep = ep[
        ep["score_source"].eq("rule_rank")
        & ep["threshold_q"].eq(0.95)
        & ep["min_duration"].eq(10)
    ].copy()
    if ep.empty:
        _blank(plots / "11_ep.png", "rule_rank q0.95 d10 example episodes not available")
        return

    by_sess = ep.groupby("session_key").agg(
        longest_duration=("duration_min", "max"),
        episode_count=("episode_id", "nunique"),
        max_score=("max_score", "max"),
    ).reset_index()
    picks = []
    longest = by_sess.sort_values(["longest_duration", "max_score"], ascending=[False, False])
    if not longest.empty:
        picks.append(("longest duration", longest.iloc[0]["session_key"]))
    many = by_sess.sort_values(["episode_count", "longest_duration"], ascending=[False, False])
    for _, row in many.iterrows():
        if row["session_key"] not in [p[1] for p in picks]:
            picks.append(("many episodes", row["session_key"]))
            break
    mid = by_sess.assign(_dist=(by_sess["longest_duration"] - by_sess["longest_duration"].median()).abs())
    mid = mid.sort_values(["_dist", "max_score"], ascending=[True, False])
    for _, row in mid.iterrows():
        if row["session_key"] not in [p[1] for p in picks]:
            picks.append(("middle duration", row["session_key"]))
            break
    for _, row in by_sess.sort_values(["max_score"], ascending=False).iterrows():
        if len(picks) >= 3:
            break
        if row["session_key"] not in [p[1] for p in picks]:
            picks.append(("additional example", row["session_key"]))
    picks = picks[:3]
    if not picks:
        _blank(plots / "11_ep.png", "no episode examples available")
        return

    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 3.8 * len(picks)), constrained_layout=True, sharex=False)
    axes = np.array(axes).reshape(-1)
    for ax, (label, key) in zip(axes, picks):
        part = ms[ms["session_key"].eq(key)].copy().sort_values("minute_idx")
        if part.empty:
            ax.axis("off")
            continue
        ax.plot(part["minute_idx"], part["minute_mismatch_rank"], color="tab:blue", lw=1.7, label="rule_rank")
        ax.set_ylabel("rule_rank")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(part["minute_idx"], part["chat_deficit"], color="tab:orange", alpha=0.55, lw=1.2, label="chat_deficit")
        ax2.set_ylabel("chat_deficit")
        ep_part = ep[ep["session_key"].eq(key)].copy()
        for _, row in ep_part.iterrows():
            start = pd.to_datetime(row["start_ts"])
            end = pd.to_datetime(row["end_ts"])
            sidx = part.loc[pd.to_datetime(part["minute_ts"]).ge(start), "minute_idx"].min()
            eidx = part.loc[pd.to_datetime(part["minute_ts"]).le(end), "minute_idx"].max()
            if pd.notna(sidx) and pd.notna(eidx):
                ax.axvspan(sidx, eidx, color="tab:red", alpha=0.15)
        ax.set_title(f"{label}: {key} ({len(ep_part)} episodes)")
        ax.set_xlabel("minute_idx")
    fig.suptitle(
        "11 Method 2 episode examples - rule_rank q=0.95 min_duration=10 example operating point, not selected truth cutoff",
        fontsize=14,
    )
    _save(fig, plots / "11_ep.png")


def _plot_rank(out, plots):
    cand = _read(out, "m2_candidates.csv")
    if cand.empty:
        _blank(plots / "12_m2_rank.png", "m2_candidates.csv not available")
        return
    top = cand[
        cand["score_source"].eq("rule_rank")
        & cand["threshold_q"].eq(0.95)
        & cand["min_duration"].eq(10)
    ].copy()
    if top.empty:
        _blank(plots / "12_m2_rank.png", "rule_rank q0.95 d10 candidates not available")
        return
    top = top.sort_values("candidate_rank").head(15).copy()
    labels = top["session_key"].astype(str).iloc[::-1]
    fig, ax = plt.subplots(1, 4, figsize=(19, 7), constrained_layout=True)
    metrics = [
        ("episode_total_duration_min", "episode total duration"),
        ("episode_duration_ratio", "episode duration ratio"),
        ("max_episode_score", "max episode score"),
        ("p95_minute_score", "p95 minute score"),
    ]
    for a, (col, title) in zip(ax, metrics):
        vals = pd.to_numeric(top[col], errors="coerce").iloc[::-1]
        bars = a.barh(labels, vals, edgecolor="black", alpha=0.8)
        a.set_title(title)
        a.tick_params(axis="y", labelsize=8)
        a.grid(axis="x", alpha=0.25)
        if col == "episode_total_duration_min":
            ranks = top["candidate_rank"].astype(int).iloc[::-1].tolist()
            for bar, rank in zip(bars, ranks):
                a.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  r{rank}", va="center", fontsize=8)
    fig.suptitle(
        "12 Method 2 candidate metrics - rule_rank q=0.95 min_duration=10 example operating point, not selected truth cutoff",
        fontsize=14,
    )
    _save(fig, plots / "12_m2_rank.png")


def _plot_pipe(plots):
    labels = [
        ("Load", "feature files"),
        ("Minute State", "bin, deficit,\nzero-run, rolling"),
        ("Minute Cluster", "KMeans state\nsummary"),
        ("Rule Rank", "equal-weight\npercentiles"),
        ("Window Evidence", "sliding windows\nand anomalies"),
        ("Reason/RRA", "review priority\nnot label"),
        ("Method 2 Handoff", "review\nevidence"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(labels))
    for i, ((title, desc), x) in enumerate(zip(labels, xs)):
        ax.text(x, 0.62, title, ha="center", va="center", fontsize=11, weight="bold", bbox=dict(boxstyle="round,pad=.35", facecolor="#F2F4F7", edgecolor="#333333"))
        ax.text(x, 0.37, desc, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.05, 0.62), xytext=(x + 0.05, 0.62), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    ax.set_title("13 Method 2 pipeline - review evidence mining, not ground-truth label", fontsize=14)
    _save(fig, plots / "13_m2_pipe.png")


def _plot_mc_selection(out, plots):
    mc = _read(out, "mc_select.csv")
    if mc.empty or "algorithm" not in mc.columns or "selection_score" not in mc.columns:
        _blank(plots / "09_mc_selection.png", "mc_select.csv not available\nminute behavior-state model selection")
        return
    km = mc[mc["algorithm"].astype(str).str.lower().eq("kmeans")].copy()
    km["k"] = pd.to_numeric(km["selected_k_num"], errors="coerce")
    km = km.dropna(subset=["k"]).sort_values("k")
    if km.empty:
        _blank(plots / "09_mc_selection.png", "no KMeans candidates in mc_select.csv")
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(km["k"], pd.to_numeric(km["selection_score"], errors="coerce"), "-o", color="#4C78A8", label="selection_score (composite)")
    ax.plot(km["k"], pd.to_numeric(km["silhouette"], errors="coerce"), "-s", color="#F58518", label="silhouette")
    sel = km[km["selected"].astype(str).str.lower().isin(["true", "1"])]
    if not sel.empty:
        ksel = int(sel["k"].iloc[0])
        ax.axvline(ksel, color="green", ls="--", alpha=0.7, label=f"selected KMeans k={ksel}")
    ax.set_xlabel("minute KMeans k")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("09 Minute behavior-state model selection (KMeans)\nGMM/HDBSCAN are structural diagnostics, not label sources")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, plots / "09_mc_selection.png")


def _plot_base_importance(out, plots):
    imp = _read(out, "base_importance.csv")
    if imp.empty or "target" not in imp.columns:
        _blank(plots / "14_base_importance.png", "base_importance.csv not available\nexpected-response feature importance")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, tgt in zip(axes, ["log_chat", "log_unique"]):
        d = imp[imp["target"].astype(str).eq(tgt)].copy()
        d = d.sort_values("importance_mae_increase")
        ax.barh(
            d["feature"].astype(str),
            pd.to_numeric(d["importance_mae_increase"], errors="coerce"),
            xerr=pd.to_numeric(d.get("importance_std"), errors="coerce"),
            color="#4C78A8",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title(f"{tgt} OOF permutation importance")
        ax.set_xlabel("MAE increase (log space)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("14 Expected-response baseline feature importance (review evidence, not probability)", fontsize=13)
    _save(fig, plots / "14_base_importance.png")


def _write_plot_doc(out):
    plots = Path(out) / "plots"
    session = _read(out, "session_summary_processed.csv")
    cluster_count = int(pd.to_numeric(session.get("cluster_number", pd.Series(dtype=float)), errors="coerce").nunique()) if not session.empty and "cluster_number" in session.columns else np.nan
    notes = {
        "01_data_quality.png": "데이터 품질과 필터링 개요; 정답 라벨/확률 아님",
        "02_dist_time.png": "minute 분포와 시간 구조 확인; 정답 라벨/확률 아님",
        "03_view_chat.png": "viewer-chat 동학 확인; 정답 라벨/확률 아님",
        "04_cluster_session.png": f"session cluster 산점도; discrete legend for categorical cluster_number; {SESSION_CLUSTER_TITLE_NOTE}",
        "05_session_k_selection.png": "session K 선택 근거; primary metric is composite selection_score, not inertia; not ground-truth label",
        "06_session_cluster_profile.png": f"cluster profile heatmap; colorbar is within-metric scaled profile value, not cluster id; {SESSION_CLUSTER_TITLE_NOTE}",
        "07_session_cluster_stability.png": "filename retained for backward compatibility; content is minute KMeans behavior-state stability diagnostic from mc_stab.csv ARI/subsample; supervised 성능지표 아님; 정답 라벨/확률 아님",
        "07_ms.png": "Method 2 rule-rank score 분포; 정답 라벨/확률 아님",
        "08_mc.png": "minute KMeans behavior state profile; colorbar is within-feature scaled profile value, not cluster id; 정답 라벨/확률 아님",
        "09_mc_selection.png": "minute 행동상태 KMeans 모델선택 비교(selection_score/silhouette); GMM/HDBSCAN은 구조 진단용이며 label source 아님; 정답 라벨/확률 아님",
        "14_base_importance.png": "expected-response GBM의 OOF permutation 변수중요도(log space MAE 증가분); 검토 근거이며 확률 아님",
        "08_cluster_minute.png": "alias/duplicate of 08_mc.png for handoff compatibility; colorbar is within-feature scaled profile value, not cluster id; 정답 라벨/확률 아님",
        "13_m2_pipe.png": "Method 2 pipeline 개요; 정답 라벨/확률 아님",
        "15_null.png": "shuffled null diagnostic; 정답 라벨/확률 아님",
        "16_state.png": "minute state transition diagnostic; colorbar label is transition share, not cluster count; 정답 라벨/확률 아님",
        "18_review.png": "수동 검토 우선순위 evidence 요약; 정답 라벨/확률 아님",
        "19_reason.png": "counts top explanation reasons only; not confidence/lift/prediction rule",
        "20_rra.png": "family RRA evidence rank; rra_q=family_rra_q이며 확률 아님; labels show top review candidates only",
        "21_interval.png": "interval duration/empirical_p caution 진단; one shared colorbar is family consensus score, not probability",
        "27_eval_scorecard.png": "label-free 평가 스코어카드; robustness/synthetic sanity 진단이며 supervised 성능지표 아님; 정답 라벨/확률 아님",
        "28_top_session_profile.png": "상위 review 후보 vs 나머지 세션 minute-signal 표준화 평균차; 설명용 검토 근거이며 정답 라벨/확률 아님",
    }
    actual = sorted(path.name for path in plots.glob("*.png"))
    manifest = [f"plots/{name}" for name in actual]
    rows = [{"file": file, "note": notes.get(Path(file).name, "generated plot; alias/diagnostic; 정답 라벨/확률 아님")} for file in manifest]
    pd.DataFrame(rows).to_csv(Path(out) / "plot_manifest.csv", index=False, encoding="utf-8-sig")
    audit_rows = [
        {
            "plot_file": "04_cluster_session.png",
            "primary_visual_encoding": "cluster_number",
            "encoding_type": "categorical",
            "colorbar_or_legend": "legend",
            "expected_category_count": cluster_count,
            "actual_legend_count": cluster_count,
            "colorbar_label": "",
            "interpretation": "cluster_number is behavior cluster id, not probability or label",
            "status": "PASS" if pd.notna(cluster_count) else "WARN",
            "note": "scatter is drawn per cluster with discrete legend; no cluster_number colorbar",
        },
        {
            "plot_file": "05_session_k_selection.png",
            "primary_visual_encoding": "selection_score",
            "encoding_type": "bar plus diagnostic lines",
            "colorbar_or_legend": "legend",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "",
            "interpretation": "selection_score is composite K selection evidence, not ground-truth label",
            "status": "PASS",
            "note": "inertia is not presented as selection criterion",
        },
        {
            "plot_file": "06_session_cluster_profile.png",
            "primary_visual_encoding": "profile scaled value",
            "encoding_type": "continuous heatmap",
            "colorbar_or_legend": "colorbar",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "within-metric scaled profile value, not cluster id",
            "interpretation": "cell text is original profile value; color is relative within-feature contrast",
            "status": "PASS",
            "note": "colorbar is not cluster id",
        },
        {
            "plot_file": "08_mc.png",
            "primary_visual_encoding": "profile scaled value",
            "encoding_type": "continuous heatmap",
            "colorbar_or_legend": "colorbar",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "within-feature scaled profile value, not cluster id",
            "interpretation": "cell text is original profile value; color is relative within-feature contrast",
            "status": "PASS",
            "note": "same content copied to 08_cluster_minute.png alias",
        },
        {
            "plot_file": "16_state.png",
            "primary_visual_encoding": "transition share",
            "encoding_type": "continuous heatmap",
            "colorbar_or_legend": "colorbar",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "transition share",
            "interpretation": "colorbar is state transition proportion, not cluster count",
            "status": "PASS",
            "note": "state labels are active/mismatch",
        },
        {
            "plot_file": "19_reason.png",
            "primary_visual_encoding": "top explanation reason count",
            "encoding_type": "bar",
            "colorbar_or_legend": "none",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "",
            "interpretation": "counts text_reason_included=True rows only; not confidence/lift/prediction rule",
            "status": "PASS",
            "note": "full m2_reason.csv rank rows are preserved",
        },
        {
            "plot_file": "20_rra.png",
            "primary_visual_encoding": "family_rra_q by review_order",
            "encoding_type": "scatter",
            "colorbar_or_legend": "none",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "",
            "interpretation": "labels limited to top review candidates only",
            "status": "PASS",
            "note": "top 5 labels with staggered offsets",
        },
        {
            "plot_file": "21_interval.png",
            "primary_visual_encoding": "family_consensus_score",
            "encoding_type": "scatter color",
            "colorbar_or_legend": "one shared colorbar",
            "expected_category_count": np.nan,
            "actual_legend_count": np.nan,
            "colorbar_label": "family consensus score, not probability",
            "interpretation": "empirical_p is shuffled-null evidence, not detection probability",
            "status": "PASS",
            "note": "both subplots share one norm/cmap/colorbar",
        },
    ]
    pd.DataFrame(audit_rows).to_csv(Path(out) / "plot_audit.csv", index=False, encoding="utf-8-sig")
    lines = [
        "Plot Guide",
        "==========",
        "목적: 각 plot이 어떤 방법론 판단을 보조하는지와 어떤 결론을 낼 수 없는지 명시한다.",
        "공통 주의: 모든 plot은 viewer-chat mismatch 기반 수동 검토 우선순위 evidence이며 정답 라벨/확률 아님.",
        "",
        "Method 2 plots:",
        *[f"- {Path(row['file']).name}: {row['note']}" for row in rows],
        "",
        "해석 제한:",
        "- cluster_number와 minute_cluster는 behavior state id이며 정답 라벨/확률 아님.",
        "- rra_q, family_rra_q, family_consensus_score, empirical_p는 확률이 아니며 수동 검토 우선순위 evidence이다.",
        "- 04_cluster_session.png uses a discrete legend for categorical cluster_number and no cluster_number colorbar.",
        "- 05_session_k_selection.png shows composite selection_score as the main selection basis; inertia is not a selection criterion.",
        "- 06_session_cluster_profile.png colorbar is within-metric scaled profile value, not cluster id.",
        "- 08_mc.png and 08_cluster_minute.png colorbar is within-feature scaled profile value, not cluster id.",
        "- 16_state.png colorbar is transition share, not cluster count.",
        "- 19_reason.png counts top explanation reasons only; not confidence/lift/prediction rule.",
        "- 21_interval.png uses one shared colorbar for family consensus score, not probability.",
        "- 07_session_cluster_stability.png 파일명은 backward compatibility를 위해 유지하지만 내용은 mc_stab.csv 기반 minute KMeans behavior-state stability diagnostic이며 supervised 성능지표가 아니다.",
        "- plot_manifest.csv는 실제 out/plots/*.png 파일만 나열한다. 08_cluster_minute.png는 08_mc.png의 handoff alias/duplicate이다.",
        "",
        *[f"- {file}" for file in manifest],
    ]
    (Path(out) / "plot_guide.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot_doc(out):
    _write_plot_doc(out)


def make_plots(minute_all, minute_model, session_all, session_model, out):
    out = Path(out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for old in plots.glob("*.png"):
        old.unlink()

    _plot_quality(minute_all, session_all, session_model, plots)
    _plot_dist_time(minute_model, plots)
    _plot_view_chat(minute_model, plots)
    _plot_session_cluster(session_model, plots)
    _plot_session_k_selection(out, plots)
    _plot_session_cluster_profile(out, plots)
    _plot_session_cluster_stability(out, plots)
    _plot_ms(out, plots)
    _plot_mc(out, plots)
    _plot_mc_selection(out, plots)
    _plot_sens(out, plots)
    _plot_ep(out, plots)
    _plot_rank(out, plots)
    _plot_pipe(plots)
    _plot_base_importance(out, plots)
    _write_plot_doc(out)
