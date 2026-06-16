from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SESSION_KEY = "160_19085934"
OUT = Path("out")
PLOT_DIR = OUT / "plots"


def _read_csv(name):
    return pd.read_csv(OUT / name, encoding="utf-8-sig")


def _num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    scores = _read_csv("m2_scores.csv")
    scan = _read_csv("m2_scan.csv")
    review = _read_csv("m2_review.csv")
    base = _read_csv("base_pred.csv") if (OUT / "base_pred.csv").exists() else pd.DataFrame()

    sess = scores.loc[scores["session_key"].astype(str).eq(SESSION_KEY)].copy()
    if sess.empty:
        raise ValueError(f"session not found in m2_scores.csv: {SESSION_KEY}")
    sess["minute_ts"] = pd.to_datetime(sess["minute_ts"], errors="coerce")
    sess = sess.sort_values("minute_ts").reset_index(drop=True)

    if not base.empty:
        base = base.loc[base["session_key"].astype(str).eq(SESSION_KEY)].copy()
        base["minute_ts"] = pd.to_datetime(base["minute_ts"], errors="coerce")
        keep = [
            "session_key",
            "minute_ts",
            "model_chat_deficit",
            "model_unique_deficit",
            "baseline_agree_chat",
            "baseline_agree_unique",
        ]
        sess = sess.merge(base[[c for c in keep if c in base.columns]], on=["session_key", "minute_ts"], how="left")

    scan_row = scan.loc[scan["session_key"].astype(str).eq(SESSION_KEY)].head(1)
    review_row = review.loc[review["session_key"].astype(str).eq(SESSION_KEY)].head(1)
    if scan_row.empty:
        raise ValueError(f"session not found in m2_scan.csv: {SESSION_KEY}")

    start = pd.to_datetime(scan_row.iloc[0].get("top_interval_start_ts"), errors="coerce")
    end = pd.to_datetime(scan_row.iloc[0].get("top_interval_end_ts"), errors="coerce")
    in_top_interval = sess["minute_ts"].between(start, end, inclusive="both")

    score = _num(sess, "minute_mismatch_score")
    score_fill = score.fillna(score.median()).fillna(0)
    norm = plt.Normalize(vmin=float(score_fill.min()), vmax=float(score_fill.max()))
    cmap = plt.get_cmap("inferno")
    colors = cmap(norm(score_fill.to_numpy()))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.35, 0.6]},
        constrained_layout=True,
    )

    title_bits = [
        f"{SESSION_KEY} suspicious-minute distribution",
        f"scan_rank={scan_row.iloc[0].get('scan_rank')}",
        f"scan_z={float(scan_row.iloc[0].get('observed_scan_z')):.2f}",
        f"top_interval={start:%Y-%m-%d %H:%M} to {end:%H:%M}",
    ]
    if not review_row.empty:
        title_bits.append(f"review_order={review_row.iloc[0].get('review_order')}")
    fig.suptitle(" | ".join(title_bits), fontsize=13, fontweight="bold")

    for ax in axes:
        ax.axvspan(start, end, color="#f3a0a0", alpha=0.22, lw=0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = sess["minute_ts"]
    axes[0].plot(x, _num(sess, "viewer_count_last"), color="#205493", linewidth=1.8, label="viewer")
    axes[0].set_ylabel("viewer")
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].plot(x, _num(sess, "chat_count"), color="#1b7f3a", linewidth=1.6, label="chat")
    axes[1].plot(x, _num(sess, "unique_chatters"), color="#8561c5", linewidth=1.3, label="unique")
    zero_mask = _num(sess, "chat_count").fillna(0).eq(0)
    if zero_mask.any():
        axes[1].scatter(x.loc[zero_mask], np.zeros(int(zero_mask.sum())), color="#c62828", s=18, label="zero chat", zorder=4)
    axes[1].set_ylabel("chat")
    axes[1].legend(loc="upper left", frameon=False, ncol=3)

    bar_width = 0.00072
    axes[2].bar(x, score_fill, width=bar_width, color=colors, edgecolor="none")
    axes[2].plot(x, score_fill.rolling(5, min_periods=1).mean(), color="#2f2f2f", linewidth=1.6, label="5-min mean")
    axes[2].scatter(x.loc[in_top_interval], score_fill.loc[in_top_interval], s=22, facecolors="none", edgecolors="#005f73", linewidths=1.0, label="top scan interval")
    axes[2].set_ylabel("mismatch score")
    axes[2].legend(loc="upper left", frameon=False)

    if "model_chat_deficit" in sess.columns:
        model_def = _num(sess, "model_chat_deficit")
        axes[3].bar(x, model_def.fillna(0), width=bar_width, color="#577590", edgecolor="none", alpha=0.8, label="model chat deficit")
        axes[3].axhline(0, color="#333333", linewidth=0.8)
        axes[3].set_ylabel("model def.")
    else:
        axes[3].bar(x, _num(sess, "rolling_zero_rate_5m").fillna(0), width=bar_width, color="#577590", edgecolor="none", alpha=0.8, label="rolling zero rate")
        axes[3].set_ylabel("zero rate")
    axes[3].legend(loc="upper left", frameon=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes[2], orientation="vertical", fraction=0.025, pad=0.01)
    cbar.set_label("minute_mismatch_score")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    axes[-1].set_xlabel("minute timestamp")

    out_png = PLOT_DIR / f"suspect_minute_distribution_{SESSION_KEY}.png"
    fig.savefig(out_png, dpi=180, facecolor="white")
    plt.close(fig)

    summary_cols = [
        "session_key",
        "minute_ts",
        "minute_idx",
        "viewer_count_last",
        "chat_count",
        "unique_chatters",
        "minute_mismatch_score",
        "minute_mismatch_rank",
        "chat_deficit",
        "unique_deficit",
        "rolling_chat_deficit_5m",
        "zero_run_len",
        "rolling_zero_rate_5m",
        "model_chat_deficit",
        "model_unique_deficit",
        "dominant_reason",
    ]
    summary = sess[[c for c in summary_cols if c in sess.columns]].copy()
    summary["in_top_scan_interval"] = in_top_interval.to_numpy()
    summary.to_csv(OUT / f"suspect_minute_distribution_{SESSION_KEY}.csv", index=False, encoding="utf-8-sig")

    print(out_png)
    print(OUT / f"suspect_minute_distribution_{SESSION_KEY}.csv")
    print(f"minutes={len(sess)} top_interval_minutes={int(in_top_interval.sum())}")


if __name__ == "__main__":
    main()
