from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_m2() -> pd.DataFrame:
    m2 = pd.read_csv(ROOT / "m2_review.csv").sort_values("review_order", kind="stable").reset_index(drop=True)
    m2["minute_rank"] = range(1, len(m2) + 1)
    return m2


def load_session_candidates() -> pd.DataFrame:
    base = load_m2()[
        [
            "session_key",
            "run_id",
            "broad_no",
            "minute_rank",
            "family_consensus_score",
            "top_interval_duration",
            "reason_set",
            "review_note",
        ]
    ].copy()

    specs = [
        ("03", ROOT / "outputs" / "03_review_candidates.csv", "review_group", "low_priority"),
        ("07a", ROOT / "outputs" / "07a_cluster_aware_review_candidates.csv", "cluster_aware_review_group", "low_priority"),
        ("07b", ROOT / "outputs" / "07b_advanced_binary_review_candidates.csv", "advanced_binary_review_group", "low_priority"),
        ("07c", ROOT / "outputs" / "07c_hidden_candidate_review_candidates.csv", "discovery_group", "normal_low_priority"),
        ("07d", ROOT / "outputs" / "07d_pu_hidden_review_candidates.csv", "pu_discovery_group", "pu_likely_normal"),
    ]

    for name, path, group_col, low_value in specs:
        df = pd.read_csv(path).copy()
        df[f"rank_{name}"] = range(1, len(df) + 1)
        df[f"flag_{name}"] = df[group_col].astype("string").ne(low_value)
        rename_cols = {
            group_col: f"group_{name}",
            "viewer_med": "viewer_med",
            "viewer_max": "viewer_max",
            "chat_mean": "chat_mean",
            "unique_mean": "unique_mean",
            "zero_rate": "zero_rate",
            "zrun_max": "zrun_max",
            "gap_med": "gap_med",
            "gap_max": "gap_max",
            "n": "n",
            "start": "start",
            "end": "end",
        }
        keep = ["session_key", f"rank_{name}", f"flag_{name}", group_col]
        if name == "03":
            keep.extend([c for c in rename_cols if c in df.columns])
        merged = df[keep].rename(columns=rename_cols)
        base = base.merge(merged, on="session_key", how="left")

    base = base.loc[:, ~pd.Index(base.columns).duplicated()].copy()
    flag_cols = [c for c in base.columns if c.startswith("flag_")]
    rank_cols = [c for c in base.columns if c.startswith("rank_")]
    base["flag_count"] = base[flag_cols].sum(axis=1)
    base["avg_session_rank"] = base[rank_cols].mean(axis=1)
    base["rank_gap_avg"] = base["minute_rank"] - base["avg_session_rank"]
    return base


def minute_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session_key, g in minute_df.groupby("session_key", sort=False):
        g = g.sort_values("minute_ts").copy()
        zero = g["chat_count"].eq(0)
        block = (zero.ne(zero.shift())).cumsum()
        longest_zero_run = int(zero.groupby(block).sum().max())
        rows.append(
            {
                "session_key": session_key,
                "minute_rows": int(len(g)),
                "chat_positive_minutes": int(g["chat_count"].gt(0).sum()),
                "unique_positive_minutes": int(g["unique_chatters"].gt(0).sum()),
                "chat_positive_ratio": float(g["chat_count"].gt(0).mean()),
                "unique_positive_ratio": float(g["unique_chatters"].gt(0).mean()),
                "max_chat_count": float(g["chat_count"].max()),
                "max_unique_chatters": float(g["unique_chatters"].max()),
                "viewer_range": float(g["viewer_count_last"].max() - g["viewer_count_last"].min()),
                "longest_zero_run_minutes": longest_zero_run,
            }
        )
    return pd.DataFrame(rows)


def pick_cases(candidate_df: pd.DataFrame) -> pd.DataFrame:
    pool = candidate_df[
        (candidate_df["flag_count"] >= 3)
        & (candidate_df["minute_rank"] >= 800)
        & (candidate_df["chat_positive_ratio"] >= 0.70)
        & (candidate_df["longest_zero_run_minutes"] <= 5)
    ].copy()

    pool["all_five_flagged"] = pool["flag_count"].eq(5)
    pool = pool.sort_values(
        ["all_five_flagged", "flag_count", "minute_rank", "avg_session_rank"],
        ascending=[False, False, False, True],
        kind="stable",
    )

    preferred = ["106_18833520", "114_18866678", "133_18973233", "138_18992515", "146_19027695", "165_19117018"]
    chosen = pool[pool["session_key"].isin(preferred)].copy()
    chosen["preferred_order"] = chosen["session_key"].map({key: idx for idx, key in enumerate(preferred)})
    chosen = chosen.sort_values("preferred_order", kind="stable").head(4).drop(columns="preferred_order")
    return chosen


def save_case_plot(case_row: pd.Series, minute_df: pd.DataFrame) -> None:
    session_key = case_row["session_key"]
    g = minute_df.loc[minute_df["session_key"].eq(session_key)].sort_values("minute_ts").copy()
    g["minute_index"] = range(len(g))

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(g["minute_index"], g["viewer_count_last"], color="#1f77b4", linewidth=2, label="viewer_count_last")
    ax1.set_xlabel("Minute Index")
    ax1.set_ylabel("Viewer Count", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.bar(g["minute_index"], g["chat_count"], width=0.8, alpha=0.35, color="#ff7f0e", label="chat_count")
    ax2.plot(g["minute_index"], g["unique_chatters"], color="#2ca02c", linewidth=1.7, marker="o", markersize=3, label="unique_chatters")
    ax2.set_ylabel("Chat / Unique", color="#333333")

    title = (
        f"{session_key} | minute rank={int(case_row['minute_rank'])} | "
        f"flag_count={int(case_row['flag_count'])} | avg session rank={case_row['avg_session_rank']:.1f}"
    )
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{session_key}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    minute = pd.read_csv(ROOT / "csv" / "minute_model.csv")
    minute["minute_ts"] = pd.to_datetime(minute["minute_ts"], errors="coerce")

    candidate_df = load_session_candidates().merge(minute_features(minute), on="session_key", how="left")
    candidate_pool = candidate_df[
        (candidate_df["flag_count"] >= 3)
        & (candidate_df["minute_rank"] >= 800)
    ].sort_values(["flag_count", "minute_rank", "avg_session_rank"], ascending=[False, False, True])

    chosen = pick_cases(candidate_df).copy()
    takeaway_map = {
        "106_18833520": "13분짜리 짧은 세션인데 채팅이 13분 중 10분에서 발생했다. session summary는 짧은 길이와 viewer-chat gap을 크게 반영했지만 minute 흐름만 보면 지속적 무응답 세션으로 보기는 어렵다.",
        "114_18866678": "67분 세션이지만 채팅이 53분에서 관측되고 longest zero-run도 4분에 그친다. minute 기준에서는 일부 구간 이상은 있어도 전체 세션을 viewbot으로 단정할 정도의 일관된 침묵 패턴은 약하다.",
        "133_18973233": "22분의 짧은 세션이며 채팅이 17분에서 발생했다. longest zero-run이 2분뿐이라 session summary가 특정 구간 mismatch를 과대대표했을 가능성을 보여준다.",
        "138_18992515": "64분 동안 채팅이 51분에서 이어지고 longest zero-run이 2분뿐이다. session-based hidden candidate로는 잡혔지만 minute evidence만 보면 방송 전체가 비정상적으로 침묵했다고 보기는 어렵다.",
        "146_19027695": "session 모델에서는 high-confidence 쪽으로 밀렸지만 실제로는 35분 중 23분에서 채팅이 존재하고 최대 채팅도 11회까지 나온다. 의심 구간이 있어도 세션 전체를 비정상으로 보기에는 애매한 사례다.",
        "165_19117018": "minute rank가 1099로 더 뒤쪽인데도 일부 session 모델에서 계속 후보로 남는다. 다만 채팅이 45분 중 35분에서 관측되어, session summary가 세션 전체의 정상적 상호작용을 충분히 반영하지 못했을 가능성이 있다.",
    }
    chosen["presentation_takeaway"] = chosen["session_key"].map(takeaway_map)

    candidate_pool.to_csv(OUT_DIR / "session_vs_minute_candidate_pool.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT_DIR / "presentation_case_studies.csv", index=False, encoding="utf-8-sig")

    for _, row in chosen.iterrows():
        save_case_plot(row, minute)

    print("Saved candidate pool:", OUT_DIR / "session_vs_minute_candidate_pool.csv")
    print("Saved chosen cases:", OUT_DIR / "presentation_case_studies.csv")
    print("Saved plots to:", PLOT_DIR)
    print()
    print(
        chosen[
            [
                "session_key",
                "minute_rank",
                "flag_count",
                "avg_session_rank",
                "group_03",
                "group_07a",
                "group_07b",
                "group_07c",
                "group_07d",
                "chat_positive_minutes",
                "minute_rows",
                "longest_zero_run_minutes",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
