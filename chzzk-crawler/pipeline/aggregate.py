import logging
import pandas as pd
import numpy as np
from sqlalchemy import text

from core.db import engine, session_scope
from core.models import MinuteFeature

logger = logging.getLogger(__name__)

def build_minute_features_for_run(run_id: int) -> int:
    chat_df = _load_chat_data(run_id)
    snap_df = _load_snapshot_data(run_id)

    if chat_df.empty and snap_df.empty:
        logger.warning(f"No data found for run_id={run_id}")
        return 0

    minute_df = _aggregate_to_minute(chat_df, snap_df, run_id)
    if minute_df is None or minute_df.empty:
        logger.warning(f"Minute aggregation is empty for run_id={run_id}")
        return 0

    rows = minute_df.to_dict(orient="records")
    inserted = _upsert_minute_features(rows)
    logger.info(f"Built {inserted} minute feature rows for run_id={run_id}")
    return inserted

def _load_chat_data(run_id: int) -> pd.DataFrame:
    query = text("""
        SELECT run_id, event_ts, broad_no, user_id, user_nick, message_raw, message_clean, message_hash
        FROM chat_messages_raw
        WHERE run_id = :run_id
        ORDER BY broad_no, event_ts
    """)
    return pd.read_sql(query, engine, params={"run_id": run_id})

def _load_snapshot_data(run_id: int) -> pd.DataFrame:
    query = text("""
        SELECT run_id, snapshot_ts, broad_no, user_id, category_id, viewer_count
        FROM live_snapshots
        WHERE run_id = :run_id
        ORDER BY broad_no, snapshot_ts
    """)
    return pd.read_sql(query, engine, params={"run_id": run_id})

def _aggregate_to_minute(chat_df: pd.DataFrame, snap_df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    chat_min = pd.DataFrame()
    if not chat_df.empty:
        chat_df["event_ts"] = pd.to_datetime(chat_df["event_ts"])
        chat_df["minute_ts"] = chat_df["event_ts"].dt.floor("min")
        chat_df["message_len"] = chat_df["message_clean"].fillna("").str.len()
        chat_df = chat_df.drop_duplicates(subset=["message_hash"], keep="first")

        def _repeat_ratio(group: pd.Series) -> float:
            counts = group.fillna("__NULL__").value_counts(dropna=False)
            repeated = counts[counts > 1].sum()
            total = counts.sum()
            return float(repeated / total) if total else 0.0

        chat_min = (
            chat_df.groupby(["broad_no", "minute_ts"], as_index=False)
            .agg(
                chat_count=("message_raw", "count"),
                unique_chatters=("user_id", pd.Series.nunique),
                avg_msg_len=("message_len", "mean"),
            )
        )

        repeat_rows = []
        for (broad_no, minute_ts), group in chat_df.groupby(["broad_no", "minute_ts"]):
            unique_known_chatters = group["user_id"].dropna().drop_duplicates().tolist()
            first_time_flags = []
            for uid in unique_known_chatters:
                seen_before = chat_df[
                    (chat_df["broad_no"] == broad_no)
                    & (chat_df["user_id"] == uid)
                    & (chat_df["event_ts"] < group["event_ts"].min())
                ]
                first_time_flags.append(1 if seen_before.empty else 0)
            
            new_chatter_ratio = float(np.mean(first_time_flags)) if first_time_flags else 0.0
            repeat_msg_ratio = _repeat_ratio(group["message_clean"])
            
            repeat_rows.append({
                "broad_no": broad_no,
                "minute_ts": minute_ts,
                "repeat_msg_ratio": repeat_msg_ratio,
                "new_chatter_ratio": new_chatter_ratio,
            })
            
        if repeat_rows:
            extra_df = pd.DataFrame(repeat_rows)
            chat_min = chat_min.merge(extra_df, on=["broad_no", "minute_ts"], how="left")
            chat_min["delta_chat_1m"] = chat_min.groupby("broad_no")["chat_count"].diff()
            
    if chat_min.empty:
        chat_min = pd.DataFrame(columns=[
            "broad_no", "minute_ts", "chat_count", "unique_chatters",
            "avg_msg_len", "repeat_msg_ratio", "new_chatter_ratio", "delta_chat_1m"
        ])

    snap_min = pd.DataFrame()
    if not snap_df.empty:
        snap_df["snapshot_ts"] = pd.to_datetime(snap_df["snapshot_ts"])
        snap_df["minute_ts"] = snap_df["snapshot_ts"].dt.floor("min")
        snap_df = snap_df.sort_values(["broad_no", "snapshot_ts"])
        snap_min = (
            snap_df.groupby(["broad_no", "minute_ts"], as_index=False)
            .agg(
                viewer_count_last=("viewer_count", "last"),
                user_id=("user_id", "last"),
                category_id=("category_id", "last"),
            )
        )
        snap_min["delta_viewer_1m"] = snap_min.groupby("broad_no")["viewer_count_last"].diff()
        
    if snap_min.empty:
        snap_min = pd.DataFrame(
            columns=["broad_no", "minute_ts", "viewer_count_last", "user_id", "category_id", "delta_viewer_1m"]
        )

    if chat_min.empty and snap_min.empty:
        return pd.DataFrame()

    minute_df = pd.merge(snap_min, chat_min, on=["broad_no", "minute_ts"], how="outer")
    minute_df["run_id"] = run_id
    minute_df["chat_count"] = minute_df["chat_count"].fillna(0).astype(int)
    minute_df["unique_chatters"] = minute_df["unique_chatters"].fillna(0).astype(int)
    minute_df["repeat_msg_ratio"] = minute_df["repeat_msg_ratio"].fillna(0.0)
    minute_df["new_chatter_ratio"] = minute_df["new_chatter_ratio"].fillna(0.0)

    minute_df = minute_df.sort_values(["broad_no", "minute_ts"])
    minute_df["viewer_count_last"] = minute_df.groupby("broad_no")["viewer_count_last"].ffill(limit=1)
    minute_df["delta_viewer_1m"] = minute_df.groupby("broad_no")["viewer_count_last"].diff()
    minute_df["delta_chat_1m"] = minute_df.groupby("broad_no")["chat_count"].diff()

    # Avoid divide by zero
    viewers = minute_df["viewer_count_last"].fillna(0)
    minute_df["chat_per_viewer"] = np.where(
        viewers > 0,
        minute_df["chat_count"] / viewers,
        0.0  # Used 0.0 instead of np.nan to avoid SQLAlchemy crash
    )

    minute_df["minute_ts"] = minute_df["minute_ts"].dt.to_pydatetime()

    # FIX: Robustly fill user_id and category_id. 
    # Try forward-backward fill
    if "user_id" not in minute_df.columns:
        minute_df["user_id"] = None
    if "category_id" not in minute_df.columns:
        minute_df["category_id"] = None

    minute_df["user_id"] = minute_df.groupby("broad_no")["user_id"].transform(lambda x: x.ffill().bfill())
    minute_df["category_id"] = minute_df.groupby("broad_no")["category_id"].transform(lambda x: x.ffill().bfill())

    # If it's STILL null, fallback so the DB constraint NOT NULL doesn't fail
    minute_df["user_id"] = minute_df["user_id"].fillna("UNKNOWN_USER_ID")
    minute_df["category_id"] = minute_df["category_id"].fillna("UNKNOWN_CATEGORY")
    minute_df["user_id"] = minute_df["user_id"].replace("", "UNKNOWN_USER_ID")
    minute_df["category_id"] = minute_df["category_id"].replace("", "UNKNOWN_CATEGORY")

    # Replace np.nan with None for safe SQLAlchemy writes
    minute_df = minute_df.replace({np.nan: None})

    return minute_df[
        [
            "broad_no", "minute_ts", "user_id", "category_id", "viewer_count_last",
            "chat_count", "unique_chatters", "avg_msg_len", "repeat_msg_ratio",
            "new_chatter_ratio", "chat_per_viewer", "delta_viewer_1m", "delta_chat_1m", "run_id"
        ]
    ]

def _upsert_minute_features(rows: list[dict]) -> int:
    if not rows:
        return 0

    cleaned_rows = []
    for row in rows:
        # Extra safety catch for NaN
        cleaned = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        
        # Schema forces minute_ts, broad_no, user_id
        if not cleaned.get("broad_no") or cleaned.get("minute_ts") is None:
            continue
            
        cleaned_rows.append(cleaned)

    if not cleaned_rows:
        return 0

    with session_scope() as session:
        for row in cleaned_rows:
            existing = session.query(MinuteFeature).filter(
                MinuteFeature.broad_no == row["broad_no"],
                MinuteFeature.minute_ts == row["minute_ts"],
            ).one_or_none()

            if existing:
                for key, value in row.items():
                    setattr(existing, key, value)
            else:
                session.add(MinuteFeature(**row))

    return len(cleaned_rows)
