from datetime import datetime
from sqlalchemy import JSON, Date, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class CrawlRun(Base):
    __tablename__ = "crawl_runs"

    run_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    window_label: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

class TargetChannel(Base):
    __tablename__ = "target_channels"

    channel_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False) # CHZZK Hash ID
    user_nick: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category_id: Mapped[str | None] = mapped_column(String(50), nullable=True)     # GAME etc.
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

class LiveSnapshot(Base):
    __tablename__ = "live_snapshots"

    snapshot_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    broad_no: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    broad_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    category_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    viewer_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_password: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

class ChatMessageRaw(Base):
    __tablename__ = "chat_messages_raw"

    chat_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    broad_no: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    user_nick: Mapped[str | None] = mapped_column(String(255), nullable=True)
    message_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_clean: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

class MinuteFeature(Base):
    __tablename__ = "minute_features"
    __table_args__ = (UniqueConstraint("broad_no", "minute_ts", name="uq_broad_minute"),)

    feature_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    broad_no: Mapped[str] = mapped_column(String(50), nullable=False)
    minute_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    category_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    viewer_count_last: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chat_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_chatters: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_msg_len: Mapped[float | None] = mapped_column(Float, nullable=True)
    repeat_msg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_chatter_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    chat_per_viewer: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_viewer_1m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    delta_chat_1m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
