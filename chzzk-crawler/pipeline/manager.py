import logging
import time
from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy import insert, select

from configs.settings import get_settings
from core.db import session_scope
from core.models import CrawlRun, LiveSnapshot, TargetChannel
from collectors.base import BufferedChatSink
from collectors.chat_wss import WssChatCollector
from collectors.live_discovery import ChzzkLiveAPIClient, LiveBroadcast
from pipeline.aggregate import build_minute_features_for_run

logger = logging.getLogger(__name__)

class CrawlManager:
    def __init__(self, window_label: str, duration_seconds: int):
        self.window_label = window_label
        self.duration_seconds = duration_seconds
        self.settings = get_settings()
        self.live_api = ChzzkLiveAPIClient()
        self.chat_sink: BufferedChatSink = None
        self.chat_collectors: Dict[str, WssChatCollector] = {}

    def run(self) -> None:
        run_id = self._create_run()
        self.chat_sink = BufferedChatSink(run_id=run_id)
        self.chat_sink.start()
        logger.info(f"Started run_id={run_id} window={self.window_label} for {self.duration_seconds}s")

        started_at = datetime.now()
        next_snapshot_at = started_at

        try:
            target_user_ids = self._load_active_target_user_ids()
            if not target_user_ids:
                logger.warning("No active target channels found in Database! Please run setup first.")

            # Initial discovery
            initial_streams = self.live_api.discover_target_lives(target_user_ids)
            self._sync_chat_collectors(initial_streams)

            next_export_at = started_at + timedelta(seconds=self.settings.export_interval_seconds)

            while (datetime.now() - started_at).total_seconds() < self.duration_seconds:
                now = datetime.now()
                if now >= next_snapshot_at:
                    streams = self.live_api.discover_target_lives(target_user_ids)
                    self._save_live_snapshots(run_id, streams)
                    self._sync_chat_collectors(streams)
                    next_snapshot_at = now + timedelta(seconds=self.settings.snapshot_interval_seconds)

                if now >= next_export_at:
                    logger.info("Performing periodic real-time data aggregation and export...")
                    try:
                        # 1. Aggregate current minute features up to now
                        feature_rows = build_minute_features_for_run(run_id)
                        logger.info(f"Aggregated {feature_rows} minute feature rows for run_id={run_id}")
                        
                        # 2. Export accumulated data to disk
                        self._export_current_data(run_id)
                    except Exception as e:
                        logger.exception(f"Failed to perform periodic aggregation/export for run_id={run_id}: {e}")
                    
                    next_export_at = now + timedelta(seconds=self.settings.export_interval_seconds)

                time.sleep(1)

            self._shutdown_collectors()
            logger.info("Aggregating final minute features...")
            feature_rows = build_minute_features_for_run(run_id)
            self._export_current_data(run_id)
            self._finish_run(run_id, status="success", notes=f"minute_features={feature_rows}")
            logger.info(f"Run finished successfully run_id={run_id}")

        except Exception as exc:
            logger.exception(f"Run failed run_id={run_id}")
            self._shutdown_collectors()
            self._finish_run(run_id, status="failed", notes=str(exc))
            raise

    def _create_run(self) -> int:
        now = datetime.now()
        with session_scope() as session:
            run = CrawlRun(
                run_date=now.date(),
                window_label=self.window_label,
                started_at=now,
                status="running",
                notes=None,
            )
            session.add(run)
            session.flush()
            return int(run.run_id)

    def _finish_run(self, run_id: int, status: str, notes: str = None) -> None:
        with session_scope() as session:
            run = session.get(CrawlRun, run_id)
            if run is None:
                return
            run.status = status
            run.notes = notes
            run.ended_at = datetime.now()

    def _load_active_target_user_ids(self) -> set[str]:
        with session_scope() as session:
            rows = session.execute(select(TargetChannel.user_id).where(TargetChannel.is_active == 1)).all()
        return {row[0] for row in rows}

    def _save_live_snapshots(self, run_id: int, streams: list[LiveBroadcast]) -> None:
        if not streams:
            logger.info("No active streams to snapshot at this time. (All 20 targets might be offline)")
            return
            
        snapshot_ts = datetime.now()
        # Truncate snapshot_ts to current minute exactly to sync with aggregate minute easier
        snapshot_ts = snapshot_ts.replace(second=0, microsecond=0)

        rows = []
        for stream in streams:
            rows.append({
                "run_id": run_id,
                "snapshot_ts": snapshot_ts,
                "broad_no": stream.broad_no,
                "user_id": stream.user_id,
                "broad_title": stream.broad_title,
                "category_id": stream.category_id,
                "viewer_count": stream.viewer_count,
                "resolution": None,     # Chzzk API usually doesn't have this on root
                "is_password": 0,       # Chzzk API usually doesn't have this bool
                "raw_json": stream.raw_json,
            })
        with session_scope() as session:
            session.execute(insert(LiveSnapshot), rows)
        logger.info(f"Saved {len(rows)} live snapshots")

    def _sync_chat_collectors(self, streams: list[LiveBroadcast]) -> None:
        if self.chat_sink is None:
            raise RuntimeError("Chat sink not initialized")

        active_broad_nos = {stream.broad_no for stream in streams}
        current_broad_nos = set(self.chat_collectors.keys())

        to_start = active_broad_nos - current_broad_nos
        to_stop = current_broad_nos - active_broad_nos

        for broad_no in to_stop:
            collector = self.chat_collectors.pop(broad_no)
            collector.stop()
            logger.info(f"Stopped chat collector broad_no={broad_no}")

        for stream in streams:
            if stream.broad_no in to_start:
                collector = WssChatCollector(
                    broad_no=stream.broad_no,
                    user_id=stream.user_id,
                    sink=self.chat_sink,
                )
                collector.start()
                self.chat_collectors[stream.broad_no] = collector
                logger.info(f"Started chat collector broad_no={stream.broad_no} user_id={stream.user_id}")

    def _shutdown_collectors(self) -> None:
        for collector in self.chat_collectors.values():
            collector.stop()
        self.chat_collectors.clear()
        
        if self.chat_sink:
            self.chat_sink.stop()
            self.chat_sink = None

    def _export_current_data(self, run_id: int) -> None:
        import pandas as pd
        from sqlalchemy import create_engine
        import os

        # Determine the exports directory at project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        export_dir = os.path.join(project_root, "exports")
        os.makedirs(export_dir, exist_ok=True)

        engine = create_engine(self.settings.sqlalchemy_url)

        # 1. Export Minute Features to Excel
        features_file = os.path.join(export_dir, f"Run_{run_id}_Features.xlsx")
        try:
            feat_query = f"SELECT * FROM minute_features WHERE run_id = {run_id}"
            feat_df = pd.read_sql(feat_query, engine)
            if not feat_df.empty:
                feat_df.to_excel(features_file, index=False)
                logger.info(f"Periodically exported {len(feat_df)} minute features to {features_file}")
            else:
                logger.warning(f"No minute features found in DB to export for run_id={run_id}")
        except Exception as e:
            logger.error(f"Failed to export minute features to Excel: {e}", exc_info=True)

        # 2. Export Chat Messages to CSV (UTF-8 with BOM for Excel Korean/Vietnamese letters)
        chats_file = os.path.join(export_dir, f"Run_{run_id}_Chats.csv")
        try:
            chat_query = f"SELECT * FROM chat_messages_raw WHERE run_id = {run_id}"
            chat_df = pd.read_sql(chat_query, engine)
            if not chat_df.empty:
                chat_df.to_csv(chats_file, index=False, encoding='utf-8-sig')
                logger.info(f"Periodically exported {len(chat_df)} chat messages to {chats_file}")
            else:
                logger.warning(f"No chat messages found in DB to export for run_id={run_id}")
        except Exception as e:
            logger.error(f"Failed to export chat messages to CSV: {e}", exc_info=True)
