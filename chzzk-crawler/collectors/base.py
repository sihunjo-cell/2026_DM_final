import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import insert

from configs.settings import get_settings
from core.db import session_scope
from core.models import ChatMessageRaw

logger = logging.getLogger(__name__)

@dataclass
class ChatEvent:
    event_ts: datetime
    broad_no: str
    user_id: str | None
    user_nick: str | None
    message_raw: str | None
    raw_json: dict[str, Any] | None = None

class BufferedChatSink:
    def __init__(self, run_id: int):
        self.run_id = run_id
        self.settings = get_settings()
        self.buffer: deque[dict[str, Any]] = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.flusher = threading.Thread(target=self._flush_loop, daemon=True)

    def start(self) -> None:
        self.flusher.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.flusher.join(timeout=5)
        self.flush(force=True)

    def push(self, event: ChatEvent) -> None:
        payload = self._normalize(event)
        with self.lock:
            self.buffer.append(payload)
            if len(self.buffer) >= self.settings.chat_flush_batch_size:
                self._flush_locked()

    def flush(self, force: bool = False) -> None:
        with self.lock:
            if force or self.buffer:
                self._flush_locked()

    def _flush_loop(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.settings.chat_flush_interval_seconds)
            self.flush()

    def _normalize(self, event: ChatEvent) -> dict[str, Any]:
        event_dict = event.__dict__.copy()
        
        # Avoid circular import, do simple hash/clean here or separate module
        message_raw = event_dict.get("message_raw")
        message_clean = self._clean_message(message_raw)
        
        # Fallback to empty string for hashing to avoid null injection
        safe_msg = message_raw or ""
        safe_uid = event_dict.get("user_id") or ""
        safe_broad = event_dict.get("broad_no") or ""
        
        import hashlib
        hash_input = f"{safe_broad}_{safe_uid}_{safe_msg}_{event.event_ts.timestamp()}".encode("utf-8")
        message_hash = hashlib.sha256(hash_input).hexdigest()

        return {
            "run_id": self.run_id,
            "event_ts": event.event_ts,
            "broad_no": safe_broad,
            "user_id": event.user_id, # Can be null
            "user_nick": event.user_nick,
            "message_raw": message_raw,
            "message_clean": message_clean,
            "message_hash": message_hash,
            "raw_json": event.raw_json,
        }

    def _clean_message(self, text: str | None) -> str | None:
        if not text:
            return text
        import re
        text = str(text)
        # Remove extra whitespaces, common cleanups
        text = re.sub(r'\\s+', ' ', text).strip()
        return text

    def _flush_locked(self) -> None:
        if not self.buffer:
            return
        rows = list(self.buffer)
        self.buffer.clear()
        
        # Ensure we catch DB errors to not crash the thread
        try:
            with session_scope() as session:
                session.execute(insert(ChatMessageRaw), rows)
            logger.info("Flushed %s chat messages", len(rows))
        except Exception as e:
            logger.error("Failed to flush chat messages to DB: %s", str(e))

class BaseChatCollector(ABC):
    def __init__(self, broad_no: str, user_id: str, sink: BufferedChatSink):
        self.broad_no = broad_no
        self.user_id = user_id # CHZZK Streamer ID
        self.sink = sink
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
