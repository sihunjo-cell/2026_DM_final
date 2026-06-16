import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()

def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]

@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_name: str = os.getenv("DB_NAME", "soop_dm")
    db_user: str = os.getenv("DB_USER", "soop_user")
    db_password: str = os.getenv("DB_PASSWORD", "")

    # Official API config (for live status query)
    chzzk_api_base: str = os.getenv("CHZZK_BASE_URL", "https://openapi.chzzk.naver.com")
    chzzk_client_id: str = os.getenv("CHZZK_CLIENT_ID", "")
    chzzk_client_secret: str = os.getenv("CHZZK_CLIENT_SECRET", "")

    # Crawler Settings
    snapshot_interval_seconds: int = int(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "60"))
    chat_flush_interval_seconds: int = int(os.getenv("CHAT_FLUSH_INTERVAL_SECONDS", "3"))
    chat_flush_batch_size: int = int(os.getenv("CHAT_FLUSH_BATCH_SIZE", "200"))
    export_interval_seconds: int = int(os.getenv("EXPORT_INTERVAL_SECONDS", "600"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
