import logging
from dataclasses import dataclass
from typing import List, Optional

import requests
from requests.exceptions import RequestException

from configs.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class LiveBroadcast:
    broad_no: str
    user_id: str
    user_nick: str
    broad_title: str
    category_id: str
    viewer_count: int
    raw_json: dict

class ChzzkLiveAPIClient:
    def __init__(self):
        self.settings = get_settings()

    def _headers(self):
        if not self.settings.chzzk_client_id or not self.settings.chzzk_client_secret:
            logger.warning("CHZZK Open API credentials are not set in .env. API calls may fail.")
        
        return {
            "Client-Id": self.settings.chzzk_client_id,
            "Client-Secret": self.settings.chzzk_client_secret,
            "Content-Type": "application/json",
        }

    def fetch_live_page(self, next_cursor: Optional[str] = None, size: int = 20) -> dict:
        url = f"{self.settings.chzzk_api_base}/open/v1/lives"
        params = {"size": size}
        if next_cursor:
            params["next"] = next_cursor

        resp = requests.get(url, headers=self._headers(), params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def discover_target_lives(self, target_user_ids: set[str], max_pages: int = 20) -> List[LiveBroadcast]:
        """
        Fetches lives from the official API and filters them by the target_user_ids.
        Returns a list of LiveBroadcast objects for the active target streams.
        """
        if not target_user_ids:
            return []

        all_lives = []
        next_cursor = None

        try:
            for _ in range(max_pages):
                payload = self.fetch_live_page(next_cursor=next_cursor, size=20)
                content = payload.get("content", {})
                data = content.get("data", [])
                page_info = content.get("page", {})

                if not data:
                    break
                
                # Filter immediately to save memory
                for item in data:
                    channel_id = str(item.get("channelId", ""))
                    if channel_id in target_user_ids:
                        all_lives.append(item)

                next_cursor = page_info.get("next")
                if not next_cursor:
                    break
        except RequestException as e:
            # We catch exceptions to avoid failing the whole pipeline
            # Mask headers if we log the exception to avoid leaking credentials
            logger.error("Failed to fetch live page from CHZZK API: %s", str(e))
        except Exception as e:
            logger.exception("Unexpected error during live discovery")

        results = []
        for item in all_lives:
            # According to Chzzk Open API response structure
            results.append(
                LiveBroadcast(
                    broad_no=str(item.get("liveId", "")),
                    user_id=str(item.get("channelId", "")),
                    user_nick=str(item.get("channelName", "")),
                    broad_title=str(item.get("liveTitle", "")),
                    category_id=str(item.get("liveCategoryValue", "")),
                    viewer_count=int(item.get("concurrentUserCount", 0) or 0),
                    raw_json=item,
                )
            )

        return results
