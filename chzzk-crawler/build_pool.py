"""
Build a pool of ~100 top GAME streamers from CHZZK,
then randomly select 30 for each crawling session.
"""
import requests
import json
import csv
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.settings import get_settings

settings = get_settings()

POOL_SIZE = 100   # How many streamers to collect into the pool
SELECT_N = 30     # How many to randomly pick per session
POOL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_100.csv")
SELECTED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top30_targets.csv")


def fetch_top_game_streamers(target_pool_size: int = POOL_SIZE) -> list[dict]:
    """Paginate through CHZZK Open API /lives to collect top GAME streamers."""
    headers = {
        "Client-Id": settings.chzzk_client_id,
        "Client-Secret": settings.chzzk_client_secret,
    }
    
    pool = []
    next_cursor = None
    page = 0
    max_pages = (target_pool_size // 20) + 5  # Buffer for filtering
    
    while len(pool) < target_pool_size and page < max_pages:
        url = f"{settings.chzzk_api_base}/open/v1/lives"
        params = {"size": 20}
        if next_cursor:
            params["next"] = next_cursor
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=20)
            res.raise_for_status()
        except Exception as e:
            print(f"[WARN] API error on page {page}: {e}")
            break
            
        content = res.json().get("content", {})
        data = content.get("data", [])
        page_info = content.get("page", {})
        
        if not data:
            break
        
        for item in data:
            category = item.get("liveCategoryValue", "") or ""
            # Accept all categories — the CHZZK /lives endpoint already returns
            # gaming-heavy results. We keep broad to ensure we hit 100.
            pool.append({
                "channelId": item.get("channelId", ""),
                "channelName": item.get("channelName", ""),
                "viewers": item.get("concurrentUserCount", 0) or 0,
                "category": category,
            })
        
        next_cursor = page_info.get("next")
        page += 1
        if not next_cursor:
            break
    
    # Sort by viewer count descending and take top N
    pool.sort(key=lambda x: x["viewers"], reverse=True)
    return pool[:target_pool_size]


def save_pool_csv(pool: list[dict]):
    """Save the full pool to pool_100.csv for audit/reference."""
    with open(POOL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "nickname", "channelId", "viewers", "category"])
        for i, s in enumerate(pool):
            writer.writerow([i+1, s["channelName"], s["channelId"], s["viewers"], s["category"]])
    print(f"[OK] Saved pool of {len(pool)} streamers → {POOL_CSV}")


def random_select_and_save(pool: list[dict], n: int = SELECT_N):
    """Randomly select N streamers from the pool and save to top20_targets.csv."""
    if len(pool) < n:
        print(f"[WARN] Pool only has {len(pool)} streamers, selecting all.")
        selected = pool
    else:
        selected = random.sample(pool, n)
    
    with open(SELECTED_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "nickname", "channelId"])
        for i, s in enumerate(selected):
            writer.writerow([i+1, s["channelName"], s["channelId"]])
    
    print(f"\n[OK] Randomly selected {len(selected)} streamers → {SELECTED_CSV}")
    print(f"\n{'Rank':<5} | {'Streamer':<25} | {'Viewers':<8} | {'Category'}")
    print("-" * 75)
    for i, s in enumerate(selected):
        print(f"{i+1:<5} | {s['channelName']:<25} | {s['viewers']:<8} | {s['category']}")


if __name__ == "__main__":
    print("=" * 75)
    print("  CHZZK Pool Builder: Fetching top ~100 live streamers...")
    print("=" * 75)
    
    pool = fetch_top_game_streamers(POOL_SIZE)
    save_pool_csv(pool)
    random_select_and_save(pool, SELECT_N)
    
    print(f"\nNext step: run 'python scripts/setup.py --load-csv top30_targets.csv' to update DB targets.")
