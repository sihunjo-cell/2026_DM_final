import requests
import json
import csv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import get_settings

settings = get_settings()

def get_top_lives():
    headers = {
        "Client-Id": settings.chzzk_client_id,
        "Client-Secret": settings.chzzk_client_secret,
        "User-Agent": "Mozilla/5.0"
    }
    
    url = f"{settings.chzzk_api_base}/open/v1/lives?size=15"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    
    data = res.json().get('content', {}).get('data', [])
    
    print(f"{'Rank':<5} | {'Streamer':<20} | {'Viewers':<8} | {'Category'}")
    print("-" * 60)
    
    new_csv = []
    
    for i, item in enumerate(data):
        rank = i + 1
        name = item.get('channelName', 'Unknown')
        cid = item.get('channelId')
        viewers = item.get('concurrentUserCount', 0)
        category = item.get('liveCategoryValue', '')
        
        new_csv.append([rank, name, cid])
        print(f"{rank:<5} | {name:<20} | {viewers:<8} | {category}")
        
    with open("top15_targets.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "nickname", "channelId"])
        writer.writerows(new_csv)
        
    print("Mới tạo xong top15_targets.csv dựa trên Top 15 hiện tại.")

if __name__ == "__main__":
    get_top_lives()
