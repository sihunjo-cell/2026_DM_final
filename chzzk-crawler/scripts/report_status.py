import sys
import os
import requests
import argparse
import subprocess
from datetime import datetime

# Add parent dir to path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import get_settings
from core.db import SessionLocal
from core.models import CrawlRun, LiveSnapshot, ChatMessageRaw

def send_discord_message(webhook_url, content, embed=None):
    payload = {"content": content}
    if embed:
        payload["embeds"] = [embed]
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        print("Discord notification sent.")
    except Exception as e:
        print(f"Failed to send Discord message: {e}")

def get_latest_run_id():
    session = SessionLocal()
    try:
        run = session.query(CrawlRun).order_by(CrawlRun.id.desc()).first()
        return run.id if run else None
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['pre', 'post'], required=True, help='Notification type')
    parser.add_argument('--window', help='Window name (evening1, evening2, night)')
    args = parser.parse_args()

    settings = get_settings()
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not found in environment.")
        return

    if args.type == 'pre':
        content = f"🔔 **[Reminder]** CHZZK Crawler session **{args.window}** will start in 5 minutes (at {datetime.now().strftime('%H:%M')}nd)."
        send_discord_message(webhook_url, content)
    
    elif args.type == 'post':
        run_id = get_latest_run_id()
        if not run_id:
            send_discord_message(webhook_url, "⚠️ **[Error]** Session finished but no run_id found in database.")
            return

        # Trigger automatic Excel export for minute features
        excel_file = f"CHZZK_Run_{run_id}_Features.xlsx"
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "export_csv.py"),
            "--table", "minute_features",
            "--run_id", str(run_id),
            "--out", excel_file,
            "--excel"
        ]
        
        export_success = False
        try:
            subprocess.run(cmd, check=True)
            export_success = True
        except Exception as e:
            print(f"Export failed: {e}")

        embed = {
            "title": f"✅ Session Complete: Run #{run_id}",
            "color": 3066993, # Green
            "fields": [
                {"name": "Window", "value": args.window or "Unknown", "inline": True},
                {"name": "Excel Export", "value": "Success" if export_success else "Failed", "inline": True},
                {"name": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": False}
            ],
            "description": f"Dữ liệu của phiên đã được thu thập và lưu trữ tại `{excel_file}` trên server."
        }
        send_discord_message(webhook_url, f"📊 **CHZZK Crawler Report**", embed=embed)

if __name__ == '__main__':
    main()
