import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import engine
import pandas as pd

run_id = sys.argv[1] if len(sys.argv) > 1 else None

if not run_id:
    # Get last run_id
    run_id = pd.read_sql("SELECT MAX(run_id) FROM crawl_runs", engine).iloc[0,0]

print(f"Checking statistics for run_id: {run_id}")

snapshots = pd.read_sql(f"SELECT COUNT(*) FROM live_snapshots WHERE run_id={run_id}", engine).iloc[0,0]
chats = pd.read_sql(f"SELECT COUNT(*) FROM chat_messages_raw WHERE run_id={run_id}", engine).iloc[0,0]
features = pd.read_sql(f"SELECT COUNT(*) FROM minute_features WHERE run_id={run_id}", engine).iloc[0,0]

print(f"Snapshots collected: {snapshots}")
print(f"Chat messages collected: {chats}")
print(f"Minute feature rows built: {features}")

if chats > 0:
    print("\nSample chats:")
    sample = pd.read_sql(f"SELECT user_nick, message_raw FROM chat_messages_raw WHERE run_id={run_id} LIMIT 5", engine)
    print(sample)
