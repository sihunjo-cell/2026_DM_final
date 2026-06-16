import sys
import os
import argparse
import csv

# Add parent dir to path to import configs/core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import engine, session_scope
from core.models import TargetChannel, Base


def init_db():
    print("Creating DB tables if not exist...")
    Base.metadata.create_all(bind=engine)
    print("Done")


def refresh_targets(csv_path: str):
    """
    Deactivate ALL current targets, then load the new random selection.
    This ensures each crawl session tracks a fresh random subset of the pool.
    """
    if not os.path.exists(csv_path):
        print(f"Error: csv file {csv_path} not found.")
        return

    print(f"Refreshing targets from {csv_path}...")
    with session_scope() as session:
        # Step 1: Deactivate all existing targets
        deactivated = session.query(TargetChannel).update({TargetChannel.is_active: 0})
        print(f"  Deactivated {deactivated} old targets.")

    # Step 2: Load new targets from CSV
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header row

        with session_scope() as session:
            count = 0
            for row in reader:
                if len(row) < 3:
                    continue
                nick = row[1]
                channel_id = row[2]

                existing = session.query(TargetChannel).filter(
                    TargetChannel.user_id == channel_id
                ).first()

                if existing:
                    existing.is_active = 1
                    existing.user_nick = nick
                    print(f"  Reactivated: {nick} ({channel_id})")
                else:
                    session.add(TargetChannel(
                        user_id=channel_id,
                        user_nick=nick,
                        category_id="GAME",
                        is_active=1
                    ))
                    print(f"  Added new:   {nick} ({channel_id})")
                count += 1

    print(f"\nTarget refresh complete. {count} streamers now active.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Init database tables")
    parser.add_argument("--load-csv", type=str, help="Path to csv to load target streamers")
    args = parser.parse_args()

    if args.init:
        init_db()

    if args.load_csv:
        refresh_targets(args.load_csv)

    if not args.init and not args.load_csv:
        print("Use --init to create tables, or --load-csv <file.csv> to load targets.")
