import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manager import CrawlManager
from configs.settings import get_settings

def setup_logging():
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File handler
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, 'crawler.log'), mode='a')
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHZZK pilot crawler runner")
    parser.add_argument("--window", required=True, help="Window label, e.g. test/morning/afternoon/evening/night")
    parser.add_argument("--duration", type=int, default=300, help="Window duration in seconds (default: 300 = 5 mins)")
    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    
    manager = CrawlManager(window_label=args.window, duration_seconds=args.duration)
    # The CrawlManager handles the graceful shutdown when duration expires
    manager.run()
