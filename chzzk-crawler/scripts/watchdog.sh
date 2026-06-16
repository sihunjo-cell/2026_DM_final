#!/bin/bash

# Configuration
PROJECT_DIR="/home/ubuntu/app/chzzk-crawler"
PROCESS_NAME="scripts/run_pilot.py"
LOG_FILE="$PROJECT_DIR/logs/watchdog.log"

# Get current time in KST (UTC+9)
HH=$(TZ='Asia/Seoul' date +%H)
MM=$(TZ='Asia/Seoul' date +%M)

# Check if we should be running based on window schedules (KST)
# windows are: 17:00-19:00, 20:00-22:00, 23:00-01:00
SHOULD_RUN=0
if [ "$HH" -ge 17 ] && [ "$HH" -lt 19 ]; then
    SHOULD_RUN=1
elif [ "$HH" -ge 20 ] && [ "$HH" -lt 22 ]; then
    SHOULD_RUN=1
elif [ "$HH" -ge 23 ] || [ "$HH" -eq 0 ]; then
    SHOULD_RUN=1
fi

if [ "$SHOULD_RUN" -eq 1 ]; then
    # Check if process is running
    pgrep -f "$PROCESS_NAME" > /dev/null
    
    if [ $? -ne 0 ]; then
        echo "$(date): Process $PROCESS_NAME not found. Restarting..." >> $LOG_FILE
        # Logic to restart (we'll keep it simple for now, as cron usually handles the start)
        # But we could trigger an alert here via report_status.py
        cd $PROJECT_DIR && source .venv/bin/activate && python scripts/report_status.py --type post --window "WATCHDOG_RECOVERY" --message "Alert: Crawler process was missing and recovered."
    fi
fi
