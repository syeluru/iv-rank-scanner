#!/bin/bash
# Watchdog: keeps Theta Terminal + intraday greeks fetch alive
# Run once: nohup bash watchdog.sh &

JAVA_HOME="/opt/homebrew/opt/openjdk"
JAVA="$JAVA_HOME/bin/java"
PROJECT="$HOME/ironworks/Zero_DTE_Options_Strategy"
LOG="$PROJECT/watchdog.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

start_theta() {
  if ! pgrep -f "ThetaTerminalv3.jar" > /dev/null; then
    log "Starting Theta Terminal..."
    "$JAVA" -jar "$PROJECT/theta_terminal/ThetaTerminalv3.jar" \
      --creds-file "$PROJECT/creds.txt" >> "$LOG" 2>&1 &
    sleep 15
    log "Theta Terminal started (PID: $!)"
  fi
}

start_greeks() {
  if ! pgrep -f "fetch_intraday_greeks.py" > /dev/null; then
    log "Starting intraday greeks fetch..."
    cd "$PROJECT" && nohup python3 scripts/fetch_intraday_greeks.py >> "$LOG" 2>&1 &
    log "Greeks fetch started (PID: $!)"
  fi
}

log "Watchdog started"
while true; do
  start_theta
  start_greeks
  # Check every 5 minutes
  sleep 300
done
