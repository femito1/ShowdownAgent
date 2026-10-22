#!/usr/bin/env bash
# Keep laddering until the account has TOTAL gen4randombattle games; restart the player when the
# websocket dies (poke-env doesn't reconnect, so a dead connection shows up as a silent log).
# Usage: [PYTHON=python] [PLAY_ARGS=--search] PS_PASSWORD=... setsid nohup scripts/ladder_watchdog.sh USER MODEL TOTAL [CONCURRENCY=2] &
set -u
USER=$1; MODEL=$2; TOTAL=$3; CONC=${4:-2}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
LOG=logs/ladder.log; STALL=900
played() {
  curl -s -m 15 "https://pokemonshowdown.com/users/$USER.json" |
    python3 -c "import json,sys; d=json.load(sys.stdin)['ratings'].get('gen4randombattle',{}); print(d.get('w',0)+d.get('l',0))"
}
while true; do
  n=$(played) || { sleep 60; continue; }
  [ "$n" -ge "$TOTAL" ] && { echo "$(date -u +%T) watchdog: $n games played, done" >> $LOG; exit 0; }
  echo "$(date -u +%T) watchdog: $n/$TOTAL games, starting player for $((TOTAL - n))" >> $LOG
  ${PYTHON:-.venv/bin/python} -m porygon.play "$MODEL" --username "$USER" --ladder $((TOTAL - n)) --concurrency "$CONC" ${PLAY_ARGS:-} >> $LOG 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    [ $(( $(date +%s) - $(stat -c %Y $LOG) )) -gt $STALL ] && {
      echo "$(date -u +%T) watchdog: no output for ${STALL}s, restarting" >> $LOG; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; }
  done
  sleep 30
done
