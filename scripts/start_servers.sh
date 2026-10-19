#!/usr/bin/env bash
# Start N local Showdown servers on ports BASE..BASE+N-1 (no auth, no throttling).
# Usage: scripts/start_servers.sh [N=1] [BASE=8000]    Stop: scripts/stop_servers.sh
set -euo pipefail
N=${1:-1}; BASE=${2:-8000}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PS="$ROOT/pokemon-showdown"
if [ ! -f "$PS/config/config.js" ]; then
  # Training servers share one directory: no disk state (avoids write races), no REPL sockets.
  cp "$PS/config/config-example.js" "$PS/config/config.js"
  printf '\nexports.nofswriting = true;\nexports.repl = false;\n' >> "$PS/config/config.js"
fi
mkdir -p "$ROOT/logs" "$PS/logs/repl"
(cd "$PS" && node build)  # once; concurrent per-server builds clobber each other
for ((i = 0; i < N; i++)); do
  PORT=$((BASE + i))
  (cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security "$PORT" < /dev/null > "$ROOT/logs/showdown-$PORT.log" 2>&1 &)
done
for ((i = 0; i < N; i++)); do
  PORT=$((BASE + i))
  for ((t = 0; t < 120; t++)); do
    grep -q "listening" "$ROOT/logs/showdown-$PORT.log" 2>/dev/null && continue 2
    sleep 0.5
  done
  echo "Showdown on port $PORT failed to start:" >&2; tail -20 "$ROOT/logs/showdown-$PORT.log" >&2; exit 1
done
echo "Showdown servers up on ports $BASE-$((BASE + N - 1))"
