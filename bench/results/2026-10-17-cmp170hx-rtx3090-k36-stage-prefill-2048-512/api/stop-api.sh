#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/server.pid"
if [[ ! -s "$PIDFILE" ]]; then
    echo "No K36 API supervisor PID file exists."
    exit 0
fi
pid="$(cat "$PIDFILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID file: $PIDFILE" >&2
    exit 1
fi
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PIDFILE"
    echo "K36 API supervisor was already stopped."
    exit 0
fi
args="$(ps -p "$pid" -o args= || true)"
if [[ "$args" != *"/api_guard.py"* ]]; then
    echo "PID $pid is not this K36 API supervisor; refusing to signal it." >&2
    exit 1
fi
kill -TERM "$pid"
for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "K36 API supervisor $pid stopped."
        exit 0
    fi
    sleep 0.5
done
echo "K36 API supervisor $pid did not stop within 60 seconds; inspect $DIR/server.log." >&2
exit 1
