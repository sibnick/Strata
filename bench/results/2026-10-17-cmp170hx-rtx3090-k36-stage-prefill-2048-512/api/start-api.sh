#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/server.pid"
mkdir -p "$DIR"
exec >>"$DIR/server.log" 2>&1

if [[ -s "$PIDFILE" ]]; then
    old_pid="$(cat "$PIDFILE")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_args="$(ps -p "$old_pid" -o args= || true)"
        if [[ "$old_args" == *"/api_guard.py"* ]]; then
            echo "K36 API supervisor is already running as PID $old_pid"
            exit 1
        fi
    fi
    rm -f "$PIDFILE"
fi

printf '%s\n' "$$" >"$PIDFILE"
echo "Starting K36 2048/512 API supervisor PID $$"
exec python3 -B "$DIR/api_guard.py"
