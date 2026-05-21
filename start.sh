#!/bin/bash
set -u

echo "Starting RE Assistant..."

ROOT="$(cd "$(dirname "$0")" && pwd)"

(cd "$ROOT/backend" && bash run.sh) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo "Backend running at http://localhost:8000  (pid $BACKEND_PID)"
echo "Frontend running at http://localhost:3000 (pid $FRONTEND_PID)"
echo "Press Ctrl+C to stop both"

cleanup() {
  echo ""
  echo "Stopping RE Assistant..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait
