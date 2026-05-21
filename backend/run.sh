#!/bin/bash
set -e

echo "Starting RE Assistant Backend..."

if [ ! -d ".venv" ]; then
  echo "No .venv found in backend/. Run:"
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
