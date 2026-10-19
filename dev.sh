#!/usr/bin/env bash
# Start backend (FastAPI :8000) and frontend (Vite :5173) together. Ctrl+C stops both.
set -e
cd "$(dirname "$0")"

if [ ! -x backend/.venv/bin/python ]; then
  echo "Creating backend/.venv (Python 3.11 or 3.12 recommended for TensorFlow)..."
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
  "$PY" -m venv backend/.venv
  backend/.venv/bin/pip install -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi

(cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000) &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT INT TERM
echo "Backend on http://localhost:8000  ·  App on http://localhost:5173"
cd frontend && npm run dev
