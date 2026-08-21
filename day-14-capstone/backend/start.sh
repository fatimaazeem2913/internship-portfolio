#!/bin/bash
# start.sh — one-command startup for the backend (requirement #9).

set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
    echo "No .env found — copying .env.example. Defaults to mock mode."
    cp .env.example .env
fi

export $(grep -v '^#' .env | xargs)

echo "Starting backend on http://localhost:8001 (mock_mode=${USE_MOCK_LLM:-true})..."
python main.py
