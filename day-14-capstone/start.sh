#!/bin/bash
# start.sh -- Requirement #9: "a startup script (start.sh) to simplify
# project execution." Starts both the backend and frontend with one
# command, in the background, so a single terminal window is enough to
# get the whole app running.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"
CLIENT_DIR="$SCRIPT_DIR/client"

echo "=================================================="
echo "Learning Adventures -- Startup"
echo "=================================================="

# --- Backend setup ---
cd "$SERVER_DIR"

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing backend dependencies..."
pip install --quiet --break-system-packages fastapi uvicorn pydantic google-genai python-dotenv 2>/dev/null \
  || pip install --quiet fastapi uvicorn pydantic google-genai python-dotenv

if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: No .env file found. Copying .env.example -- edit it with your"
    echo "real GEMINI_API_KEY before real (non-mock) responses will work."
    cp .env.example .env
fi

# Load .env into this shell so uvicorn's process inherits the variables.
set -a
source .env
set +a

echo "Starting backend on http://127.0.0.1:8000 ..."
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# --- Frontend setup ---
cd "$CLIENT_DIR"

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies (this may take a minute)..."
    npm install
fi

echo "Starting frontend on http://localhost:5173 ..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo "Both servers are starting up."
echo "Backend:  http://127.0.0.1:8000  (API docs at /docs)"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."
echo "=================================================="

# Forward Ctrl+C to both background processes and wait for them.
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" INT TERM
wait $BACKEND_PID $FRONTEND_PID
