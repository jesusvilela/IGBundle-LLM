#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo ""
echo "  ===================================================="
echo "       IACS // Inter-Agent Communication System"
echo "  ===================================================="
echo ""

# Start Python server in background
python -m uvicorn iacs.server.main:app --host localhost --port 9100 --log-level info &
PYTHON_PID=$!
echo "  [1/2] IACS Server PID=$PYTHON_PID on http://localhost:9100"
sleep 3

# Start Node.js dashboard
cd "$SCRIPT_DIR/dashboard"
if [ ! -d "node_modules" ]; then
    echo "  Installing dashboard dependencies..."
    npm install --production
fi
echo "  [2/2] Dashboard on http://localhost:9110"
node server.js &
NODE_PID=$!

echo ""
echo "  ===================================================="
echo "  Server:    http://localhost:9100"
echo "  Dashboard: http://localhost:9110"
echo "  ===================================================="
echo "  Press Ctrl+C to stop"
echo ""

trap "kill $PYTHON_PID $NODE_PID 2>/dev/null; exit 0" INT TERM
wait
