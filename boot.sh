#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: ./boot.sh [--autonomous]
  --autonomous   Start the autonomous crew supervisor (auxiliary_crew.py)
  -h, --help     Show this help
EOF
}

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: python not found in PATH." >&2
    exit 1
  fi
fi

case "${1:-}" in
  --autonomous|autonomous)
    exec "$PYTHON_BIN" auxiliary_crew.py
    ;;
  -h|--help|"")
    usage
    exit 0
    ;;
  *)
    echo "Error: unknown option: ${1}" >&2
    usage
    exit 1
    ;;
esac
