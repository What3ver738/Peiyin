#!/bin/bash
# Run the local app with a supported Python version.
set -e
cd "$(dirname "$0")/.."
PYBIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))'; then
    PYBIN="$candidate"
    break
  fi
done
if [ -z "$PYBIN" ]; then
  echo "Install Python 3.12 (supported: 3.10–3.12), then run this launcher again."
  exit 1
fi
if [ ! -d venv ]; then
  "$PYBIN" -m venv venv
fi
./venv/bin/python -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else "This venv uses unsupported Python. Rename venv and run the launcher again.")'
./venv/bin/python -m pip install -r requirements.txt
exec ./venv/bin/python -m peiyin
