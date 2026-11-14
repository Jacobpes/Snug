#!/bin/bash
# Start Snug on macOS with the TC001 plugged in.
# First run: macOS asks Terminal for camera access – click Allow, then run again.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt imageio-ffmpeg
python - <<'PY'
from snug.capture_mac import _ffmpeg_bin, list_avfoundation_devices
f=_ffmpeg_bin(); print("ffmpeg:", f); print("video devices:", list_avfoundation_devices(f))
PY
( sleep 3; open "http://localhost:8080" ) &
python server.py --host 127.0.0.1 --port 8080 "$@"
