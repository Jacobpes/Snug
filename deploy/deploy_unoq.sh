#!/usr/bin/env bash
# Deploy Snug to an Arduino UNO Q over USB (adb) from a Mac/Linux host.
# The board needs no internet: aarch64/cp313 wheels are downloaded on the host.
# Usage: ./deploy/deploy_unoq.sh [WIFI_SSID WIFI_PASSWORD]
set -euo pipefail
cd "$(dirname "$0")/.."
ADB=${ADB:-adb}
command -v "$ADB" >/dev/null || ADB=~/platform-tools/adb
"$ADB" devices -l
DEPS=$(mktemp -d)/unoq_deps
python3 -m pip install -q --target "$DEPS" --platform manylinux2014_aarch64 \
  --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64 \
  --python-version 3.13 --implementation cp --only-binary=:all: \
  numpy opencv-python-headless fastapi uvicorn pillow onnxruntime
COPYFILE_DISABLE=1 tar czf /tmp/unoq_deps.tgz -C "$(dirname "$DEPS")" unoq_deps
COPYFILE_DISABLE=1 tar czf /tmp/snug_src.tgz --exclude .venv --exclude __pycache__ --exclude .git -C .. "$(basename "$PWD")"
"$ADB" push /tmp/unoq_deps.tgz /tmp/snug_src.tgz /root/
"$ADB" push deploy/snug.service /etc/systemd/system/snug.service
"$ADB" shell "date -u -s '$(date -u '+%Y-%m-%d %H:%M:%S')' >/dev/null; cd /root && rm -rf unoq_deps && tar xzf unoq_deps.tgz && tar xzf snug_src.tgz && rm -f *.tgz && chmod 644 /etc/systemd/system/snug.service && systemctl daemon-reload && systemctl enable --now snug && systemctl restart snug"
if [ $# -ge 2 ]; then
  "$ADB" shell "nmcli dev wifi connect '$1' password '$2'; ip -4 addr show wlan0 | grep inet"
fi
echo "Snug installed. Open http://<board-ip>:8080 from a device on the same network."
