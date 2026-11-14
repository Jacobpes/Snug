"""Optional STM32 / Modulino alarm stub for Arduino UNO Q.

On the UNO Q, Python on the Qualcomm MPU talks to the STM32U585 sketch
through the official Arduino App Lab Bridge (MessagePack-RPC via the
`arduino-router` service). Do **not** open `/dev/ttyHS1` or `Serial1`
yourself — the router locks those ports.

Documented API (https://docs.arduino.cc/software/app-lab/bridge/bridge-api/):

    from arduino.app_utils import Bridge
    Bridge.call("set_alarm", level, timeout=2)   # Python → sketch
    Bridge.provide("on_button", handler)         # sketch → Python (optional)

The matching sketch is `sketches/AlarmBridge/AlarmBridge.ino`. It should
`Bridge.provide("set_alarm", ...)` and drive an LED / buzzer / Modulino.

This module is a no-op unless `arduino.app_utils` is importable and the
router is running. Failures are logged and never crash the dashboard.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("snug.bridge")

SEVERITY_LEVEL = {"ok": 0, "low": 1, "medium": 2, "high": 3}


class AlarmBridge:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._bridge: Any = None
        self._last_level = 0
        if not enabled:
            return
        try:
            from arduino.app_utils import Bridge  # type: ignore

            self._bridge = Bridge
            log.info("Arduino App Lab Bridge loaded")
        except Exception as exc:
            log.warning("Bridge unavailable (%s). Alarm I/O disabled.", exc)
            self._bridge = None

    @property
    def active(self) -> bool:
        return self._bridge is not None

    def set_from_findings(self, findings: list[Any]) -> None:
        if not findings:
            self.set_level(0)
            return
        level = max(SEVERITY_LEVEL.get(getattr(f, "severity", "ok"), 0) for f in findings)
        self.set_level(level)

    def set_level(self, level: int) -> None:
        level = int(max(0, min(3, level)))
        if level == self._last_level:
            return
        self._last_level = level
        if self._bridge is None:
            return
        try:
            # Official call signature: Bridge.call(name, *args, timeout=10)
            self._bridge.call("set_alarm", level, timeout=2)
        except Exception as exc:
            log.warning("Bridge.call(set_alarm) failed: %s", exc)
