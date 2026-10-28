"""Arduino serial link. Importing pyserial is lazy so tests can pass
a fake sender instead of real hardware.
"""

from __future__ import annotations

import time


class ArduinoLink:
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser = None
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)  # Arduino resets when the port opens
            print(f"[INFO] Serial open: {port} @ {baud}")
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[WARN] Serial not available ({port}): {exc}. "
                  "Running without servos.")
            self.ser = None

    def send(self, line: str):
        if self.ser is None or not line:
            return
        try:
            self.ser.write(line.encode("utf-8"))
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Serial write failed: {exc}")

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
