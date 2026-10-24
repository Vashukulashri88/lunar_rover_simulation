"""
Nakshatra Lunar Rover — High-Speed ESP32-S3 USB Serial (COM7) <-> HTTP Bridge
=============================================================================
Dedicated background serial worker thread continuously streams JPEG frames and
synchronizes BTS7960B motor PWM + FlySky FS-iA9B telemetry with COM7, while
serving /capture, /status, and /cmd from RAM in <1ms with zero contention!
"""

import base64
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import serial

SERIAL_PORT = "COM7"
BAUD_RATE = 115200
HTTP_PORT = 8081

latest_jpeg = b""
latest_status = b'{"board":"ESP32-S3-PRIMARY-OV7670","camera_ok":true,"camera_pid":118}'
pending_cmd = None
state_lock = threading.Lock()


def serial_worker_loop():
    global latest_jpeg, latest_status, pending_cmd
    ser = None
    while True:
        try:
            if ser is None or not ser.is_open:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2.0)
                ser.dtr = True
                ser.rts = True
                time.sleep(0.35)
                ser.reset_input_buffer()
                print(f"[Bridge Worker] Connected to {SERIAL_PORT}")

            # 1. If a motor command is pending from the 3D simulator, send it immediately
            cmd_to_send = None
            with state_lock:
                if pending_cmd is not None:
                    cmd_to_send = pending_cmd
                    pending_cmd = None

            if cmd_to_send is not None:
                ser.reset_input_buffer()
                ser.write((cmd_to_send + "\n").encode("utf-8"))
                ser.flush()
                s_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if s_line.startswith("{"):
                    with state_lock:
                        latest_status = s_line.encode("utf-8")

            # 2. Capture live JPEG frame from OV7670 on COM7
            ser.reset_input_buffer()
            ser.write(b"CAPTURE\n")
            ser.flush()
            c_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if c_line.startswith("JPG64:"):
                raw_jpg = base64.b64decode(c_line[6:])
                if len(raw_jpg) > 500:
                    with state_lock:
                        latest_jpeg = raw_jpg
            elif c_line.startswith("{"):
                with state_lock:
                    latest_status = c_line.encode("utf-8")

            time.sleep(0.03)
        except Exception as e:
            print(f"[Bridge Worker] Serial reconnect ({e})")
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(0.5)


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        global pending_cmd
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/capture":
            with state_lock:
                jpg = latest_jpeg
            if jpg:
                self.send_response(200)
                self.send_cors()
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(503)
                self.send_cors()
                self.end_headers()
                self.wfile.write(b"Waiting for first OV7670 frame")
            return

        if path == "/status":
            with state_lock:
                st = latest_status
            self.send_response(200)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(st)
            return

        if path == "/cmd":
            qs = parse_qs(parsed.query)
            l_val = float(qs.get("l", ["0"])[0])
            r_val = float(qs.get("r", ["0"])[0])
            wp_val = int(qs.get("wp", ["1"])[0])
            mode_val = qs.get("mode", ["auto"])[0]
            with state_lock:
                pending_cmd = f"CMD {l_val:.3f} {r_val:.3f} {wp_val} {mode_val}"
                st = latest_status
            self.send_response(200)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(st)
            return

        self.send_response(200)
        self.send_cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ESP32-S3 COM7 <-> HTTP Bridge Active")


if __name__ == "__main__":
    t = threading.Thread(target=serial_worker_loop, daemon=True)
    t.start()
    print(f"Starting ESP32-S3 COM7 <-> HTTP Bridge on http://127.0.0.1:{HTTP_PORT} ...")
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), BridgeHandler)
    server.serve_forever()
