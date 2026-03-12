"""
serve_dashboard.py
==================
Zero-dependency HTTP server for the AI-IDPS dashboard.

Usage:
    python3 serve_dashboard.py           # serves on port 8765
    python3 serve_dashboard.py --port 9000

Opens the dashboard automatically in your default browser.
"""

import argparse
import collections
import json
import logging
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from threading import Timer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_logger = logging.getLogger("dashboard_server")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DIR         = Path(__file__).resolve().parent
DASHBOARD_HTML = _DIR / "dashboard.html"
LOGS_DIR     = _DIR / "logs"
IDPS_LOG     = LOGS_DIR / "idps.log"
ATTACK_LOG   = LOGS_DIR / "attacks.log"

# ---------------------------------------------------------------------------
# Honeypot event store (populated by POST /api/new_attack)
# ---------------------------------------------------------------------------
_honeypot_lock   = threading.Lock()
_honeypot_events = collections.deque(maxlen=200)   # newest-first ring buffer

# ---------------------------------------------------------------------------
# Log parsers
# ---------------------------------------------------------------------------
_RE_ATTACK = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*?ATTACK DETECTED\s+"
    r"(?P<src>[\d.:]+)\s+[→>]+\s+(?P<dst>[\d.:]+)\s+"
    r"\((?P<cls>[^)0-9]+?)\s+(?P<prob>[\d.]+%)\)"
)
_RE_BENIGN = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*?BENIGN flow"
)


def _tail(path: Path, n: int = 2000) -> list[str]:
    """Read the last n lines of a file efficiently."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()[-n:]


def build_stats() -> dict:
    attacks       = []
    attack_classes: dict[str, int] = {}
    recent_events = []
    benign_count  = 0

    # ── Parse attacks.log ───────────────────────────────────────────────
    for line in _tail(ATTACK_LOG, 2000):
        m = _RE_ATTACK.search(line)
        if m:
            cls = m.group("cls").strip()
            entry = {
                "timestamp":      m.group("ts"),
                "src":            m.group("src"),
                "dst":            m.group("dst"),
                "classification": cls,
                "confidence":     m.group("prob"),
            }
            attacks.append(entry)
            attack_classes[cls] = attack_classes.get(cls, 0) + 1
            recent_events.append({**entry, "type": "attack"})

    # Sort newest-first
    attacks.sort(key=lambda x: x["timestamp"], reverse=True)

    # ── Parse idps.log for benign count & recent benign events ──────────
    for line in _tail(IDPS_LOG, 5000):
        if "BENIGN flow" in line:
            benign_count += 1
            m_ts = _RE_BENIGN.search(line)
            if m_ts:
                # extract IPs from "BENIGN flow A → B"
                rest = line[m_ts.end():].strip()
                ips = re.findall(r"[\d.]+", rest)
                src = ips[0] if len(ips) > 0 else "?"
                dst = ips[1] if len(ips) > 1 else "?"
                recent_events.append({
                    "timestamp":      m_ts.group("ts"),
                    "src":            src,
                    "dst":            dst,
                    "classification": "BENIGN",
                    "type":           "benign",
                })

    # ── Merge honeypot events into the main data ────────────────────────
    with _honeypot_lock:
        hp_snapshot = list(_honeypot_events)

    for hp in hp_snapshot:
        cls = hp.get("attack_type", "Unknown")
        conf_raw = hp.get("confidence", 0)
        conf_str = f"{float(conf_raw) * 100:.2f}%" if isinstance(conf_raw, (int, float)) else str(conf_raw)
        entry = {
            "timestamp":      hp.get("timestamp", ""),
            "src":            hp.get("source_ip", "?"),
            "dst":            "Honeypot",
            "classification": cls,
            "confidence":     conf_str,
        }
        attacks.append(entry)
        attack_classes[cls] = attack_classes.get(cls, 0) + 1
        recent_events.append({**entry, "type": "attack"})

    # Sort & truncate
    attacks.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_events.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_events = recent_events[:40]

    return {
        "attack_count":   len(attacks),
        "benign_count":   benign_count,
        "attacks":        attacks[:50],
        "attack_classes": attack_classes,
        "recent_events":  recent_events,
        "honeypot_events": hp_snapshot,
    }


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass  # silence default access log

    def _set_headers(self, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers("text/plain")

    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/dashboard":
                if not DASHBOARD_HTML.exists():
                    self._set_headers("text/plain", 404)
                    self.wfile.write(b"dashboard.html not found")
                    return
                self._set_headers("text/html; charset=utf-8")
                self.wfile.write(DASHBOARD_HTML.read_bytes())

            elif self.path == "/api/stats":
                data = build_stats()
                payload = json.dumps(data).encode("utf-8")
                self._set_headers("application/json")
                self.wfile.write(payload)

            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Not found")
        except BrokenPipeError:
            pass

    def do_POST(self):
        try:
            if self.path == "/api/new_attack":
                try:
                    length  = int(self.headers.get("Content-Length", 0))
                    raw     = self.rfile.read(length)
                    event   = json.loads(raw.decode("utf-8"))

                    # Validate required fields
                    required = {"source_ip", "attack_type", "confidence", "timestamp"}
                    missing  = required - event.keys()
                    if missing:
                        self._set_headers("application/json", 400)
                        self.wfile.write(json.dumps(
                            {"error": f"Missing fields: {sorted(missing)}"}
                        ).encode())
                        return

                    if not isinstance(event["confidence"], (int, float)):
                        raise ValueError("confidence must be numeric")

                    with _honeypot_lock:
                        _honeypot_events.appendleft(event)  # newest first

                    _logger.info(
                        "HONEYPOT ATTACK received: %s from %s (conf=%.2f)",
                        event["attack_type"], event["source_ip"], event["confidence"],
                    )

                    self._set_headers("application/json", 200)
                    self.wfile.write(json.dumps({"status": "ok"}).encode())

                except (json.JSONDecodeError, ValueError) as exc:
                    _logger.warning("Bad POST /api/new_attack: %s", exc)
                    self._set_headers("application/json", 400)
                    self.wfile.write(json.dumps({"error": str(exc)}).encode())
                except Exception as exc:
                    _logger.error("Internal error in /api/new_attack: %s", exc)
                    self._set_headers("application/json", 500)
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Not found")
        except BrokenPipeError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-IDPS Dashboard Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    host, port = "0.0.0.0", args.port
    url = f"http://localhost:{port}/"

    server = ThreadingHTTPServer((host, port), DashboardHandler)

    print("=" * 55)
    print("  🛡️  AI-IDPS Dashboard Server")
    print("=" * 55)
    print(f"  URL    : {url}")
    print(f"  API    : http://localhost:{port}/api/stats")
    print(f"  Logs   : {LOGS_DIR}")
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
