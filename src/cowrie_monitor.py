"""
cowrie_monitor.py
=================
AI-IDPS — Cowrie Honeypot Integration Module

Continuously tails /home/cowrieuser/cowrie/var/log/cowrie/cowrie.json,
parses events, detects SSH brute-force behaviour, and pushes
classification results to the AI-IDPS dashboard via POST /api/new_attack.

Usage:
    python3 src/cowrie_monitor.py
    python3 src/cowrie_monitor.py --log /path/to/cowrie.json --threshold 5 --window 60

Event types handled:
    cowrie.login.failed   — failed SSH authentication attempt
    cowrie.login.success  — successful SSH authentication
    cowrie.session.connect — new SSH session opened
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional

# Import prevention module (same package)
try:
    from prevention import block_ip
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prevention import block_ip


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_COWRIE_LOG = Path("/home/cowrieuser/cowrie/var/log/cowrie/cowrie.json")
_DEFAULT_DASHBOARD  = "http://127.0.0.1:8765/api/new_attack"
_DEFAULT_THRESHOLD  = 5      # failed logins from same IP to trigger alert
_DEFAULT_WINDOW_SEC = 60     # rolling time window (seconds)
_DEFAULT_POLL_SEC   = 1.0    # how often to poll for new lines

_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_LOGS_DIR    = _PROJECT_ROOT / "logs"

# Cowrie event types we care about
_EVENT_FAILED  = "cowrie.login.failed"
_EVENT_SUCCESS = "cowrie.login.success"
_EVENT_CONNECT = "cowrie.session.connect"

_INTERESTING_EVENTS = {_EVENT_FAILED, _EVENT_SUCCESS, _EVENT_CONNECT}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> logging.Logger:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cowrie_log_path = _LOGS_DIR / "cowrie_monitor.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        fh = logging.FileHandler(cowrie_log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger("cowrie_monitor")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CowrieEvent:
    """Parsed representation of a single Cowrie JSON log line."""
    event_id:   str
    src_ip:     str
    timestamp:  str
    session:    str = ""
    username:   str = ""
    password:   str = ""
    raw:        dict = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Output format sent to the dashboard API."""
    source_ip:   str
    attack_type: str
    confidence:  float
    timestamp:   str

    def to_dict(self) -> dict:
        return {
            "source_ip":   self.source_ip,
            "attack_type": self.attack_type,
            "confidence":  self.confidence,
            "timestamp":   self.timestamp,
        }


# ---------------------------------------------------------------------------
# JSON line parser
# ---------------------------------------------------------------------------

class CowrieLogParser:
    """Parses individual JSON lines from cowrie.json."""

    def parse(self, line: str) -> Optional[CowrieEvent]:
        """
        Parse a single cowrie JSON log line.
        Returns a CowrieEvent if the line is valid and interesting,
        None otherwise.
        """
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        event_id = data.get("eventid", "")
        if event_id not in _INTERESTING_EVENTS:
            return None

        src_ip = data.get("src_ip", "0.0.0.0")
        timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())

        return CowrieEvent(
            event_id  = event_id,
            src_ip    = src_ip,
            timestamp = timestamp,
            session   = data.get("session", ""),
            username  = data.get("username", ""),
            password  = data.get("password", ""),
            raw       = data,
        )


# ---------------------------------------------------------------------------
# Brute-force detector
# ---------------------------------------------------------------------------

class BruteForceDetector:
    """
    Tracks failed login attempts per source IP in a rolling time window.
    Fires an alert when the threshold is exceeded within the window.
    """

    def __init__(self, threshold: int, window_sec: float) -> None:
        self._threshold  = threshold
        self._window_sec = window_sec
        # src_ip -> deque of epoch timestamps of failed attempts
        self._failures: Dict[str, Deque[float]] = defaultdict(deque)
        # src_ip -> epoch of last alert (to avoid alert spam)
        self._last_alert: Dict[str, float] = {}

    def record_failure(self, src_ip: str, event_time: float) -> Optional[ClassificationResult]:
        """
        Record a failed login attempt. Returns a ClassificationResult if
        brute-force threshold is crossed, None otherwise.
        """
        dq = self._failures[src_ip]
        dq.append(event_time)

        # Evict entries older than the window
        cutoff = event_time - self._window_sec
        while dq and dq[0] < cutoff:
            dq.popleft()

        count = len(dq)
        if count >= self._threshold:
            # Suppress duplicate alerts within the same window
            last = self._last_alert.get(src_ip, 0.0)
            if event_time - last < self._window_sec:
                return None

            self._last_alert[src_ip] = event_time
            confidence = min(0.50 + (count - self._threshold) * 0.05, 0.99)
            return ClassificationResult(
                source_ip   = src_ip,
                attack_type = "SSH-Patator",
                confidence  = round(confidence, 4),
                timestamp   = datetime.fromtimestamp(
                    event_time, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )
        return None

    def record_success(self, src_ip: str, event_time: float) -> Optional[ClassificationResult]:
        """
        A successful login after prior failures is suspicious — flag as
        SSH Brute Force (successful compromise).
        """
        if src_ip in self._failures and len(self._failures[src_ip]) > 0:
            last = self._last_alert.get(src_ip, 0.0)
            if event_time - last < self._window_sec:
                return None
            self._last_alert[src_ip] = event_time
            return ClassificationResult(
                source_ip   = src_ip,
                attack_type = "SSH-Bruteforce-Success",
                confidence  = 0.99,
                timestamp   = datetime.fromtimestamp(
                    event_time, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )
        return None


# ---------------------------------------------------------------------------
# Dashboard reporter
# ---------------------------------------------------------------------------

class DashboardReporter:
    """POSTs ClassificationResult objects to the dashboard /api/new_attack."""

    def __init__(self, api_url: str, timeout: float = 5.0, max_retries: int = 3) -> None:
        self._url         = api_url
        self._timeout     = timeout
        self._max_retries = max_retries
        self._logger      = logging.getLogger("cowrie_monitor.reporter")

    def send(self, result: ClassificationResult, status: str = "") -> bool:
        """
        Send a classification result to the dashboard.
        Retries up to max_retries times with 1s backoff.
        Returns True on success, False on failure (non-fatal).
        """
        data = result.to_dict()
        if status:
            data["status"] = status
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data    = payload,
            headers = {
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
            method  = "POST",
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = resp.read()
                    if resp.status == 200:
                        self._logger.info(
                            "Attack sent to dashboard successfully: [%s] from %s (conf=%.2f)",
                            result.attack_type, result.source_ip, result.confidence,
                        )
                        return True
                    else:
                        self._logger.warning(
                            "Dashboard returned HTTP %d for attack from %s",
                            resp.status, result.source_ip,
                        )
            except urllib.error.URLError as exc:
                self._logger.warning(
                    "Dashboard unreachable (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )
            except Exception as exc:
                self._logger.error(
                    "Unexpected error sending to dashboard (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )

            if attempt < self._max_retries:
                time.sleep(1)

        self._logger.error(
            "Failed to send attack to dashboard after %d retries: [%s] from %s",
            self._max_retries, result.attack_type, result.source_ip,
        )
        return False


# ---------------------------------------------------------------------------
# File tailer (rotation-safe)
# ---------------------------------------------------------------------------

class RotationSafeTailer:
    """
    Reads new lines appended to a file using inode-based rotation detection.

    When a log rotation is detected (different inode or file shrinks),
    the tailer re-opens the file from the beginning so no events are missed.
    """

    def __init__(self, path: Path, poll_interval: float = 1.0) -> None:
        self._path          = path
        self._poll_interval = poll_interval
        self._file          = None
        self._inode         = None
        self._logger        = logging.getLogger("cowrie_monitor.tailer")

    def _open(self) -> None:
        """Open (or reopen) the log file, seeking to end on first open."""
        try:
            self._file  = open(self._path, "r", encoding="utf-8", errors="replace")
            self._inode = os.stat(self._path).st_ino
        except OSError as exc:
            self._logger.warning("Cannot open %s: %s", self._path, exc)
            self._file  = None
            self._inode = None

    def _detect_rotation(self) -> bool:
        """Return True if the file has been rotated or truncated."""
        try:
            stat = os.stat(self._path)
        except OSError:
            return False

        if stat.st_ino != self._inode:
            self._logger.info("Rotation detected (inode changed) — reopening %s", self._path)
            return True

        if self._file and self._file.tell() > stat.st_size:
            self._logger.info("Rotation detected (file truncated) — reopening %s", self._path)
            return True

        return False

    def readlines(self) -> list[str]:
        """
        Return any new lines available in the file.
        Handles file-not-yet-existing and rotation transparently.
        """
        if not self._path.exists():
            if self._file:
                self._file.close()
                self._file = None
            return []

        if self._file is None:
            self._open()
            if self._file is None:
                return []
            # Seek to end so we only process new data going forward
            self._file.seek(0, 2)
            return []

        if self._detect_rotation():
            if self._file:
                self._file.close()
            self._open()
            if self._file is None:
                return []

        return self._file.readlines()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# Main monitor orchestrator
# ---------------------------------------------------------------------------

class CowrieMonitor:
    """
    Orchestrates tailing, parsing, brute-force detection, and reporting.
    """

    def __init__(
        self,
        cowrie_log:    Path,
        dashboard_url: str,
        threshold:     int,
        window_sec:    float,
        poll_interval: float,
    ) -> None:
        self._logger    = _configure_logging()
        self._tailer    = RotationSafeTailer(cowrie_log, poll_interval)
        self._parser    = CowrieLogParser()
        self._detector  = BruteForceDetector(threshold, window_sec)
        self._reporter  = DashboardReporter(dashboard_url)
        self._running   = False
        self._poll      = poll_interval

        self._logger.info("CowrieMonitor initialised")
        self._logger.info("  Log file    : %s", cowrie_log)
        self._logger.info("  Dashboard   : %s", dashboard_url)
        self._logger.info("  Threshold   : %d failures in %ds window", threshold, int(window_sec))

    # ── event handlers ──────────────────────────────────────────────────────

    def _handle_event(self, event: CowrieEvent) -> None:
        """Route a parsed event to the appropriate detector logic."""
        try:
            epoch = datetime.fromisoformat(
                event.timestamp.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            epoch = time.time()

        result: Optional[ClassificationResult] = None

        if event.event_id == _EVENT_FAILED:
            self._logger.info(
                "LOGIN FAILED  src=%s user=%s pass=%s",
                event.src_ip, event.username, event.password,
            )
            result = self._detector.record_failure(event.src_ip, epoch)

        elif event.event_id == _EVENT_SUCCESS:
            self._logger.warning(
                "LOGIN SUCCESS src=%s user=%s — possible compromise",
                event.src_ip, event.username,
            )
            result = self._detector.record_success(event.src_ip, epoch)

        elif event.event_id == _EVENT_CONNECT:
            self._logger.info(
                "SESSION CONNECT src=%s session=%s",
                event.src_ip, event.session,
            )

        if result:
            self._logger.warning(
                "ATTACK CLASSIFIED  type=%s src=%s conf=%.2f",
                result.attack_type, result.source_ip, result.confidence,
            )

            # Send ONE event to dashboard (server determines NEW/BLOCKED status)
            self._reporter.send(result)
            self._logger.warning(
                "ATTACK REPORTED to dashboard for %s", result.source_ip,
            )

            # Schedule iptables block after 60s delay (dashboard stays visible)
            def _delayed_block(r=result):
                self._logger.info(
                    "Waiting 60s before blocking %s (keeping dashboard live)...",
                    r.source_ip,
                )
                time.sleep(60)
                self._logger.warning(
                    "PREVENTION TRIGGERED for IP %s", r.source_ip,
                )
                block_ip(r.source_ip, r.attack_type)

            threading.Thread(
                target=_delayed_block, daemon=True, name="block-" + result.source_ip,
            ).start()

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the monitoring loop. Blocks until interrupted."""
        self._running = True
        self._logger.info("CowrieMonitor started — press Ctrl+C to stop")

        try:
            while self._running:
                lines = self._tailer.readlines()
                for line in lines:
                    event = self._parser.parse(line)
                    if event:
                        self._handle_event(event)
                time.sleep(self._poll)
        except KeyboardInterrupt:
            self._logger.info("CowrieMonitor stopped by user")
        finally:
            self._tailer.close()

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-IDPS — Cowrie Honeypot Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=_DEFAULT_COWRIE_LOG,
        help="Path to cowrie.json log file",
    )
    parser.add_argument(
        "--dashboard",
        default=_DEFAULT_DASHBOARD,
        help="URL of the dashboard /api/new_attack endpoint",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=_DEFAULT_THRESHOLD,
        help="Number of failed logins from same IP to trigger brute-force alert",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=_DEFAULT_WINDOW_SEC,
        help="Rolling time window in seconds for brute-force detection",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=_DEFAULT_POLL_SEC,
        help="Log file poll interval in seconds",
    )

    args = parser.parse_args()

    monitor = CowrieMonitor(
        cowrie_log    = args.log,
        dashboard_url = args.dashboard,
        threshold     = args.threshold,
        window_sec    = args.window,
        poll_interval = args.poll,
    )
    monitor.run()


if __name__ == "__main__":
    main()
