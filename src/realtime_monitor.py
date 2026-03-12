"""
realtime_monitor.py
AI-IDPS Phase 2 — Real-Time Detection Engine
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"
_DEFAULT_LOGS_DIR = _PROJECT_ROOT / "logs"
_DEFAULT_SNORT_LOG = Path("/var/log/snort/alert")
_DEFAULT_ATTACK_LOG = _DEFAULT_LOGS_DIR / "attacks.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(attack_log_path: Path) -> logging.Logger:
    attack_log_path.parent.mkdir(parents=True, exist_ok=True)
    idps_log_path = _DEFAULT_LOGS_DIR / "idps.log"
    idps_log_path.parent.mkdir(parents=True, exist_ok=True)

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

        fh = logging.FileHandler(idps_log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    attack_handler = logging.FileHandler(attack_log_path, encoding="utf-8")
    attack_handler.setLevel(logging.WARNING)
    attack_handler.setFormatter(fmt)

    logger = logging.getLogger("realtime_monitor")
    logger.addHandler(attack_handler)

    return logger


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ParsedAlert:
    timestamp: str = ""
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "TCP"
    ttl: int = 64
    tos: int = 0
    packet_len: int = 0
    raw_lines: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return self.src_ip != "0.0.0.0"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class SnortLogParser:
    _RE_NET = re.compile(
        r"(?P<ts>\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+"
        r"(?P<src>[\d\.]+)(?::(?P<sport>\d+))?\s+->\s+"
        r"(?P<dst>[\d\.]+)(?::(?P<dport>\d+))?"
    )

    _RE_PROTO = re.compile(r"^(TCP|UDP|ICMP)", re.IGNORECASE)
    _RE_TTL = re.compile(r"TTL:(\d+)", re.IGNORECASE)
    _RE_TOS = re.compile(r"TOS:0x([0-9a-fA-F]+)", re.IGNORECASE)
    _RE_DGM = re.compile(r"DgmLen:(\d+)", re.IGNORECASE)

    def parse_block(self, lines: list[str]) -> Optional[ParsedAlert]:
        if not lines:
            return None

        alert = ParsedAlert(raw_lines=lines)

        for line in lines:
            line = line.strip()

            m = self._RE_NET.search(line)
            if m:
                alert.timestamp = m.group("ts")
                alert.src_ip = m.group("src")
                alert.dst_ip = m.group("dst")
                alert.src_port = int(m.group("sport") or 0)
                alert.dst_port = int(m.group("dport") or 0)

            if self._RE_PROTO.match(line):
                alert.protocol = self._RE_PROTO.match(line).group(0).upper()

                ttl = self._RE_TTL.search(line)
                if ttl:
                    alert.ttl = int(ttl.group(1))

                tos = self._RE_TOS.search(line)
                if tos:
                    alert.tos = int(tos.group(1), 16)

                dgm = self._RE_DGM.search(line)
                if dgm:
                    alert.packet_len = int(dgm.group(1))

        return alert if alert.is_valid() else None


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Converts a reconstructed Flow (from FlowTracker) into a numpy feature vector
    that matches the order and names stored in selected_features.pkl.
    """

    _PROTO_MAP: dict[str, int] = {"TCP": 0, "UDP": 1, "ICMP": 2}

    _FIELD_MAP: list[tuple[str, object]] = [
        ("destination port",     lambda f: float(f["dst_port"])),
        ("total fwd packets",    lambda f: float(f["packet_count"])),
        ("total backward",       lambda _: 0.0),
        ("total length of fwd",  lambda f: float(f["total_bytes"])),
        ("fwd packet length max",lambda f: float(f["max_packet_size"])),
        ("fwd packet length mean",lambda f: float(f["avg_packet_size"])),
        ("fwd packet length min", lambda f: float(f["min_packet_size"])),
        ("bwd packet length max", lambda _: 0.0),
        ("bwd packet length mean",lambda _: 0.0),
        ("bwd packet length min", lambda _: 0.0),
        ("flow bytes/s",         lambda f: float(f["bytes_per_second"])),
        ("flow packets/s",       lambda f: float(f["packets_per_second"])),
        ("fwd iat total",        lambda f: float(f["duration"])),
        ("fwd iat mean",         lambda f: float(f["duration"] / max(1.0, f["packet_count"]))),
        ("protocol",             lambda f: float(FeatureBuilder._PROTO_MAP.get(f["protocol"], 0))),
        ("ttl",                  lambda f: float(f["ttl"])),
        ("tos",                  lambda f: float(f["tos"])),
        ("source port",          lambda f: float(f["src_port"])),
        ("flow duration",        lambda f: float(f["duration"])),
    ]

    def __init__(self, selected_features: list[str]) -> None:
        self.selected_features = selected_features
        self._n = len(selected_features)

    def build(self, flow_stats: dict) -> np.ndarray:
        vec = np.zeros(self._n, dtype=np.float32)

        for idx, feat_name in enumerate(self.selected_features):
            feat_lower = feat_name.strip().lower()
            for keyword, extractor in self._FIELD_MAP:
                if keyword in feat_lower:
                    try:
                        vec[idx] = extractor(flow_stats)
                    except Exception:
                        vec[idx] = 0.0
                    break

        return vec.reshape(1, -1)


# ---------------------------------------------------------------------------
# Flow Tracker (FIXED 5-TUPLE)
# ---------------------------------------------------------------------------

class FlowTracker:
    def __init__(self, timeout: float = 10.0):
        self._flows = {}
        self._timeout = timeout

    def add_packet(self, alert: ParsedAlert):
        key = (
            alert.src_ip,
            alert.src_port,
            alert.dst_ip,
            alert.dst_port,
            alert.protocol,
        )

        now = time.time()

        if key not in self._flows:
            self._flows[key] = {
                "src_ip": alert.src_ip,
                "dst_ip": alert.dst_ip,
                "src_port": alert.src_port,
                "dst_port": alert.dst_port,
                "protocol": alert.protocol,
                "ttl": alert.ttl,
                "tos": alert.tos,
                "packet_count": 0,
                "total_bytes": 0,
                "first_seen": now,
                "last_seen": now,
                "max_packet_size": alert.packet_len,
                "min_packet_size": alert.packet_len,
                "initial_timestamp": alert.timestamp,
            }

        flow = self._flows[key]

        flow["packet_count"] += 1
        flow["total_bytes"] += alert.packet_len
        flow["last_seen"] = now

        flow["max_packet_size"] = max(flow["max_packet_size"], alert.packet_len)
        flow["min_packet_size"] = min(flow["min_packet_size"], alert.packet_len)

    def check_timeouts(self):
        now = time.time()
        finalized = []

        for key in list(self._flows.keys()):
            flow = self._flows[key]

            if now - flow["last_seen"] >= self._timeout:
                duration = max(flow["last_seen"] - flow["first_seen"], 1e-6)

                flow["duration"] = duration * 1e6
                flow["avg_packet_size"] = flow["total_bytes"] / flow["packet_count"]
                flow["packets_per_second"] = flow["packet_count"] / duration
                flow["bytes_per_second"] = flow["total_bytes"] / duration

                finalized.append(flow)
                del self._flows[key]

        return finalized


# ---------------------------------------------------------------------------
# RealTimeDetector
# ---------------------------------------------------------------------------

class RealTimeDetector:

    def __init__(
        self,
        snort_log: Path,
        models_dir: Path,
        attack_log: Path,
        poll_interval: float = 0.5,
        confidence_threshold: float = 0.95,
        debug: bool = False,
    ):
        self._threshold = confidence_threshold
        self._debug = debug
        self._running = False
        self._poll_interval = poll_interval

        self._logger = _configure_logging(attack_log)

        self._model = joblib.load(models_dir / "rf_model.pkl")
        self._scaler = joblib.load(models_dir / "scaler.pkl")
        self._selected_features = joblib.load(models_dir / "selected_features.pkl")
        self._label_encoder = joblib.load(models_dir / "label_encoder.pkl")

        self._class_names = list(self._label_encoder.classes_)

        self._feature_builder = FeatureBuilder(self._selected_features)
        self._parser = SnortLogParser()
        self._flow_tracker = FlowTracker()
        self._buffer = []

        self._snort_log = snort_log
        self._file = None

    def _predict(self, flow):
        vec = self._feature_builder.build(flow)
        
        if self._debug:
            self._logger.info("DEBUG [Vector pre-scaling]: %s", vec.tolist()[0][:5] + ["..."])
            
        vec_scaled = self._scaler.transform(vec)

        idx = int(self._model.predict(vec_scaled)[0])
        prob = float(self._model.predict_proba(vec_scaled)[0][idx])

        label = self._class_names[idx]
        return label, prob

    def _handle_flow(self, flow):
        label, confidence = self._predict(flow)
        
        if self._debug:
            self._logger.info(
                "DEBUG [Raw Prediction]: label='%s' prob=%.4f (threshold=%.2f)", 
                label, confidence, self._threshold
            )

        if confidence < self._threshold and not self._debug:
            label = "BENIGN"

        if label == "BENIGN" and not self._debug:
            self._logger.info("BENIGN flow %s → %s", flow["src_ip"], flow["dst_ip"])
        else:
            self._logger.warning(
                "ATTACK DETECTED %s:%s → %s:%s (%s %.2f%%)",
                flow["src_ip"],
                flow["src_port"],
                flow["dst_ip"],
                flow["dst_port"],
                label,
                confidence * 100,
            )

    def run(self):
        self._running = True
        self._file = open(self._snort_log, "r")
        self._file.seek(0, 2)

        while self._running:
            lines = self._file.readlines()
            if lines:
                for line in lines:
                    if line.strip() == "":
                        if self._buffer:
                            alert = self._parser.parse_block(self._buffer)
                            if alert:
                                self._flow_tracker.add_packet(alert)
                            self._buffer = []
                    else:
                        self._buffer.append(line.strip())

            flows = self._flow_tracker.check_timeouts()
            for flow in flows:
                if self._debug:
                    self._logger.info("DEBUG [Flow Finalized]: %s:%s -> %s:%s (%s), pacckets=%d, duration=%.2fus", 
                                      flow["src_ip"], flow["src_port"], flow["dst_ip"], flow["dst_port"], flow["protocol"], flow["packet_count"], flow["duration"])
                self._handle_flow(flow)

            time.sleep(self._poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snort_log", type=Path, default=_DEFAULT_SNORT_LOG)
    parser.add_argument("--models_dir", type=Path, default=_DEFAULT_MODELS_DIR)
    parser.add_argument("--attack_log", type=Path, default=_DEFAULT_ATTACK_LOG)
    parser.add_argument("--poll_interval", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and disable threshold filtering")

    args = parser.parse_args()

    detector = RealTimeDetector(
        args.snort_log,
        args.models_dir,
        args.attack_log,
        args.poll_interval,
        args.threshold,
        args.debug,
    )

    detector.run()


if __name__ == "__main__":
    main()