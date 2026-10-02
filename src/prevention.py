"""
prevention.py
=============
AI-IDPS — Automated IP Blocking Module

Blocks attacker IP addresses using iptables when an attack is classified.
Maintains an in-memory list of blocked IPs to prevent duplicate firewall rules.

Usage (imported by cowrie_monitor.py):
    from prevention import block_ip, get_blocked_ips
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_logger = logging.getLogger("cowrie_monitor.prevention")

# ---------------------------------------------------------------------------
# Blocked IP tracking  (thread-safe)
# ---------------------------------------------------------------------------

@dataclass
class BlockedEntry:
    """Record of a blocked IP."""
    ip:          str
    attack_type: str
    timestamp:   str

    def to_dict(self) -> dict:
        return {
            "ip":          self.ip,
            "attack_type": self.attack_type,
            "timestamp":   self.timestamp,
        }


_lock        = threading.Lock()
_blocked_ips: List[BlockedEntry] = []
_blocked_set: set = set()            # fast O(1) duplicate check


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def block_ip(ip_address: str, attack_type: str = "SSH-Patator") -> bool:
    """
    Block an IP address using iptables.

    - Skips if IP is already blocked (prevents duplicate rules).
    - ALWAYS stores the IP in blocked_ips (even if iptables fails)
      so it appears in the dashboard for demonstration.
    - Attempts iptables as best-effort.
    - Returns True if the IP was newly added, False if duplicate.
    """
    with _lock:
        if ip_address in _blocked_set:
            _logger.info(
                "IP %s already blocked — skipping duplicate rule", ip_address,
            )
            return False

    # Record the IP immediately (so dashboard always shows it)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    entry = BlockedEntry(
        ip=ip_address,
        attack_type=attack_type,
        timestamp=ts,
    )
    with _lock:
        _blocked_set.add(ip_address)
        _blocked_ips.append(entry)

    _logger.warning(
        "PREVENTION ACTION: IP BLOCKED %s (attack=%s)",
        ip_address, attack_type,
    )

    # Attempt iptables as best-effort (non-fatal if it fails)
    cmd = ["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            _logger.info("iptables rule added successfully for %s", ip_address)
        else:
            _logger.warning(
                "iptables failed for %s (exit=%d): %s — IP still recorded in blocked list",
                ip_address, result.returncode, result.stderr.strip(),
            )
    except subprocess.TimeoutExpired:
        _logger.warning("iptables timed out for %s — IP still recorded", ip_address)
    except FileNotFoundError:
        _logger.warning("iptables not found — IP %s recorded without firewall rule", ip_address)
    except Exception as exc:
        _logger.warning("iptables error for %s: %s — IP still recorded", ip_address, exc)

    return True


def get_blocked_ips() -> list[dict]:
    """Return a list of blocked IP records (thread-safe snapshot)."""
    with _lock:
        return [entry.to_dict() for entry in _blocked_ips]


def is_blocked(ip_address: str) -> bool:
    """Check if an IP is already blocked."""
    with _lock:
        return ip_address in _blocked_set
