"""JSON-over-HTTP services: binding a fixed port that a stopped service still holds."""

from __future__ import annotations

import socket
import threading

import pytest

from steward.net import JsonServer


def _held_port() -> socket.socket:
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    return held


def test_a_fixed_port_freed_within_the_wait_is_bound():
    held = _held_port()
    port = held.getsockname()[1]
    threading.Timer(0.6, held.close).start()
    server = JsonServer({}, port=port, name="probe", bind_wait_s=5.0).start()
    try:
        assert server.port == port
    finally:
        server.stop()


def test_a_fixed_port_still_held_after_the_wait_fails_loudly():
    held = _held_port()
    try:
        with pytest.raises(OSError):
            JsonServer({}, port=held.getsockname()[1], name="probe", bind_wait_s=0.6)
    finally:
        held.close()
