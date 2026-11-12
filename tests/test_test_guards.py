"""The suite's own guard rails (spec §8: no network in tests)."""

import socket

import pytest


def test_outbound_network_is_blocked():
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="network access blocked"):
            s.connect(("1.1.1.1", 443))
    finally:
        s.close()


def test_loopback_is_allowed():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    c = socket.create_connection(("127.0.0.1", port), timeout=1)
    c.close()
    srv.close()
