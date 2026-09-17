"""Cross-platform regression checks for the offline test harness."""

import asyncio
import socket

import pytest
from conftest import _no_network


def test_socketpair_fallback_allowed_but_other_connections_blocked(monkeypatch):
    monkeypatch.undo()  # Reinstall the guard around the emulated implementation.
    # Exercise the loopback implementation Windows uses, even on Unix.
    def fallback_pair():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client = socket.socket()
            try:
                client.connect(listener.getsockname())
                server, _ = listener.accept()
                return server, client
            except BaseException:
                client.close()
                raise

    monkeypatch.setattr(socket, "socketpair", fallback_pair)
    _no_network.__wrapped__(monkeypatch)
    left, right = socket.socketpair()
    with left, right:
        right.sendall(b"ok")
        assert left.recv(2) == b"ok"
    with socket.socket() as other:
        with pytest.raises(AssertionError, match="network request"):
            other.connect(("127.0.0.1", 9))


def test_asyncio_can_create_and_close_its_internal_sockets():
    async def value():
        return 42
    assert asyncio.run(value()) == 42
