"""Shared test setup.

Three guarantees for the whole suite:

  * it runs against a throwaway PEIYIN_HOME, so it never reads or writes a
    real installation's config, caches or output folder;
  * a known profile is active, so cast-dependent assertions are stable;
  * the network is blocked, so a test that accidentally reaches for an API
    fails loudly instead of making a real call.
"""

import os
import tempfile

os.environ["PEIYIN_HOME"] = tempfile.mkdtemp(prefix="peiyin-tests-")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="peiyin-mpl-tests-")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test in this suite is offline. Nothing here should ever call out."""
    import socket
    from contextvars import ContextVar

    import httpx

    for name in ("FISH_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def _blocked(*a, **k):
        raise AssertionError(
            "a test tried to make a network request; mock it instead")

    # Windows implements socketpair with a loopback connection. asyncio needs
    # this internal pair even when the test makes no network requests.
    making_pair = ContextVar("making_socketpair", default=False)
    original_pair = socket.socketpair
    original_connect = socket.socket.connect

    def _socketpair(*args, **kwargs):
        token = making_pair.set(True)
        try:
            return original_pair(*args, **kwargs)
        finally:
            making_pair.reset(token)

    def _connect(sock, address):
        if making_pair.get():
            return original_connect(sock, address)
        return _blocked()

    monkeypatch.setattr(socket, "socketpair", _socketpair)
    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(httpx.Client, "send", _blocked)
    monkeypatch.setattr(httpx.Client, "request", _blocked)
    monkeypatch.setattr(httpx.Client, "get", _blocked)
    monkeypatch.setattr(httpx.Client, "post", _blocked)


@pytest.fixture(autouse=True)
def _active_profile():
    """Pin the example profile, so cast-dependent assertions are stable."""
    from peiyin import profiles
    profiles.set_active_profile("example_show")
    yield
    profiles.active.cache_clear()


@pytest.fixture
def tmp(tmp_path):
    """A scratch directory, named as the original self-test named it."""
    return tmp_path


@pytest.fixture
def tone_wav():
    """Write a harmonic tone at a known pitch, for the f0 / gender checks."""
    def _write(path, f0, dur=2.0, sr=22050):
        tt = np.linspace(0, dur, int(sr * dur), endpoint=False)
        x = sum((0.6 / h) * np.sin(2 * np.pi * f0 * h * tt) for h in range(1, 5))
        sf.write(str(path), (0.3 * x / np.max(np.abs(x))).astype("float32"), sr)
        return path
    return _write


@pytest.fixture
def synthetic_profile():
    """A profile with an invented cast, so behaviour is tested against the
    profile system rather than against any particular show."""
    from peiyin import profiles
    pid = "synthetic-test"
    try:
        profiles.create_profile(pid, show_name="Synthetic Test")
    except profiles.ProfileError:
        pass
    profiles.set_active_profile(pid)
    return profiles.active()
