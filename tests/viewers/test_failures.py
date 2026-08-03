"""``viewers/_failures.py`` — safe_slot + handler registry."""
from __future__ import annotations

import pytest

from apeGmsh.viewers._failures import (
    clear_error_handlers,
    register_error_handler,
    report,
    safe_connect,
    safe_slot,
    unregister_error_handler,
)


@pytest.fixture(autouse=True)
def _isolated_handlers():
    """Clear the global handler registry around each test."""
    clear_error_handlers()
    yield
    clear_error_handlers()


def test_safe_slot_swallows_exception_returns_none():
    @safe_slot
    def boom():
        raise RuntimeError("nope")

    # Should not raise.
    result = boom()
    assert result is None


def test_safe_slot_passes_through_normal_return():
    @safe_slot
    def fine():
        return 42

    assert fine() == 42


def test_safe_slot_passes_args_kwargs():
    captured = {}

    @safe_slot
    def echo(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "ok"

    assert echo(1, 2, key="v") == "ok"
    assert captured == {"args": (1, 2), "kwargs": {"key": "v"}}


def test_handler_called_on_failure():
    seen: list[tuple[str, BaseException]] = []
    register_error_handler(lambda name, exc: seen.append((name, exc)))

    @safe_slot
    def boom():
        raise ValueError("kaboom")

    boom()
    assert len(seen) == 1
    name, exc = seen[0]
    assert "boom" in name
    assert isinstance(exc, ValueError)
    assert str(exc) == "kaboom"


def test_handler_not_called_on_success():
    seen: list[tuple[str, BaseException]] = []
    register_error_handler(lambda name, exc: seen.append((name, exc)))

    @safe_slot
    def fine():
        return "ok"

    fine()
    assert seen == []


def test_unregister_handler():
    cb_called: list[bool] = []
    cb = lambda name, exc: cb_called.append(True)
    register_error_handler(cb)
    unregister_error_handler(cb)

    @safe_slot
    def boom():
        raise RuntimeError("x")

    boom()
    assert cb_called == []


def test_handlers_isolated_from_each_other():
    """A handler raising must not block other handlers."""
    seen: list[str] = []

    def good_handler(name, exc):
        seen.append("good")

    def bad_handler(name, exc):
        raise RuntimeError("handler bug")

    register_error_handler(bad_handler)
    register_error_handler(good_handler)

    @safe_slot
    def boom():
        raise ValueError("x")

    boom()
    assert "good" in seen


def test_safe_connect_wraps_lambda(monkeypatch):
    """safe_connect should let a lambda raise without propagating."""
    seen: list[str] = []

    class _FakeSignal:
        def __init__(self):
            self._slot = None

        def connect(self, slot):
            self._slot = slot

        def emit(self, *args):
            return self._slot(*args)

    register_error_handler(lambda name, exc: seen.append(name))
    sig = _FakeSignal()
    safe_connect(sig, lambda x: 1 / 0, name="safe_connect_test")

    # Should not raise.
    sig.emit(0)
    assert seen and seen[0] == "safe_connect_test"


def test_safe_slot_preserves_function_name():
    @safe_slot
    def my_named_slot():
        pass

    assert my_named_slot.__name__ == "my_named_slot"


# ── ADR 0084 D4 — the strict pump-failure collector ────────────────────
# ``_isolated_handlers`` above clears the registry, which drops the
# autouse ``pump_failures`` collector; re-register it explicitly so
# these tests exercise the real report path regardless of ordering.


@pytest.mark.allow_pump_failures
def test_pump_report_reaches_the_strict_collector(pump_failures):
    """What a pump catch site emits is what the fixture collects."""
    register_error_handler(pump_failures)
    report("pump.step", RuntimeError("layer blew up"))

    assert [name for name, _ in pump_failures.failures] == ["pump.step"]
    assert isinstance(pump_failures.failures[0][1], RuntimeError)
    assert "layer blew up" in pump_failures.summary()


def test_strict_collector_ignores_ordinary_slot_failures(pump_failures):
    """Only ``pump.*`` reports arm the teardown gate."""
    register_error_handler(pump_failures)
    report("some_button_slot", ValueError("unrelated"))

    assert pump_failures.failures == []
