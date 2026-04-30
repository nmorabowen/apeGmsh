"""Make slot failures visible.

Qt's signal-slot dispatcher silently swallows exceptions raised inside
slot callbacks — a click that hits an ImportError or a runtime failure
just looks like nothing happened, with no traceback in either the
console or the window. Same goes for inline lambdas connected to
``QComboBox.currentIndexChanged`` etc.

This module provides:

* :func:`safe_slot` — decorator that wraps a callable so exceptions are
  caught, logged with a full traceback to stderr, and forwarded to any
  registered UI handler.
* :func:`safe_connect` — convenience for connecting a free function or
  lambda where you can't decorate the source.
* :func:`register_error_handler` / :func:`unregister_error_handler` —
  plug in UI delivery (e.g. ``ResultsViewer`` registers a status-bar
  callback at ``show()``).

The decorator is Qt-agnostic and works the same way in headless
tests.
"""
from __future__ import annotations

import functools
import sys
import traceback
from typing import Any, Callable


# Module-level handler registry. UI callers (the active viewer window)
# register; tests use the same hook to capture-and-assert.
_HANDLERS: list[Callable[[str, BaseException], None]] = []


def register_error_handler(cb: Callable[[str, BaseException], None]) -> None:
    """Add a callback that receives ``(slot_name, exception)`` per catch."""
    if cb not in _HANDLERS:
        _HANDLERS.append(cb)


def unregister_error_handler(
    cb: Callable[[str, BaseException], None],
) -> None:
    if cb in _HANDLERS:
        _HANDLERS.remove(cb)


def clear_error_handlers() -> None:
    """Drop every registered handler. Useful for test isolation."""
    _HANDLERS.clear()


def report(name: str, exc: BaseException) -> None:
    """Log a slot failure and notify all registered handlers.

    Always writes to stderr with a full traceback so CI logs and
    consoles capture it. Handler exceptions are themselves caught so a
    bad handler doesn't take the catch-and-report path with it.
    """
    print(
        f"[viewer] {name} raised {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    traceback.print_exception(
        type(exc), exc, exc.__traceback__, file=sys.stderr,
    )
    for cb in list(_HANDLERS):
        try:
            cb(name, exc)
        except Exception as cb_exc:
            print(
                f"[viewer] error handler raised: {cb_exc}",
                file=sys.stderr,
            )


def safe_slot(fn: Callable | None = None) -> Callable:
    """Wrap a slot callable so exceptions are caught and reported.

    Usage::

        @safe_slot
        def _on_add(self) -> None:
            ...

    The wrapped slot returns ``None`` on exception (matching what Qt's
    swallowing default would do anyway), but stderr + registered
    handlers see the failure.
    """
    def _decorate(target: Callable) -> Callable:
        name = getattr(target, "__qualname__", None) or str(target)

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return target(*args, **kwargs)
            except Exception as exc:
                report(name, exc)
                return None

        return wrapper

    if fn is None:
        return _decorate
    return _decorate(fn)


def safe_connect(
    signal: Any,
    slot: Callable,
    *,
    name: str | None = None,
) -> None:
    """Connect a Qt signal to a slot, wrapping it with :func:`safe_slot`.

    Use when the slot is a lambda or a free function you can't decorate
    at the definition site::

        safe_connect(combo.currentIndexChanged, lambda i: refresh(i),
                     name="refresh_on_index_change")
    """
    if name is not None:
        try:
            slot.__name__ = name        # type: ignore[attr-defined]
            slot.__qualname__ = name    # type: ignore[attr-defined]
        except Exception:
            pass
    signal.connect(safe_slot(slot))


__all__ = [
    "clear_error_handlers",
    "register_error_handler",
    "report",
    "safe_connect",
    "safe_slot",
    "unregister_error_handler",
]
