"""OutputDock — a docked log panel showing color-coded captured messages.

Composes the content widget of a dock (toolbar + read-only text view)
that consumes a :class:`LogRouter`'s ``message`` signal and appends
each ``(text, severity)`` pair color-coded:

* ``"info"``    — palette ``overlay`` (muted)
* ``"warning"`` — palette ``warning``
* ``"error"``   — palette ``error``

Also tracks running counts per severity (``self.counts``) for an
external status-bar badge to consult.

Usage::

    router = LogRouter()
    router.install()
    output = OutputDock(router)        # → a QWidget
    # Embed it in a QDockWidget yourself, or register via DockSpec:
    dock_spec = output_dock_spec(router)
    results_window = ResultsWindow(
        title="...",
        extension_docks=[dock_spec],
    )
    # On viewer close:
    router.uninstall()

The dock should start hidden by default — the View menu toggle and
the (future) status-bar badge are the way users surface it. Auto-
raising on every message would yank focus during scrubbing.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _severity_colors() -> "dict[str, str]":
    """Per-severity text colors from the ACTIVE palette's semantic
    roles (ADR 0087 D1.1) — info reads muted, warning/error semantic.

    Resolved per append so the colors track theme switches for new
    messages; already-appended spans keep the colors they were logged
    under (same as before, when the colors were fixed hex).
    """
    from .theme import THEME
    p = THEME.current
    return {"info": p.overlay, "warning": p.warning, "error": p.error}

# Max number of message blocks before the oldest are dropped — keeps
# memory bounded for long-running viewers without losing recent
# debugging info.
_MAX_BLOCKS = 2000


class OutputDock:
    """Content widget for the Output dock. Wraps the inner text view +
    a small toolbar.

    Not a ``QObject`` / ``QWidget`` subclass itself — composes one.
    The exposed :attr:`widget` is the ``QWidget`` ready to be put inside
    a ``QDockWidget`` (or registered via :func:`output_dock_spec`).

    Parameters
    ----------
    router
        :class:`LogRouter` whose ``message`` signal feeds this dock.
        The dock connects to ``router.message`` on construction and
        disconnects in :meth:`close`.
    parent
        Optional Qt parent.
    """

    def __init__(self, router: Any, *, parent: Any = None) -> None:
        from qtpy import QtWidgets

        # ── Toolbar — slim header row, Clear right-aligned ────────
        toolbar = QtWidgets.QWidget(parent)
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.setSpacing(4)
        clear_btn = QtWidgets.QPushButton("Clear", toolbar)
        clear_btn.setFlat(True)
        clear_btn.setToolTip("Clear output log")
        clear_btn.clicked.connect(self.clear)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(clear_btn)

        # ── Text view — themed via the OutputConsole QSS block in
        # theme.py (surface bg, mono stack, palette text; ADR 0087
        # INV-6/INV-7), no local font / stylesheet.
        text = QtWidgets.QPlainTextEdit(parent)
        text.setObjectName("OutputConsole")
        text.setReadOnly(True)
        text.setUndoRedoEnabled(False)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text.setMaximumBlockCount(_MAX_BLOCKS)

        # ── Outer container ──────────────────────────────────────
        container = QtWidgets.QWidget(parent)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(toolbar)
        container_layout.addWidget(text, stretch=1)

        # ── State ────────────────────────────────────────────────
        self._text = text
        self._clear_btn = clear_btn
        self._router = router
        self._widget = container
        # Per-severity counts. Plain dict — no synchronization (Qt
        # marshals append() to UI thread).
        self._counts: dict[str, int] = {"info": 0, "warning": 0, "error": 0}
        # Observers called on every append (for status-bar badge etc.).
        self._on_append_observers: list[Callable[[str, str], None]] = []
        # Observers fired when counts change for any reason — append OR
        # clear. Status-bar badges subscribe here so the badge resets to
        # zero when the user hits Clear, not just on the next append.
        self._on_counts_changed_observers: list[Callable[[], None]] = []

        # ── Wire to router ───────────────────────────────────────
        router.message.connect(self.append)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self) -> Any:
        """The container ``QWidget`` — embed in a dock / panel."""
        return self._widget

    @property
    def text_view(self) -> Any:
        """The inner ``QPlainTextEdit`` (for tests / introspection)."""
        return self._text

    @property
    def counts(self) -> dict[str, int]:
        """Read-only snapshot of ``{severity → count}``.

        Status-bar badges can poll this; the dict is rebuilt on
        read so callers can mutate the returned copy freely.
        """
        return dict(self._counts)

    def append(self, text: str, severity: str) -> None:
        """Append a color-coded message to the log.

        Slot for :attr:`LogRouter.message`. Severity must be one of
        ``"info"`` / ``"warning"`` / ``"error"`` — unknown values are
        treated as ``"info"``.
        """
        if not text:
            return
        if severity not in self._counts:
            severity = "info"
        self._counts[severity] += 1
        colors = _severity_colors()
        color = colors.get(severity, colors["info"])

        # HTML-escape so messages containing < or > don't get parsed
        # as tags; preserve newlines.
        from html import escape
        safe = escape(text).replace("\n", "<br/>")
        html = (
            f'<span style="color:{color};white-space:pre;">'
            f'{safe}</span>'
        )
        self._text.appendHtml(html)
        # Scroll to bottom — newest message in view.
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

        # Notify observers (status-bar badge, …).
        for obs in list(self._on_append_observers):
            try:
                obs(text, severity)
            except Exception:
                pass
        self._fire_counts_changed()

    def clear(self) -> None:
        """Wipe the log buffer and reset counts."""
        self._text.clear()
        self._counts = {"info": 0, "warning": 0, "error": 0}
        self._fire_counts_changed()

    def on_append(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback fired on every :meth:`append`.

        Receives ``(text, severity)``. Used by status-bar badges or
        similar peripherals. Registering the same callback twice
        subscribes it twice.
        """
        self._on_append_observers.append(callback)

    def on_counts_changed(self, callback: Callable[[], None]) -> None:
        """Register a callback fired whenever counts change.

        Fires on both :meth:`append` and :meth:`clear`. No arguments —
        the consumer reads :attr:`counts` to learn the new state.
        Use this for widgets that need to refresh even when the cause
        of the change is a Clear (not an Append).
        """
        self._on_counts_changed_observers.append(callback)

    def _fire_counts_changed(self) -> None:
        for obs in list(self._on_counts_changed_observers):
            try:
                obs()
            except Exception:
                pass

    def close(self) -> None:
        """Disconnect from the router. Idempotent.

        Doesn't destroy the widget — Qt's parent ownership handles
        teardown. Just severs the signal so the router can outlive
        the dock cleanly.
        """
        import warnings
        # PySide6 raises RuntimeWarning (not RuntimeError) when
        # disconnecting an already-disconnected signal. Suppress it
        # to keep close() truly idempotent under both bindings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                self._router.message.disconnect(self.append)
            except (TypeError, RuntimeError):
                # PyQt raises TypeError; some bindings raise RuntimeError.
                pass


def make_output_dock(
    router: Any,
    *,
    dock_id: str = "dock_output",
    title: str = "Output",
    default_area: str = "bottom",
    default_visible: bool = False,
    tabify_with: Optional[str] = None,
) -> tuple["OutputDock", Any]:
    """Construct an :class:`OutputDock` and a :class:`DockSpec` for it.

    Returns ``(output_dock, dock_spec)`` so the caller can hold a
    reference to the dock (for :attr:`OutputDock.counts`,
    :meth:`OutputDock.clear`, :meth:`OutputDock.close`) while passing
    the spec to :class:`ResultsWindow`'s ``extension_docks=`` argument.

    Defaults to bottom-area, hidden — discoverable through the View
    menu toggle without grabbing screen real estate at startup.

    Example::

        router = LogRouter(); router.install()
        output, spec = make_output_dock(router)
        window = ResultsWindow(title="...", extension_docks=[spec])
        # `output` is alive; .counts, .clear(), .close() all work.
    """
    from ._dock_registry import DockSpec

    output = OutputDock(router)

    # The factory ignores the parent arg — the widget was constructed
    # parentless and Qt reparents on dock.setWidget(). Capturing the
    # pre-built widget closure-keeps it alive for the caller.
    def _factory(parent: Any) -> Any:    # noqa: ARG001
        return output.widget

    spec = DockSpec(
        dock_id=dock_id,
        title=title,
        factory=_factory,
        default_area=default_area,
        default_visible=default_visible,
        tabify_with=tabify_with,
    )
    return output, spec
