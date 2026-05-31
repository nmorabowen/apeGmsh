"""
_SessionBase — shared base for objects that own a Gmsh session.
================================================================

Composite Parent Contract
-------------------------
Composites may access the following on ``self._parent``:

* ``_parent._verbose: bool``   — logging verbosity flag
* ``_parent.name: str``        — session / model name
* ``_parent.is_active: bool``  — property, True when Gmsh session is open
* ``_parent.model``            — the Model composite (Selection uses ``model._metadata``)
* ``_parent.physical``         — PhysicalGroups composite (Selection uses ``physical.add()``)

Subclasses MUST define ``_COMPOSITES`` as a class-level tuple of
``(attr_name, relative_module, class_name, is_optional)`` entries.
"""

from __future__ import annotations

import importlib
import threading
from typing import ClassVar

import gmsh

from ._optional import MissingOptionalDependency


# ----------------------------------------------------------------------
# gmsh init refcount
# ----------------------------------------------------------------------
# gmsh holds a single process-wide runtime: ``gmsh.initialize()`` is
# idempotent (extra calls log a warning) but ``gmsh.finalize()`` tears
# the runtime down regardless of who is still using it.  When sessions
# nest — e.g. a ``Part`` opened inside an ``apeGmsh`` session, or a
# ``from_msh`` helper that pops its own session inside a larger
# workflow — the inner ``end()`` would otherwise finalize gmsh out
# from under the outer session.
#
# The module-level refcount below is shared by every ``_SessionBase``
# subclass and every standalone helper that needs gmsh briefly.  Pair
# every ``_gmsh_acquire()`` with exactly one ``_gmsh_release()``; the
# release is the only thing that may call ``gmsh.finalize()``, and
# only when the refcount returns to zero.
_GMSH_INIT_LOCK = threading.Lock()
_GMSH_INIT_COUNT = 0


def _gmsh_acquire() -> None:
    """Increment the gmsh init refcount.

    Calls ``gmsh.initialize()`` exactly once (the first acquire when
    gmsh is not already initialized).  Subsequent acquires are no-ops
    on the gmsh runtime.  Safe across threads via the lock; safe
    across nested sessions via the refcount.  Idempotent on already-
    initialized gmsh (defensive against external init).
    """
    global _GMSH_INIT_COUNT
    with _GMSH_INIT_LOCK:
        if _GMSH_INIT_COUNT == 0:
            if not gmsh.isInitialized():
                gmsh.initialize()
        _GMSH_INIT_COUNT += 1


def _gmsh_release() -> None:
    """Decrement the gmsh init refcount.

    Calls ``gmsh.finalize()`` only when the last session releases
    (refcount hits 0).  Raises ``RuntimeError`` if the refcount
    underflows — a release without a matching acquire indicates a
    session-lifecycle bug.
    """
    global _GMSH_INIT_COUNT
    with _GMSH_INIT_LOCK:
        if _GMSH_INIT_COUNT <= 0:
            raise RuntimeError(
                "gmsh release without matching acquire — session lifecycle bug"
            )
        _GMSH_INIT_COUNT -= 1
        if _GMSH_INIT_COUNT == 0:
            if gmsh.isInitialized():
                gmsh.finalize()


class _SessionBase:
    """Base class for objects that own a Gmsh session and parent composites."""

    _COMPOSITES: ClassVar[tuple[tuple[str, str, str, bool], ...]] = ()

    def __init__(self, name: str, *, verbose: bool = False) -> None:
        self.name: str = name
        self._verbose: bool = verbose
        self._active: bool = False
        # When True, ``Model._register`` auto-creates a physical group
        # for every entity that has a user-supplied ``label=``.  Set to
        # True only on ``Part`` — the main ``apeGmsh`` session leaves
        # this False so labels in the assembly don't produce unwanted PGs.
        self._auto_pg_from_label: bool = False
        # Pre-declare composite slots as None
        for attr_name, _, _, _ in self._COMPOSITES:
            setattr(self, attr_name, None)

    # ------------------------------------------------------------------
    # Composite parent interface
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when the wrapped Gmsh session is open."""
        return self._active

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin(self, *, verbose: bool | None = None) -> "_SessionBase":
        """Open a Gmsh session, create composites.

        Parameters
        ----------
        verbose : bool or None
            Override the verbosity set in ``__init__``.  ``None`` keeps
            the current value.

        Returns ``self`` for chaining.
        """
        if self._active:
            raise RuntimeError(
                f"{type(self).__name__} '{self.name}' session is already open."
            )
        if verbose is not None:
            self._verbose = verbose
        _gmsh_acquire()
        gmsh.model.add(self.name)
        if self._verbose:
            print(f"Gmsh version: {gmsh.__version__}")
        self._create_composites()
        self._active = True
        return self

    def end(self) -> None:
        """Close the Gmsh session.

        If the subclass set a ``_save_to`` path (autosave configured at
        construction), the broker snapshot is written before
        ``gmsh.finalize()``.  Save failures are logged and swallowed —
        the gmsh process must still finalize.
        """
        if self._active:
            save_to = getattr(self, "_save_to", None)
            if save_to is not None:
                try:
                    # A directory save_to resolves to ``<dir>/<name>.h5``
                    # so autosave matches g.save(); a bare directory
                    # truncate-opens as a file and fails on Windows.
                    resolve = getattr(self, "_resolve_save_target", None)
                    target = resolve(None) if resolve is not None else save_to
                    self._do_save(target)
                except Exception as exc:  # noqa: BLE001
                    import warnings
                    warnings.warn(
                        f"autosave to {save_to} failed: {exc!r}",
                        stacklevel=2,
                    )
            _gmsh_release()
            self._active = False

    # Context-manager support
    def __enter__(self) -> "_SessionBase":
        return self.begin()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Return None (implicitly) rather than False so mypy doesn't
        # flag this as an overly-narrow __exit__ return type.  Python
        # treats a falsy/None return as "propagate the exception".
        self.end()

    # ------------------------------------------------------------------
    # Chain-phase freeze guard (Phase 3B.2d / ADR 0038)
    # ------------------------------------------------------------------

    def _check_chain_phase(self, operation: str) -> None:
        """Raise :class:`ChainPhaseError` when the session is post-extraction.

        A session is in *chain phase* once it has produced its first
        :class:`FEMData` snapshot (``self._fem is not None``).  After
        that the broker is canonical and any geometry / mesh / PG /
        label / parts / sections mutation would silently desync the
        broker from gmsh.  Build-phase APIs guard themselves by
        calling this helper at their entry point with a short
        ``operation`` name for the error message.

        Vanilla sessions / subclasses without ``_fem`` (e.g. low-level
        test fixtures) are not gated — the attribute is treated as
        ``None`` when absent, mirroring the rest of the cache helpers.

        Parameters
        ----------
        operation : str
            Short identifier of the API being called (e.g.
            ``"g.model.geometry.add_box"``, ``"g.mesh.generation.generate"``).
            Surfaces in the error message so users can pinpoint the
            offending call.
        """
        if getattr(self, "_fem", None) is None:
            return
        from .core._compose_errors import ChainPhaseError
        raise ChainPhaseError(
            f"{operation}: model frozen after first get_fem_data() / "
            f"compose; reload from H5 or restart to mutate geometry. "
            f"Chain-phase composition (g.compose) and the "
            f"interface-bridging constraints "
            f"(g.constraints.embedded / tied_contact / equalDOF / "
            f"rigid_link / rigid_diaphragm) remain available."
        )


    # ------------------------------------------------------------------
    # Composite creation
    # ------------------------------------------------------------------

    def _create_composites(self) -> None:
        """Instantiate each composite declared in ``_COMPOSITES``."""
        for attr_name, module_path, class_name, is_optional in self._COMPOSITES:
            try:
                mod = importlib.import_module(module_path, package=__package__)
                cls = getattr(mod, class_name)
                setattr(self, attr_name, cls(self))
            except ImportError as exc:
                if is_optional:
                    setattr(
                        self,
                        attr_name,
                        MissingOptionalDependency(
                            f"{class_name} support",
                            class_name.lower(),
                            extra=attr_name,
                            cause=exc,
                        ),
                    )
                else:
                    raise

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "active" if self._active else "closed"
        return f"{type(self).__name__}('{self.name}', {status})"
