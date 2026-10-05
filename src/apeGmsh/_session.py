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
import uuid
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

import gmsh

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator
    from pathlib import Path

    from .mesh._geometry_h5_io import GeometryCapture
    from .mesh.FEMData import FEMData

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

# ----------------------------------------------------------------------
# gmsh runtime lock — one thread inside gmsh at a time
# ----------------------------------------------------------------------
# ``_GMSH_INIT_LOCK`` above guards the *refcount*, nothing more.  Two
# threads could each hold a valid session and still interleave their
# ``gmsh.model.*`` calls, and gmsh is one process-global C++ runtime
# that is neither thread-safe nor reentrant: interleaved model calls
# trip a C-level assert and ``abort()`` the interpreter, below the
# reach of any Python ``except``.
#
# This RLock closes that hole.  It is held from ``_gmsh_acquire`` to
# the matching ``_gmsh_release`` — i.e. for a session's whole lifetime
# — so gmsh work from different threads serializes instead of
# interleaving.  Re-entrant because sessions nest (a ``Part`` opened
# inside an ``apeGmsh`` session) and because helpers acquire briefly
# inside a session; those are the same thread and must not self-block.
#
# **The deadlock rule**: never wait on another thread's gmsh work while
# holding this lock.  The one caller that could — the ADR 0080 B6
# properties worker, whose UI thread joins it — takes the lock with a
# timeout via :func:`gmsh_runtime_lock` and reports a busy runtime
# rather than blocking.  A thread must release what it acquired
# (``RLock`` is owner-bound), so a session begun on one thread cannot
# be ended on another.
_GMSH_RUNTIME_LOCK = threading.RLock()


class GmshBusyError(RuntimeError):
    """Another thread holds the gmsh runtime and the wait timed out.

    Raised only by :func:`gmsh_runtime_lock` with a ``timeout``; the
    plain :func:`_gmsh_acquire` path waits as long as it takes.
    """


@contextmanager
def gmsh_runtime_lock(timeout: "float | None" = None) -> "Iterator[None]":
    """Hold the gmsh runtime lock across a block.

    For callers that must **not** block indefinitely — a background
    worker whose UI thread is waiting on it. Sessions opened inside the
    block re-enter the lock for free (same thread), so this is a
    "reserve the runtime, then do gmsh work" wrapper rather than a
    second locking scheme.

    Raises :class:`GmshBusyError` when ``timeout`` elapses first.
    """
    if timeout is None:
        _GMSH_RUNTIME_LOCK.acquire()
    elif not _GMSH_RUNTIME_LOCK.acquire(timeout=timeout):
        raise GmshBusyError(
            f"another thread has held the Gmsh runtime for more than "
            f"{timeout:g} s. Gmsh is a single process-global runtime, so "
            f"only one thread may drive it at a time."
        )
    try:
        yield
    finally:
        _GMSH_RUNTIME_LOCK.release()


def _gmsh_acquire() -> None:
    """Take the gmsh runtime lock and increment the init refcount.

    Calls ``gmsh.initialize()`` exactly once (the first acquire when
    gmsh is not already initialized).  Subsequent acquires are no-ops
    on the gmsh runtime.  Safe across threads via the lock; safe
    across nested sessions via the refcount.  Idempotent on already-
    initialized gmsh (defensive against external init).

    Blocks until any other thread's gmsh work finishes — see the
    runtime-lock note above, and use :func:`gmsh_runtime_lock` when
    waiting forever is not acceptable.
    """
    global _GMSH_INIT_COUNT
    _GMSH_RUNTIME_LOCK.acquire()
    try:
        _gmsh_init_locked()
    except BaseException:
        # nothing was counted, so nothing will release for us
        _GMSH_RUNTIME_LOCK.release()
        raise


def _gmsh_init_locked() -> None:
    """The refcount + ``gmsh.initialize`` half of :func:`_gmsh_acquire`,
    with the runtime lock already held."""
    global _GMSH_INIT_COUNT
    with _GMSH_INIT_LOCK:
        if not gmsh.isInitialized():
            # gmsh runtime is down.  Either this is the first acquire, or
            # it was finalized out-of-band (a direct ``gmsh.finalize()``,
            # a crashed session, or a sibling that bypassed the refcount).
            # Any count we still hold refers to zombie sessions whose gmsh
            # state is already gone, so reset it to track reality before
            # re-initializing — otherwise the stale count would keep this
            # acquire from re-initializing and every later session would
            # operate on a dead runtime.
            _GMSH_INIT_COUNT = 0
            # ``gmsh.initialize`` installs a SIGINT handler when
            # ``interruptible`` is set, and ``signal.signal`` only works
            # on the main thread — so off the main thread (e.g. the ADR
            # 0080 B6 properties worker) request the non-interruptible
            # init. Main-thread behaviour (Ctrl-C aborts a long mesh) is
            # unchanged. Fall back for gmsh builds without the kwarg.
            on_main = threading.current_thread() is threading.main_thread()
            try:
                gmsh.initialize(interruptible=on_main)
            except TypeError:  # pragma: no cover - very old gmsh
                gmsh.initialize()
        _GMSH_INIT_COUNT += 1


def _gmsh_release() -> None:
    """Decrement the gmsh init refcount and drop the runtime lock.

    Calls ``gmsh.finalize()`` only when the last session releases
    (refcount hits 0).  Raises ``RuntimeError`` if the refcount
    underflows — a release without a matching acquire indicates a
    session-lifecycle bug — **without** touching the runtime lock,
    since this thread never took it.
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
    try:
        _GMSH_RUNTIME_LOCK.release()
    except RuntimeError as exc:      # owner-bound: wrong thread released
        raise RuntimeError(
            "gmsh released from a thread that did not acquire it. A Gmsh "
            "session must begin and end on the same thread — the runtime "
            "is process-global and its lock is owner-bound."
        ) from exc


class _SessionBase:
    """Base class for objects that own a Gmsh session and parent composites."""

    _COMPOSITES: ClassVar[tuple[tuple[str, str, str, bool], ...]] = ()

    def __init__(self, name: str, *, verbose: bool = False) -> None:
        self.name: str = name
        self._verbose: bool = verbose
        self._active: bool = False
        # ADR 0112 D1: a session that is *the model* writes its artifacts
        # (``model.h5`` and the ``<stem>.geometry.h5`` sibling) on every
        # ``end()``, unconditionally.  Only ``apeGmsh`` turns this on
        # (its private ``_artifacts=`` constructor flag lets a
        # library-internal session, a section mesh worker or a solver
        # cross-check, opt out; there is no user-facing opt-out).  A
        # ``Part`` or a bare fixture session owns no artifact.  A
        # subclass that turns it on must define ``_resolve_save_target``
        # and ``_do_save``.
        self._writes_artifacts: bool = False
        # ``overwrite=False`` on the constructor is honoured by the
        # automatic write as well as by ``g.save()``.
        self._overwrite: bool = True
        # The session's identity (h5-schema.md, "/meta/session_id"):
        # one uuid4 minted by ``begin()``, handed to every snapshot
        # ``FEMData.from_gmsh`` extracts for this session and stamped
        # on the geometry sibling, so a reader pairs the two files by
        # equality.  ``None`` until ``begin()``.
        self._session_id: str | None = None
        # The geometry captured at the exit of the last successful
        # ``g.mesh.generation.generate(dim >= 2)``; ``None`` means the
        # session never meshed its surfaces and ``end()`` builds a 2-D
        # temporary mesh to capture them (V0 decision 6, amendment 3).
        self._geometry_capture: "GeometryCapture | None" = None
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
        self._session_id = str(uuid.uuid4())
        self._geometry_capture = None
        _gmsh_acquire()
        try:
            gmsh.model.add(self.name)
            if self._verbose:
                print(f"Gmsh version: {gmsh.__version__}")
            self._create_composites()
        except BaseException:
            # ``_active`` is still False, so ``end()`` will never run for
            # this session — release the acquire here or the refcount
            # leaks and gmsh can never finalize for the process lifetime.
            # BaseException: a KeyboardInterrupt mid-begin in a notebook
            # leaves the kernel alive and must not leak either.
            _gmsh_release()
            raise
        self._active = True
        return self

    def end(self) -> None:
        """Close the Gmsh session.

        An artifact-writing session (``_WRITES_ARTIFACTS``, i.e.
        ``apeGmsh``) first writes its artifacts **unconditionally** (ADR
        0112 D1): ``model.h5`` at ``save_to`` or, when none was given, at
        the conventional path ``<artifact dir>/<model_name>.h5``, and the
        geometry sibling ``<stem>.geometry.h5`` beside it.  Every failure
        is a warning, never an exception: the gmsh process must still
        finalize.
        """
        if self._active:
            try:
                if self._writes_artifacts:
                    self._write_artifacts()
            finally:
                # A KeyboardInterrupt during the write must not leak the
                # acquire: release runs whatever happened above.
                _gmsh_release()
                self._active = False

    def _write_artifacts(self) -> None:
        """Write ``model.h5`` and ``<stem>.geometry.h5`` before finalize.

        Order matters: the model file is extracted from the session's
        real state first; the geometry capture may then build and clear
        a temporary 2-D mesh for a session that never meshed.  The
        sibling is stamped with the ``session_id`` of the snapshot the
        model file carries (``self._fem`` after the save), so the two
        files pair by equality even when that snapshot was minted by
        ``compose`` rather than inherited from this session.

        Both files are written to ``<target>.tmp-<uuid>`` beside the
        target and moved into place with ``os.replace``, so a failed
        write leaves the previous file untouched and no temp behind.

        Whether a target may be replaced is the module rule
        :func:`~apeGmsh.mesh._geometry_h5_io._artifact_target_is_ours`,
        shared with the bridge's automatic write (V2d, #1307): a
        ``model.h5`` the bridge already wrote for this session (same
        ``session_id``, every zone this write would produce) is kept
        silently; a foreign file or another session's file is skipped
        with a warning.
        """
        from ._atomic_io import replace_with_retry
        from .mesh._geometry_h5_io import (
            GeometryArtifactWarning,
            _artifact_target_is_ours,
            capture_fallback,
            geometry_sibling_path,
            write_geometry_h5,
        )
        from .opensees._internal.schema_version import GEOMETRY, NEUTRAL, PROVENANCE

        target: "Path | None" = None
        try:
            target = self._resolve_save_target(None)
            fem = self._snapshot_to_save()
            # ``write_fem_h5`` writes the neutral zone and, only when the
            # snapshot carries a provenance table (V2c, ADR 0112 D3),
            # ``/provenance``: a target holding ``/provenance`` that this
            # snapshot would drop is refused like any other zone.
            model_zones = frozenset({NEUTRAL}) | (
                frozenset({PROVENANCE}) if fem.provenance is not None else frozenset()
            )
            if _artifact_target_is_ours(
                target, writes=model_zones, overwrite=self._overwrite,
                session_id=fem.session_id,
            ):
                tmp = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
                try:
                    self._do_save(tmp, fem=fem)
                    replace_with_retry(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"autosave to {target if target is not None else '<unresolved>'} "
                f"failed: {exc!r}",
                stacklevel=3,
            )
        if target is None:
            return
        sibling = geometry_sibling_path(target)
        try:
            if not _artifact_target_is_ours(
                sibling, writes=frozenset({GEOMETRY}), overwrite=self._overwrite,
            ):
                return
            fem = getattr(self, "_fem", None)
            session_id = (
                fem.session_id if fem is not None else self._session_id
            )
            if session_id is None:
                raise RuntimeError("session has no session_id (begin() never ran)")
            capture = self._geometry_capture
            if capture is None:
                capture = capture_fallback()
            from . import __version__ as _ver
            write_geometry_h5(
                sibling, capture,
                session_id=session_id, model_name=self.name,
                apegmsh_version=_ver,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"geometry sibling for {target} not written: {exc!r}",
                GeometryArtifactWarning,
                stacklevel=3,
            )

    def _resolve_save_target(self, path: "str | Path | None") -> "Path":
        """The ``model.h5`` path; defined by the artifact-writing subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} writes no artifacts (_writes_artifacts is False)"
        )

    def _snapshot_to_save(self) -> "FEMData":
        """The snapshot a save writes; defined by the artifact-writing subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} writes no artifacts (_writes_artifacts is False)"
        )

    def _do_save(self, path: "Path", fem: "FEMData | None" = None) -> None:
        """Write the broker snapshot; defined by the artifact-writing subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} writes no artifacts (_writes_artifacts is False)"
        )

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
