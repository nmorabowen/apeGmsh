"""Existence probe for the ``model.h5`` OpenSees orientation zone.

The post-solve :class:`apeGmsh.viewers.ResultsViewer` auto-resolves an
effective ``model_h5`` from ``results._path`` when the results were
opened from disk (``Results.from_native``) **and** the file actually
carries beam-orientation data — the
``/opensees/transforms`` + ``/opensees/element_meta`` pair that
:meth:`apeGmsh.opensees.emitter.h5_reader.H5Model.element_local_axes_vecxz`
joins on. This module is the single predicate that gates that fallback.

Producer-agnostic: both the bridge writer
(``apeSees(fem).h5()``) and :class:`apeGmsh.opensees.ModelData` write
the same byte-equivalent zone (ADR 0018 INV-16), so the probe needs no
provenance check — it asks the file directly.

Probe MUST use ``in`` (HDF5 ``H5Lexists``), never ``Group.get(...)`` —
see project memory ``project_h5py_optional_child_get_hazard`` and PR
[#261](https://github.com/nmorabowen/apeGmsh/pull/261) for the regression
this rule prevents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

_PathLike = Union[str, Path]


def has_opensees_orientation(path: _PathLike) -> bool:
    """Return ``True`` iff ``path`` is a model.h5 carrying both the
    ``/opensees/transforms`` and ``/opensees/element_meta`` groups.

    A missing or unreadable file returns ``False`` without raising —
    the caller's contract is "should I auto-resolve?", not "is this
    file healthy?".

    ADR 0026 PR7-c — the probe logic now lives on
    :meth:`apeGmsh.opensees.emitter.h5_reader.H5Model.has_opensees_orientation`
    (the method on an already-open reader).  This standalone function
    is a thin shim for path-only callers: it confirms the file exists
    and is openable as raw h5py (the cheap probe — no schema
    validation), then defers to the same group-membership check.

    Producer-agnostic: both the bridge writer (``apeSees(fem).h5()``)
    and :class:`apeGmsh.opensees.ModelData` write the same
    byte-equivalent zone (ADR 0018 INV-16), so the probe needs no
    provenance check — it asks the file directly.

    Parameters
    ----------
    path
        Filesystem path to a candidate ``model.h5``.
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(str(p), "r") as f:
            # The single-source-of-truth membership check.  Keeping it
            # at raw-h5py rather than ``h5_reader.open`` preserves the
            # original "no schema validation, no SchemaVersionError"
            # contract — a stale or wrong-version file still returns
            # ``True`` if both groups are present, matching the
            # method's behaviour on an open reader.
            return "opensees/transforms" in f and "opensees/element_meta" in f
    except (OSError, KeyError):
        # Not a valid HDF5 file, or any low-level read failure — the
        # caller treats this as "no orientation zone available."
        return False


def _is_readable_h5(path: Path) -> bool:
    """Cheap "can the viewer read this at all?" probe — file exists and
    opens as HDF5. No group or schema checks (the paired branch's gate
    is coverage-based, not content-based); guards only against on-disk
    corruption/replacement between ``Results`` construction and scene
    build, degrading to the ``from_fem`` fallback instead of raising."""
    if not path.is_file():
        return False
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(str(path), "r"):
            return True
    except OSError:
        return False


def resolve_orientation_source(results: Any) -> Optional[Path]:
    """Return the file path that carries an orientation zone for
    ``results``, or ``None`` when the viewer must degrade.

    Single source of truth for the "should we read from the file or
    fall back to the live FEMData?" decision that the post-solve
    viewer makes in two places — :meth:`ResultsViewer._build_viewer_data`
    (scene snapshot) and :meth:`ResultsViewer._apply_pending_cuts`
    (director binding).  Both paths previously inlined the same
    ``Path + has_opensees_orientation`` block, drifting independently.

    The probe gate is, in order:

    * **Paired open** (ADR 0043 slice 1.3): when element-level reads on
      ``results`` come back relabelled into ``fem_eid`` space
      (``results._element_reads_fem_keyed()`` — the reader's attached
      pairing covers every recorded element id, so the all-or-nothing
      translation actually fires), the scene MUST be built from the
      ``model_h5=`` sibling (``results._model_path``, whose neutral
      ``/mesh`` zone is fem_eid-keyed), never from the reader's
      ops-tag-keyed embedded snapshot. A merely *attached* pairing is
      not enough: a deliberately-unrelated stub ``model_h5=`` (the
      sanctioned test-suite pattern) attaches but never fires — reads
      stay in ops space and the embedded-snapshot fallback below stays
      the consistent choice. This branch deliberately does NOT require
      :func:`has_opensees_orientation`: a solid-only model records
      ``element_meta`` but no ``transforms`` group, and every consumer
      of the returned path degrades gracefully without it
      (``ViewerData.from_h5`` → empty ``vecxz``). The file must still
      open as HDF5 (guards on-disk corruption after open).
    * Otherwise ``results._path`` must be a non-None filesystem path
      (in-memory ``Results`` and recorder flavours yield ``None``) and
      satisfy :func:`has_opensees_orientation` (both
      ``/opensees/transforms`` and ``/opensees/element_meta`` groups
      present).

    Returns ``None`` if every gate fails. Producer-agnostic by
    construction — the probe inspects the file, not its provenance.

    .. note::
       ADR 0026 (proposed) widens this to return an ``H5ModelReader``
       instead of a ``Path``. Until that lands, the return is the
       ``Path`` consumed by :func:`ViewerData.from_h5` and
       :func:`FemToOpsTagMap.from_h5`.  The signature change at
       adoption time is a one-line update at each call site.

    Parameters
    ----------
    results
        Any object exposing a ``_path`` attribute — typically a
        :class:`apeGmsh.results.Results` instance.  Duck-typed via
        :func:`getattr`; no import edge into :mod:`apeGmsh.results`
        is created (ADR 0014 INV-1 is preserved).
    """
    # Paired mpco/ladruno open — when the reader relabels element reads
    # into fem_eid space, the id space the scene joins against must come
    # from the model.h5 that defined the pairing. Duck-typed like the
    # rest of this module (no import edge into apeGmsh.results):
    # ``Results._element_reads_fem_keyed`` owns the coverage semantics.
    model_path = getattr(results, "_model_path", None)
    fem_keyed = getattr(results, "_element_reads_fem_keyed", None)
    if model_path is not None and callable(fem_keyed):
        try:
            keyed = bool(fem_keyed())
        except Exception:
            keyed = False
        if keyed:
            mp = Path(model_path)
            if _is_readable_h5(mp):
                return mp
    results_path = getattr(results, "_path", None)
    if results_path is None:
        return None
    p = Path(results_path)
    if not has_opensees_orientation(p):
        return None
    return p


__all__ = ["has_opensees_orientation", "resolve_orientation_source"]
