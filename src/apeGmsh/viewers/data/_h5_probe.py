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
from typing import Union

_PathLike = Union[str, Path]


def has_opensees_orientation(path: _PathLike) -> bool:
    """Return ``True`` iff ``path`` is a model.h5 carrying both the
    ``/opensees/transforms`` and ``/opensees/element_meta`` groups.

    A missing or unreadable file returns ``False`` without raising —
    the caller's contract is "should I auto-resolve?", not "is this
    file healthy?". The downstream
    :func:`apeGmsh.viewers.data.ViewerData.from_h5` path performs the
    real schema check on open.

    Parameters
    ----------
    path
        Filesystem path to a candidate ``model.h5``.

    Notes
    -----
    Existence of *both* groups is the contract — having transforms but
    not element_meta (or vice versa) cannot resolve a vecxz at all
    (the reader's join requires both, ``h5_reader.py:309-310``). The
    probe stays cheap: it checks for the groups, not their contents,
    so a file with the zones but only ``-1`` sentinel rows still gets
    a ``True`` here and degrades gracefully inside ``from_h5``
    (ADR 0018 INV-11 — "write+degrade on nothing-injected").
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
            return "opensees/transforms" in f and "opensees/element_meta" in f
    except (OSError, KeyError):
        # Not a valid HDF5 file, or any low-level read failure — the
        # caller treats this as "no orientation zone available."
        return False


__all__ = ["has_opensees_orientation"]
