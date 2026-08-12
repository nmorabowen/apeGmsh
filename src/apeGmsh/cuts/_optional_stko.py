"""Lazy import shim for the optional ``STKO_to_python`` dependency.

``apeGmsh.cuts`` produces :class:`SectionCutDef` objects from CAD-level
geometry. The handoff to STKO_to_python — converting to its
``SectionCutSpec`` — only matters when the user actually wants to run a
cut against MPCO output. Modeling-only users shouldn't be forced to
install STKO_to_python and its h5py / pandas / matplotlib transitive
deps just to import ``apeGmsh``.

This module owns the single point where STKO_to_python is imported and
keeps the error message consistent. Callers do::

    from apeGmsh.cuts._optional_stko import load_stko_cuts
    cuts_mod = load_stko_cuts()        # raises ImportError if missing
    plane = cuts_mod.Plane(...)
    spec = cuts_mod.SectionCutSpec(...)
"""
from __future__ import annotations

from types import ModuleType


_INSTALL_HINT = (
    "STKO_to_python is required to convert SectionCutDef to a "
    "SectionCutSpec. Install with `pip install STKO_to_python` or, "
    "for an editable dev install, clone the repo and "
    "`pip install -e .` from its root."
)


def load_stko_cuts() -> ModuleType:
    """Return the ``STKO_to_python.cuts`` module, importing it lazily.

    Raises
    ------
    ImportError
        If STKO_to_python is not installed. The message includes a
        ``pip install`` hint.
    """
    try:
        from STKO_to_python import cuts as cuts_mod  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return cuts_mod
