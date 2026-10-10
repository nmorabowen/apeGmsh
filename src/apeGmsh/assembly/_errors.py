"""The assembly's error type (ADR 0117)."""
from __future__ import annotations

__all__ = ["AssemblyError"]


class AssemblyError(Exception):
    """Raised for an invalid :class:`~apeGmsh.assembly.Assembly`
    declaration, an archive that fails validation, or a ``bridge()`` that
    cannot build what was declared."""
