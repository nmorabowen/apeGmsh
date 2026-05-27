"""Error and warning types for the geometry composite.

Kept narrow: one class per failure mode, with a clear ``__str__``
that names the offending entity.  Mirrors the
``_compose_errors`` module that hosts the compose-side typed errors
(:class:`ComposeInterfaceSizeWarning`, etc.) — both modules live in
``core/`` so callers in ``core/``, ``mesh/``, and ``opensees/`` can
import them without introducing a layering cycle.
"""
from __future__ import annotations


class GeometryValidationError(RuntimeError):
    """Raised by :meth:`_Geometry.validate_pre_mesh` when the model
    carries orphan geometry that would silently poison meshing.

    Mirrors :meth:`MassesComposite.validate_pre_mesh` /
    :meth:`LoadsComposite.validate_pre_mesh` /
    :meth:`ConstraintsComposite.validate_pre_mesh` — the failure mode
    is the same (a name/state that would crash deeper in the pipeline)
    and the recovery is the same (fix the source then retry).

    The message names the orphan dimtags so the caller can either
    delete them by hand or run :meth:`_Geometry.remove_orphans` to
    sweep them.
    """


class WarnGeomCoincidentFace(UserWarning):
    """Advisory: a cutting plane / surface is coplanar with an
    existing face of an operand within OCC's positional tolerance.

    OCC's :func:`fragment` consumes such a tool but typically leaves a
    free-floating surface at the coincident location.  The post-op
    sweep (:func:`sweep_dangling`) catches the orphan, but the warning
    fires first so users who built the geometry on purpose can
    refactor to avoid the coincidence entirely.

    Subclass of :class:`UserWarning` so it can be silenced with
    ``warnings.simplefilter('ignore', WarnGeomCoincidentFace)`` when
    callers accept the cost.
    """


class WarnGeomOneSidedCut(UserWarning):
    """Advisory: a plane-cut produced fragments on only one side of
    the plane.

    Means the plane sits at or outside the operand's bounding box, so
    the "cut" was effectively a no-op classification (every fragment
    landed on the same side).  Almost always a sign that the
    ``offset`` argument is wrong; the operation still returns the
    expected tuple shape so downstream code that pattern-matches on
    ``(above, below)`` doesn't crash.

    Subclass of :class:`UserWarning` so it can be silenced with
    ``warnings.simplefilter('ignore', WarnGeomOneSidedCut)``.
    """
