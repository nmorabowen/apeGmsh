"""
sections — parametric structural cross-section factories.
==========================================================

Each function returns a :class:`~apeGmsh.core.Part.Part` with
labeled volumes/surfaces ready for assembly placement.

Solid sections (3D volumes, hex-compatible via internal slicing)::

    from apeGmsh.sections import (
        W_solid, rect_solid, rect_hollow,
        pipe_solid, pipe_hollow,
        angle_solid, channel_solid, tee_solid,
    )

Shell sections (3D mid-surface rectangles for shell elements)::

    from apeGmsh.sections import W_shell

Profile-only (2D cross-section for fiber analysis or sweep)::

    from apeGmsh.sections import W_profile
"""

from .solid import (
    W_solid,
    angle_solid,
    channel_solid,
    pipe_hollow,
    pipe_solid,
    rect_hollow,
    rect_solid,
    tee_solid,
)
from .shell import W_shell
from .profile import W_profile

__all__ = [
    "W_solid",
    "W_shell",
    "W_profile",
    "rect_solid",
    "rect_hollow",
    "pipe_solid",
    "pipe_hollow",
    "angle_solid",
    "channel_solid",
    "tee_solid",
]
