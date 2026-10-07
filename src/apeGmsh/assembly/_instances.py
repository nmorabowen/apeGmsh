"""Declarations of an :class:`~apeGmsh.assembly.Assembly`: instances and ties.

Both are frozen records validated when they are declared, so a refused
call leaves the assembly unchanged (ADR 0117 INV-1, INV-7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

from ._v1 import AssemblyError

__all__ = [
    "Instance", "Tie", "check_label", "check_rotate", "check_translate",
    "split_port",
]


@dataclass(frozen=True)
class Instance:
    """One placed copy of a saved ``model.h5`` (ADR 0117 D1).

    ``rotate`` is ``((ax, ay, az), theta)`` with ``theta`` in radians,
    applied about the origin before ``translate``.
    """

    label: str
    source: Path
    translate: tuple[float, float, float]
    rotate: "tuple[tuple[float, float, float], float] | None"

    def compose_rotate(self) -> "tuple[float, float, float, float] | None":
        """``rotate`` in the merge engine's axis-angle form ``(x, y, z, theta)``."""
        if self.rotate is None:
            return None
        (ax, ay, az), theta = self.rotate
        return (ax, ay, az, theta)


@dataclass(frozen=True)
class Tie:
    """One assembly-level ``tie`` between two instance ports (ADR 0117 D3)."""

    master: str
    slave: str
    enforce: str
    method: str
    dofs: "tuple[int, ...] | None"
    tolerance: float
    name: "str | None"
    #: The ``TieDef`` built (and so validated) when ``tie()`` was called.
    definition: Any = field(compare=False, repr=False)


def check_label(label: object, *, what: str) -> str:
    """Return ``label`` if it is a valid instance or assembly-object name.

    The rule is the merge engine's (``Compose._validate_label``), checked
    here so ``instance()`` refuses before it records anything: a
    non-empty string with no ``.``, ``/`` or whitespace, and no leading
    or trailing ``_``.
    """
    if not isinstance(label, str) or not label:
        raise AssemblyError(f"{what} must be a non-empty string, got {label!r}.")
    if "." in label or "/" in label:
        raise AssemblyError(
            f"{what} {label!r} contains '.' or '/'; an instance-owned name "
            f"is '{{instance}}.{{name}}', so neither may appear in a label."
        )
    if any(ch.isspace() for ch in label):
        raise AssemblyError(f"{what} {label!r} contains whitespace.")
    if label.startswith("_") or label.endswith("_"):
        raise AssemblyError(f"{what} {label!r} cannot start or end with '_'.")
    return label


def check_rotate(
    rotate: object,
) -> "tuple[tuple[float, float, float], float] | None":
    """Normalise ``rotate=((ax, ay, az), theta)``; refuse a zero axis and
    a non-finite (``nan``, ``inf``) component."""
    if rotate is None:
        return None
    try:
        axis, theta = cast("tuple[Sequence[float], float]", rotate)
        ax, ay, az = (float(a) for a in axis)
        th = float(theta)
    except (TypeError, ValueError) as exc:
        raise AssemblyError(
            f"rotate={rotate!r}: expected ((ax, ay, az), theta_radians)."
        ) from exc
    if not all(math.isfinite(v) for v in (ax, ay, az, th)):
        raise AssemblyError(
            f"rotate={rotate!r}: the axis and angle must be finite numbers.")
    if math.hypot(ax, ay, az) == 0.0:
        raise AssemblyError(f"rotate={rotate!r}: the rotation axis is zero.")
    return ((ax, ay, az), th)


def check_translate(translate: Sequence[float]) -> tuple[float, float, float]:
    """Normalise ``translate=(x, y, z)``; refuse a non-finite component."""
    try:
        t = tuple(float(v) for v in translate)
    except (TypeError, ValueError) as exc:
        raise AssemblyError(
            f"translate={translate!r}: expected (x, y, z).") from exc
    if len(t) != 3:
        raise AssemblyError(f"translate={translate!r}: expected (x, y, z).")
    if not all(math.isfinite(v) for v in t):
        raise AssemblyError(
            f"translate={translate!r}: every component must be a finite number.")
    return (t[0], t[1], t[2])


def split_port(port: object, labels: Sequence[str]) -> tuple[str, str]:
    """Split ``"{instance}.{pg|label}"`` on its first dot.

    A port with no dot names an assembly-owned object; P1 declares none,
    so it raises listing the instances. The instance must already be
    declared, and the local name must be non-empty.
    """
    if not isinstance(port, str) or not port:
        raise AssemblyError(f"port must be a non-empty string, got {port!r}.")
    inst, dot, local = port.partition(".")
    if not dot:
        raise AssemblyError(
            f"port {port!r} names no assembly object (this assembly declares "
            f"none); an instance port is '{{instance}}.{{pg}}', with instances "
            f"{list(labels)}."
        )
    if inst not in labels:
        raise AssemblyError(
            f"port {port!r} names unknown instance {inst!r}; declared "
            f"instances are {list(labels)}."
        )
    if not local:
        raise AssemblyError(
            f"port {port!r} has an empty name after the instance label."
        )
    return inst, local
