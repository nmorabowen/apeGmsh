"""Assembly couplings beyond ``tie`` (ADR 0117 D3).

Each coupling verb of :class:`~apeGmsh.assembly.Assembly` records its
options as one JSON object, the ``params`` of its ``/assembly/ties`` row,
and :func:`coupling_definition` builds the chain-phase constraint def from
exactly that object. The verb and ``Assembly.from_h5`` both go through it,
so a row the verb would refuse is refused on read as well, and a re-listed
coupling carries the def the declared one did.

=========================  =================================  ===============
kind                       def (``_chain_phase_router``)      master / slave
=========================  =================================  ===============
``equal_dof``              ``EqualDOFDef``                    port / port
``rigid_link``             ``RigidLinkDef``                   port / port
``rigid_diaphragm``        ``RigidDiaphragmDef``              port / port
``embedded``               ``EmbeddedDef``                    host / embedded
``kinematic_coupling``     ``KinematicCouplingDef`` (RBE2)    reference / target
``distributing_coupling``  ``DistributingCouplingDef`` (RBE3) reference / target
=========================  =================================  ===============

A port is ``{instance}.{pg|label}`` or, where the verb accepts one, the name
of an assembly reference node (``Assembly.node``). The reference of an RBE2
or RBE3 coupling is always a reference node, and its coordinates are the
def's ``master_point``.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

from ._h5 import TIE_PARAMS
from ._v1 import AssemblyError

__all__ = [
    "COUPLING_KINDS", "NODE_PORTS", "canonical_params", "coupling_definition",
]

#: The coupling kinds, in the order of the table above.
COUPLING_KINDS: tuple[str, ...] = (
    "equal_dof", "rigid_link", "rigid_diaphragm", "embedded",
    "kinematic_coupling", "distributing_coupling",
)

#: Per kind, whether the master / slave port may name a reference node
#: instead of an instance port. ``embedded`` and the RBE2 / RBE3 targets
#: resolve elements or a node set of an instance, so they never do.
NODE_PORTS: dict[str, tuple[bool, bool]] = {
    "equal_dof": (True, True),
    "rigid_link": (True, True),
    "rigid_diaphragm": (True, True),
    "embedded": (False, False),
    "kinematic_coupling": (True, False),
    "distributing_coupling": (True, False),
}


def canonical_params(params: Mapping[str, Any]) -> str:
    """The canonical JSON of ``params``: sorted keys, no spaces."""
    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"))


def coupling_definition(
    kind: str,
    master: str,
    slave: str,
    params: Mapping[str, Any],
    name: "str | None",
    nodes: Mapping[str, tuple[float, float, float]],
) -> Any:
    """Build the constraint def of one coupling; raise for a bad option.

    ``nodes`` maps each declared reference node to its coordinates. Raises
    :class:`AssemblyError` for an unknown kind, params whose keys are not
    the kind's (``TIE_PARAMS``), an invalid option, and an RBE2 / RBE3
    reference that is not a declared reference node.
    """
    from apeGmsh._kernel.defs import constraints as c

    what = f"{kind}({master!r}, {slave!r})"
    if kind not in COUPLING_KINDS:
        raise AssemblyError(
            f"{what}: unknown coupling kind; the kinds are {list(COUPLING_KINDS)}.")
    if set(params) != TIE_PARAMS[kind]:
        raise AssemblyError(
            f"{what}: params carry {sorted(params)}, expected "
            f"{sorted(TIE_PARAMS[kind])}."
        )
    p = dict(params)
    try:
        if kind == "equal_dof":
            return c.EqualDOFDef(
                master_label=master, slave_label=slave, name=name,
                dofs=_dofs(p["dofs"], "dofs", optional=True),
                tolerance=_positive(p["tolerance"], "tolerance"),
            )
        if kind == "rigid_link":
            if p["link_type"] not in ("beam", "rod"):
                raise AssemblyError(
                    f"link_type={p['link_type']!r}: expected 'beam' or 'rod'.")
            mp = p["master_point"]
            return c.RigidLinkDef(
                master_label=master, slave_label=slave, name=name,
                link_type=p["link_type"],
                master_point=None if mp is None else _point(mp, "master_point"),
            )
        if kind == "rigid_diaphragm":
            normal = _point(p["plane_normal"], "plane_normal")
            if math.hypot(*normal) == 0.0:
                raise AssemblyError(f"plane_normal={normal!r} is zero.")
            return c.RigidDiaphragmDef(
                master_label=master, slave_label=slave, name=name,
                master_point=_point(p["master_point"], "master_point"),
                plane_normal=normal,
                constrained_dofs=_required_dofs(
                    p["constrained_dofs"], "constrained_dofs"),
                plane_tolerance=_positive(p["plane_tolerance"], "plane_tolerance"),
            )
        if kind == "embedded":
            stiffness = p["stiffness"]
            if stiffness != "auto":
                stiffness = _positive(stiffness, "stiffness")
            return c.EmbeddedDef(
                master_label=master, slave_label=slave, name=name,
                tolerance=_non_negative(p["tolerance"], "tolerance"),
                stiffness=stiffness,
            )
        # RBE2 / RBE3: the master is a reference node.
        if master not in nodes:
            raise AssemblyError(
                f"reference {master!r} is not a declared reference node "
                f"(declared: {sorted(nodes)}); declare it with "
                f"Assembly.node(name, coords) first."
            )
        if kind == "kinematic_coupling":
            return c.KinematicCouplingDef(
                master_label=master, slave_label=slave, name=name,
                master_point=nodes[master],
                dofs=_dofs(p["dofs"], "dofs", optional=True),
            )
        if p["weighting"] not in ("uniform", "area"):
            raise AssemblyError(
                f"weighting={p['weighting']!r}: expected 'uniform' or 'area'.")
        return c.DistributingCouplingDef(
            master_label=master, slave_label=slave, name=name,
            master_point=nodes[master], weighting=p["weighting"],
        )
    except AssemblyError as exc:
        raise AssemblyError(f"{what}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        # The def's own __post_init__ refusals (EmbeddedDef stiffness, ...).
        raise AssemblyError(f"{what}: {exc}") from exc


def _dofs(value: object, what: str, *, optional: bool) -> "list[int] | None":
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise AssemblyError(
            f"{what}={value!r}: expected a non-empty list of 1-based DOFs"
            + (" or None." if optional else "."))
    out: list[int] = []
    for d in value:
        if not isinstance(d, int) or isinstance(d, bool) or not 1 <= d <= 6:
            raise AssemblyError(f"{what}={value!r}: DOF {d!r} is not an int in 1..6.")
        if d in out:
            raise AssemblyError(f"{what}={value!r}: DOF {d} repeats.")
        out.append(d)
    return out


def _required_dofs(value: object, what: str) -> list[int]:
    out = _dofs(value, what, optional=False)
    assert out is not None  # optional=False never returns None
    return out


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{what}={value!r}: expected a number.")
    v = float(value)
    if not math.isfinite(v):
        raise AssemblyError(f"{what}={value!r}: expected a finite number.")
    return v


def _positive(value: object, what: str) -> float:
    v = _real(value, what)
    if v <= 0.0:
        raise AssemblyError(f"{what}={value!r}: expected a number > 0.")
    return v


def _non_negative(value: object, what: str) -> float:
    v = _real(value, what)
    if v < 0.0:
        raise AssemblyError(f"{what}={value!r}: expected a number >= 0.")
    return v


def _point(value: object, what: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssemblyError(f"{what}={value!r}: expected (x, y, z).")
    x, y, z = (_real(v, what) for v in value)
    return (x, y, z)
