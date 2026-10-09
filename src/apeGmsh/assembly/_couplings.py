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
``equal_dof_mixed``        ``EqualDOFMixedDef``               port / port
``rigid_link``             ``RigidLinkDef``                   port / port
``rigid_diaphragm``        ``RigidDiaphragmDef``              port / port
``rigid_body``             ``RigidBodyDef``                   port / port
``embedded``             ``EmbeddedDef``                    host / embedded
``kinematic_coupling``     ``KinematicCouplingDef`` (RBE2)    reference / target
``distributing_coupling``  ``DistributingCouplingDef`` (RBE3) reference / target
=========================  =================================  ===============

A port is ``{instance}.{pg|label}`` or, where the verb accepts one, the name
of an assembly reference node (``Assembly.node``). The reference of an RBE2
or RBE3 coupling is always a reference node, and its coordinates are the
def's ``master_point``.

:func:`tie_definition` does the same for ``tie``: the verb and the reader
build its ``TieDef`` from one params object, and :func:`tie_params` turns a
``TieDef`` back into it. The penalty / enforcement knobs of ``tie`` and of
the RBE2 / RBE3 couplings (``TIE_KNOBS``) are written only when set.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Mapping

from ._h5 import TIE_KNOBS, params_key_error, row_params
from ._v1 import AssemblyError

__all__ = [
    "COUPLING_KINDS", "NODE_PORTS", "canonical_params", "coupling_definition",
    "tie_definition", "tie_params",
]

#: The coupling kinds, in the order of the table above.
COUPLING_KINDS: tuple[str, ...] = (
    "equal_dof", "equal_dof_mixed", "rigid_link", "rigid_diaphragm",
    "rigid_body", "embedded", "kinematic_coupling", "distributing_coupling",
)

#: Per kind, whether the master / slave port may name a reference node
#: instead of an instance port. ``embedded`` and the RBE2 / RBE3 targets
#: resolve elements or a node set of an instance, so they never do.
NODE_PORTS: dict[str, tuple[bool, bool]] = {
    "equal_dof": (True, True),
    "equal_dof_mixed": (True, True),
    "rigid_link": (True, True),
    "rigid_diaphragm": (True, True),
    "rigid_body": (True, True),
    "embedded": (False, False),
    "kinematic_coupling": (True, False),
    "distributing_coupling": (True, False),
}


def canonical_params(params: Mapping[str, Any], kind: "str | None" = None) -> str:
    """The canonical JSON of ``params``: sorted keys, no spaces.

    With ``kind``, every knob of that kind at its default is left out
    first (``row_params``): the form a ``/assembly/ties`` row stores.
    """
    if kind is not None:
        params = row_params(kind, dict(params))
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
    the kind's (``TIE_PARAMS``, plus any ``TIE_KNOBS`` knob not at its
    default), an invalid option, and an RBE2 / RBE3 reference that is not
    a declared reference node.
    """
    from apeGmsh._kernel.defs import constraints as c

    what = f"{kind}({master!r}, {slave!r})"
    if kind not in COUPLING_KINDS:
        raise AssemblyError(
            f"{what}: unknown coupling kind; the kinds are {list(COUPLING_KINDS)}.")
    problem = params_key_error(kind, dict(params))
    if problem is not None:
        raise AssemblyError(f"{what}: {problem}.")
    p = {**TIE_KNOBS.get(kind, {}), **params}
    try:
        if kind == "equal_dof":
            return c.EqualDOFDef(
                master_label=master, slave_label=slave, name=name,
                dofs=_dofs(p["dofs"], "dofs", optional=True),
                tolerance=_positive(p["tolerance"], "tolerance"),
            )
        if kind == "equal_dof_mixed":
            return c.EqualDOFMixedDef(
                master_label=master, slave_label=slave, name=name,
                dof_pairs=_dof_pairs(p["dof_pairs"]),
                tolerance=_positive(p["tolerance"], "tolerance"),
            )
        if kind == "rigid_body":
            mass, omega = p["mass"], p["omega"]
            return c.RigidBodyDef(
                master_label=master, slave_label=slave, name=name,
                master_point=_point(p["master_point"], "master_point"),
                as_element=_flag(p["as_element"], "as_element"),
                mass=None if mass is None else _non_negative(mass, "mass"),
                omega=None if omega is None else _point(omega, "omega"),
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
                control=_rbe_control(p),
            )
        if p["weighting"] not in ("uniform", "area"):
            raise AssemblyError(
                f"weighting={p['weighting']!r}: expected 'uniform' or 'area'.")
        return c.DistributingCouplingDef(
            master_label=master, slave_label=slave, name=name,
            master_point=nodes[master], weighting=p["weighting"],
            control=_rbe_control(p),
        )
    except AssemblyError as exc:
        raise AssemblyError(f"{what}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        # The def's own __post_init__ refusals (EmbeddedDef stiffness, ...).
        raise AssemblyError(f"{what}: {exc}") from exc


def tie_definition(
    master: str, slave: str, params: Mapping[str, Any], name: "str | None",
) -> Any:
    """Build the ``TieDef`` of one assembly ``tie``; raise for a bad option.

    ``params`` is the row's object: the ``TIE_PARAMS["tie"]`` keys plus any
    ``TIE_KNOBS["tie"]`` knob that is set. The knobs reach the def, and so
    the chain-phase route ``g.constraints.tie`` takes, unchanged. Raises
    :class:`AssemblyError`, without a ``tie(...)`` prefix (each caller adds
    its own), for foreign keys, a knob of the wrong type, and every option
    ``TieDef`` refuses.
    """
    from apeGmsh._kernel.defs.constraints import TieDef

    problem = params_key_error("tie", dict(params))
    if problem is not None:
        raise AssemblyError(f"{problem}.")
    p = {**TIE_KNOBS["tie"], **params}
    stiffness = p["stiffness"]
    if stiffness != "auto":
        stiffness = _positive(stiffness, "stiffness")
    sp = p["stiffness_p"]
    outward = p["outward"]
    if outward is not None:
        outward = _point(outward, "outward")
        if math.hypot(*outward) == 0.0:
            raise AssemblyError(f"outward={outward!r} is zero.")
    for key in ("enforce", "method"):
        if not isinstance(p[key], str):
            raise AssemblyError(f"{key}={p[key]!r}: expected a string.")
    try:
        return TieDef(
            master_label=master, slave_label=slave,
            dofs=None if p["dofs"] is None else [int(d) for d in p["dofs"]],
            tolerance=float(p["tolerance"]), enforce=p["enforce"],
            method=p["method"], name=name, stiffness=stiffness,
            stiffness_p=None if sp is None else _positive(sp, "stiffness_p"),
            rotational=_flag(p["rotational"], "rotational"),
            pressure=_flag(p["pressure"], "pressure"),
            control=_control(p["control"]), outward=outward,
        )
    except (TypeError, ValueError) as exc:
        raise AssemblyError(str(exc)) from exc


def tie_params(defn: Any) -> dict[str, Any]:
    """The params object of a ``TieDef`` that :func:`tie_definition` built,
    every knob included (``canonical_params(..., "tie")`` drops the ones at
    their default)."""
    return {
        "dofs": None if defn.dofs is None else list(defn.dofs),
        "enforce": defn.enforce,
        "method": defn.method,
        "tolerance": defn.tolerance,
        "stiffness": defn.stiffness,
        "stiffness_p": defn.stiffness_p,
        "rotational": defn.rotational,
        "pressure": defn.pressure,
        "control": (None if defn.control is None
                    else dataclasses.asdict(defn.control)),
        "outward": None if defn.outward is None else list(defn.outward),
    }


def _control(value: object) -> Any:
    """A tie's ``control`` param, a ``CouplingControl`` stored as an object
    with every field, as the ``CouplingControl`` (``None`` stays)."""
    from apeGmsh._kernel._coupling_control import CouplingControl

    if value is None:
        return None
    fields = {f.name for f in dataclasses.fields(CouplingControl)}
    if not isinstance(value, dict) or set(value) != fields:
        raise AssemblyError(
            f"control={value!r}: expected a CouplingControl, stored as an "
            f"object with the keys {sorted(fields)}.")
    return CouplingControl(**value)


def _rbe_control(p: Mapping[str, Any]) -> Any:
    """The ``CouplingControl`` of an RBE2 / RBE3 row: ``k``, ``kr``,
    ``enforce`` and, on RBE2, ``al_update``, as ``g.constraints`` takes them.

    ``k="auto"`` and ``k_alpha`` scale the penalty off a host element
    (``-host``), which an assembly coupling does not take, so they are
    refused here, before ``CouplingControl`` would ask for ``host=``.
    """
    from apeGmsh._kernel._coupling_control import CouplingControl

    k, k_alpha = p["k"], p["k_alpha"]
    if k == "auto" or k_alpha is not None:
        raise AssemblyError(
            f"k={k!r}, k_alpha={k_alpha!r}: k='auto' and k_alpha scale the "
            f"penalty off a host element (-host), which an assembly "
            f"coupling does not take; give k as a number > 0.")
    if p["enforce"] not in ("penalty", "al"):
        raise AssemblyError(
            f"enforce={p['enforce']!r}: expected 'penalty' or 'al'.")
    al_update = p.get("al_update")
    if al_update not in (None, "commit", "iter"):
        raise AssemblyError(
            f"al_update={al_update!r}: expected None, 'commit' or 'iter'.")
    kr = p["kr"]
    return CouplingControl(
        k=None if k is None else _positive(k, "k"),
        kr=None if kr is None else _positive(kr, "kr"),
        enforce=p["enforce"], al_update=al_update,
    )


def _dof_pairs(value: object) -> list[tuple[int, int]]:
    """``[(retained, constrained), ...]``: non-empty, every DOF an int in
    1..6, and no constrained DOF twice."""
    if not isinstance(value, (list, tuple)) or not value:
        raise AssemblyError(
            f"dof_pairs={value!r}: expected a non-empty list of "
            f"(retained_dof, constrained_dof) couples.")
    out: list[tuple[int, int]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AssemblyError(
                f"dof_pairs={value!r}: {pair!r} is not a (retained_dof, "
                f"constrained_dof) couple.")
        (r,) = _required_dofs([pair[0]], "dof_pairs")
        (cd,) = _required_dofs([pair[1]], "dof_pairs")
        if any(cd == seen for _, seen in out):
            raise AssemblyError(
                f"dof_pairs={value!r}: constrained DOF {cd} repeats.")
        out.append((r, cd))
    return out


def _flag(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise AssemblyError(f"{what}={value!r}: expected True or False.")
    return value


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
