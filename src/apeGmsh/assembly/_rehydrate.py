"""Rehydrate an instance's archived model content onto the assembly bridge.

ADR 0117 option B: an instance's ``/opensees`` model content is read through
:meth:`OpenSeesModel.from_h5` and re-declared on the one forward
:class:`~apeGmsh.opensees.apeSees` through the public namespaces, each name
prefixed ``{instance}.``. Nothing here allocates a tag: the bridge plans
every tag at build (ADR 0114 D4), in registration order, which is instance
order, then the archived family order below.

Model content travels; analysis content (fixes, masses, patterns, time
series, recorders, stages, analysis) does not (ADR 0117 D4).

The archive stores each declaration as its emitted OpenSees argument tail,
so every supported type has a tail parser here. Rehydration runs in two
passes: the first parses and checks every archived declaration of the
instance, the second registers. Anything unsupported raises
:class:`AssemblyError` in the first pass, so an instance is never half
registered.

AS2a scope: materials, sections, transforms, beam integrations, dampings and
element specs, one spec per physical group (or a disjoint set of groups).
AS2b (#1542): rows whose args vary inside a group become one spec per
distinct args row, each on an element group the rehydrator registers on
the merged FEM as ``{instance}.{pg}#<k>`` (ADR 0117 D8).
Bars of a rebar cage (``fem_eid = -1`` ``CorotTruss`` rows) are not
re-declared: they travel as the carried ``/rebar_elements`` stream, whose
material is ``{instance}.{name}`` and binds to the rehydrated material.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from apeGmsh.mesh._compose import _prefix_namespaced_name
from apeGmsh.opensees._internal.typed_records import SectionSimpleRecord

from ._v1 import AssemblyError

if TYPE_CHECKING:
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.typed_records import ElementRecord
    from apeGmsh.opensees._internal.types import Primitive
    from apeGmsh.opensees.opensees_model import OpenSeesModel

__all__ = ["refuse_region_dampings", "rehydrate"]

#: A ``(kind, tag)`` reference, ``kind`` as ``OpenSeesModel.names()`` spells it.
_Key = tuple[str, int]
_Ref = Callable[[str, int], "Primitive"]
_Make = Callable[["apeSees", _Ref], "Primitive"]


@dataclass(frozen=True)
class _Decl:
    """One parsed declaration: what it references and how to register it."""

    key: "_Key | None"          # None for an element spec
    refs: tuple[_Key, ...]
    make: _Make


# ---------------------------------------------------------------------------
# Tail parsing
# ---------------------------------------------------------------------------

def _numbers(params: "tuple[Any, ...] | list[Any]", n_min: int, n_max: int,
             what: str) -> list[float]:
    """``params`` as floats, refusing a count outside ``[n_min, n_max]``."""
    if not n_min <= len(params) <= n_max or any(
        isinstance(p, (str, bool)) for p in params
    ):
        raise AssemblyError(
            f"{what}: archived params {tuple(params)!r} do not have a shape "
            f"the assembly rehydrates ({n_min}..{n_max} numbers)."
        )
    return [float(p) for p in params]


def _tag(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer():
        raise AssemblyError(f"{what}: archived value {v!r} is not a tag.")
    return int(v)


def _split(
    tail: "tuple[Any, ...]", flags: dict[str, int], what: str,
) -> tuple[list[Any], dict[str, list[Any]]]:
    """Split a tail into leading positionals and ``{flag: values}``.

    ``flags`` maps each accepted flag to its value count. An unknown or
    repeated flag, or a flag short of values, raises.
    """
    items = list(tail)
    i = 0
    while i < len(items) and not (isinstance(items[i], str) and items[i].startswith("-")):
        i += 1
    positional, rest = items[:i], items[i:]
    out: dict[str, list[Any]] = {}
    j = 0
    while j < len(rest):
        flag = rest[j]
        if not isinstance(flag, str) or flag not in flags:
            raise AssemblyError(
                f"{what}: archived flag {flag!r} is not rehydrated; "
                f"supported: {sorted(flags)}."
            )
        if flag in out:
            raise AssemblyError(f"{what}: archived flag {flag!r} repeats.")
        n = flags[flag]
        vals = rest[j + 1:j + 1 + n]
        if len(vals) != n or any(isinstance(v, str) for v in vals):
            raise AssemblyError(
                f"{what}: archived flag {flag!r} needs {n} numeric values, "
                f"got {vals!r}."
            )
        out[flag] = vals
        j += 1 + n
    return positional, out


def _opt(flags: dict[str, list[Any]], flag: str) -> "float | None":
    return float(flags[flag][0]) if flag in flags else None


# ---------------------------------------------------------------------------
# Materials and sections: (kind, type token) -> parser(params, name, what)
# ---------------------------------------------------------------------------

_Parser = Callable[[tuple[Any, ...], "str | None", str], _Make]


def _nd_elastic_isotropic(p: tuple[Any, ...], name: "str | None", what: str) -> _Make:
    E, nu, rho = _numbers(p, 3, 3, what)
    return lambda ops, ref: ops.nDMaterial.ElasticIsotropic(E=E, nu=nu, rho=rho, name=name)


def _uni_elastic(p: tuple[Any, ...], name: "str | None", what: str) -> _Make:
    v = _numbers(p, 1, 3, what)
    return lambda ops, ref: ops.uniaxialMaterial.ElasticMaterial(
        E=v[0], eta=v[1] if len(v) > 1 else 0.0,
        Eneg=v[2] if len(v) > 2 else None, name=name)


def _sec_elastic_membrane_plate(p: tuple[Any, ...], name: "str | None", what: str) -> _Make:
    v = _numbers(p, 4, 5, what)
    return lambda ops, ref: ops.section.ElasticMembranePlateSection(
        E=v[0], nu=v[1], h=v[2], rho=v[3],
        Ep_mod=v[4] if len(v) == 5 else 1.0, name=name)


def _sec_elastic(p: tuple[Any, ...], name: "str | None", what: str) -> _Make:
    v = _numbers(p, 3, 8, what)
    if len(v) in (3, 4, 5):     # 2-D: E A Iz [G [alphaY]]
        return lambda ops, ref: ops.section.Elastic(
            E=v[0], A=v[1], Iz=v[2], G=v[3] if len(v) > 3 else None,
            alphaY=v[4] if len(v) > 4 else None, name=name)
    if len(v) in (6, 8):        # 3-D: E A Iz Iy G J [alphaY alphaZ]
        return lambda ops, ref: ops.section.Elastic(
            E=v[0], A=v[1], Iz=v[2], Iy=v[3], G=v[4], J=v[5],
            alphaY=v[6] if len(v) == 8 else None,
            alphaZ=v[7] if len(v) == 8 else None, name=name)
    raise AssemblyError(f"{what}: {len(v)} params is neither the 2-D nor the 3-D form.")


#: Material families of ``/opensees/materials`` -> ``names()`` kind.
_MATERIAL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("uniaxial", "uniaxialMaterial"),
    ("nd", "nDMaterial"),
)

_MATERIALS: dict[tuple[str, str], _Parser] = {
    ("uniaxialMaterial", "Elastic"): _uni_elastic,
    ("nDMaterial", "ElasticIsotropic"): _nd_elastic_isotropic,
}

_SECTIONS: dict[str, _Parser] = {
    "Elastic": _sec_elastic,
    "ElasticMembranePlateSection": _sec_elastic_membrane_plate,
}


# ---------------------------------------------------------------------------
# Transforms, beam integrations, dampings
# ---------------------------------------------------------------------------

_TRANSFORMS = ("Linear", "PDelta", "Corotational")

#: Uniform-section quadrature rules: args ``(secTag, nIP)``.
_INTEGRATIONS = ("Legendre", "Lobatto", "NewtonCotes", "Radau", "Trapezoidal")

_WINDOW = {"-activateTime": 1, "-deactivateTime": 1}


def _transform(token: str, vec: tuple[float, ...], name: "str | None", what: str) -> _Make:
    if len(vec) not in (0, 3):
        raise AssemblyError(f"{what}: archived vecxz {vec!r} is not a 3-vector.")
    vecxz = (float(vec[0]), float(vec[1]), float(vec[2])) if vec else None
    # ``token`` is one of _TRANSFORMS, each a public ``ops.geomTransf`` verb.
    return lambda ops, ref: getattr(ops.geomTransf, token)(vecxz=vecxz, name=name)


def _integration(
    token: str, args: tuple[Any, ...], name: "str | None", what: str,
) -> tuple[tuple[_Key, ...], _Make]:
    if len(args) != 2:
        raise AssemblyError(f"{what}: archived args {args!r} are not (secTag, nIP).")
    sec = ("section", _tag(args[0], what))
    n_ip = _tag(args[1], what)
    return (sec,), lambda ops, ref: getattr(ops.beamIntegration, token)(
        section=cast(Any, ref(*sec)), n_ip=n_ip, name=name)


def _damping(token: str, args: tuple[Any, ...], name: "str | None", what: str) -> _Make:
    flags_ok = dict(_WINDOW, **{"-factor": 1})
    pos, flags = _split(args, flags_ok, what)
    if "-factor" in flags:
        raise AssemblyError(
            f"{what}: '-factor' references a time series, which is analysis "
            f"content and not carried (ADR 0117 D4)."
        )
    ta, td = _opt(flags, "-activateTime"), _opt(flags, "-deactivateTime")
    if token == "Uniform":
        z, f1, f2 = _numbers(pos, 3, 3, what)
        return lambda ops, ref: ops.damping.uniform(
            ratio=z, freq_lower=f1, freq_upper=f2, name=name,
            activate_time=ta, deactivate_time=td)
    if token == "SecStif":
        (beta,) = _numbers(pos, 1, 1, what)
        return lambda ops, ref: ops.damping.sec_stif(beta=beta, name=name,
            activate_time=ta, deactivate_time=td)
    if token in ("URD", "URDbeta"):
        n = _tag(pos[0], what) if pos else 0
        if n < 2:
            raise AssemblyError(f"{what}: archived params {tuple(pos)!r} hold no point table.")
        v = _numbers(pos[1:], 2 * n, 2 * n, what)
        points = [(v[2 * k], v[2 * k + 1]) for k in range(n)]
        method = "urd" if token == "URD" else "urd_beta"
        return lambda ops, ref: getattr(ops.damping, method)(points=points, name=name,
            activate_time=ta, deactivate_time=td)
    raise AssemblyError(
        f"{what} is not rehydrated; supported: ['SecStif', 'URD', 'URDbeta', 'Uniform']."
    )


# ---------------------------------------------------------------------------
# Elements: type token -> parser(pg, args tail, what) -> (refs, make)
# ---------------------------------------------------------------------------

_ElementParser = Callable[[str, tuple[Any, ...], str], tuple[tuple[_Key, ...], _Make]]


def _solid(token: str, pg: str, args: tuple[Any, ...], what: str) -> tuple[tuple[_Key, ...], _Make]:
    """``stdBrick`` / ``FourNodeTetrahedron``: ``matTag [b1 b2 b3]``."""
    if len(args) not in (1, 4):
        raise AssemblyError(
            f"{what}: archived args {args!r} carry flags the assembly does not "
            f"rehydrate (only matTag and an optional body force)."
        )
    mat = ("nDMaterial", _tag(args[0], what))
    body = _numbers(args[1:], 3, 3, what) if len(args) == 4 else None
    return (mat,), lambda ops, ref: getattr(ops.element, token)(
        pg=pg, material=ref(*mat),
        body_force=(body[0], body[1], body[2]) if body is not None else None)


def _el_shell_mitc4(pg: str, args: tuple[Any, ...], what: str) -> tuple[tuple[_Key, ...], _Make]:
    if len(args) != 1:
        raise AssemblyError(
            f"{what}: archived args {args!r} carry flags the assembly does not "
            f"rehydrate (only secTag)."
        )
    sec = ("section", _tag(args[0], what))
    return (sec,), lambda ops, ref: ops.element.ShellMITC4(
        pg=pg, section=cast(Any, ref(*sec)))


def _damp_ref(flags: dict[str, list[Any]], what: str) -> "_Key | None":
    return ("damping", _tag(flags["-damp"][0], what)) if "-damp" in flags else None


def _el_elastic_beam(pg: str, args: tuple[Any, ...], what: str) -> tuple[tuple[_Key, ...], _Make]:
    pos, flags = _split(args, {"-mass": 1, "-cMass": 0, "-damp": 1}, what)
    G: "float | None"
    J: "float | None"
    Iy: "float | None"
    if len(pos) == 7:       # 3-D: A E G J Iy Iz transfTag
        A, E, G, J, Iy, Iz = _numbers(pos[:6], 6, 6, what)
    elif len(pos) == 4:     # 2-D: A E Iz transfTag
        A, E, Iz = _numbers(pos[:3], 3, 3, what)
        G = J = Iy = None
    else:
        raise AssemblyError(f"{what}: archived args {args!r} are neither the 2-D nor the 3-D form.")
    transf = ("geomTransf", _tag(pos[-1], what))
    damp = _damp_ref(flags, what)
    mass, c_mass = _opt(flags, "-mass"), "-cMass" in flags
    refs = (transf,) + ((damp,) if damp else ())
    return refs, lambda ops, ref: ops.element.elasticBeamColumn(
        pg=pg, transf=cast(Any, ref(*transf)), A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
        mass=mass, c_mass=c_mass, damp=cast(Any, ref(*damp)) if damp else None)


def _beam_with_integration(
    token: str, pg: str, args: tuple[Any, ...], what: str,
) -> tuple[tuple[_Key, ...], _Make]:
    accepted = {"-mass": 1, "-damp": 1}
    accepted.update({"-iter": 2} if token == "forceBeamColumn" else {"-cMass": 0})
    pos, flags = _split(args, accepted, what)
    if len(pos) != 2:
        raise AssemblyError(f"{what}: archived args {args!r} are not (transfTag, integTag, ...).")
    transf = ("geomTransf", _tag(pos[0], what))
    integ = ("beamIntegration", _tag(pos[1], what))
    damp = _damp_ref(flags, what)
    mass = _opt(flags, "-mass")
    refs = (transf, integ) + ((damp,) if damp else ())

    def make(ops: "apeSees", ref: _Ref) -> "Primitive":
        common: dict[str, Any] = {
            "pg": pg, "transf": ref(*transf), "integration": ref(*integ),
            "mass": mass, "damp": ref(*damp) if damp else None,
        }
        if token == "forceBeamColumn":
            it = flags.get("-iter")
            return ops.element.forceBeamColumn(
                **common, max_iter=_tag(it[0], what) if it else None,
                tol=float(it[1]) if it else None)
        return ops.element.dispBeamColumn(**common, c_mass="-cMass" in flags)

    return refs, make


_ELEMENTS: dict[str, _ElementParser] = {
    "stdBrick": lambda pg, a, w: _solid("stdBrick", pg, a, w),
    "FourNodeTetrahedron": lambda pg, a, w: _solid("FourNodeTetrahedron", pg, a, w),
    "ShellMITC4": _el_shell_mitc4,
    "elasticBeamColumn": _el_elastic_beam,
    "forceBeamColumn": lambda pg, a, w: _beam_with_integration("forceBeamColumn", pg, a, w),
    "dispBeamColumn": lambda pg, a, w: _beam_with_integration("dispBeamColumn", pg, a, w),
}


# ---------------------------------------------------------------------------
# Element specs
# ---------------------------------------------------------------------------

def _rebar_rows(
    label: str, model: "OpenSeesModel", names: dict[_Key, str],
) -> set[int]:
    """Tags of the ``CorotTruss`` rows the carried ``/rebar_elements`` emit.

    A row matches a bar cell of the source's ``rebar_elements`` by its two
    nodes, its area and its material name. Those rows are not re-declared:
    the carried stream emits them on the assembly bridge.
    """
    cells: dict[tuple[int, int], tuple[float, str]] = {}
    for rec in model.fem.elements.rebar_elements:
        for i, j in rec.connectivity:
            cells[(int(i), int(j))] = (float(rec.area), rec.material)
    out: set[int] = set()
    for row in model.elements():
        if row.fem_eid >= 0 or row.type_token != "CorotTruss" or len(row.args) != 4:
            continue
        i, j, area, mat = row.args
        cell = cells.get((int(i), int(j)))
        if cell is not None and math.isclose(float(area), cell[0], rel_tol=1e-12) \
                and names.get(("uniaxialMaterial", _tag(mat, "CorotTruss"))) == cell[1]:
            out.add(row.tag)
    return out


@dataclass(frozen=True)
class _Spec:
    """One element spec: on source group ``pg``, or on its row subset."""

    token: str
    args: tuple[Any, ...]
    pg: str
    #: ``(k, source FEM ids)`` when the args vary inside ``pg``: the spec
    #: takes the synthesized group ``{instance}.{pg}#<k>``.
    rows: "tuple[int, tuple[int, ...]] | None" = None


def _element_specs(
    label: str, model: "OpenSeesModel", skip: set[int],
) -> list[_Spec]:
    """One :class:`_Spec` per archived element spec, in tag order.

    Rows sharing ``(type, args)`` are one spec per physical group: the
    group whose elements (of the row's element types) are exactly the
    rows' FEM ids, else a disjoint set of groups that covers them. Rows
    whose args vary inside a group (AS2b, #1542) are one spec per
    distinct args row on the first group (by name) that holds them all,
    numbered ``k = 1, 2, ...`` per group in tag order. Rows no single
    group holds raise.
    """
    fem = model.fem
    rows: dict[tuple[str, tuple[Any, ...]], list["ElementRecord"]] = {}
    for rec in sorted(model.elements(), key=lambda r: r.tag):
        if rec.tag in skip:
            continue
        if rec.fem_eid < 0:
            raise AssemblyError(
                f"instance {label!r}: element {rec.type_token} tag {rec.tag} "
                f"has no FEM element (a node-pair or synthesised row that is "
                f"not a carried rebar bar); the assembly rehydrates "
                f"physical-group element specs only."
            )
        rows.setdefault((rec.type_token, rec.args), []).append(rec)

    type_of: dict[int, int] = {}
    for code, group in enumerate(fem.elements):
        for eid in group.ids:
            type_of[int(eid)] = code
    pg_ids = {
        pg: frozenset(int(e) for e in fem.elements.select(pg=pg).ids)
        for pg in sorted(set(fem.elements.physical.names()))
    }

    specs: list[tuple[int, _Spec]] = []
    for (token, args), recs in rows.items():
        eids = frozenset(r.fem_eid for r in recs)
        missing = sorted(e for e in eids if e not in type_of)
        if missing:
            raise AssemblyError(
                f"instance {label!r}: {token} rows name FEM elements "
                f"{missing[:5]} that the neutral zone does not carry."
            )
        codes = {type_of[e] for e in eids}
        same_type = {
            pg: frozenset(e for e in ids if type_of.get(e) in codes)
            for pg, ids in pg_ids.items()
        }
        tag_of = {r.fem_eid: r.tag for r in recs}
        exact = [pg for pg, ids in same_type.items() if ids == eids]
        if exact:
            chosen = exact[:1]
        else:
            chosen = [pg for pg, ids in same_type.items() if ids and ids <= eids]
            covered = [e for pg in chosen for e in same_type[pg]]
            if len(covered) != len(set(covered)) or set(covered) != eids:
                holder = [pg for pg, ids in same_type.items() if eids <= ids]
                if not holder:
                    raise AssemblyError(
                        f"instance {label!r}: {len(recs)} {token} rows with "
                        f"args {args!r} match no physical group, disjoint set "
                        f"of groups, or single group that holds them all."
                    )
                specs.append((min(tag_of.values()), _Spec(
                    token, args, holder[0], (0, tuple(sorted(eids))))))
                continue
        for pg in chosen:
            specs.append((min(tag_of[e] for e in same_type[pg]),
                          _Spec(token, args, pg)))
    specs.sort(key=lambda s: s[0])
    out: list[_Spec] = []
    count: dict[str, int] = {}
    for _, spec in specs:
        if spec.rows is not None:
            count[spec.pg] = count.get(spec.pg, 0) + 1
            spec = _Spec(spec.token, spec.args, spec.pg,
                         (count[spec.pg], spec.rows[1]))
        out.append(spec)
    return out


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------

#: ``(group name, merged parent PG, source parent PG, source FEM ids)``.
_Group = tuple[str, str, str, tuple[int, ...]]


def _plan(label: str, model: "OpenSeesModel") -> tuple[list[_Decl], list[_Group]]:
    """Parse and check every archived declaration; register nothing.

    Returns the declarations and the element groups to synthesize.
    """
    names = {(kind, tag): name for name, kind, tag in model.names()}

    def prefixed(kind: str, tag: int) -> "str | None":
        name = names.get((kind, tag))
        return f"{label}.{name}" if name is not None else None

    def what(kind: str, token: str, tag: int) -> str:
        return f"instance {label!r}: {kind} {token!r} (tag {tag})"

    unknown = set(model.materials_by_family()) - {fam for fam, _ in _MATERIAL_FAMILIES}
    if unknown:
        raise AssemblyError(
            f"instance {label!r}: unknown material families {sorted(unknown)}."
        )

    plan: list[_Decl] = []
    by_family = model.materials_by_family()
    for family, kind in _MATERIAL_FAMILIES:
        for mrec in sorted(by_family.get(family, ()), key=lambda r: r.tag):
            parse = _MATERIALS.get((kind, mrec.type_token))
            if parse is None:
                raise AssemblyError(
                    f"{what(kind, mrec.type_token, mrec.tag)} is not "
                    f"rehydrated; supported: "
                    f"{sorted(t for k, t in _MATERIALS if k == kind)}."
                )
            plan.append(_Decl((kind, mrec.tag), (), parse(
                mrec.params, prefixed(kind, mrec.tag), what(kind, mrec.type_token, mrec.tag))))

    for sec in sorted(model.sections(), key=lambda r: r.tag):
        parse_sec = _SECTIONS.get(sec.type_token)
        if parse_sec is None or not isinstance(sec, SectionSimpleRecord):
            raise AssemblyError(
                f"{what('section', sec.type_token, sec.tag)} is not "
                f"rehydrated; supported: {sorted(_SECTIONS)}."
            )
        plan.append(_Decl(("section", sec.tag), (), parse_sec(
            sec.params, prefixed("section", sec.tag),
            what("section", sec.type_token, sec.tag))))

    for tr in sorted(model.transforms(), key=lambda r: r.tag):
        if tr.type_token not in _TRANSFORMS:
            raise AssemblyError(
                f"{what('geomTransf', tr.type_token, tr.tag)} is not "
                f"rehydrated; supported: {sorted(_TRANSFORMS)}."
            )
        plan.append(_Decl(("geomTransf", tr.tag), (), _transform(
            tr.type_token, tr.vec, prefixed("geomTransf", tr.tag),
            what("geomTransf", tr.type_token, tr.tag))))

    for bi in sorted(model.beam_integration(), key=lambda r: r.tag):
        if bi.type_token not in _INTEGRATIONS:
            raise AssemblyError(
                f"{what('beamIntegration', bi.type_token, bi.tag)} is not "
                f"rehydrated; supported: {sorted(_INTEGRATIONS)}."
            )
        refs, make = _integration(
            bi.type_token, bi.args, prefixed("beamIntegration", bi.tag),
            what("beamIntegration", bi.type_token, bi.tag))
        plan.append(_Decl(("beamIntegration", bi.tag), refs, make))

    for dm in sorted(model.dampings(), key=lambda r: r.tag):
        plan.append(_Decl(("damping", dm.tag), (), _damping(
            dm.type_token, dm.args, prefixed("damping", dm.tag),
            what("damping", dm.type_token, dm.tag))))

    skip = _rebar_rows(label, model, names)
    groups: list[_Group] = []
    for spec in _element_specs(label, model, skip):
        parse_el = _ELEMENTS.get(spec.token)
        if parse_el is None:
            raise AssemblyError(
                f"instance {label!r}: element {spec.token!r} is not "
                f"rehydrated; supported: {sorted(_ELEMENTS)}."
            )
        # The merge engine's own rule, so the spec names the PG compose wrote.
        pg = str(_prefix_namespaced_name(label, spec.pg))
        if spec.rows is not None:
            groups.append((f"{pg}#{spec.rows[0]}", pg, spec.pg, spec.rows[1]))
            pg = groups[-1][0]
        refs, make = parse_el(pg, spec.args,
                              f"instance {label!r}: element {spec.token!r}")
        plan.append(_Decl(None, refs, make))

    declared: set[_Key] = set()
    used: set[_Key] = set()
    for decl in plan:
        for r in decl.refs:
            if r not in declared:
                raise AssemblyError(
                    f"instance {label!r}: a declaration references {r[0]} tag "
                    f"{r[1]}, which the archive does not declare before it."
                )
        used.update(decl.refs)
        if decl.key is not None:
            declared.add(decl.key)
    unattached = sorted(t for k, t in declared if k == "damping" and ("damping", t) not in used)
    if unattached:
        raise AssemblyError(
            f"instance {label!r}: damping tags {unattached} are attached by "
            f"region (on=), which is analysis content and not carried; attach "
            f"them with an element's damp= in the source, or declare the "
            f"damping on the assembly bridge."
        )
    return plan, groups


def _group_ids(
    label: str, ops: "apeSees", model: "OpenSeesModel", groups: list[_Group],
) -> list[tuple[str, list[int]]]:
    """Each synthesized group's merged FEM ids; raise on a name in use.

    Compose relocates an instance's ids by one offset, so the k-th
    smallest id of the source group is the k-th smallest of its merged
    counterpart.
    """
    fem = ops.fem
    taken = set(fem.nodes.physical.names()) | set(fem.elements.physical.names())
    out: list[tuple[str, list[int]]] = []
    for name, merged_pg, source_pg, rows in groups:
        if name in taken:
            raise AssemblyError(
                f"instance {label!r}: the per-row element group {name!r} "
                f"would shadow a physical group of that name; rename the "
                f"source's group."
            )
        src = sorted(int(e) for e in model.fem.elements.select(pg=source_pg).ids)
        dst = sorted(int(e) for e in fem.elements.select(pg=merged_pg).ids)
        if len(src) != len(dst):
            raise AssemblyError(
                f"instance {label!r}: merged group {merged_pg!r} holds "
                f"{len(dst)} elements, its source {source_pg!r} {len(src)}."
            )
        to_merged = dict(zip(src, dst))
        ids = [to_merged[e] for e in rows]
        if not ids:
            raise AssemblyError(f"instance {label!r}: element group {name!r} is empty.")
        out.append((name, ids))
    return out


def _region_params(g: Any) -> list[Any]:
    """A region group's ``params`` tail (the writer's float/str attr pair)."""
    nums = [float(v) for v in g.attrs["params"]] if "params" in g.attrs else []
    strs: list[str] = []
    if "params_str" in g.attrs:
        strs = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in g.attrs["params_str"]]
    return [strs[i] if i < len(strs) and strs[i] != "" else v
            for i, v in enumerate(nums)]


def refuse_region_dampings(label: str, source: "str | Path") -> None:
    """Raise if ``source`` attaches a damping object through a region.

    A ``region ... -damp`` attach, global or stage-scoped, is analysis
    content that does not travel (ADR 0117 D4), and
    :meth:`OpenSeesModel.from_h5` does not read ``/opensees/regions``, so
    a damping also attached by an element's ``damp=`` would bridge with
    its region half silently dropped. The archive is read here, read-only.
    """
    import h5py

    found: dict[int, list[str]] = {}
    with h5py.File(str(source), "r") as f:
        if "opensees" not in f:
            return
        zone = f["opensees"]
        pools: list[tuple[str, Any]] = []
        if "regions" in zone:
            pools.append(("/opensees/regions", zone["regions"]))
        if "stages" in zone:
            for sname in sorted(zone["stages"]):
                stage = zone["stages"][sname]
                if "regions" in stage:
                    pools.append((f"/opensees/stages/{sname}/regions", stage["regions"]))
        for where, grp in pools:
            for rname in sorted(grp):
                args = _region_params(grp[rname])
                for k, a in enumerate(args[:-1]):
                    if a == "-damp":
                        tag = _tag(args[k + 1], f"{where}/{rname}")
                        found.setdefault(tag, []).append(f"{where}/{rname}")
    if found:
        raise AssemblyError(
            f"instance {label!r}: damping tags {sorted(found)} are attached "
            f"by region ({sorted({w for ws in found.values() for w in ws})}), "
            f"which is analysis content and not carried (ADR 0117 D4); "
            f"attach them with an element's damp= in the source, or declare "
            f"the damping on the assembly bridge."
        )


def rehydrate(ops: "apeSees", label: str, model: "OpenSeesModel") -> None:
    """Register ``model``'s model content on ``ops`` under ``{label}.``.

    Names become ``{label}.{name}`` (the carried rebar material's rule in
    the merge engine); physical groups take the merge engine's
    ``_prefix_namespaced_name`` (ADR 0038 alternation): ``{label}.{pg}``
    when ``pg`` holds an even number of ``.``/``/``, else ``{label}/{pg}``.
    Registration order: uniaxial materials, nD materials, sections,
    transforms, beam integrations, dampings, then element specs, each in
    archived tag order. Every declaration is parsed and checked before the
    first registration, so a refusal leaves ``ops`` untouched.
    """
    plan, groups = _plan(label, model)
    for name, ids in _group_ids(label, ops, model, groups):
        ops.fem._add_element_group(name, ids)
    by_ref: dict[_Key, "Primitive"] = {}

    def ref(kind: str, tag: int) -> "Primitive":
        return by_ref[(kind, tag)]       # _plan proved every ref is declared

    for decl in plan:
        prim = decl.make(ops, ref)
        if decl.key is not None:
            by_ref[decl.key] = prim
