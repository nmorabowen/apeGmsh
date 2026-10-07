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
Rows whose args vary inside a group raise; their selector is AS2b (#1542).
Bars of a rebar cage (``fem_eid = -1`` ``CorotTruss`` rows) are not
re-declared: they travel as the carried ``/rebar_elements`` stream, whose
material is ``{instance}.{name}`` and binds to the rehydrated material.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

from apeGmsh.opensees._internal.typed_records import SectionSimpleRecord

from ._v1 import AssemblyError

if TYPE_CHECKING:
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.typed_records import ElementRecord
    from apeGmsh.opensees._internal.types import Primitive
    from apeGmsh.opensees.opensees_model import OpenSeesModel

__all__ = ["rehydrate"]

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
    window = {"activate_time": _opt(flags, "-activateTime"),
              "deactivate_time": _opt(flags, "-deactivateTime")}
    if token == "Uniform":
        z, f1, f2 = _numbers(pos, 3, 3, what)
        return lambda ops, ref: ops.damping.uniform(
            ratio=z, freq_lower=f1, freq_upper=f2, name=name, **window)
    if token == "SecStif":
        (beta,) = _numbers(pos, 1, 1, what)
        return lambda ops, ref: ops.damping.sec_stif(beta=beta, name=name, **window)
    if token in ("URD", "URDbeta"):
        n = _tag(pos[0], what) if pos else 0
        if n < 2:
            raise AssemblyError(f"{what}: archived params {tuple(pos)!r} hold no point table.")
        v = _numbers(pos[1:], 2 * n, 2 * n, what)
        points = [(v[2 * k], v[2 * k + 1]) for k in range(n)]
        method = "urd" if token == "URD" else "urd_beta"
        return lambda ops, ref: getattr(ops.damping, method)(points=points, name=name, **window)
    raise AssemblyError(
        f"{what} is not rehydrated; supported: ['SecStif', 'URD', 'URDbeta', 'Uniform']."
    )


# ---------------------------------------------------------------------------
# Elements: type token -> parser(pg, args tail, what) -> (refs, make)
# ---------------------------------------------------------------------------

_ElementParser = Callable[[str, tuple[Any, ...], str], tuple[tuple[_Key, ...], _Make]]


def _el_std_brick(pg: str, args: tuple[Any, ...], what: str) -> tuple[tuple[_Key, ...], _Make]:
    if len(args) not in (1, 4):
        raise AssemblyError(
            f"{what}: archived args {args!r} carry flags the assembly does not "
            f"rehydrate (only matTag and an optional body force)."
        )
    mat = ("nDMaterial", _tag(args[0], what))
    body = _numbers(args[1:], 3, 3, what) if len(args) == 4 else None
    return (mat,), lambda ops, ref: ops.element.stdBrick(
        pg=pg, material=cast(Any, ref(*mat)),
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
    "stdBrick": _el_std_brick,
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


def _element_specs(
    label: str, model: "OpenSeesModel", skip: set[int],
) -> list[tuple[str, tuple[Any, ...], str]]:
    """``(type_token, args, pg)`` per archived element spec, in tag order.

    Rows sharing ``(type, args)`` are one spec per physical group: the
    group whose elements (of the row's element types) are exactly the
    rows' FEM ids, else a disjoint set of groups that covers them. Any
    other shape (args that vary inside a group) raises: its selector is
    AS2b (#1542).
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

    specs: list[tuple[int, str, tuple[Any, ...], str]] = []
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
        exact = [pg for pg, ids in same_type.items() if ids == eids]
        if exact:
            chosen = exact[:1]
        else:
            chosen = [pg for pg, ids in same_type.items() if ids and ids <= eids]
            covered = [e for pg in chosen for e in same_type[pg]]
            if len(covered) != len(set(covered)) or set(covered) != eids:
                raise AssemblyError(
                    f"instance {label!r}: {len(recs)} {token} rows with args "
                    f"{args!r} match no physical group or disjoint set of "
                    f"groups (args vary inside a group); per-row element "
                    f"rehydration needs the AS2b selector (#1542)."
                )
        tag_of = {r.fem_eid: r.tag for r in recs}
        for pg in chosen:
            specs.append((min(tag_of[e] for e in same_type[pg]), token, args, pg))
    specs.sort(key=lambda s: s[0])
    return [(token, args, pg) for _, token, args, pg in specs]


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------

def _plan(label: str, model: "OpenSeesModel") -> list[_Decl]:
    """Parse and check every archived declaration; register nothing."""
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
    for token, args, pg in _element_specs(label, model, skip):
        parse_el = _ELEMENTS.get(token)
        if parse_el is None:
            raise AssemblyError(
                f"instance {label!r}: element {token!r} is not rehydrated; "
                f"supported: {sorted(_ELEMENTS)}."
            )
        refs, make = parse_el(f"{label}.{pg}", args, f"instance {label!r}: element {token!r}")
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
    return plan


def rehydrate(ops: "apeSees", label: str, model: "OpenSeesModel") -> None:
    """Register ``model``'s model content on ``ops`` under ``{label}.``.

    Names become ``{label}.{name}``; physical groups ``{label}.{pg}``.
    Registration order: uniaxial materials, nD materials, sections,
    transforms, beam integrations, dampings, then element specs, each in
    archived tag order. Every declaration is parsed and checked before the
    first registration, so a refusal leaves ``ops`` untouched.
    """
    plan = _plan(label, model)
    by_ref: dict[_Key, "Primitive"] = {}

    def ref(kind: str, tag: int) -> "Primitive":
        return by_ref[(kind, tag)]       # _plan proved every ref is declared

    for decl in plan:
        prim = decl.make(ops, ref)
        if decl.key is not None:
            by_ref[decl.key] = prim
