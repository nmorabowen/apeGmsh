"""Rehydrate an instance's archived model content onto the assembly bridge.

ADR 0117 option B: an instance's ``/opensees`` model content is read through
:meth:`OpenSeesModel.from_h5` and re-declared on the one forward
:class:`~apeGmsh.opensees.apeSees` through the public namespaces, each name
prefixed ``{instance}.``. Nothing here allocates a tag: the bridge plans
every tag at build (ADR 0114 D4), in registration order, which is instance
order, then the archived family order below.

Model content travels; analysis content (fixes, masses, patterns, time
series, recorders, stages, analysis) does not (ADR 0117 D4).

AS1 rehydrates the common case: one element spec per physical group, and
the primitive types in the tables below. Anything else raises
:class:`AssemblyError` naming it; AS2 generalises (per-row element args,
transforms, beam integrations, dampings).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from apeGmsh.opensees._internal.typed_records import SectionSimpleRecord

from ._v1 import AssemblyError

if TYPE_CHECKING:
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.typed_records import ElementRecord
    from apeGmsh.opensees._internal.types import NDMaterial, Primitive
    from apeGmsh.opensees.opensees_model import OpenSeesModel

__all__ = ["rehydrate"]

_Ref = Callable[[str, Any], "Primitive"]


def _numbers(params: tuple[Any, ...], n_min: int, n_max: int, what: str) -> list[float]:
    """``params`` as floats, refusing a count outside ``[n_min, n_max]``."""
    if not n_min <= len(params) <= n_max or any(
        isinstance(p, str) or isinstance(p, bool) for p in params
    ):
        raise AssemblyError(
            f"{what}: archived params {params!r} do not have the shape AS1 "
            f"rehydrates ({n_min}..{n_max} numbers)."
        )
    return [float(p) for p in params]


# ---------------------------------------------------------------------------
# Materials and sections: (kind, type token) -> builder(ops, params, name)
# ---------------------------------------------------------------------------

def _nd_elastic_isotropic(ops: "apeSees", p: tuple[Any, ...], name: "str | None") -> "Primitive":
    E, nu, rho = _numbers(p, 3, 3, "nDMaterial ElasticIsotropic")
    return ops.nDMaterial.ElasticIsotropic(E=E, nu=nu, rho=rho, name=name)


def _sec_elastic_membrane_plate(
    ops: "apeSees", p: tuple[Any, ...], name: "str | None",
) -> "Primitive":
    v = _numbers(p, 4, 5, "section ElasticMembranePlateSection")
    return ops.section.ElasticMembranePlateSection(
        E=v[0], nu=v[1], h=v[2], rho=v[3],
        Ep_mod=v[4] if len(v) == 5 else 1.0, name=name,
    )


_Builder = Callable[["apeSees", tuple[Any, ...], "str | None"], "Primitive"]

#: Material families of ``/opensees/materials`` -> tag-allocator kind.
_MATERIAL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("uniaxial", "uniaxialMaterial"),
    ("nd", "nDMaterial"),
)

_MATERIALS: dict[tuple[str, str], _Builder] = {
    ("nDMaterial", "ElasticIsotropic"): _nd_elastic_isotropic,
}

_SECTIONS: dict[str, _Builder] = {
    "ElasticMembranePlateSection": _sec_elastic_membrane_plate,
}


# ---------------------------------------------------------------------------
# Elements: type token -> builder(ops, pg, args tail, ref)
# ---------------------------------------------------------------------------

def _tag_arg(args: tuple[Any, ...], i: int, what: str) -> int:
    v = args[i] if i < len(args) else None
    if not isinstance(v, int) or isinstance(v, bool):
        raise AssemblyError(f"{what}: archived arg {i} is {v!r}, not a tag.")
    return v


def _el_std_brick(ops: "apeSees", pg: str, args: tuple[Any, ...], ref: _Ref) -> "Primitive":
    if len(args) not in (1, 4):
        raise AssemblyError(
            f"element stdBrick: archived args {args!r} carry flags AS1 does "
            f"not rehydrate (only matTag and an optional body force)."
        )
    mat = ref("nDMaterial", _tag_arg(args, 0, "element stdBrick"))
    body = _numbers(args[1:], 3, 3, "element stdBrick") if len(args) == 4 else None
    # ``ref`` looks the tag up in its own kind, so the type holds.
    return ops.element.stdBrick(
        pg=pg, material=cast("NDMaterial", mat),
        body_force=(body[0], body[1], body[2]) if body is not None else None,
    )


def _el_shell_mitc4(ops: "apeSees", pg: str, args: tuple[Any, ...], ref: _Ref) -> "Primitive":
    if len(args) != 1:
        raise AssemblyError(
            f"element ShellMITC4: archived args {args!r} carry flags AS1 "
            f"does not rehydrate (only secTag)."
        )
    sec = ref("section", _tag_arg(args, 0, "element ShellMITC4"))
    return ops.element.ShellMITC4(pg=pg, section=cast(Any, sec))


_ElementBuilder = Callable[["apeSees", str, tuple[Any, ...], _Ref], "Primitive"]

_ELEMENTS: dict[str, _ElementBuilder] = {
    "stdBrick": _el_std_brick,
    "ShellMITC4": _el_shell_mitc4,
}


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------

def _refuse_unsupported(label: str, model: "OpenSeesModel") -> None:
    """Carried families AS1 cannot rehydrate yet raise, never drop."""
    pending = {
        "transforms": len(model.transforms()),
        "beam integrations": len(model.beam_integration()),
        "dampings": len(model.dampings()),
    }
    found = {k: n for k, n in pending.items() if n}
    if found:
        raise AssemblyError(
            f"instance {label!r} carries {found}; AS1 rehydrates materials, "
            f"sections and elements only (AS2 adds the rest)."
        )
    known = {fam for fam, _ in _MATERIAL_FAMILIES}
    unknown = set(model.materials_by_family()) - known
    if unknown:
        raise AssemblyError(
            f"instance {label!r}: unknown material families {sorted(unknown)}."
        )


def _element_specs(
    label: str, model: "OpenSeesModel",
) -> list[tuple[str, tuple[Any, ...], str]]:
    """``(type_token, args, pg)`` per archived element spec, in tag order.

    Rows sharing ``(type, args)`` are one spec per physical group: the
    group whose elements (of the row's element types) are exactly the
    rows' FEM ids, else a disjoint set of groups that covers them. Any
    other shape is per-row rehydration (AS2) and raises.
    """
    fem = model.fem
    rows: dict[tuple[str, tuple[Any, ...]], list["ElementRecord"]] = {}
    for rec in sorted(model.elements(), key=lambda r: r.tag):
        if rec.fem_eid < 0:
            raise AssemblyError(
                f"instance {label!r}: element {rec.type_token} tag {rec.tag} "
                f"has no FEM element (a node-pair or synthesised row); AS1 "
                f"rehydrates physical-group element specs only."
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
                    f"groups; per-row element rehydration is AS2."
                )
        tag_of = {r.fem_eid: r.tag for r in recs}
        for pg in chosen:
            specs.append((min(tag_of[e] for e in same_type[pg]), token, args, pg))
    specs.sort(key=lambda s: s[0])
    return [(token, args, pg) for _, token, args, pg in specs]


def rehydrate(ops: "apeSees", label: str, model: "OpenSeesModel") -> None:
    """Register ``model``'s materials, sections and element specs on ``ops``.

    Names become ``{label}.{name}``; physical groups ``{label}.{pg}``.
    Registration order: uniaxial materials, nD materials, sections, then
    element specs, each in archived tag order.
    """
    _refuse_unsupported(label, model)
    names = {(kind, tag): name for name, kind, tag in model.names()}
    by_ref: dict[tuple[str, int], "Primitive"] = {}

    def prefixed(kind: str, tag: int) -> "str | None":
        name = names.get((kind, tag))
        return f"{label}.{name}" if name is not None else None

    def ref(kind: str, tag: Any) -> "Primitive":
        prim = by_ref.get((kind, int(tag)))
        if prim is None:
            raise AssemblyError(
                f"instance {label!r}: an element references {kind} tag {tag}, "
                f"which the archive does not declare."
            )
        return prim

    by_family = model.materials_by_family()
    for family, kind in _MATERIAL_FAMILIES:
        for rec in sorted(by_family.get(family, ()), key=lambda r: r.tag):
            build = _MATERIALS.get((kind, rec.type_token))
            if build is None:
                raise AssemblyError(
                    f"instance {label!r}: {kind} {rec.type_token!r} is not "
                    f"rehydrated in AS1; supported: "
                    f"{sorted(t for k, t in _MATERIALS if k == kind)}."
                )
            by_ref[(kind, rec.tag)] = build(ops, rec.params, prefixed(kind, rec.tag))

    for sec in sorted(model.sections(), key=lambda r: r.tag):
        build_sec = _SECTIONS.get(sec.type_token)
        if build_sec is None or not isinstance(sec, SectionSimpleRecord):
            raise AssemblyError(
                f"instance {label!r}: section {sec.type_token!r} is not "
                f"rehydrated in AS1; supported: {sorted(_SECTIONS)}."
            )
        by_ref[("section", sec.tag)] = build_sec(
            ops, sec.params, prefixed("section", sec.tag))

    for token, args, pg in _element_specs(label, model):
        build_el = _ELEMENTS.get(token)
        if build_el is None:
            raise AssemblyError(
                f"instance {label!r}: element {token!r} is not rehydrated in "
                f"AS1; supported: {sorted(_ELEMENTS)}."
            )
        build_el(ops, f"{label}.{pg}", args, ref)
