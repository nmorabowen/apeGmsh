"""STKO element and physical properties -> apeSees primitives (ADR 0111 D3, D5).

The registry is two dicts keyed by STKO's ``XOBJ_META``:
:data:`ELEMENT_HANDLERS` (element properties) and :data:`PROPERTY_HANDLERS`
(sections, materials, ``BeamSectionProperty``). Each entry has a status
(``"supported"`` or ``"tier_only"``), the options it cannot translate, and a
builder. A type that is not a key is an unknown STKO type. Adding a type is
one entry plus its builder.

:func:`check_supported` scans what STKO would export for the analysis
elements: their element properties and the physical-property closure
reachable from them (:func:`~.translate_types.property_references`, which
includes the fiber-group materials held inside a fiber section's custom
object). It returns one :class:`~.translate_types.Unsupported` per offending
type or option, never the first one only.

:func:`build_props` builds each reachable physical property once (memoised
by STKO id, dependencies first) and declares one element per
:class:`~.translate_types.ElementGroup`. Values are STKO's, value for value
(rules P1-P10 of ``internal_docs/stko_translator_rules.md``); the two
computations STKO does itself are ported: the ``ASDConcrete`` "Concrete (9P)"
preset (backbones and ``-autoRegularization`` length) and fiber extraction
from the stored ``FIBERS`` rows offset by the stored centroid.

Two deviations from the rules file, both forced by the Tier-1 documents:

- ``-crackPlanes`` on ``ASDConcrete3D`` (1D's concretes) is translated, through
  :class:`ASDConcrete3DCrackPlanes`, a subclass of the bridge primitive that
  adds the flag. The bridge's ``ASDConcrete3D`` has no such field yet.
- A damage value STKO computes as a round-off negative (``-2.2e-16``, from
  ``1 - s/q`` with ``s == q``; 1D's ``ASDConcrete1D``) is written as ``0.0``:
  the bridge refuses damage outside ``[0, 1)``. Anything below
  ``-DAMAGE_ROUNDOFF`` still raises.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ...opensees.material.nd import ASDConcrete3D
from ...opensees.material.uniaxial import ASDConcrete1D
from ...opensees.section.fiber import FiberPoint
from ...opensees.section.plate import ShellLayer
from .model import ScdModel, XObject
from .translate_types import (
    ElementGroup,
    PropsResult,
    Unsupported,
    UnsupportedSTKOTypes,
    Vec3,
    local_axis,
    property_references,
)

__all__ = [
    "ELEMENT_HANDLERS",
    "PROPERTY_HANDLERS",
    "ASDConcrete3DCrackPlanes",
    "ElementHandler",
    "PropertyHandler",
    "asdconcrete_9p",
    "build_props",
    "check_supported",
    "fiber_rows",
]

Status = Literal["supported", "tier_only"]
#: ``(attribute, reason)`` of one option a handler cannot translate.
Option = tuple[str, str]

_TIER_ONLY = "tier-only: not in this version"
_UNKNOWN = "unknown STKO type"
#: Largest round-off below zero accepted (and written as 0.0) in a damage list.
DAMAGE_ROUNDOFF = 1.0e-12

_TRANSFORMS = ("Linear", "PDelta", "Corotational")
_INTEGRATIONS = ("Lobatto", "Legendre", "Radau", "NewtonCotes", "Trapezoidal")
_FIBER_GROUPS = ("PUNCTUAL_FIBER_GROUPS", "SURFACE_FIBER_GROUPS", "LINEAR_FIBER_GROUPS")
_FIBER_SECTIONS = frozenset({"sections.Fiber", "sections.RectangularFiberSection"})
_SHELL_SECTIONS = frozenset({"sections.ElasticMembranePlateSection", "sections.LayeredShell"})


# ── handlers ──────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ElementHandler:
    """One element-property type: status, the physical-property types it
    takes, its untranslatable options (given the element and its physical
    property) and the builder of one element declaration per group."""

    status: Status
    dim: int = 0
    accepts: frozenset[str] = frozenset()
    options: Callable[[ScdModel, XObject, XObject], list[Option]] | None = None
    build: Callable[["_Builder", ElementGroup, XObject, XObject], Any] | None = None


@dataclass(frozen=True, slots=True)
class PropertyHandler:
    """One physical-property type: status, its untranslatable options and the
    builder of its primitive (``None`` builder: never emitted on its own)."""

    status: Status
    options: Callable[[ScdModel, XObject], list[Option]] | None = None
    build: Callable[["_Builder", XObject], Any] | None = None


class _Builder:
    """Memoised construction of primitives on one bridge."""

    def __init__(self, ops: Any, scd: ScdModel) -> None:
        self.ops = ops
        self.scd = scd
        self.prims: dict[int, Any] = {}
        self.transforms: dict[tuple[str, Vec3], Any] = {}
        self._open: set[int] = set()

    def prim(self, pid: int) -> Any:
        """The primitive of physical property ``pid``, built on first use."""
        if pid in self.prims:
            return self.prims[pid]
        if pid in self._open:
            raise ValueError(f"STKO physical property {pid} references itself")
        x = self.scd.physical_properties[pid]
        handler = PROPERTY_HANDLERS[x.type]
        if handler.build is None:
            raise ValueError(
                f"STKO physical property {pid} ({x.type}) is not emitted on its own"
            )
        self._open.add(pid)
        try:
            self.prims[pid] = handler.build(self, x)
        finally:
            self._open.discard(pid)
        return self.prims[pid]

    def transform(self, kind: str, vecxz: Vec3) -> Any:
        """Rule P4: one ``geomTransf`` per distinct ``(type, vecxz)``."""
        key = (kind, vecxz)
        if key not in self.transforms:
            ns = self.ops.geomTransf
            make = {"Linear": ns.Linear, "PDelta": ns.PDelta, "Corotational": ns.Corotational}
            self.transforms[key] = make[kind](vecxz=vecxz)
        return self.transforms[key]


# ── STKO computations (ported) ────────────────────────────────────────

def _bezier3(xi: float, x0: float, x1: float, x2: float,
             y0: float, y1: float, y2: float) -> float:
    A = x0 - 2.0 * x1 + x2
    B = 2.0 * (x1 - x0)
    C = x0 - xi
    if abs(A) < 1.0e-12:
        x1 = x1 + 1.0e-6 * (x2 - x0)
        A = x0 - 2.0 * x1 + x2
        B = 2.0 * (x1 - x0)
        C = x0 - xi
    if A == 0.0:
        return 0.0
    D = B * B - 4.0 * A * C
    t = (math.sqrt(D) - B) / (2.0 * A)
    return (y0 - 2.0 * y1 + y2) * t * t + 2.0 * (y1 - y0) * t + y0


def _lch_ref(E: float, ft: float, Gt: float, fc: float, ec: float, Gc: float) -> float:
    hmin_t = Gt / (ft * (ft / E) / 2.0) / 100.0
    ec1 = fc / E
    ec_pl = (ec - ec1) * 0.4 + ec1
    hmin_c = Gc / (fc * (ec - ec_pl) / 2.0) / 100.0
    return min(hmin_t, hmin_c)


def _tension(E: float, ft: float, Gt: float, pscale: float) -> tuple[list[float], ...]:
    f0 = ft * 0.9
    f1 = ft
    e0 = f0 / E
    e1 = ft / E * 1.5
    ep = e1 - ft / E
    f2 = 0.2 * ft
    f3 = 1.0e-3 * ft
    w2 = Gt / ft
    w3 = w2 / 0.2
    e2 = w2 + f2 / E + ep
    if e2 <= e1:
        e2 = e1 * 1.001
    e3 = w3 + f3 / E + ep
    if e3 <= e2:
        e3 = e2 * 1.001
    e4 = e3 * 10.0
    Te = [0.0, e0, e1, e2, e3, e4]
    Ts = [0.0, f0, f1, f2, f3, f3]
    Td = [0.0] * 6
    Tpl = [0.0, 0.0, ep, e2 * 0.9, e3 * 0.8, e3 * 0.8]
    if 0.0 <= pscale < 1.0:
        Tpl = [v * pscale for v in Tpl]
    for i in range(2, len(Te)):
        xipl = min(Tpl[i], Te[i] - Ts[i] / E)
        Td[i] = 1.0 - Ts[i] / ((Te[i] - xipl) * E)
    return Te, Ts, Td


def _compression(E: float, fc: float, fc0: float, fcr: float, ec: float,
                 Gc: float, pscale: float) -> tuple[list[float], ...]:
    ec0 = fc0 / E
    ec1 = fc / E
    ec_pl = (ec - ec1) * 0.4 + ec1
    Gc1 = fc * (ec - ec_pl) / 2.0
    Gc2 = max(Gc1 * 1.0e-2, Gc - Gc1)
    ecr = ec + 2.0 * Gc2 / (fc + fcr)
    Ce, Cs, Cpl = [0.0, ec0], [0.0, fc0], [0.0, 0.0]
    nc = 10
    dec = (ec - ec0) / (nc - 1)
    for i in range(nc - 1):
        iec = ec0 + (i + 1.0) * dec
        Ce.append(iec)
        Cs.append(_bezier3(iec, ec0, ec1, ec, fc0, fc, fc))
        Cpl.append(Cpl[-1] + (iec - Cpl[-1]) * 0.7)
    Ce.append(ecr)
    Cs.append(fcr)
    Cpl.append(Cpl[-1] + (ecr - Cpl[-1]) * 0.7)
    Ce.append(ecr + ec0)
    Cs.append(fcr)
    Cpl.append(Cpl[-1])
    if 0.0 <= pscale < 1.0:
        Cpl = [v * pscale for v in Cpl]
    Cd = [0.0] * len(Ce)
    for i in range(2, len(Ce)):
        xipl = min(Cpl[i], Ce[i] - Cs[i] / E)
        Cd[i] = 1.0 - Cs[i] / ((Ce[i] - xipl) * E)
    return Ce, Cs, Cd


def asdconcrete_9p(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Rule P9: STKO's ``Concrete (9P)`` preset (``ASDConcrete3D.py`` /
    ``ASDConcrete1D.py``, ``_make_hl_concrete_9p``), as written to the deck.

    Returns ``Te, Ts, Td, Ce, Cs, Cd`` (tuples) and ``lch_ref`` (the
    ``-autoRegularization`` length). Damage values are exactly STKO's.
    """
    a = attributes
    E, fc, fc0, fcr, ec = (float(a[k]) for k in ("E", "fcp", "fc0", "fcr", "ecp"))
    ft, Gt, Gc = (float(a[k]) for k in ("ft", "Gt", "Gc"))
    lch = _lch_ref(E, ft, Gt, fc, ec, Gc)
    Te, Ts, Td = _tension(E, ft, Gt / lch, float(a["PScale Tension"]))
    Ce, Cs, Cd = _compression(E, fc, fc0, fcr, ec, Gc / lch, float(a["PScale Compression"]))
    return {"Te": tuple(Te), "Ts": tuple(Ts), "Td": tuple(Td),
            "Ce": tuple(Ce), "Cs": tuple(Cs), "Cd": tuple(Cd), "lch_ref": lch}


def _damage(values: tuple[float, ...], where: str) -> tuple[float, ...]:
    out = []
    for d in values:
        if -DAMAGE_ROUNDOFF < d < 0.0:
            d = 0.0
        elif d < 0.0:
            raise ValueError(f"{where}: STKO damage {d!r} is below zero beyond round-off")
        out.append(d)
    return tuple(out)


def _items(groups: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Fiber-group items in STKO's order (``ITEM_k`` by ``k``)."""
    groups = groups or {}
    return [groups[k] for k in sorted(groups, key=lambda k: int(k.rsplit("_", 1)[1]))]


def _fiber_material(item: Mapping[str, Any]) -> int:
    return int(np.asarray(item["@attrs"]["PHYS_PROP_ID"]).ravel()[0])


def fiber_rows(x: XObject) -> list[tuple[float, float, float, int]]:
    """Rule P8: the ``fiber y z A matTag`` rows STKO writes for a
    ``sections.Fiber`` / ``sections.RectangularFiberSection``: punctual,
    then surface, then linear groups, items in ``k`` order, each stored fiber
    ``(x, y, area)`` at ``(x - cx, y - cy)`` with ``(cx, cy)`` the stored
    ``CENTER_AND_AREA`` (``(0, 0)`` with ``-noCentroid``)."""
    fs = x.attributes["Fiber section"]
    ca = np.asarray(fs["CENTER_AND_AREA"], dtype=float).ravel()
    cx, cy = (0.0, 0.0) if x.attributes.get("-noCentroid") else (float(ca[0]), float(ca[1]))
    rows: list[tuple[float, float, float, int]] = []
    for grp in _FIBER_GROUPS:
        for item in _items(fs.get(grp)):
            mat = _fiber_material(item)
            for fx, fy, fa in np.asarray(item["FIBERS"], dtype=float).reshape(-1, 3):
                rows.append((float(fx) - cx, float(fy) - cy, float(fa), mat))
    return rows


# ── a bridge primitive STKO needs and the bridge lacks ────────────────

@dataclass(frozen=True, kw_only=True, slots=True)
class ASDConcrete3DCrackPlanes(ASDConcrete3D):
    """``nDMaterial ASDConcrete3D`` with ``-crackPlanes $nct $ncc
    $smoothingAngle`` (STKO's ``-crackPlanes`` option; the parser takes the
    flag in any position). Everything else is :class:`ASDConcrete3D`'s."""

    crack_planes: tuple[int, int, float]

    def _emit(self, emitter: Any, tag: int) -> None:
        args: list[float | int | str] = [
            self.E, self.v,
            "-Te", *self.Te, "-Ts", *self.Ts, "-Td", *self.Td,
            "-Ce", *self.Ce, "-Cs", *self.Cs, "-Cd", *self.Cd,
            "-rho", self.rho, "-Kc", self.Kc,
        ]
        if self.eta:
            args += ["-eta", self.eta]
        if self.cdf:
            args += ["-cdf", self.cdf]
        if self.implex:
            args.append("-implex")
        nct, ncc, angle = self.crack_planes
        args += ["-crackPlanes", int(nct), int(ncc), float(angle)]
        if self.tangent == "numerical":
            args.append("-tangent")
        if self.auto_regularize:
            args += ["-autoRegularization", self.lch_ref]
        emitter.nDMaterial("ASDConcrete3D", tag, *args)


# ── option checks ─────────────────────────────────────────────────────

def _is_2d(x: XObject) -> bool:
    return bool(x.attributes.get("2D")) or x.attributes.get("Dimension") == "2D"


def _dim_option(x: XObject) -> list[Option]:
    return [("Dimension", "2-D model: only 3-D (ndm 3, ndf 6) is translated")] if _is_2d(x) else []


def _offset_options(x: XObject) -> list[Option]:
    out = []
    for name in ("Y/section_offset", "Z/section_offset"):
        if float(x.attributes.get(name) or 0.0) != 0.0:
            out.append((name, "a section offset makes STKO write -jntOffset: not in this version"))
    return out


def _transform_option(a: Mapping[str, Any], name: str) -> list[Option]:
    if a.get(name) not in _TRANSFORMS:
        return [(name, f"geomTransf {a.get(name)!r} is not one of {list(_TRANSFORMS)}")]
    return []


def _shell_options(scd: ScdModel, ep: XObject, pp: XObject) -> list[Option]:
    a = ep.attributes
    out: list[Option] = []
    if a.get("Kinematics") not in ("Linear", "Corotational"):
        out.append(("Kinematics", f"Kinematics {a.get('Kinematics')!r} is not Linear / Corotational"))
    if a.get("Drilling DOF Type") == "Elastic Drilling DOF":
        stab = float(f"{float(a['Drilling Stabilization']):.6g}")
        if not 0.0 <= stab <= 1.0:
            out.append(("Drilling Stabilization", f"{stab} is outside [0, 1]"))
    return out


def _elastic_beam_options(scd: ScdModel, ep: XObject, pp: XObject) -> list[Option]:
    a = ep.attributes
    out = _dim_option(ep) + _transform_option(a, "transfType")
    for flag in ("-releasey", "-releasez", "-cMass"):
        if a.get(flag):
            out.append((flag, f"{flag} is not in this version"))
    return out


def _force_beam_options(scd: ScdModel, ep: XObject, pp: XObject) -> list[Option]:
    a = ep.attributes
    out = _dim_option(ep) + _transform_option(a, "transType")
    if a.get("-cMass"):
        out.append(("-cMass", "-cMass is not in this version"))
    return out


def _elastic_section_options(scd: ScdModel, x: XObject) -> list[Option]:
    out = _dim_option(x) + _offset_options(x)
    if x.attributes.get("Use Uniaxial Materials"):
        out.append(("Use Uniaxial Materials", "STKO writes an Aggregator section: not in this version"))
    props = np.asarray((x.attributes.get("Section") or {}).get("PROPS", ()), dtype=float).ravel()
    if len(props) < 4:
        out.append(("Section", f"Section/PROPS has {len(props)} values; A, I, I, J need 4"))
    elif props[1] != props[2]:
        out.append(("Section", (
            f"Section/PROPS[1] = {props[1]!r} != PROPS[2] = {props[2]!r}: which one is Iyy "
            "and which Izz is unverified (STKO's exporter reads Section.properties.Iyy / "
            ".Izz from the compiled MpcBeamSection; every section in the 281 readable "
            "San Ramon documents is square), so a non-square section is refused")))
    return out


def _layered_options(scd: ScdModel, x: XObject) -> list[Option]:
    mats = tuple(x.attributes.get("matTag") or ())
    thick = tuple(x.attributes.get("thickness") or ())
    out: list[Option] = []
    if len(mats) != len(thick):
        out.append(("matTag", f"{len(mats)} materials for {len(thick)} thicknesses"))
    if len(mats) < 3:
        out.append(("matTag", f"{len(mats)} layers: OpenSees' LayeredShell needs at least 3"))
    for m in mats:
        p = scd.physical_properties.get(int(m or 0))
        if p is None or not p.type.startswith("materials.nD."):
            out.append(("matTag", f"layer material {m} is not an nD material of the document"))
    return out


def _fiber_options(scd: ScdModel, x: XObject) -> list[Option]:
    a = x.attributes
    out = _dim_option(x) + _offset_options(x)
    if a.get("-torsion"):
        out.append(("-torsion", "-torsion (a torsion material) is not in this version"))
    if a.get("-noCentroid"):
        out.append(("-noCentroid", "-noCentroid is not in this version"))
    if x.type == "sections.RectangularFiberSection" and not a.get("Concrete (Core) Material"):
        out.append(("Concrete (Core) Material",
                    "no core material: STKO generates a confined material on the fly"))
    fs = a.get("Fiber section")
    if not isinstance(fs, Mapping):
        return out + [("Fiber section", "the section holds no fiber data")]
    rows = 0
    for grp in _FIBER_GROUPS:
        for item in _items(fs.get(grp)):
            fib = np.asarray(item.get("FIBERS", ()), dtype=float).reshape(-1, 3)
            rows += len(fib)
            mid = _fiber_material(item)
            p = scd.physical_properties.get(mid)
            if p is None or not p.type.startswith("materials.uniaxial."):
                out.append(("Fiber section", f"fiber material {mid} is not a uniaxial material of the document"))
            if len(fib) and float(fib[:, 2].min()) <= 0.0:
                out.append(("Fiber section", "a fiber has a non-positive area"))
    if rows == 0:
        out.append(("Fiber section", "the section has no fibers"))
    return out


def _beam_section_property_options(scd: ScdModel, x: XObject) -> list[Option]:
    a = x.attributes
    if a.get("Option") != "StandardIntegrationTypes":
        return [("Option", f"integration option {a.get('Option')!r} is not in this version")]
    out: list[Option] = []
    if a.get("IntegrationType/1") not in _INTEGRATIONS:
        out.append(("IntegrationType/1", f"{a.get('IntegrationType/1')!r} is not one of {list(_INTEGRATIONS)}"))
    sec = scd.physical_properties.get(int(a.get("secTag/1") or 0))
    if sec is None:
        out.append(("secTag/1", "no section"))
    elif sec.type not in _FIBER_SECTIONS:
        out.append(("secTag/1", f"a {sec.type} integration section is not in this version"))
    else:
        out += [(f"secTag/1:{n}", r) for n, r in _offset_options(sec)]
    if int(a.get("numIntPts/1") or 0) < 1:
        out.append(("numIntPts/1", "fewer than 1 integration point"))
    return out


def _asd_concrete_options(scd: ScdModel, x: XObject) -> list[Option]:
    a = x.attributes
    out: list[Option] = []
    if a.get("Preset") != "Concrete (9P)":
        out.append(("Preset", f"preset {a.get('Preset')!r}: only 'Concrete (9P)' is in this version"))
    implex = a.get("integration")
    if implex not in ("Implicit", "IMPL-EX"):
        out.append(("integration", f"integration {implex!r} is not Implicit / IMPL-EX"))
    if implex == "IMPL-EX" and float(a.get("implexAlpha", 1.0)) != 1.0:
        out.append(("implexAlpha", "implexAlpha != 1.0: apeSees writes no -implexAlpha"))
    if a.get("constitutiveTensorType") == "Tangent":
        if x.type == "materials.uniaxial.ASDConcrete1D":
            out.append(("constitutiveTensorType", "-tangent on ASDConcrete1D is not in this version"))
        elif implex == "IMPL-EX":
            out.append(("constitutiveTensorType", "-tangent with IMPL-EX (OpenSees ignores it)"))
    return out


# ── builders ──────────────────────────────────────────────────────────

def _build_shell(b: _Builder, g: ElementGroup, ep: XObject, pp: XObject) -> Any:
    """Rule P1."""
    a = ep.attributes
    elastic = a["Drilling DOF Type"] == "Elastic Drilling DOF"
    return b.ops.element.ASDShellQ4(
        pg=g.pg,
        section=b.prim(pp.id),
        corotational=a["Kinematics"] == "Corotational",
        no_eas=not a["Use EAS"],
        drilling_stab=float(f"{float(a['Drilling Stabilization']):.6g}") if elastic else None,
        drilling_nl=not elastic,
        local_cs=tuple(g.local_axis),
    )


def _build_elastic_beam(b: _Builder, g: ElementGroup, ep: XObject, pp: XObject) -> Any:
    """Rule P2: the section's values times its modifiers; the section is
    not referenced (and not emitted). ``PROPS[1] == PROPS[2]`` is guaranteed by
    :func:`check_supported` (the Iyy / Izz order of ``PROPS`` is unverified)."""
    a, s = ep.attributes, pp.attributes
    props = np.asarray(s["Section"]["PROPS"], dtype=float).ravel()
    return b.ops.element.elasticBeamColumn(
        pg=g.pg,
        transf=b.transform(a["transfType"], g.local_axis),
        A=float(props[0]) * float(s["A_modifier"]),
        E=float(s["E"]),
        Iz=float(props[2]) * float(s["Izz_modifier"]),
        Iy=float(props[1]) * float(s["Iyy_modifier"]),
        G=float(s["G"]),
        J=float(props[3]) * float(s["J_modifier"]),
        mass=float(a["massDens"]) if a.get("-mass") else None,
    )


def _build_force_beam(b: _Builder, g: ElementGroup, ep: XObject, pp: XObject) -> Any:
    """Rule P3: the BeamSectionProperty resolves to the integration."""
    a = ep.attributes
    it = bool(a.get("-iter"))
    return b.ops.element.forceBeamColumn(
        pg=g.pg,
        transf=b.transform(a["transType"], g.local_axis),
        integration=b.prim(pp.id),
        mass=float(a["massDens"]) if a.get("-mass") else None,
        max_iter=int(a["maxIters"]) if it else None,
        tol=float(a["tol"]) if it else None,
    )


def _build_beam_section_property(b: _Builder, x: XObject) -> Any:
    a = x.attributes
    ns = b.ops.beamIntegration
    make = {"Lobatto": ns.Lobatto, "Legendre": ns.Legendre, "Radau": ns.Radau,
            "NewtonCotes": ns.NewtonCotes, "Trapezoidal": ns.Trapezoidal}
    return make[a["IntegrationType/1"]](section=b.prim(int(a["secTag/1"])), n_ip=int(a["numIntPts/1"]))


def _build_emps(b: _Builder, x: XObject) -> Any:
    """Rule P5."""
    a = x.attributes
    return b.ops.section.ElasticMembranePlateSection(
        E=float(a["E"]), nu=float(a["nu"]), h=float(a["h"]),
        rho=float(a["rho"]), Ep_mod=float(a["Ep_mod"]),
    )


def _build_layered(b: _Builder, x: XObject) -> Any:
    """Rule P6: layers in order, each ``(matTag_i, thickness_i)``."""
    a = x.attributes
    layers = tuple(
        ShellLayer(material=b.prim(int(m)), thickness=float(t))
        for m, t in zip(a["matTag"], a["thickness"], strict=True)
    )
    return b.ops.section.LayeredShell(layers=layers)


def _build_fiber(b: _Builder, x: XObject) -> Any:
    """Rule P8: the stored fibers, never recomputed from the outline."""
    a = x.attributes
    fibers = tuple(
        FiberPoint(material=b.prim(mat), y=y, z=z, area=area)
        for y, z, area, mat in fiber_rows(x)
    )
    return b.ops.section.Fiber(fibers=fibers, GJ=float(a["GJ"]) if a.get("-GJ") else None)


def _build_plate_from_plane_stress(b: _Builder, x: XObject) -> Any:
    """Rule P7."""
    a = x.attributes
    return b.ops.nDMaterial.PlateFromPlaneStress(
        material=b.prim(int(a["matTag"])), G_out=float(a["OutofPlaneModulus"]))


def _build_plate_rebar(b: _Builder, x: XObject) -> Any:
    """Rule P7."""
    a = x.attributes
    return b.ops.nDMaterial.PlateRebar(material=b.prim(int(a["matTag"])), angle=float(a["sita"]))


def _build_asd_concrete_3d(b: _Builder, x: XObject) -> Any:
    """Rule P9, 3-D: the raw backbone class, registered."""
    a = x.attributes
    law = asdconcrete_9p(a)
    kw: dict[str, Any] = dict(
        E=float(a["E"]), v=float(a["v"]),
        Te=law["Te"], Ts=law["Ts"], Td=_damage(law["Td"], x.name),
        Ce=law["Ce"], Cs=law["Cs"], Cd=_damage(law["Cd"], x.name),
        lch_ref=law["lch_ref"], rho=float(a["rho"]), Kc=float(a["Kc"]),
        eta=float(a["eta"]), cdf=float(a["CDF"]),
        implex=a["integration"] == "IMPL-EX",
        tangent="numerical" if a["constitutiveTensorType"] == "Tangent" else "secant",
    )
    if a.get("-crackPlanes"):
        prim: ASDConcrete3D = ASDConcrete3DCrackPlanes(
            **kw, crack_planes=(int(a["nct"]), int(a["ncc"]), float(a["smoothingAngle"])))
    else:
        prim = ASDConcrete3D(**kw)
    return b.ops.register(prim)


def _build_asd_concrete_1d(b: _Builder, x: XObject) -> Any:
    """Rule P9, 1-D: no ``v``, ``rho``, ``Kc``, ``cdf``."""
    a = x.attributes
    law = asdconcrete_9p(a)
    return b.ops.register(ASDConcrete1D(
        E=float(a["E"]),
        Te=law["Te"], Ts=law["Ts"], Td=_damage(law["Td"], x.name),
        Ce=law["Ce"], Cs=law["Cs"], Cd=_damage(law["Cd"], x.name),
        lch_ref=law["lch_ref"], eta=float(a["eta"]),
        implex=a["integration"] == "IMPL-EX",
    ))


def _build_hysteretic(b: _Builder, x: XObject) -> Any:
    """Rule P10: the third points when ``Optional``, ``beta`` when ``use_beta``."""
    a = x.attributes
    third = bool(a.get("Optional"))
    return b.ops.uniaxialMaterial.Hysteretic(
        s1p=float(a["s1p"]), e1p=float(a["e1p"]), s2p=float(a["s2p"]), e2p=float(a["e2p"]),
        s1n=float(a["s1n"]), e1n=float(a["e1n"]), s2n=float(a["s2n"]), e2n=float(a["e2n"]),
        pinch_x=float(a["pinchx"]), pinch_y=float(a["pinchy"]),
        damage1=float(a["damage1"]), damage2=float(a["damage2"]),
        s3p=float(a["s3p"]) if third else None, e3p=float(a["e3p"]) if third else None,
        s3n=float(a["s3n"]) if third else None, e3n=float(a["e3n"]) if third else None,
        beta=float(a["beta"]) if a.get("use_beta") else 0.0,
    )


# ── the registry ──────────────────────────────────────────────────────

_TIER = ElementHandler("tier_only")
ELEMENT_HANDLERS: dict[str, ElementHandler] = {
    "shell.ASDShellQ4": ElementHandler(
        "supported", 2, _SHELL_SECTIONS, _shell_options, _build_shell),
    "beam_column_elements.elasticBeamColumn": ElementHandler(
        "supported", 1, frozenset({"sections.Elastic"}), _elastic_beam_options, _build_elastic_beam),
    "beam_column_elements.forceBeamColumn": ElementHandler(
        "supported", 1, frozenset({"special_purpose.BeamSectionProperty"}),
        _force_beam_options, _build_force_beam),
    "brick_elements.stdBrick": _TIER,
    "zero_length_elements.zeroLength": _TIER,
    "absorbingBoundaries.ASDAbsorbingBoundary3DAuto": _TIER,
}

_TIER_P = PropertyHandler("tier_only")
PROPERTY_HANDLERS: dict[str, PropertyHandler] = {
    # consumed by elasticBeamColumn (rule P2), never emitted on its own
    "sections.Elastic": PropertyHandler("supported", _elastic_section_options),
    "sections.ElasticMembranePlateSection": PropertyHandler(
        "supported", lambda scd, x: [], _build_emps),
    "sections.LayeredShell": PropertyHandler("supported", _layered_options, _build_layered),
    "sections.Fiber": PropertyHandler("supported", _fiber_options, _build_fiber),
    "sections.RectangularFiberSection": PropertyHandler("supported", _fiber_options, _build_fiber),
    # resolves to the beam integration of rule P3
    "special_purpose.BeamSectionProperty": PropertyHandler(
        "supported", _beam_section_property_options, _build_beam_section_property),
    "materials.nD.ASDConcrete3D": PropertyHandler(
        "supported", _asd_concrete_options, _build_asd_concrete_3d),
    "materials.nD.PlateFromPlaneStress": PropertyHandler(
        "supported", lambda scd, x: [], _build_plate_from_plane_stress),
    "materials.nD.PlateRebar": PropertyHandler(
        "supported", lambda scd, x: [], _build_plate_rebar),
    "materials.uniaxial.ASDConcrete1D": PropertyHandler(
        "supported", _asd_concrete_options, _build_asd_concrete_1d),
    "materials.uniaxial.Hysteretic": PropertyHandler(
        "supported", lambda scd, x: [], _build_hysteretic),
    "materials.nD.ElasticIsotropic": _TIER_P,
    "materials.nD.ASDAbsorbingBoundary3DMaterial": _TIER_P,
    "special_purpose.zeroLengthMaterial": _TIER_P,
    "materials.uniaxial.Elastic": _TIER_P,
}


# ── public entry points ───────────────────────────────────────────────

def check_supported(scd: ScdModel) -> list[Unsupported]:
    """The props slice of the registry (rules doc section 4).

    Scans the element properties of the analysis elements on sub-shapes, the
    element and physical properties every interaction carries (the
    interaction itself belongs to the mesh slice; its properties are named
    here, e.g. 2A's ``zeroLength`` links), the physical-property closure
    reachable from all of them, their options, the element/physical-property
    pairing, and the transforms (one ``vecxz`` cannot carry two
    ``geomTransf`` types). Every offender is listed; nothing is skipped.
    """
    found: dict[tuple[str, str, str], tuple[set[int], set[str]]] = {}

    def add(category: str, meta: str, x: XObject, reason: str) -> None:
        ids, names = found.setdefault((category, meta, reason), (set(), set()))
        ids.add(x.id)
        names.add(x.name)

    pairs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for a in scd.analysis_elements().values():
        if a.interaction is None:
            pairs[(a.element_property, a.physical_property)].append(a.id)

    roots: set[int] = set()
    transfer: dict[Vec3, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for (epid, ppid), eids in sorted(pairs.items()):
        ep = scd.element_properties[epid]
        eh = ELEMENT_HANDLERS.get(ep.type)
        if eh is None or eh.status != "supported":
            add("element_property", ep.type, ep, _UNKNOWN if eh is None else _TIER_ONLY)
            if ppid in scd.physical_properties:
                roots.add(ppid)
            continue
        pp = scd.physical_properties.get(ppid)
        if pp is None:
            add("option", f"{ep.type}:physical property", ep,
                "an analysis element without a physical property")
            continue
        roots.add(ppid)
        if pp.type not in eh.accepts:
            add("option", f"{ep.type}:physical property", ep,
                f"{pp.type} is not a physical property this element takes "
                f"(takes {sorted(eh.accepts)})")
            continue
        assert eh.options is not None
        for attr, reason in eh.options(scd, ep, pp):
            add("option", f"{ep.type}:{attr}", ep, reason)
        kind = ep.attributes.get("transfType", ep.attributes.get("transType"))
        if eh.dim == 1 and kind in _TRANSFORMS:
            for e in eids:
                transfer[local_axis(scd, e)][str(kind)].add(epid)

    for vec, kinds in transfer.items():
        if len(kinds) > 1:
            for epid in sorted(set().union(*kinds.values())):
                ep = scd.element_properties[epid]
                add("option", f"{ep.type}:geomTransf", ep,
                    f"vecxz {vec} carries geomTransf types {sorted(kinds)}")

    # element / physical properties an interaction carries (2A's zeroLength links):
    # interaction elements are never translated, so a supported type there is refused
    # too, and its physical-property closure is scanned like any other.
    for inter in sorted(scd.interactions.values(), key=lambda i: i.id):
        iep = scd.element_properties.get(inter.element_property)
        if iep is not None:     # a missing id: the mesh slice reports the interaction itself
            ieh = ELEMENT_HANDLERS.get(iep.type)
            if ieh is None or ieh.status != "supported":
                add("element_property", iep.type, iep, _UNKNOWN if ieh is None else _TIER_ONLY)
            else:
                add("option", f"{iep.type}:on interaction", iep,
                    f"carried by interaction {inter.id} ({inter.name!r}): elements generated "
                    "by an interaction are not translated")
        if inter.physical_property in scd.physical_properties:
            roots.add(inter.physical_property)

    closure = set(roots)
    for r in roots:
        closure |= property_references(scd, r)
    for pid in sorted(closure):
        x = scd.physical_properties[pid]
        ph = PROPERTY_HANDLERS.get(x.type)
        if ph is None or ph.status != "supported":
            add("physical_property", x.type, x, _UNKNOWN if ph is None else _TIER_ONLY)
            continue
        if ph.options is not None:
            for attr, reason in ph.options(scd, x):
                add("option", f"{x.type}:{attr}", x, reason)

    return [
        Unsupported(category=cat, xobj_meta=meta, ids=tuple(sorted(ids)),  # type: ignore[arg-type]
                    names=tuple(sorted(names)), reason=reason)
        for (cat, meta, reason), (ids, names) in sorted(found.items())
    ]


def build_props(ops: Any, scd: ScdModel, groups: Sequence[ElementGroup]) -> PropsResult:
    """Rules P1-P10: declare one element per group and every physical
    property it reaches, once, dependencies first.

    Raises :class:`UnsupportedSTKOTypes` (listing everything) before
    touching ``ops`` when :func:`check_supported` finds anything, so the
    function is safe to call on its own. ``PropsResult.primitives`` maps
    each built physical property's STKO id to its primitive (a
    ``BeamSectionProperty`` to its beam integration; a ``sections.Elastic``
    read by ``elasticBeamColumn`` is not built). ``transforms`` maps each
    ``vecxz`` to its ``geomTransf``.
    """
    problems = check_supported(scd)
    if problems:
        raise UnsupportedSTKOTypes(tuple(problems))
    b = _Builder(ops, scd)
    elements: dict[str, Any] = {}
    for g in groups:
        ep = scd.element_properties[g.element_property]
        pp = scd.physical_properties[g.physical_property]
        eh = ELEMENT_HANDLERS[ep.type]
        if eh.dim != g.dim:
            raise ValueError(
                f"element group {g.pg!r}: {ep.type} is a {eh.dim}-D element, the group is {g.dim}-D"
            )
        if g.pg in elements:
            raise ValueError(f"two element groups share the physical group {g.pg!r}")
        assert eh.build is not None
        elements[g.pg] = eh.build(b, g, ep, pp)
    return PropsResult(
        primitives=dict(b.prims),
        elements=elements,
        transforms={vec: t for (_, vec), t in b.transforms.items()},
    )
