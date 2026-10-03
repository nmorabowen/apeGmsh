"""STKO translator, properties (ADR 0111; rules P1-P11 and the props slice
of the registry in ``internal_docs/stko_translator_rules.md``).

The synthetic tests build an :class:`ScdModel` directly (the reader's
encodings are covered by ``test_stko_reader.py``): one geometry whose faces
and edges each carry one element, an element property and a physical
property. Expected numbers are STKO's own, copied from its Tcl exports of
the San Ramon documents. The oracle test at the end compares every
reachable property with STKO's decks when ``APEGMSH_STKO_ORACLES`` points
at ``models/stko-rev0-tcl``.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh.interop.stko.model import (
    Geometry,
    Interaction,
    Mesh,
    MeshElement,
    ScdModel,
    SubShapeRef,
    XObject,
)
from apeGmsh.interop.stko.translate_props import (
    ELEMENT_HANDLERS,
    PROPERTY_HANDLERS,
    ASDConcrete3DCrackPlanes,
    asdconcrete_9p,
    build_props,
    check_supported,
    fiber_rows,
)
from apeGmsh.interop.stko.translate_types import (
    ElementGroup,
    UnsupportedSTKOTypes,
    local_axis,
)
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ASDConcrete3D
from apeGmsh.opensees.material.uniaxial import ASDConcrete1D

# ── synthetic documents ───────────────────────────────────────────────

IDENTITY = (0.0, 0.0, 0.0, 1.0)
#: Quaternion of a 90 deg rotation about z: local x = global y, vecxz = z.
ROT_Z90 = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))


def _x(oid: int, xtype: str, name: str | None = None, refs: tuple[str, ...] = (), **attrs: Any) -> XObject:
    return XObject(id=oid, name=name or f"{xtype.rsplit('.', 1)[-1]}_{oid}", type=xtype,
                   attributes=dict(attrs), references=frozenset(refs))


def shell_ep(oid: int, **kw: Any) -> XObject:
    a = {"Drilling DOF Type": "Elastic Drilling DOF", "Drilling Stabilization": 0.01,
         "Kinematics": "Linear", "Use EAS": True}
    a.update(kw)
    return _x(oid, "shell.ASDShellQ4", **a)


def elastic_beam_ep(oid: int, **kw: Any) -> XObject:
    a = {"-alpha": False, "-cMass": False, "-depth": False, "-mass": False, "-releasey": False,
         "-releasez": False, "2D": False, "3D": True, "Dimension": "3D", "massDens": 0.0,
         "transfType": "Linear"}
    a.update(kw)
    return _x(oid, "beam_column_elements.elasticBeamColumn", **a)


def force_beam_ep(oid: int, **kw: Any) -> XObject:
    a = {"-cMass": False, "-iter": False, "-mass": False, "2D": False, "3D": True,
         "Dimension": "3D", "massDens": 0.0, "maxIters": 10, "tol": 1e-11, "transType": "Linear"}
    a.update(kw)
    return _x(oid, "beam_column_elements.forceBeamColumn", **a)


def emps(oid: int, **kw: Any) -> XObject:
    a = {"E": 25000.0, "Ep_mod": 1.0, "h": 200.0, "nu": 0.2, "rho": 0.0}
    a.update(kw)
    return _x(oid, "sections.ElasticMembranePlateSection", **a)


def elastic_section(oid: int, props: tuple[float, ...], **kw: Any) -> XObject:
    a = {"A_modifier": 1.0, "Iyy_modifier": 1.0, "Izz_modifier": 1.0, "J_modifier": 1.0,
         "Dimension": "3D", "E": 25000.0, "G": 10500.0, "Eb material": 0, "Em material": 0,
         "G material": 0, "Section": {"PROPS": np.asarray(props, dtype=float)},
         "Shear Deformable": False, "Use Uniaxial Materials": False,
         "Y/section_offset": 0.0, "Z/section_offset": 0.0}
    a.update(kw)
    return _x(oid, "sections.Elastic", refs=("Eb material", "Em material", "G material"), **a)


#: 1B's ASDConcrete_UC (STKO deck: materials.tcl, nDMaterial 6).
UC = {"E": 25000.0, "v": 0.2, "fcp": 28.0, "fc0": 14.0, "fcr": 5.6, "ecp": 0.003, "ft": 2.8,
      "Gt": 0.1, "Gc": 30.0, "PScale Tension": 0.3, "PScale Compression": 0.3,
      "Preset": "Concrete (9P)", "integration": "IMPL-EX", "implexAlpha": 1.0,
      "constitutiveTensorType": "Secant", "rho": 0.0, "eta": 0.0, "Kc": 2.0 / 3.0, "CDF": 0.0,
      "-crackPlanes": False, "nct": 4, "ncc": 4, "smoothingAngle": 45.0}


def asd3d(oid: int, **kw: Any) -> XObject:
    a = dict(UC)
    a.update(kw)
    return _x(oid, "materials.nD.ASDConcrete3D", **a)


def asd1d(oid: int, **kw: Any) -> XObject:
    a = {k: v for k, v in UC.items() if k not in ("v", "rho", "Kc", "CDF", "-crackPlanes", "nct", "ncc", "smoothingAngle")}
    a.update(kw)
    return _x(oid, "materials.uniaxial.ASDConcrete1D", **a)


def hysteretic(oid: int, **kw: Any) -> XObject:
    a = {"Optional": True, "beta": 0.0, "use_beta": False, "damage1": 0.0, "damage2": 0.0,
         "s1p": 414.0, "e1p": 0.002, "s2p": 415.0, "e2p": 0.02, "s3p": 550.0, "e3p": 0.1,
         "s1n": -414.0, "e1n": -0.002, "s2n": -415.0, "e2n": -0.02, "s3n": -550.0, "e3n": -0.1,
         "pinchx": 0.5, "pinchy": 0.8}
    a.update(kw)
    return _x(oid, "materials.uniaxial.Hysteretic", **a)


def fiber_section(oid: int, groups: dict[str, list[tuple[int, list[tuple[float, float, float]]]]],
                  center: tuple[float, float] = (0.0, 0.0), *, rectangular: bool = False,
                  **kw: Any) -> XObject:
    """``groups``: group name -> [(material, [(x, y, area), ...]), ...] in ITEM order."""
    fs: dict[str, Any] = {"CENTER_AND_AREA": np.array([*center, 1.0])}
    for grp in ("PUNCTUAL_FIBER_GROUPS", "SURFACE_FIBER_GROUPS", "LINEAR_FIBER_GROUPS"):
        fs[grp] = {
            f"ITEM_{k}": {"FIBERS": np.asarray(rows, dtype=float),
                          "@attrs": {"PHYS_PROP_ID": np.array([mat], dtype=np.int32)}}
            for k, (mat, rows) in enumerate(groups.get(grp, []))
        }
    a = {"-GJ": True, "GJ": 1.0e15, "-torsion": False, "2D": False, "3D": True, "Dimension": "3D",
         "Fiber section": fs, "Y/section_offset": 0.0, "Z/section_offset": 0.0, "torsionMatTag": 0}
    refs: tuple[str, ...] = ("torsionMatTag",)
    if rectangular:
        a.update({"Concrete (Core) Material": 0, "Concrete (Cover) Material": 0,
                  "Rebars Material": 0, "Width": 700.0, "Height": 700.0})
        refs += ("Concrete (Core) Material", "Concrete (Cover) Material", "Rebars Material")
        xtype = "sections.RectangularFiberSection"
    else:
        a["-noCentroid"] = False
        xtype = "sections.Fiber"
    a.update(kw)
    return _x(oid, xtype, refs=refs, **a)


def bsp(oid: int, sec: int, *, kind: str = "Lobatto", n: int = 5, **kw: Any) -> XObject:
    a = {"Option": "StandardIntegrationTypes", "IntegrationType/1": kind, "secTag/1": sec,
         "numIntPts/1": n, "secTagE/3": 0, "secTagI/3": 0, "secTagJ/3": 0}
    a.update(kw)
    return _x(oid, "special_purpose.BeamSectionProperty",
              refs=("secTag/1", "secTagE/3", "secTagI/3", "secTagJ/3"), **a)


def model(pps: list[XObject], eps: list[XObject], *, faces: list[tuple[int, int]] = (),
          edges: list[tuple[int, int]] = (), quats: dict[str, list[tuple[float, ...]]] | None = None,
          ) -> ScdModel:
    """One geometry: face ``i`` carries quad ``100 + i`` with ``faces[i] = (ep, pp)``,
    edge ``j`` carries line ``200 + j`` with ``edges[j] = (ep, pp)``."""
    quats = quats or {}
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elements: dict[int, MeshElement] = {}
    orientation: dict[int, tuple[float, float, float, float]] = {}
    domains: dict[tuple[int, str], dict[int, np.ndarray]] = {(1, "faces"): {}, (1, "edges"): {}}
    for i, _ in enumerate(faces):
        elements[100 + i] = MeshElement(303, 201, (1, 2, 3, 4))
        orientation[100 + i] = tuple(quats.get("faces", [IDENTITY] * len(faces))[i])  # type: ignore[assignment]
        domains[(1, "faces")][i] = np.array([100 + i])
    for j, _ in enumerate(edges):
        elements[200 + j] = MeshElement(102, 2, (1, 5))
        orientation[200 + j] = tuple(quats.get("edges", [IDENTITY] * len(edges))[j])  # type: ignore[assignment]
        domains[(1, "edges")][j] = np.array([200 + j])
    geom = Geometry(
        id=1, name="g", shape_index=1, counts={"faces": len(faces), "edges": len(edges)},
        element_property={"faces": np.array([e for e, _ in faces], dtype=int),
                          "edges": np.array([e for e, _ in edges], dtype=int)},
        physical_property={"faces": np.array([p for _, p in faces], dtype=int),
                           "edges": np.array([p for _, p in edges], dtype=int)},
        local_axes={},
    )
    mesh = Mesh(node_ids=np.arange(1, 6), coordinates=coords, node_flags=np.zeros(5, dtype=int),
                elements=elements, domains=domains, vertex_nodes={1: np.arange(1, 6)},
                orientation=orientation)
    return ScdModel(path=Path("synthetic.scd"), version=(4, 1, 0), geometries={1: geom}, mesh=mesh,
                    selection_sets={}, physical_properties={x.id: x for x in pps},
                    element_properties={x.id: x for x in eps}, conditions={}, interactions={},
                    local_axes={}, definitions={}, analysis_steps={})


def groups(scd: ScdModel) -> list[ElementGroup]:
    """Element groups keyed as the mesh module keys them (rule M6)."""
    acc: dict[tuple[int, int, tuple[float, ...]], list[int]] = defaultdict(list)
    for a in scd.analysis_elements().values():
        if a.interaction is None:
            acc[(a.element_property, a.physical_property, local_axis(scd, a.id))].append(a.id)
    return [
        ElementGroup(pg=f"g{i}", dim=2 if len(scd.mesh.elements[ids[0]].nodes) == 4 else 1,
                     element_property=ep, physical_property=pp, local_axis=ax,  # type: ignore[arg-type]
                     element_ids=tuple(sorted(ids)))
        for i, ((ep, pp, ax), ids) in enumerate(sorted(acc.items()))
    ]


def bridge() -> apeSees:
    ops = apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    return ops


def emitted(prim: Any, tags: dict[int, int] | None = None, tag: int = 1) -> list[tuple]:
    rec = RecordingEmitter()
    tags = tags or {}
    set_tag_resolver(rec, lambda p: tags[id(p)])
    prim._emit(rec, tag)
    return rec.calls


def nl_model() -> ScdModel:
    """A 1C/1D-like document: a layered-shell wall and a fiber column."""
    pps = [
        asd3d(6), hysteretic(8), asd1d(9), asd1d(10, Gc=90.0, ecp=0.006, fc0=20.0, fcp=35.0, fcr=7.0, ft=3.5),
        _x(13, "materials.nD.PlateFromPlaneStress", refs=("matTag",), matTag=6, OutofPlaneModulus=10500.0),
        _x(15, "materials.nD.PlateRebar", refs=("matTag",), matTag=8, sita=0.0),
        _x(16, "materials.nD.PlateRebar", refs=("matTag",), matTag=8, sita=90.0),
        _x(17, "sections.LayeredShell", refs=("matTag",), matTag=(13, 15, 16, 6, 16, 15, 13),
           thickness=(25.0, 0.5, 0.4, 250.0, 0.4, 0.5, 25.0)),
        fiber_section(11, {
            "PUNCTUAL_FIBER_GROUPS": [(8, [(110.0, 210.0, 380.0)]), (8, [(90.0, 190.0, 380.0)])],
            "SURFACE_FIBER_GROUPS": [(10, [(100.0, 200.0, 1000.0), (101.0, 199.0, 1000.0)]),
                                     (9, [(80.0, 180.0, 50.0)])],
        }, center=(100.0, 200.0), rectangular=True,
            **{"Concrete (Core) Material": 10, "Concrete (Cover) Material": 9, "Rebars Material": 8}),
        bsp(12, 11, n=5),
    ]
    eps = [shell_ep(3, **{"Drilling DOF Type": "Non-Linear Drilling DOF", "Use EAS": False}),
           force_beam_ep(7)]
    return model(pps, eps, faces=[(3, 17), (3, 17)], edges=[(7, 12), (7, 12)])


# ── registry ──────────────────────────────────────────────────────────

TIER1_ELEMENTS = ("shell.ASDShellQ4", "beam_column_elements.elasticBeamColumn",
                  "beam_column_elements.forceBeamColumn")
TIER1_PROPERTIES = ("sections.Elastic", "sections.ElasticMembranePlateSection", "sections.LayeredShell",
                    "sections.Fiber", "sections.RectangularFiberSection",
                    "special_purpose.BeamSectionProperty", "materials.nD.ASDConcrete3D",
                    "materials.nD.PlateFromPlaneStress", "materials.nD.PlateRebar",
                    "materials.uniaxial.ASDConcrete1D", "materials.uniaxial.Hysteretic")


def test_registry_statuses() -> None:
    assert all(ELEMENT_HANDLERS[t].status == "supported" for t in TIER1_ELEMENTS)
    assert all(PROPERTY_HANDLERS[t].status == "supported" for t in TIER1_PROPERTIES)
    for t in ("brick_elements.stdBrick", "zero_length_elements.zeroLength",
              "absorbingBoundaries.ASDAbsorbingBoundary3DAuto"):
        assert ELEMENT_HANDLERS[t].status == "tier_only"
    for t in ("materials.nD.ElasticIsotropic", "materials.nD.ASDAbsorbingBoundary3DMaterial",
              "special_purpose.zeroLengthMaterial", "materials.uniaxial.Elastic"):
        assert PROPERTY_HANDLERS[t].status == "tier_only"


# ── P1, P2, P4, P5: the elastic case (1A) ─────────────────────────────

def elastic_model(**shell: Any) -> ScdModel:
    # square (PROPS[1] == PROPS[2]: the Iyy / Izz order of PROPS is unverified);
    # the modifiers differ, so Iy and Iz still tell their modifier apart
    pps = [elastic_section(1, (490000.0, 3.0e10, 3.0e10, 4.0e10), A_modifier=0.5, Iyy_modifier=0.7,
                           Izz_modifier=0.3, J_modifier=0.1),
           emps(2), emps(4, h=300.0, Ep_mod=0.01)]
    eps = [elastic_beam_ep(1, **{"-mass": True, "massDens": 1.2}), shell_ep(2, **shell)]
    return model(pps, eps, faces=[(2, 2), (2, 4)], edges=[(1, 1), (1, 1)],
                 quats={"faces": [IDENTITY, ROT_Z90], "edges": [IDENTITY, ROT_Z90]})


def test_elastic_case_builds() -> None:
    scd = elastic_model(**{"Drilling Stabilization": 0.0123456789})
    assert check_supported(scd) == []
    gs = groups(scd)
    res = build_props(bridge(), scd, gs)
    # the sections.Elastic is read by the element, never built (rule P2)
    assert sorted(res.primitives) == [2, 4]
    e4 = res.primitives[4]
    assert (e4.E, e4.nu, e4.h, e4.rho, e4.Ep_mod) == (25000.0, 0.2, 300.0, 0.0, 0.01)
    shells = {tuple(round(v, 9) + 0.0 for v in g.local_axis): res.elements[g.pg] for g in gs if g.dim == 2}
    assert set(shells) == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)}
    s = shells[(1.0, 0.0, 0.0)]
    assert s.section is res.primitives[2]
    assert s.drilling_stab == 0.0123457 and not s.drilling_nl     # STKO prints %.6g
    assert not s.no_eas and not s.corotational
    assert s.local_cs == (1.0, 0.0, 0.0)
    beams = [res.elements[g.pg] for g in gs if g.dim == 1]
    assert len(beams) == 1 and len(res.transforms) == 1            # both beams: vecxz (0, 0, 1)
    b = beams[0]
    assert (b.A, b.E, b.G, b.mass) == (490000.0 * 0.5, 25000.0, 10500.0, 1.2)
    assert b.Iy == 3.0e10 * 0.7 and b.Iz == 3.0e10 * 0.3 and b.J == 4.0e10 * 0.1
    assert res.transforms[(0.0, 0.0, 1.0)] is b.transf
    assert type(b.transf).__name__ == "Linear" and b.transf.vecxz == (0.0, 0.0, 1.0)


def test_a_non_square_elastic_section_is_refused_not_guessed() -> None:
    """MAJOR-3: STKO writes ``Section.properties.Iyy / .Izz`` of the compiled
    MpcBeamSection; which of ``PROPS[1]`` / ``PROPS[2]`` is which is unverified
    (every Tier-1 section is square), so a non-square one is refused, typed."""
    scd = elastic_model()
    scd.physical_properties[1] = elastic_section(1, (490000.0, 3.0e10, 2.0e10, 4.0e10))
    items = check_supported(scd)
    assert [(u.category, u.xobj_meta, u.ids) for u in items] == [
        ("option", "sections.Elastic:Section", (1,))]
    assert "unverified" in items[0].reason
    with pytest.raises(UnsupportedSTKOTypes, match="PROPS"):
        build_props(bridge(), scd, groups(scd))



def test_properties_an_interaction_carries_are_named() -> None:
    """2A's zeroLength links: the interaction is the mesh slice's to refuse, but its
    element property, physical property and that property's closure are named here
    (a supported element type on an interaction is refused too)."""
    import dataclasses

    scd = elastic_model()
    scd.element_properties[7] = _x(7, "zero_length_elements.zeroLength")
    scd.element_properties[8] = shell_ep(8)
    scd.physical_properties[9] = _x(9, "materials.uniaxial.Elastic")
    scd.physical_properties[10] = _x(10, "special_purpose.zeroLengthMaterial",
                                     refs=("matTag",), matTag=9)
    scd.physical_properties[11] = _x(11, "materials.uniaxial.Steel01")
    ref = (SubShapeRef(1, 1, 0),)
    inters = {
        1: Interaction(id=1, name="springs", type="NN", masters=ref, slaves=ref,
                       element_property=7, physical_property=10),
        2: Interaction(id=2, name="odd", type="NN", masters=ref, slaves=ref,
                       element_property=8, physical_property=11),
    }
    scd = dataclasses.replace(scd, interactions=inters)
    got = {(u.category, u.xobj_meta, u.ids, u.reason) for u in check_supported(scd)}
    assert got == {
        ("element_property", "zero_length_elements.zeroLength", (7,), "tier-only: not in this version"),
        ("physical_property", "special_purpose.zeroLengthMaterial", (10,), "tier-only: not in this version"),
        ("physical_property", "materials.uniaxial.Elastic", (9,), "tier-only: not in this version"),
        ("physical_property", "materials.uniaxial.Steel01", (11,), "unknown STKO type"),
        ("option", "shell.ASDShellQ4:on interaction", (8,),
         "carried by interaction 2 ('odd'): elements generated by an interaction are not translated"),
    }

def test_shell_flags() -> None:
    scd = elastic_model(**{"Drilling DOF Type": "Non-Linear Drilling DOF", "Use EAS": False,
                           "Kinematics": "Corotational"})
    res = build_props(bridge(), scd, groups(scd))
    s = next(e for e in res.elements.values() if type(e).__name__ == "ASDShellQ4")
    assert s.drilling_nl and s.drilling_stab is None and s.no_eas and s.corotational


# ── P3, P6-P10: the nonlinear case (1C/1D) ────────────────────────────

def test_nl_case_builds_each_property_once() -> None:
    scd = nl_model()
    assert check_supported(scd) == []
    gs = groups(scd)
    res = build_props(bridge(), scd, gs)
    assert sorted(res.primitives) == [6, 8, 9, 10, 11, 12, 13, 15, 16, 17]
    p = res.primitives
    wall = res.elements[next(g.pg for g in gs if g.dim == 2)]
    assert wall.section is p[17]
    layers = p[17].layers
    assert [lay.material for lay in layers] == [p[13], p[15], p[16], p[6], p[16], p[15], p[13]]
    assert [lay.thickness for lay in layers] == [25.0, 0.5, 0.4, 250.0, 0.4, 0.5, 25.0]
    assert p[13].material is p[6] and p[13].G_out == 10500.0
    assert p[15].material is p[8] and p[15].angle == 0.0 and p[16].angle == 90.0
    col = res.elements[next(g.pg for g in gs if g.dim == 1)]
    assert col.integration is p[12]
    assert type(p[12]).__name__ == "Lobatto" and p[12].n_ip == 5 and p[12].section is p[11]
    assert col.max_iter is None and col.mass is None


def test_fiber_rows_order_and_centroid() -> None:
    """Rule P8: punctual, surface, linear; items in k order; minus the stored centroid."""
    x = nl_model().physical_properties[11]
    assert fiber_rows(x) == [
        (10.0, 10.0, 380.0, 8), (-10.0, -10.0, 380.0, 8),
        (0.0, 0.0, 1000.0, 10), (1.0, -1.0, 1000.0, 10), (-20.0, -20.0, 50.0, 9),
    ]
    res = build_props(bridge(), nl_model(), groups(nl_model()))
    sec = res.primitives[11]
    assert sec.GJ == 1.0e15
    assert [(f.y, f.z, f.area) for f in sec.fibers] == [r[:3] for r in fiber_rows(x)]
    assert [f.material for f in sec.fibers] == [res.primitives[r[3]] for r in fiber_rows(x)]


def test_fiber_items_sort_numerically() -> None:
    rows = [(1, [(float(k), 0.0, 1.0)]) for k in range(12)]
    x = fiber_section(5, {"SURFACE_FIBER_GROUPS": rows})
    assert [r[0] for r in fiber_rows(x)] == [float(k) for k in range(12)]   # ITEM_10 after ITEM_9


def test_no_centroid_offset() -> None:
    x = fiber_section(5, {"SURFACE_FIBER_GROUPS": [(1, [(3.0, 4.0, 1.0)])]}, center=(1.0, 1.0),
                      **{"-noCentroid": True})
    assert fiber_rows(x) == [(3.0, 4.0, 1.0, 1)]


def test_force_beam_options_and_shared_transforms() -> None:
    scd = nl_model()
    scd.element_properties[7].attributes.update({"-iter": True, "maxIters": 20, "tol": 1e-9,
                                                  "-mass": True, "massDens": 2.5})
    res = build_props(bridge(), scd, groups(scd))
    col = next(e for e in res.elements.values() if type(e).__name__ == "forceBeamColumn")
    assert (col.max_iter, col.tol, col.mass) == (20, 1e-9, 2.5)
    assert list(res.transforms) == [(0.0, 0.0, 1.0)]


@pytest.mark.parametrize("kind", ["Lobatto", "Legendre", "Radau", "NewtonCotes", "Trapezoidal"])
def test_integration_types(kind: str) -> None:
    scd = nl_model()
    scd.physical_properties[12].attributes.update({"IntegrationType/1": kind, "numIntPts/1": 4})
    res = build_props(bridge(), scd, groups(scd))
    assert type(res.primitives[12]).__name__ == kind and res.primitives[12].n_ip == 4


def test_hysteretic_optional_and_beta() -> None:
    scd = nl_model()
    res = build_props(bridge(), scd, groups(scd))
    h = res.primitives[8]
    assert (h.s3p, h.e3p, h.s3n, h.e3n, h.pinch_x, h.pinch_y, h.beta) == (550.0, 0.1, -550.0, -0.1, 0.5, 0.8, 0.0)
    scd.physical_properties[8].attributes.update({"Optional": False, "use_beta": True, "beta": 0.3})
    h = build_props(bridge(), scd, groups(scd)).primitives[8]
    assert (h.s3p, h.s3n, h.beta) == (None, None, 0.3)
    assert emitted(h)[0][1] == ("Hysteretic", 1, 414.0, 0.002, 415.0, 0.02, -414.0, -0.002, -415.0,
                                -0.02, 0.5, 0.8, 0.0, 0.0, 0.3)


# ── P9: ASDConcrete "Concrete (9P)" against STKO's printed numbers ────

def test_asdconcrete_9p_matches_stko_deck() -> None:
    """1B/1C ``nDMaterial ASDConcrete3D 6`` as STKO wrote it."""
    law = asdconcrete_9p(UC)
    assert law["lch_ref"] == 6.377551020408164
    assert law["Te"] == (0.0, 0.0001008, 0.000168, 0.005678399999999999, 0.028056111999999998, 0.28056112)
    assert law["Ts"] == (0.0, 2.52, 2.8, 0.5599999999999999, 0.0028, 0.0028)
    assert law["Td"] == (0.0, 0.0, 0.2592592592592593, 0.994596201129394, 0.999994747368379, 0.9999995909836032)
    assert law["Ce"][-2:] == (0.28206, 0.28262)
    assert law["Cs"][2:5] == (18.962292550544525, 21.981982603914037, 24.038053872296533)
    assert law["Cd"][4] == 0.03827139403405799 and law["Cd"][-1] == 0.9989960912900835


def test_asdconcrete_3d_primitive_and_crack_planes() -> None:
    """1D's ``ASDConcrete_UC``: ``-cdf 0.25 -implex ... -crackPlanes 4 4 45.0``."""
    scd = nl_model()
    scd.physical_properties[6].attributes.update({"-crackPlanes": True, "CDF": 0.25, "Gt": 0.05,
                                                   "PScale Tension": 0.1, "PScale Compression": 0.25})
    m = build_props(bridge(), scd, groups(scd)).primitives[6]
    assert isinstance(m, ASDConcrete3DCrackPlanes) and isinstance(m, ASDConcrete3D)
    assert m.lch_ref == 3.188775510204082
    assert m.Td == (0.0, 0.0, 0.31034482758620685, 0.9956650844224809, 0.9999956608695305, 0.9999995975806419)
    assert m.Ce[-2:] == (0.56206, 0.56262)
    (name, args, _), = emitted(m)
    assert name == "nDMaterial" and args[:4] == ("ASDConcrete3D", 1, 25000.0, 0.2)
    tail = args[args.index("-rho"):]
    assert tail == ("-rho", 0.0, "-Kc", 2.0 / 3.0, "-cdf", 0.25, "-implex",
                    "-crackPlanes", 4, 4, 45.0, "-autoRegularization", 3.188775510204082)


def test_asdconcrete_without_crack_planes_is_the_bridge_class() -> None:
    res = build_props(bridge(), nl_model(), groups(nl_model()))
    m = res.primitives[6]
    assert type(m) is ASDConcrete3D and m.implex and m.cdf == 0.0 and m.tangent == "secant"
    assert type(res.primitives[9]) is ASDConcrete1D and res.primitives[9].implex


def test_asdconcrete_1d_roundoff_damage_written_as_zero() -> None:
    """1D's ``ASDConcrete_CC_1D``: STKO prints ``-Td ... -2.220446049250313e-16``."""
    cc = {**UC, "Gc": 90.0, "ecp": 0.006, "fc0": 20.0, "fcp": 35.0, "fcr": 7.0, "ft": 3.5,
          "PScale Tension": 1.0, "PScale Compression": 1.0}
    law = asdconcrete_9p(cc)
    assert law["Td"][2] == -2.220446049250313e-16 and law["Cd"][3] == -2.220446049250313e-16
    assert law["lch_ref"] == 4.081632653061225
    scd = nl_model()
    scd.physical_properties[10].attributes.update({"PScale Tension": 1.0, "PScale Compression": 1.0})
    m = build_props(bridge(), scd, groups(scd)).primitives[10]
    assert m.Td[2] == 0.0 and m.Cd[3] == 0.0
    assert m.Td[3:] == law["Td"][3:] and m.Cs == law["Cs"]


def test_negative_damage_beyond_roundoff_raises() -> None:
    from apeGmsh.interop.stko.translate_props import _damage
    assert _damage((0.0, -1e-13, 0.5), "m") == (0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="below zero"):
        _damage((0.0, -1e-6), "m")


# ── unsupported: every offender listed, nothing touched ───────────────

def test_everything_unsupported_listed_at_once() -> None:
    pps = [
        emps(2),
        _x(3, "sections.LayeredShell", refs=("matTag",), matTag=(30, 30), thickness=(1.0, 1.0)),
        _x(30, "materials.nD.ElasticIsotropic", E=1.0, v=0.2),
        elastic_section(1, (1.0, 1.0, 1.0, 1.0), **{"Use Uniaxial Materials": True, "Y/section_offset": 50.0}),
        asd3d(6, Preset="Concrete (1P)", implexAlpha=0.5),
        asd1d(9, constitutiveTensorType="Tangent"),
        _x(13, "materials.nD.PlateFromPlaneStress", refs=("matTag",), matTag=6, OutofPlaneModulus=1.0),
        _x(40, "sections.Mystery"),
        fiber_section(11, {"SURFACE_FIBER_GROUPS": [(9, [(0.0, 0.0, 1.0)])]}, rectangular=True,
                      **{"-torsion": True}),
        bsp(12, 11, kind="CompositeSimpson"),
        _x(14, "special_purpose.zeroLengthMaterial"),       # an unassigned template: not scanned
    ]
    eps = [shell_ep(2), shell_ep(5), elastic_beam_ep(1, **{"-releasey": True, "-cMass": True}),
           force_beam_ep(7, transType="Weird"), _x(8, "zero_length_elements.zeroLength"),
           _x(9, "beam_column_elements.mysteryBeam"), shell_ep(10)]
    scd = model(pps, eps, faces=[(2, 3), (5, 13), (10, 1)],
                edges=[(1, 1), (7, 12), (8, 40), (9, 2)])
    found = {(u.category, u.xobj_meta, u.reason.split(":")[0]) for u in check_supported(scd)}
    expected = {
        ("element_property", "zero_length_elements.zeroLength", "tier-only"),
        ("element_property", "beam_column_elements.mysteryBeam", "unknown STKO type"),
        ("physical_property", "materials.nD.ElasticIsotropic", "tier-only"),
        ("physical_property", "sections.Mystery", "unknown STKO type"),
        ("option", "shell.ASDShellQ4:physical property",
         "sections.Elastic is not a physical property this element takes (takes ['sections.ElasticMembranePlateSection', 'sections.LayeredShell'])"),
        ("option", "shell.ASDShellQ4:physical property",
         "materials.nD.PlateFromPlaneStress is not a physical property this element takes (takes ['sections.ElasticMembranePlateSection', 'sections.LayeredShell'])"),
        ("option", "beam_column_elements.elasticBeamColumn:-releasey", "-releasey is not in this version"),
        ("option", "beam_column_elements.elasticBeamColumn:-cMass", "-cMass is not in this version"),
        ("option", "beam_column_elements.forceBeamColumn:transType", "geomTransf 'Weird' is not one of ['Linear', 'PDelta', 'Corotational']"),
        ("option", "sections.Elastic:Use Uniaxial Materials", "STKO writes an Aggregator section"),
        ("option", "sections.Elastic:Y/section_offset", "a section offset makes STKO write -jntOffset"),
        ("option", "sections.LayeredShell:matTag", "2 layers"),
        ("option", "materials.nD.ASDConcrete3D:Preset", "preset 'Concrete (1P)'"),
        ("option", "materials.nD.ASDConcrete3D:implexAlpha", "implexAlpha != 1.0"),
        ("option", "materials.uniaxial.ASDConcrete1D:constitutiveTensorType", "-tangent on ASDConcrete1D is not in this version"),
        ("option", "sections.RectangularFiberSection:-torsion", "-torsion (a torsion material) is not in this version"),
        ("option", "sections.RectangularFiberSection:Concrete (Core) Material", "no core material"),
        ("option", "special_purpose.BeamSectionProperty:IntegrationType/1", "'CompositeSimpson' is not one of ['Lobatto', 'Legendre', 'Radau', 'NewtonCotes', 'Trapezoidal']"),
    }
    assert found == expected
    ops = MagicMock(name="ops")
    with pytest.raises(UnsupportedSTKOTypes) as err:
        build_props(ops, scd, groups(scd))
    assert len(err.value.items) == len(check_supported(scd))
    assert ops.mock_calls == []                                   # raised before touching the bridge
    assert "zeroLengthMaterial" not in str(err.value)            # unassigned template not scanned


def test_option_rows() -> None:
    def problems(scd: ScdModel) -> set[str]:
        return {u.xobj_meta for u in check_supported(scd)}

    scd = nl_model()
    scd.physical_properties[6].attributes.update({"constitutiveTensorType": "Tangent"})
    scd.physical_properties[11].attributes.update({"Z/section_offset": 10.0})
    scd.physical_properties[17].attributes.update({"matTag": (13, 15, 8, 6, 16, 15, 13)})
    scd.element_properties[7].attributes.update({"-cMass": True, "2D": True})
    scd.element_properties[3].attributes.update({"Kinematics": "Weird",
                                                 "Drilling DOF Type": "Elastic Drilling DOF",
                                                 "Drilling Stabilization": 2.0})
    assert problems(scd) == {
        "materials.nD.ASDConcrete3D:constitutiveTensorType",
        "sections.RectangularFiberSection:Z/section_offset",
        "special_purpose.BeamSectionProperty:secTag/1:Z/section_offset",
        "sections.LayeredShell:matTag",                    # a uniaxial material as a layer
        "beam_column_elements.forceBeamColumn:-cMass",
        "beam_column_elements.forceBeamColumn:Dimension",
        "shell.ASDShellQ4:Kinematics",
        "shell.ASDShellQ4:Drilling Stabilization",
    }
    scd = nl_model()
    scd.physical_properties[12].attributes.update({"Option": "UserDefined"})
    scd.physical_properties[11].attributes["Fiber section"]["SURFACE_FIBER_GROUPS"]["ITEM_1"]["@attrs"]["PHYS_PROP_ID"] = np.array([0])
    assert problems(scd) == {"special_purpose.BeamSectionProperty:Option",
                             "sections.RectangularFiberSection:Fiber section"}


def test_fiber_noncentroid_and_elastic_integration_section_unsupported() -> None:
    scd = nl_model()
    scd.physical_properties[20] = elastic_section(20, (1.0, 1.0, 1.0, 1.0))
    scd.physical_properties[12].attributes["secTag/1"] = 20
    metas = {(u.xobj_meta, u.reason) for u in check_supported(scd)}
    assert ("special_purpose.BeamSectionProperty:secTag/1",
            "a sections.Elastic integration section is not in this version") in metas
    x = fiber_section(21, {"SURFACE_FIBER_GROUPS": [(9, [(0.0, 0.0, 1.0)])]}, **{"-noCentroid": True})
    scd.physical_properties[21] = x
    scd.physical_properties[12].attributes["secTag/1"] = 21
    assert "sections.Fiber:-noCentroid" in {u.xobj_meta for u in check_supported(scd)}


def test_element_without_physical_property_and_mixed_transforms() -> None:
    pps = [elastic_section(1, (1.0, 1.0, 1.0, 1.0))]
    eps = [elastic_beam_ep(1), elastic_beam_ep(2, transfType="PDelta"), elastic_beam_ep(3)]
    scd = model(pps, eps, edges=[(1, 1), (2, 1), (3, 0)])
    rows = {(u.xobj_meta, u.ids, u.reason) for u in check_supported(scd)}
    assert ("beam_column_elements.elasticBeamColumn:physical property", (3,),
            "an analysis element without a physical property") in rows
    assert ("beam_column_elements.elasticBeamColumn:geomTransf", (1, 2),
            "vecxz (0.0, 0.0, 1.0) carries geomTransf types ['Linear', 'PDelta']") in rows


def test_pdelta_and_corotational_transforms() -> None:
    pps = [elastic_section(1, (1.0, 1.0, 1.0, 1.0))]
    eps = [elastic_beam_ep(1, transfType="PDelta"), elastic_beam_ep(2, transfType="Corotational")]
    scd = model(pps, eps, edges=[(1, 1), (2, 1)],
                quats={"edges": [IDENTITY, (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))]})
    res = build_props(bridge(), scd, groups(scd))
    assert sorted(type(t).__name__ for t in res.transforms.values()) == ["Corotational", "PDelta"]


def test_group_dimension_mismatch_raises() -> None:
    scd = elastic_model()
    gs = [g if g.dim == 2 else ElementGroup(g.pg, 2, g.element_property, g.physical_property,
                                            g.local_axis, g.element_ids) for g in groups(scd)]
    with pytest.raises(ValueError, match="1-D element"):
        build_props(bridge(), scd, gs)


# ── oracle: STKO's own Tcl exports (local only) ───────────────────────

ORACLES = os.environ.get("APEGMSH_STKO_ORACLES")
DECKS = {"1A": "1A/input files", "1B": "1B/campaign_sta2_rup1", "1C": "1C/campaign_sta2_rup1",
         "1D": "1D/campaign_sta2_rup1"}


def _deck(path: Path) -> list[list[str]]:
    text = path.read_text().replace("\\\n", " ")
    toks = (ln.replace("{", " ").replace("}", " ").split() for ln in text.splitlines()
            if not ln.lstrip().startswith("#"))
    return [t for t in toks if t]


def _num(t: str) -> Any:
    for cast_ in (int, float):
        try:
            return cast_(t)
        except ValueError:
            pass
    return t


@pytest.mark.skipif(not ORACLES, reason="set APEGMSH_STKO_ORACLES to the research repo's "
                    "models/stko-rev0-tcl to compare with STKO's Tier-1 Tcl exports (local data)")
@pytest.mark.parametrize("case", sorted(DECKS))
def test_oracle_props_match_stko_decks(case: str) -> None:
    from apeGmsh.interop.stko import read_scd

    root = Path(cast(str, ORACLES)) / "Tier_1"
    scd = read_scd(root / case / f"{case}_TH_000.scd")
    assert check_supported(scd) == []
    res = build_props(bridge(), scd, groups(scd))
    deck = root / DECKS[case]
    stko: dict[int, list[Any]] = {}
    fibers: dict[int, list[tuple]] = defaultdict(list)
    current = None
    for tok in _deck(deck / "materials.tcl") + _deck(deck / "sections.tcl"):
        if tok[0] in ("nDMaterial", "uniaxialMaterial", "section"):
            stko[int(tok[2])] = [tok[1], *(_num(t) for t in tok[3:])]
            current = int(tok[2]) if tok[1] == "Fiber" else None
        elif tok[0] == "fiber" and current is not None:
            fibers[current].append(tuple(_num(t) for t in tok[1:5]))
    tags = {id(p): pid for pid, p in res.primitives.items()}
    defaults = {"-rho": [0.0], "-eta": [0.0], "-Kc": [2.0 / 3.0], "-cdf": [0.0]}
    for pid, prim in res.primitives.items():
        if scd.physical_properties[pid].type == "special_purpose.BeamSectionProperty":
            continue
        calls = emitted(prim, tags, pid)
        args = list(calls[0][1])
        ours, theirs = [args[0], *args[2:]], stko[pid]
        if args[0] in ("ASDConcrete3D", "ASDConcrete1D"):
            # flags in any order; STKO writes the defaults and -implexAlpha 1.0 explicitly
            o, t = _flags(ours), _flags(theirs)
            assert t.pop("-implexAlpha", [1.0]) == [1.0]
            for flag, value in defaults.items():
                if args[0] == "ASDConcrete3D" or flag == "-eta":
                    o.setdefault(flag, value)
            assert o.keys() == t.keys(), (pid, sorted(o), sorted(t))
            for flag in o:
                assert o[flag] == pytest.approx(t[flag], rel=1e-12, abs=3e-16), (pid, flag)
            continue
        if args[0] == "ElasticMembranePlateSection" and len(ours) == 5:
            ours.append(1.0)                       # apeSees omits Ep_mod at its default
        assert len(ours) == len(theirs), (pid, ours, theirs)
        for o_, t_ in zip(ours, theirs):
            if isinstance(t_, str):
                assert o_ == t_, (pid, o_, t_)
            else:
                assert o_ == pytest.approx(t_, rel=1e-12), (pid, o_, t_)
        if args[0] == "Fiber":
            got = [c[1] for c in calls if c[0] == "fiber"]
            assert got == fibers[pid]              # exact, in STKO's order

    # elements: one STKO line per element id, all partitions
    lines: dict[int, list[Any]] = {}
    transf: dict[int, tuple[str, list[float], list[str]]] = {}
    for f in sorted(deck.glob("elements*.tcl")):
        for tok in _deck(f):
            if tok[0] == "element":
                lines[int(tok[2])] = [tok[1], *(_num(t) for t in tok[3:])]
            elif tok[0] == "geomTransf":
                transf[int(tok[2])] = (tok[1], [float(t) for t in tok[3:6]], tok[6:])
    vec_of = {id(t): vec for vec, t in res.transforms.items()}
    for g in groups(scd):
        prim = res.elements[g.pg]
        for eid in g.element_ids:
            typ, *rest = lines[eid]
            rest = rest[len(scd.mesh.elements[eid].nodes):]
            assert typ == type(prim).__name__
            if typ == "ASDShellQ4":
                flags = ["-corotational"] * prim.corotational + ["-noeas"] * prim.no_eas
                flags += ["-drillingStab", prim.drilling_stab] if prim.drilling_stab is not None else []
                flags += ["-drillingNL"] * prim.drilling_nl
                assert rest[:-4] == [tags[id(prim.section)], *flags], eid
                assert rest[-4] == "-local"
                assert max(abs(a - b) for a, b in zip(rest[-3:], prim.local_cs)) <= 1e-6
                continue
            kind, vec, extra = transf[eid]
            assert kind == type(prim.transf).__name__ and extra == [] and rest[0 if typ == "forceBeamColumn" else 6] == eid
            assert max(abs(a - b) for a, b in zip(vec, vec_of[id(prim.transf)])) <= 1e-11
            if typ == "elasticBeamColumn":      # STKO: A E G J Iy Iz transf
                assert rest[:6] == pytest.approx([prim.A, prim.E, prim.G, prim.J, prim.Iy, prim.Iz], rel=1e-12)
            else:                                # STKO: transf Type secTag np
                integ = prim.integration
                assert rest[1:] == [type(integ).__name__, tags[id(integ.section)], integ.n_ip]
    assert len(lines) == sum(len(g.element_ids) for g in groups(scd))


def _flags(args: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {"_pos": []}
    key = "_pos"
    for a in args:
        if isinstance(a, str) and a.startswith("-"):
            key = a
            out[key] = []
        else:
            out[key].append(a)
    return out
