"""Strut-and-tie consumer — FE overlays for an apeConcrete model JSON.

apeConcrete ADR-0014 §6: the finite-element side of the strut-and-tie
tool lives *beside* apeConcrete, never inside it. This module reads the
model JSON that ``apeConcrete.stm.model_to_dict`` writes (schema 1),
meshes the D-region and runs it on the live OpenSees domain, returning
an ``overlays`` block for
``apeConcrete.plotting.write_stm_viewer(..., overlays=...)``.

Two runs share one FE model:

:func:`strut_tie_overlays` — **linear elastic**, one static step.
    ``fe_trajectories``: principal-stress glyphs, one pair per element
    at its centroid — a compression segment along σ₃ and a tension
    segment along σ₁ — as ``[[x, y, z], [x, y, z], sigma1, sigma3]`` in
    the model's base units (N, mm), lengths scaled by element size and
    by stress magnitude normalised to the 90th percentile. The viewer
    colours compression like struts and tension like ties, so the truss
    the engineer drew can be compared with the elastic load path.
:func:`strut_tie_pushover` — **nonlinear**, displacement control.
    ``fe_curve``: the load–deformation curve ``[[delta, P], ...]`` with
    the fork's ``LadrunoConcrete3D`` plastic-damage concrete, driven by
    displacement control on the mesh node under the load plate in the
    dominant load direction; ``capacity`` is the peak load. This is the
    physics oracle of ADR-0014 §9: by the lower-bound theorem the FE
    capacity should not fall below the strut-and-tie nominal capacity.
    By default the model's ties are placed as conformal ``Steel02``
    truss bars of the tie's steel area (perfect bond, no anchorage
    model); ``reinforced=False`` gives the plain concrete's curve.

Both carry ``fe_summary``: applied and reacted resultants, node and
element counts, mesh size and material parameters.

How the STM model maps to the FE model (documented, not clever):

* **Region** — a plane region meshes its ``outline`` polygon with
  ``Tri31`` plane-stress triangles of the region thickness; a solid
  region meshes its bounding box (``extrude`` for *z*, the plates and
  nodes for *x*, *y*) with ``FourNodeTetrahedron``. Concrete is
  ``ElasticIsotropic`` with ``E = 4700·√f'c`` (MPa, ACI 318 Eq.
  19.2.2.1b) and ν = 0.2 for the linear run; ``LadrunoConcrete3D`` with
  the same *E*, ``ft = 0.33·√f'c``, ``Gf = 0.073·f'c^0.18`` N/mm (Model
  Code 2010) and ``Gc = 250·Gf`` for the nonlinear one, each overridable.
* **Support plates** (``kind == "support"``) are fixed along their
  normal on every mesh node inside the plate, plus whatever directions
  the model's supports restrain at the node on that plate; the loads
  the model applies at those nodes (rigid-cap pile reactions, for
  instance) are **not** applied — the FE finds its own reactions.
* **Load plates** receive the model's nodal load at that node split
  equally over the mesh nodes inside the plate.
* **Nodes without a plate** map to the nearest mesh node, for both
  loads and supports. ``extra_fixed_planes`` lets the caller fix a
  whole face (a corbel's far column face, say) when the STM supports
  are not the physical boundary.
* Restraints along an inclined direction are applied on the dominant
  global axis (OpenSees fixities are per DOF).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

EC_COEFFICIENT_4700: float = 4700.0  # ACI 318 Eq. (19.2.2.1b), MPa
POISSON_RATIO: float = 0.2
FT_COEFFICIENT_0_33: float = 0.33  # direct tensile strength ≈ 0.33·√f'c, MPa
GF_MC2010_COEFFICIENT: float = 0.073  # Gf = 73·f'c^0.18 N/m → 0.073·f'c^0.18 N/mm
GF_MC2010_EXPONENT: float = 0.18
GC_OVER_GF_250: float = 250.0
REBAR_MATERIAL_NAME: str = "stm_rebar"
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class StrutTieOverlays:
    """What :func:`strut_tie_overlays` / :func:`strut_tie_pushover` return."""

    overlays: dict[str, Any]
    case: str
    n_nodes: int
    n_elements: int
    mesh_size: float
    applied: tuple[float, float, float]
    reactions: tuple[float, float, float]
    fixed_nodes: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FEModel:
    """The meshed, restrained, loaded D-region both runs share."""

    fem: Any
    ids: np.ndarray
    coords: np.ndarray
    plane: bool
    ndm: int
    z0: float
    size: float
    thickness: float
    diag: float
    fc: float
    case: str
    fixed: dict[int, set[int]]
    nodal: dict[int, np.ndarray]
    applied: np.ndarray
    control_row: int
    fy: float
    Es: float
    ties: dict[str, float]
    """Tie id → steel area (mm²) of the bars placed in the mesh; empty when not reinforced."""


def _v(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _nearest(coords: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(coords - point[None, :], axis=1)))


def _plate_nodes(
    coords: np.ndarray,
    plate: Mapping[str, Any],
    *,
    plane: bool,
    tol_n: float,
    tol_t: float,
) -> np.ndarray:
    c, n = _v(plate["center"]), _v(plate["normal"])
    rel = coords - c[None, :]
    on_face = np.abs(rel @ n) <= tol_n
    if plate.get("shape") == "circle":
        radial = rel - np.outer(rel @ n, n)
        inside = np.linalg.norm(radial, axis=1) <= plate["length_l1"] / 2.0 + tol_t
    else:
        a1 = _v(plate.get("axis1", [1.0, 0.0, 0.0]))
        a2 = np.cross(n, a1)
        inside = np.abs(rel @ a1) <= plate["length_l1"] / 2.0 + tol_t
        if not plane and plate.get("length_l2") is not None:
            inside &= np.abs(rel @ a2) <= plate["length_l2"] / 2.0 + tol_t
    return np.where(on_face & inside)[0]


def _dominant_dofs(directions: Sequence[Sequence[float]], ndf: int) -> set[int]:
    dofs: set[int] = set()
    for d in directions:
        vec = _v(d)[:ndf]
        if np.linalg.norm(vec) == 0.0:
            continue
        dofs.add(int(np.argmax(np.abs(vec))))
    return dofs


def _build(
    model: Mapping[str, Any],
    *,
    case: str | None,
    mesh_size: float | None,
    extra_fixed_planes: Sequence[tuple[str, float, Sequence[int]]],
    verbose: bool,
    reinforced: bool = False,
) -> _FEModel:
    """Mesh the region and resolve supports and loads onto mesh nodes.

    With ``reinforced=True`` every tie of the model becomes one CAD line
    between its two nodes (nodes shared between ties), embedded in the
    host before meshing so the mesh conforms to it (shared nodes,
    perfect bond); :func:`_elements` then emits one ``CorotTruss`` per
    line cell with the tie's steel area against the uniaxial material
    named :data:`REBAR_MATERIAL_NAME`, which the caller registers.
    Anchorage is not modelled — a bar ends where the tie ends.
    """
    if model.get("schema_version") != 1:
        raise ValueError(
            f"unsupported strut-and-tie schema_version {model.get('schema_version')!r}"
        )
    region = model["region"]
    plane = region["kind"] == "plane"
    ndm = 2 if plane else 3
    nodes_stm = {n["id"]: n for n in model["nodes"]}
    plates = {p["id"]: p for p in model.get("plates", [])}
    cases = model["load_cases"]
    if not cases:
        raise ValueError("the model has no load case")
    chosen = cases[0] if case is None else next(c for c in cases if c["name"] == case)

    # -- bounding geometry -------------------------------------------------
    pts = [_v(n["xyz"]) for n in nodes_stm.values()] + [
        _v(p["center"]) for p in plates.values()
    ]
    for p in plates.values():
        r = max(p["length_l1"], p.get("length_l2") or 0.0) / 2.0
        pts.append(_v(p["center"]) + r)
        pts.append(_v(p["center"]) - r)
    lo, hi = np.min(np.array(pts), axis=0), np.max(np.array(pts), axis=0)
    outline = [tuple(map(float, q)) for q in region.get("outline", [])]
    if plane and len(outline) < 3:
        outline = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    if plane:
        xs = [q[0] for q in outline]
        ys = [q[1] for q in outline]
        lo = np.array([min(xs), min(ys), lo[2]])
        hi = np.array([max(xs), max(ys), hi[2]])
    z0 = float(next(iter(nodes_stm.values()))["xyz"][2]) if plane else 0.0
    extrude = region.get("extrude")
    if not plane:
        zlo, zhi = (
            (float(extrude[0]), float(extrude[1]))
            if extrude
            else (float(lo[2]), float(hi[2]))
        )
        if len(outline) >= 3:
            xs = [q[0] for q in outline]
            ys = [q[1] for q in outline]
            lo = np.array([min(xs), min(ys), zlo])
            hi = np.array([max(xs), max(ys), zhi])
        else:
            lo[2], hi[2] = zlo, zhi
    diag = float(np.linalg.norm(hi - lo))
    size = mesh_size if mesh_size is not None else diag / 25.0
    thickness = float(region.get("thickness") or 0.0)

    # -- mesh ---------------------------------------------------------------
    with apeGmsh(model_name="stm_overlay", verbose=verbose) as g:
        geo = g.model.geometry
        if plane:
            tags = [geo.add_point(x, y, z0, mesh_size=size) for x, y in outline]
            lines = [
                geo.add_line(tags[i], tags[(i + 1) % len(tags)])
                for i in range(len(tags))
            ]
            loop = geo.add_curve_loop(lines)
            entity = geo.add_plane_surface(loop, label="Region")
            g.model.sync()
            g.physical.add(2, [entity], name="Region")
        else:
            entity = geo.add_box(
                float(lo[0]),
                float(lo[1]),
                float(lo[2]),
                float(hi[0] - lo[0]),
                float(hi[1] - lo[1]),
                float(hi[2] - lo[2]),
                label="Region",
            )
            g.model.sync()
            g.physical.add(3, [entity], name="Region")
        ties_modelled: dict[str, float] = {}
        if reinforced and model.get("ties"):
            # One CAD point per STM node (shared by every tie ending there —
            # a second point at the same place would give the mesher
            # zero-volume elements), one line per tie, all embedded in the
            # host so the mesh conforms to them. Each line is its own
            # physical group, which is how the bridge finds its cells.
            point_of: dict[str, int] = {}

            def point(nid: str) -> int:
                if nid not in point_of:
                    x, y, z = (float(c) for c in nodes_stm[nid]["xyz"])
                    point_of[nid] = geo.add_point(x, y, z, mesh_size=size)
                return point_of[nid]

            line_tags: list[int] = []
            for t in model["ties"]:
                a, b = t["nodes"]
                tag = geo.add_line(point(a), point(b))
                g.model.sync()
                g.physical.add(1, [tag], name=f"tie_{t['id']}")
                line_tags.append(tag)
                ties_modelled[t["id"]] = float(t["n_bars"]) * float(t["bar"]["Ab"])
            g.mesh.editing.embed(line_tags, entity, dim=1, in_dim=ndm)
        g.mesh.sizing.set_global_size(size)
        g.mesh.generation.generate(ndm)
        fem = g.mesh.queries.get_fem_data(dim=None if ties_modelled else ndm)

    ids = np.asarray(fem.nodes.ids, dtype=int)
    coords = np.asarray(fem.nodes.coords, dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.full(len(coords), z0)])
    tol_n = 1e-6 * diag + 1e-9
    tol_t = 0.5 * size

    # -- supports ------------------------------------------------------------
    fixed: dict[int, set[int]] = {}

    def fix(rows: np.ndarray | list[int], dofs: set[int]) -> None:
        for r in rows:
            fixed.setdefault(int(r), set()).update(d for d in dofs if d < ndm)

    support_dirs: dict[str, list[Sequence[float]]] = {}
    for s in model.get("supports", []):
        support_dirs.setdefault(s["node"], []).extend(s["directions"])
    plate_of_node = {nid: n.get("plate") for nid, n in nodes_stm.items()}
    support_plate_ids = {pid for pid, p in plates.items() if p.get("kind") == "support"}

    def rows_for(nid: str) -> np.ndarray:
        pid = plate_of_node.get(nid)
        if pid is not None:
            rows = _plate_nodes(
                coords, plates[pid], plane=plane, tol_n=tol_n, tol_t=tol_t
            )
            if rows.size:
                return rows
        return np.array([_nearest(coords, _v(nodes_stm[nid]["xyz"]))])

    for pid in support_plate_ids:
        p = plates[pid]
        rows = _plate_nodes(coords, p, plane=plane, tol_n=tol_n, tol_t=tol_t)
        if rows.size == 0:
            rows = np.array([_nearest(coords, _v(p["center"]))])
        fix(rows, _dominant_dofs([p["normal"]], ndm))
    for nid, dirs in support_dirs.items():
        fix(rows_for(nid), _dominant_dofs(dirs, ndm))
    for axis, value, dofs in extra_fixed_planes:
        k = _AXES[axis]
        rows = np.where(np.abs(coords[:, k] - value) <= tol_n * 10.0 + 1e-6)[0]
        fix(rows, {int(d) for d in dofs})
    if not fixed:
        raise ValueError("no restraint could be placed; give extra_fixed_planes")

    # -- loads ---------------------------------------------------------------
    nodal: dict[int, np.ndarray] = {}
    applied = np.zeros(3)
    control_row = -1
    control_force = 0.0
    for ld in chosen["loads"]:
        nid = ld["node"]
        pid = plate_of_node.get(nid)
        if pid is not None and pid in support_plate_ids:
            continue  # a reaction the STM model carried as a load
        force = _v(ld["force"])
        rows = rows_for(nid)
        share = force / float(rows.size)
        for r in rows:
            nodal[int(r)] = nodal.get(int(r), np.zeros(3)) + share
        applied += force
        if float(np.linalg.norm(force)) > control_force:
            control_force = float(np.linalg.norm(force))
            control_row = _nearest(coords, _v(nodes_stm[nid]["xyz"]))
    if not nodal:
        raise ValueError(
            f"load case {chosen['name']!r} applies no load outside the supports"
        )

    return _FEModel(
        fem=fem,
        ids=ids,
        coords=coords,
        plane=plane,
        ndm=ndm,
        z0=z0,
        size=float(size),
        thickness=thickness,
        diag=diag,
        fc=float(model["concrete"]["fc"]),
        case=chosen["name"],
        fixed=fixed,
        nodal=nodal,
        applied=applied,
        control_row=control_row,
        fy=float(model["steel"]["fy"]),
        Es=float(model["steel"]["Es"]),
        ties=ties_modelled,
    )


def _elements(ops: Any, fe: _FEModel, mat: Any) -> None:
    if fe.plane:
        ops.element.Tri31(
            pg="Region", thickness=fe.thickness, material=mat, plane_type="PlaneStress"
        )
    else:
        ops.element.FourNodeTetrahedron(pg="Region", material=mat)
    for tie_id, area in fe.ties.items():
        ops.element.CorotTruss(pg=f"tie_{tie_id}", A=area, material=REBAR_MATERIAL_NAME)


def _restraints_and_loads(ops: Any, fe: _FEModel) -> None:
    by_pattern: dict[tuple[int, ...], list[int]] = {}
    for row, dofs in fe.fixed.items():
        pattern = tuple(1 if k in dofs else 0 for k in range(fe.ndm))
        by_pattern.setdefault(pattern, []).append(int(fe.ids[row]))
    for pattern, node_list in by_pattern.items():
        ops.fix(nodes=node_list, dofs=pattern)
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for row, force in fe.nodal.items():
            p.load(
                node=int(fe.ids[row]), forces=tuple(float(x) for x in force[: fe.ndm])
            )


def _reactions(live: Any, fe: _FEModel) -> np.ndarray:
    live.reactions()
    reactions = np.zeros(3)
    for row in fe.fixed:
        reactions[: fe.ndm] += _v(live.nodeReaction(int(fe.ids[row])))[: fe.ndm]
    return reactions


def _summary(
    fe: _FEModel, live: Any, reactions: np.ndarray, **extra: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "applied": fe.applied.tolist(),
        "reactions": reactions.tolist(),
        "n_nodes": len(fe.ids),
        "n_elements": len(live.getEleTags() or []),
        "mesh_size": fe.size,
        "ties_modelled": dict(fe.ties),
    }
    out.update(extra)
    return out


def _result(
    fe: _FEModel,
    live: Any,
    overlays: dict[str, Any],
    reactions: np.ndarray,
    **extras: Any,
) -> StrutTieOverlays:
    return StrutTieOverlays(
        overlays=overlays,
        case=fe.case,
        n_nodes=len(fe.ids),
        n_elements=len(live.getEleTags() or []),
        mesh_size=fe.size,
        applied=(float(fe.applied[0]), float(fe.applied[1]), float(fe.applied[2])),
        reactions=(float(reactions[0]), float(reactions[1]), float(reactions[2])),
        fixed_nodes=len(fe.fixed),
        extras=dict(extras),
    )


# ---------------------------------------------------------------------------
# Linear elastic — trajectories
# ---------------------------------------------------------------------------
def strut_tie_overlays(
    model: Mapping[str, Any],
    *,
    case: str | None = None,
    mesh_size: float | None = None,
    extra_fixed_planes: Sequence[tuple[str, float, Sequence[int]]] = (),
    glyph_fraction: float = 0.6,
    reinforced: bool = False,
    verbose: bool = False,
) -> StrutTieOverlays:
    """Mesh, solve linearly and glyph the D-region of an apeConcrete model JSON.

    Parameters
    ----------
    model
        ``apeConcrete.stm.model_to_dict`` output (schema 1).
    case
        Load case name; the first one when ``None``.
    mesh_size
        Global characteristic length (mm); default is the bounding-box
        diagonal / 25.
    extra_fixed_planes
        ``(axis, value, dofs)`` triples: every mesh node with that
        coordinate is fixed in those DOFs (0-based: ``(0, 1)`` is x and y).
    glyph_fraction
        Longest glyph as a fraction of the element size.
    reinforced
        Place the ties as elastic truss bars (see :func:`_build`); off by
        default so the glyphs show the plain elastic load path.
    """
    fe = _build(
        model,
        case=case,
        mesh_size=mesh_size,
        extra_fixed_planes=extra_fixed_planes,
        verbose=verbose,
        reinforced=reinforced,
    )
    E = EC_COEFFICIENT_4700 * math.sqrt(fe.fc)
    ops = apeSees(fe.fem)
    ops.model(ndm=fe.ndm, ndf=fe.ndm)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=POISSON_RATIO)
    if fe.ties:
        ops.uniaxialMaterial.ElasticMaterial(E=fe.Es, name=REBAR_MATERIAL_NAME)
    _elements(ops, fe, mat)
    _restraints_and_loads(ops, fe)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    rc = emitter.analyze(steps=1)
    if rc != 0:
        raise RuntimeError(f"OpenSees analyze returned {rc}")
    live = emitter.ops

    # -- principal-stress glyphs --------------------------------------------
    ncomp = 3 if fe.plane else 6
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for tag in live.getEleTags() or []:
        enodes = live.eleNodes(int(tag))
        xyz = np.array(
            [list(live.nodeCoord(int(n))) + [fe.z0] * (3 - fe.ndm) for n in enodes]
        )
        centroid = xyz.mean(axis=0)
        h = 2.0 * float(np.mean(np.linalg.norm(xyz - centroid[None, :], axis=1)))
        raw = _v(live.eleResponse(int(tag), "stresses"))
        if raw.size % ncomp != 0 or raw.size == 0:
            continue
        s = raw.reshape(-1, ncomp).mean(axis=0)
        if fe.plane:
            tensor = np.array([[s[0], s[2], 0.0], [s[2], s[1], 0.0], [0.0, 0.0, 0.0]])
        else:
            tensor = np.array(
                [[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]]
            )
        w, v = np.linalg.eigh(tensor)  # ascending
        records.append((centroid, w, v, h))
    # Normalise glyph lengths by a robust magnitude: the 90th percentile of
    # the principal stresses, so the singular peaks under a bearing plate do
    # not shrink every other glyph to a dot. Lengths are capped at one.
    magnitudes = (
        np.array([abs(x) for _, w, _, _ in records for x in w])
        if records
        else np.array([1.0])
    )
    sigma_max = float(np.max(magnitudes)) or 1.0
    sigma_ref = float(np.percentile(magnitudes, 90.0)) or sigma_max
    segments: list[list[Any]] = []
    for centroid, w, v, h in records:
        s3, s1 = float(w[0]), float(w[2])
        if fe.plane:
            # drop the null out-of-plane eigenpair
            in_plane = [k for k in range(3) if abs(v[2, k]) < 0.5]
            if len(in_plane) == 2:
                s3, s1 = float(w[in_plane[0]]), float(w[in_plane[1]])
                e3, e1 = v[:, in_plane[0]], v[:, in_plane[1]]
            else:
                e3, e1 = v[:, 0], v[:, 2]
        else:
            e3, e1 = v[:, 0], v[:, 2]
        if s3 < 0.0:
            length = glyph_fraction * h * min(1.0, abs(s3) / sigma_ref)
            a, b = centroid - e3 * length / 2.0, centroid + e3 * length / 2.0
            segments.append([a.tolist(), b.tolist(), 0.0, s3])
        if s1 > 0.0:
            length = glyph_fraction * h * min(1.0, s1 / sigma_ref)
            a, b = centroid - e1 * length / 2.0, centroid + e1 * length / 2.0
            segments.append([a.tolist(), b.tolist(), s1, 0.0])

    reactions = _reactions(live, fe)
    element_name = "Tri31 plane stress" if fe.plane else "FourNodeTetrahedron"
    overlays: dict[str, Any] = {
        "fe_trajectories": {
            "segments": segments,
            "source": f"apeGmsh linear elastic ({element_name})",
            "case": fe.case,
            "n_elements": len(live.getEleTags() or []),
            "sigma_max": sigma_max,
            "sigma_ref": sigma_ref,
        },
        "fe_summary": _summary(fe, live, reactions, E=E, nu=POISSON_RATIO),
    }
    return _result(fe, live, overlays, reactions)


# ---------------------------------------------------------------------------
# Nonlinear — load–deformation curve
# ---------------------------------------------------------------------------
def strut_tie_pushover(
    model: Mapping[str, Any],
    *,
    case: str | None = None,
    mesh_size: float | None = None,
    extra_fixed_planes: Sequence[tuple[str, float, Sequence[int]]] = (),
    target_displacement: float | None = None,
    steps: int = 25,
    ft: float | None = None,
    Gf: float | None = None,
    Gc: float | None = None,
    max_halvings: int = 6,
    reinforced: bool = True,
    hardening_b: float = 0.01,
    verbose: bool = False,
) -> StrutTieOverlays:
    """Push the D-region under displacement control with the fork's
    plastic-damage concrete and return the load–deformation curve.

    Parameters
    ----------
    target_displacement
        Displacement of the control node along the dominant load
        direction at which to stop (mm); default ``diagonal / 200``.
    steps
        Number of displacement increments to the target; increments are
        halved (up to ``max_halvings`` times per step) when Newton fails,
        and grow back afterwards.
    ft, Gf, Gc
        Tensile strength (MPa), tensile and compressive fracture energies
        (N/mm); defaults in the module docstring.

    reinforced, hardening_b
        Place the model's ties as conformal truss bars with ``Steel02``
        (``fy``, ``Es`` from the model's steel, kinematic hardening ratio
        ``hardening_b``); ``reinforced=False`` gives the plain concrete's
        curve.

    Notes
    -----
    Requires the Ladruno fork of OpenSees (``LadrunoConcrete3D``); on
    stock ``openseespy`` the bridge raises at emit. Bars are perfectly
    bonded and end where the tie ends: anchorage is not modelled.
    """
    fe = _build(
        model,
        case=case,
        mesh_size=mesh_size,
        extra_fixed_planes=extra_fixed_planes,
        verbose=verbose,
        reinforced=reinforced,
    )
    fc = fe.fc
    E = EC_COEFFICIENT_4700 * math.sqrt(fc)
    ft_v = ft if ft is not None else FT_COEFFICIENT_0_33 * math.sqrt(fc)
    gf_v = Gf if Gf is not None else GF_MC2010_COEFFICIENT * fc**GF_MC2010_EXPONENT
    gc_v = Gc if Gc is not None else GC_OVER_GF_250 * gf_v
    target = target_displacement if target_displacement is not None else fe.diag / 200.0
    if steps < 1 or target <= 0.0:
        raise ValueError("steps must be >= 1 and target_displacement > 0")

    dof0 = int(np.argmax(np.abs(fe.applied[: fe.ndm])))
    sign = 1.0 if fe.applied[dof0] >= 0.0 else -1.0
    control_node = int(fe.ids[fe.control_row])
    dof = dof0 + 1  # OpenSees numbering
    reference = float(np.linalg.norm(fe.applied[: fe.ndm]))

    ops = apeSees(fe.fem)
    ops.model(ndm=fe.ndm, ndf=fe.ndm)
    mat = ops.nDMaterial.LadrunoConcrete3D(
        E=E, nu=POISSON_RATIO, fc=fc, ft=ft_v, Gf=gf_v, Gc=gc_v, auto_regularize=True
    )
    if fe.ties:
        ops.uniaxialMaterial.Steel02(
            fy=fe.fy, E=fe.Es, b=hardening_b, name=REBAR_MATERIAL_NAME
        )
    _elements(ops, fe, mat)
    _restraints_and_loads(ops, fe)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-6, max_iter=60)
    ops.algorithm.Newton()
    du = sign * target / steps
    ops.integrator.DisplacementControl(node=control_node, dof=dof, dU=du)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    live = emitter.ops

    points: list[list[float]] = [[0.0, 0.0]]
    lambdas: list[float] = [0.0]
    stopped = "target"
    reached = 0.0
    current = du
    while abs(reached) < target - 1e-12:
        rc = live.analyze(1)
        halvings = 0
        while rc != 0 and halvings < max_halvings:
            halvings += 1
            current *= 0.5
            live.integrator("DisplacementControl", control_node, dof, current)
            rc = live.analyze(1)
        if rc != 0:
            stopped = "divergence"
            break
        reached = float(live.nodeDisp(control_node, dof))
        lam = float(live.getTime())
        lambdas.append(lam)
        points.append([abs(reached), lam * reference])
        if halvings == 0 and abs(current) < abs(du):
            current = min(abs(du), 2.0 * abs(current)) * sign
            live.integrator("DisplacementControl", control_node, dof, current)
        peak = max(p[1] for p in points)
        if peak > 0.0 and points[-1][1] < 0.5 * peak:
            stopped = "post_peak"
            break

    capacity = max(p[1] for p in points)
    reactions = _reactions(live, fe)
    element_name = "Tri31 plane stress" if fe.plane else "FourNodeTetrahedron"
    overlays: dict[str, Any] = {
        "fe_curve": {
            "points": points,
            "capacity": capacity,
            "capacity_factor": capacity / reference if reference else 0.0,
            "source": (
                f"apeGmsh + LadrunoConcrete3D ({element_name}, "
                f"{'ties as Steel02 truss bars' if fe.ties else 'plain concrete'})"
            ),
            "case": fe.case,
            "control": {
                "node": control_node,
                "dof": dof,
                "target_displacement": target,
                "reached": abs(reached),
            },
            "stopped": stopped,
            "load_factors": lambdas,
        },
        "fe_summary": _summary(
            fe, live, reactions, E=E, nu=POISSON_RATIO, fc=fc, ft=ft_v, Gf=gf_v, Gc=gc_v
        ),
    }
    return _result(fe, live, overlays, reactions, capacity=capacity, stopped=stopped)


def write_strut_tie_overlays(
    model_json: str | Path,
    out_json: str | Path,
    *,
    case: str | None = None,
    mesh_size: float | None = None,
    extra_fixed_planes: Sequence[tuple[str, float, Sequence[int]]] = (),
    pushover: bool = False,
    **pushover_kwargs: Any,
) -> Path:
    """Read a model JSON file, run the linear overlay (and, with
    ``pushover=True``, the nonlinear curve as well), write one overlays JSON."""
    model = json.loads(Path(model_json).read_text(encoding="utf-8"))
    result = strut_tie_overlays(
        model, case=case, mesh_size=mesh_size, extra_fixed_planes=extra_fixed_planes
    )
    overlays = dict(result.overlays)
    if pushover:
        curve = strut_tie_pushover(
            model,
            case=case,
            mesh_size=mesh_size,
            extra_fixed_planes=extra_fixed_planes,
            **pushover_kwargs,
        )
        overlays["fe_curve"] = curve.overlays["fe_curve"]
        overlays["fe_summary_nonlinear"] = curve.overlays["fe_summary"]
    target = Path(out_json)
    target.write_text(json.dumps(overlays, allow_nan=False), encoding="utf-8")
    return target


__all__ = [
    "EC_COEFFICIENT_4700",
    "FT_COEFFICIENT_0_33",
    "GC_OVER_GF_250",
    "GF_MC2010_COEFFICIENT",
    "GF_MC2010_EXPONENT",
    "POISSON_RATIO",
    "StrutTieOverlays",
    "strut_tie_overlays",
    "strut_tie_pushover",
    "write_strut_tie_overlays",
]
