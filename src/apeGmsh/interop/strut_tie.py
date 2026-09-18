"""Strut-and-tie consumer — FE overlays for an apeConcrete model JSON.

apeConcrete ADR-0014 §6: the finite-element side of the strut-and-tie
tool lives *beside* apeConcrete, never inside it. This module reads the
model JSON that ``apeConcrete.stm.model_to_dict`` writes (schema 1),
meshes the D-region, runs a **linear elastic** static step on the live
OpenSees domain and returns an ``overlays`` block for
``apeConcrete.plotting.write_stm_viewer(..., overlays=...)``:

``fe_trajectories``
    Principal-stress glyphs, one pair per element at its centroid —
    a compression segment along the σ₃ direction and a tension segment
    along σ₁ — as ``[[x, y, z], [x, y, z], sigma1, sigma3]`` in the
    model's base units (N, mm). Lengths scale with the element size and
    the stress magnitude. The viewer colours compression like struts and
    tension like ties, so the truss the engineer drew can be compared
    with the elastic load path.
``fe_summary``
    Applied and reacted resultants, node/element counts and the mesh
    size, so a page can say what the overlay came from.

How the STM model maps to the FE model (documented, not clever):

* **Region** — a plane region meshes its ``outline`` polygon with
  ``Tri31`` plane-stress triangles of the region thickness; a solid
  region meshes its bounding box (``extrude`` for *z*, the plates and
  nodes for *x*, *y*) with ``FourNodeTetrahedron``. Concrete is
  ``ElasticIsotropic`` with ``E = 4700·√f'c`` (MPa, ACI 318 Eq.
  19.2.2.1b) and ν = 0.2.
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

The nonlinear load–deformation mode (``LadrunoConcrete3D``) is the
next step and is not here yet.
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
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class StrutTieOverlays:
    """What :func:`strut_tie_overlays` returns, plus the dict the viewer takes."""

    overlays: dict[str, Any]
    case: str
    n_nodes: int
    n_elements: int
    mesh_size: float
    applied: tuple[float, float, float]
    reactions: tuple[float, float, float]
    fixed_nodes: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


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


def _principal(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, v = np.linalg.eigh(tensor)  # ascending
    return w, v


def strut_tie_overlays(
    model: Mapping[str, Any],
    *,
    case: str | None = None,
    mesh_size: float | None = None,
    extra_fixed_planes: Sequence[tuple[str, float, Sequence[int]]] = (),
    glyph_fraction: float = 0.6,
    verbose: bool = False,
) -> StrutTieOverlays:
    """Mesh, solve and glyph the D-region of an apeConcrete model JSON.

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
        coordinate is fixed in those DOFs (1-based OpenSees numbering
        is *not* used here — pass ``(0, 1)`` for x and y).
    glyph_fraction
        Longest glyph as a fraction of the element size.
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
            entity = geo.add_plane_surface(loop)
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
            )
            g.model.sync()
            g.physical.add(3, [entity], name="Region")
        g.mesh.sizing.set_global_size(size)
        g.mesh.generation.generate(ndm)
        fem = g.mesh.queries.get_fem_data(dim=ndm)

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
    for pid in support_plate_ids:
        p = plates[pid]
        rows = _plate_nodes(coords, p, plane=plane, tol_n=tol_n, tol_t=tol_t)
        if rows.size == 0:
            rows = np.array([_nearest(coords, _v(p["center"]))])
        fix(rows, _dominant_dofs([p["normal"]], ndm))
    for nid, dirs in support_dirs.items():
        dofs = _dominant_dofs(dirs, ndm)
        pid = plate_of_node.get(nid)
        if pid is not None:
            rows = _plate_nodes(
                coords, plates[pid], plane=plane, tol_n=tol_n, tol_t=tol_t
            )
            if rows.size == 0:
                rows = np.array([_nearest(coords, _v(nodes_stm[nid]["xyz"]))])
        else:
            rows = np.array([_nearest(coords, _v(nodes_stm[nid]["xyz"]))])
        fix(rows, dofs)
    for axis, value, dofs in extra_fixed_planes:
        k = _AXES[axis]
        rows = np.where(np.abs(coords[:, k] - value) <= tol_n * 10.0 + 1e-6)[0]
        fix(rows, {int(d) for d in dofs})
    if not fixed:
        raise ValueError("no restraint could be placed; give extra_fixed_planes")

    # -- loads ---------------------------------------------------------------
    nodal: dict[int, np.ndarray] = {}
    applied = np.zeros(3)
    for ld in chosen["loads"]:
        nid = ld["node"]
        pid = plate_of_node.get(nid)
        if pid is not None and pid in support_plate_ids:
            continue  # a reaction the STM model carried as a load
        force = _v(ld["force"])
        if pid is not None:
            rows = _plate_nodes(
                coords, plates[pid], plane=plane, tol_n=tol_n, tol_t=tol_t
            )
            if rows.size == 0:
                rows = np.array([_nearest(coords, _v(nodes_stm[nid]["xyz"]))])
        else:
            rows = np.array([_nearest(coords, _v(nodes_stm[nid]["xyz"]))])
        share = force / float(rows.size)
        for r in rows:
            nodal[int(r)] = nodal.get(int(r), np.zeros(3)) + share
        applied += force

    # -- OpenSees ------------------------------------------------------------
    fc = float(model["concrete"]["fc"])
    E = EC_COEFFICIENT_4700 * math.sqrt(fc)
    ops = apeSees(fem)
    ops.model(ndm=ndm, ndf=ndm)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=POISSON_RATIO)
    if plane:
        ops.element.Tri31(
            pg="Region", thickness=thickness, material=mat, plane_type="PlaneStress"
        )
    else:
        ops.element.FourNodeTetrahedron(pg="Region", material=mat)
    by_pattern: dict[tuple[int, ...], list[int]] = {}
    for row, dofs in fixed.items():
        pattern = tuple(1 if k in dofs else 0 for k in range(ndm))
        by_pattern.setdefault(pattern, []).append(int(ids[row]))
    for pattern, node_list in by_pattern.items():
        ops.fix(nodes=node_list, dofs=pattern)
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for row, force in nodal.items():
            p.load(node=int(ids[row]), forces=tuple(float(x) for x in force[:ndm]))
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
    ncomp = 3 if plane else 6
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for tag in live.getEleTags() or []:
        enodes = live.eleNodes(int(tag))
        xyz = np.array(
            [list(live.nodeCoord(int(n))) + [z0] * (3 - ndm) for n in enodes]
        )
        centroid = xyz.mean(axis=0)
        h = 2.0 * float(np.mean(np.linalg.norm(xyz - centroid[None, :], axis=1)))
        raw = _v(live.eleResponse(int(tag), "stresses"))
        if raw.size % ncomp != 0 or raw.size == 0:
            continue
        s = raw.reshape(-1, ncomp).mean(axis=0)
        if plane:
            tensor = np.array([[s[0], s[2], 0.0], [s[2], s[1], 0.0], [0.0, 0.0, 0.0]])
        else:
            tensor = np.array(
                [[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]]
            )
        w, v = _principal(tensor)
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
        if plane:
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

    live.reactions()
    reactions = np.zeros(3)
    for row in fixed:
        r = live.nodeReaction(int(ids[row]))
        reactions[:ndm] += _v(r)[:ndm]

    n_elements = len(live.getEleTags() or [])
    overlays: dict[str, Any] = {
        "fe_trajectories": {
            "segments": segments,
            "source": f"apeGmsh linear elastic ({'Tri31 plane stress' if plane else 'FourNodeTetrahedron'})",
            "case": chosen["name"],
            "n_elements": n_elements,
            "sigma_max": sigma_max,
            "sigma_ref": sigma_ref,
        },
        "fe_summary": {
            "applied": applied.tolist(),
            "reactions": reactions.tolist(),
            "n_nodes": len(ids),
            "n_elements": n_elements,
            "mesh_size": float(size),
            "E": E,
            "nu": POISSON_RATIO,
        },
    }
    return StrutTieOverlays(
        overlays=overlays,
        case=chosen["name"],
        n_nodes=len(ids),
        n_elements=n_elements,
        mesh_size=float(size),
        applied=(float(applied[0]), float(applied[1]), float(applied[2])),
        reactions=(float(reactions[0]), float(reactions[1]), float(reactions[2])),
        fixed_nodes=len(fixed),
    )


def write_strut_tie_overlays(
    model_json: str | Path,
    out_json: str | Path,
    *,
    case: str | None = None,
    mesh_size: float | None = None,
    extra_fixed_planes: Sequence[tuple[str, float, Sequence[int]]] = (),
) -> Path:
    """Read a model JSON file, run :func:`strut_tie_overlays`, write the overlays JSON."""
    model = json.loads(Path(model_json).read_text(encoding="utf-8"))
    result = strut_tie_overlays(
        model, case=case, mesh_size=mesh_size, extra_fixed_planes=extra_fixed_planes
    )
    target = Path(out_json)
    target.write_text(json.dumps(result.overlays, allow_nan=False), encoding="utf-8")
    return target


__all__ = [
    "EC_COEFFICIENT_4700",
    "POISSON_RATIO",
    "StrutTieOverlays",
    "strut_tie_overlays",
    "write_strut_tie_overlays",
]
