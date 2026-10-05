"""``<stem>.geometry.h5`` — the geometry sibling (ADR 0112 D2a, V0 Q1).

The session writes its tessellated geometry as an artifact of its own,
next to ``model.h5``, so a reader that knows nothing of apeGmsh (the
browser app) can draw the CAD entities, named by label and physical
group, without a BRep kernel.  The layout is the ``/geometry`` section of
``architecture/h5-schema.md``: concatenated arrays plus offsets per
dimension (CSR), ``int32`` indices, ``float64`` coordinates, one level of
detail, no normals and no UVs.

This module is **gmsh-only**: it reads the current gmsh model and writes
one HDF5 file.  It never imports ``apeGmsh.viewers`` (frozen by ADR 0112
D8) and never touches the FEMData broker.

Two capture routes (V0 decision 6, amendment 3):

* :func:`capture_geometry` with ``source="mesh"`` — at the exit of
  ``g.mesh.generation.generate()``: surfaces are triangulated from the
  real mesh, curves are sampled parametrically with
  :data:`CURVE_SAMPLES` points.
* :func:`capture_temp_mesh` — at ``end()`` when the session never
  meshed: a **2-D, surface-only** temporary mesh at
  ``lod_size = LOD_FRACTION * bbox diagonal`` is generated, captured and
  cleared again.  Never a 3-D mesh.

Integer policy (V0 decision 3, amendment 2): every id, offset and count
is checked against ``int32`` **before the file is opened**; a value that
does not fit raises :class:`GeometryInt32Overflow` and nothing is
written.  The writer never truncates or wraps.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import gmsh
import h5py
import numpy as np

__all__ = [
    "CURVE_SAMPLES",
    "LOD_FRACTION",
    "GeometryArtifactWarning",
    "GeometryCapture",
    "GeometryInt32Overflow",
    "artifact_session_id",
    "artifact_zones",
    "capture_fallback",
    "capture_geometry",
    "capture_temp_mesh",
    "geometry_sibling_path",
    "is_apegmsh_artifact",
    "write_geometry_h5",
]

#: Parametric samples per curve (V0 Q9).
CURVE_SAMPLES: int = 32
#: Display level of detail as a fraction of the model's bbox diagonal (Q9).
LOD_FRACTION: float = 0.03

_I4_MIN = -(2 ** 31)
_I4_MAX = 2 ** 31 - 1
_SOURCES = ("mesh", "temp_mesh")


class GeometryArtifactWarning(UserWarning):
    """A geometry capture or write did not complete.

    Raised as a warning, never an exception: the gmsh session must still
    finalize, and a missing or partial sibling is a reader's problem to
    report, not a reason to lose the model (V0 decision 6).
    """


class GeometryInt32Overflow(OverflowError):
    """An id, offset or count of the geometry zone does not fit ``int32``.

    The writer refuses before opening the file (V0 amendment 2).  2³¹ is
    the documented limit of the ADR 0112 zones.
    """


# ---------------------------------------------------------------------------
# Capture record
# ---------------------------------------------------------------------------


def _empty_i(shape: tuple[int, ...] = (0,)) -> np.ndarray:
    return np.zeros(shape, dtype=np.int64)


def _empty_f(shape: tuple[int, ...] = (0, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


@dataclass(frozen=True)
class GeometryCapture:
    """One tessellation of the current gmsh model, in the file's layout.

    Integer arrays are kept as ``int64`` here and narrowed to ``int32`` by
    the writer after the overflow check.  Offsets are CSR: row ``i`` of an
    entity table owns ``[offsets[i], offsets[i + 1])`` of the concatenated
    array it indexes.
    """

    source: str
    gmsh_version: str
    curve_samples: int
    lod_size: float
    bbox: np.ndarray                     # (6,) xmin ymin zmin xmax ymax zmax
    status: str                          # "ok" | "partial"
    entities_dim: np.ndarray = field(default_factory=_empty_i)
    entities_tag: np.ndarray = field(default_factory=_empty_i)
    entities_bbox: np.ndarray = field(default_factory=lambda: _empty_f((0, 6)))
    entities_ok: np.ndarray = field(default_factory=_empty_i)
    points_entity: np.ndarray = field(default_factory=_empty_i)
    points_xyz: np.ndarray = field(default_factory=_empty_f)
    curves_entity: np.ndarray = field(default_factory=_empty_i)
    curves_vertex_offsets: np.ndarray = field(default_factory=lambda: _empty_i((1,)))
    curves_vertices: np.ndarray = field(default_factory=_empty_f)
    surfaces_entity: np.ndarray = field(default_factory=_empty_i)
    surfaces_vertex_offsets: np.ndarray = field(default_factory=lambda: _empty_i((1,)))
    surfaces_vertices: np.ndarray = field(default_factory=_empty_f)
    surfaces_triangle_offsets: np.ndarray = field(default_factory=lambda: _empty_i((1,)))
    surfaces_triangles: np.ndarray = field(default_factory=lambda: _empty_i((0, 3)))
    volumes_entity: np.ndarray = field(default_factory=_empty_i)
    volumes_face_offsets: np.ndarray = field(default_factory=lambda: _empty_i((1,)))
    volumes_faces: np.ndarray = field(default_factory=_empty_i)
    memberships_dim: np.ndarray = field(default_factory=_empty_i)
    memberships_tag: np.ndarray = field(default_factory=_empty_i)
    memberships_kind: tuple[str, ...] = ()
    memberships_name: tuple[str, ...] = ()
    memberships_pg: np.ndarray = field(default_factory=_empty_i)

    @property
    def n_entities(self) -> int:
        return int(self.entities_tag.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.surfaces_triangles.shape[0])


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def geometry_sibling_path(model_path: "str | Path") -> Path:
    """``<dir>/<stem>.geometry.h5`` for a ``<dir>/<stem>.h5`` model file."""
    p = Path(model_path)
    return p.with_name(f"{p.stem}.geometry.h5")


def is_apegmsh_artifact(path: "str | Path") -> bool:
    """True when ``path`` is an HDF5 file whose ``/meta`` carries one of
    apeGmsh's per-zone version keys, or (a file from before the per-zone
    split) the envelope key together with ``apeGmsh_version``.

    The automatic D1 write replaces only such files (or files that do
    not exist): anything else is a foreign file and is left alone.  A
    file h5py cannot open, or one without ``/meta``, is foreign.  The
    generic envelope key ``schema_version`` alone is not proof: a
    third-party file may well carry one.
    """
    from apeGmsh.opensees._internal.schema_version import ENVELOPE_KEY, _ZONE_KEY

    zone_keys = tuple(_ZONE_KEY.values())
    try:
        with h5py.File(str(path), "r") as f:
            if "meta" not in f:
                return False
            attrs = f["meta"].attrs
            if any(k in attrs for k in zone_keys):
                return True
            return ENVELOPE_KEY in attrs and "apeGmsh_version" in attrs
    except OSError:
        return False


def artifact_zones(path: "str | Path") -> frozenset[str]:
    """The zones an apeGmsh artifact holds, judged by root group.

    ``opensees`` is ``/opensees``, ``results`` is ``/stages``, ``geometry``
    is ``/geometry``, ``provenance`` is ``/provenance`` (h5-schema.md,
    "Zone registry"); ``neutral`` is any other root group than ``/meta``.
    Root groups, not ``/meta`` keys: the neutral writer stamps
    ``opensees_schema_version`` on a file with no ``/opensees`` zone.

    The automatic D1 write never replaces a target that holds a zone it
    would drop: an ``apeSees(fem).h5("<model_name>.h5")`` written inside
    the ``with`` block keeps its ``/opensees`` zone.
    """
    from apeGmsh.opensees._internal.schema_version import (
        GEOMETRY, NEUTRAL, OPENSEES, PROVENANCE, RESULTS,
    )

    root_of = {OPENSEES: "opensees", RESULTS: "stages",
               GEOMETRY: "geometry", PROVENANCE: "provenance"}
    with h5py.File(str(path), "r") as f:
        roots = set(f.keys()) - {"meta"}
    zones = {zone for zone, root in root_of.items() if root in roots}
    if roots - set(root_of.values()):
        zones.add(NEUTRAL)
    return frozenset(zones)


def artifact_session_id(path: "str | Path") -> str | None:
    """The ``/meta/session_id`` an artifact carries, or ``None`` when the
    file has none (written before #1304) or cannot be opened."""
    try:
        with h5py.File(str(path), "r") as f:
            if "meta" not in f or "session_id" not in f["meta"].attrs:
                return None
            raw = f["meta"].attrs["session_id"]
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    except OSError:
        return None


def _artifact_target_is_ours(
    target: "Path",
    *,
    writes: frozenset[str],
    overwrite: bool,
    session_id: str | None = None,
) -> bool:
    """May an automatic D1 write replace ``target``?

    The V2b ownership rule, lifted to a module function (V2d, #1307) so
    the end-of-session write and the bridge's write share it.  Yes when
    ``target`` does not exist, or when it is an apeGmsh artifact (a
    ``/meta`` zone version key is present), ``overwrite`` is on, and
    every zone it holds is in ``writes`` (the zones the write produces).
    A foreign file, an existing one under ``overwrite=False``, or one
    holding a zone the write would drop is never replaced: one warning,
    and that file is skipped (never written elsewhere).

    ``session_id`` (the end-of-session write passes the id its snapshot
    would stamp) adds one silent case: a target stamped with the same
    ``/meta/session_id`` that already holds every zone in ``writes`` is
    this run's own, fuller output (the bridge wrote neutral +
    ``/opensees`` first), and it is kept without a warning.  A foreign
    file, or one from another session (an older run), keeps the warning.

    Warnings are raised at ``stacklevel=4``: this function, the writer's
    private method, its public caller (``end()``, ``tcl()``, ...), the
    user's line.
    """
    if not target.exists():
        return True
    if not overwrite:
        warnings.warn(
            f"{target} exists and overwrite=False; not written",
            stacklevel=4,
        )
        return False
    if not is_apegmsh_artifact(target):
        warnings.warn(
            f"{target} exists and is not an apeGmsh artifact (no /meta "
            f"schema key); not overwritten. Pass save_to= (or ops.h5(path)) "
            f"to write the model elsewhere.",
            stacklevel=4,
        )
        return False
    held = artifact_zones(target)
    dropped = sorted(held - writes)
    if dropped:
        if (
            session_id is not None
            and writes <= held
            and artifact_session_id(target) == session_id
        ):
            return False
        warnings.warn(
            f"{target} holds the {', '.join(dropped)} zone(s) that the "
            f"automatic write would drop; not overwritten. Write that "
            f"file under another name, or pass save_to= / ops.h5(path).",
            stacklevel=4,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _model_bbox() -> np.ndarray:
    """The model's bounding box, or NaN when the model has no entity."""
    if not gmsh.model.getEntities():
        return np.full(6, np.nan, dtype=np.float64)
    return np.asarray(gmsh.model.getBoundingBox(-1, -1), dtype=np.float64)


def _lod_size(bbox: np.ndarray) -> float:
    if not np.all(np.isfinite(bbox)):
        return 0.0
    diag = float(np.linalg.norm(bbox[3:] - bbox[:3]))
    return LOD_FRACTION * diag


def _sample_curve(tag: int, n: int) -> np.ndarray:
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    params = np.linspace(float(lo[0]), float(hi[0]), n)
    xyz = np.asarray(gmsh.model.getValue(1, tag, params), dtype=np.float64)
    return xyz.reshape(-1, 3)


def _node_table() -> tuple[np.ndarray, np.ndarray]:
    """Every mesh node of the model, sorted by tag: ``(tags, xyz)``.

    One call for the whole capture.  A per-surface ``getNodes(2, tag,
    includeBoundary=True)`` is not enough: a surface's elements also use
    the nodes of entities *embedded* in it (a column line in a slab),
    which are classified on the embedded entity, not on the surface or
    its boundary (San Ramon 1A: 13 slabs failed that way).
    """
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(tags)
    return tags[order], xyz[order]


def _tessellate_surface(
    tag: int, node_tags: np.ndarray, node_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (V,3), triangles (T,3))`` of a meshed surface.

    Triangles index the surface's own vertex slice (0-based, local).  Only
    corner nodes are used: a quad becomes two triangles, a high-order
    element its primary corners.  A surface without 2-D elements (not
    meshed), or one whose elements name a node the model does not have,
    raises, so the caller marks it ``ok = 0``.
    """
    etypes, _etags, enodes = gmsh.model.mesh.getElements(2, tag)
    tris: list[np.ndarray] = []
    for etype, nodes in zip(etypes, enodes):
        _name, _dim, _order, n_nodes, _loc, n_primary = \
            gmsh.model.mesh.getElementProperties(etype)
        conn = np.asarray(nodes, dtype=np.int64).reshape(-1, int(n_nodes))
        corners = conn[:, : int(n_primary)]
        if n_primary == 3:
            tris.append(corners)
        elif n_primary == 4:
            tris.append(corners[:, (0, 1, 2)])
            tris.append(corners[:, (0, 2, 3)])
        else:
            raise ValueError(
                f"surface {tag}: element type {etype} has {n_primary} "
                f"corners; expected 3 or 4"
            )
    if not tris:
        raise ValueError(f"surface {tag} has no 2-D mesh elements")
    tri_tags = np.vstack(tris)
    uniq, inverse = np.unique(tri_tags, return_inverse=True)
    idx = np.searchsorted(node_tags, uniq)
    found = (idx < node_tags.shape[0])
    found[found] = node_tags[idx[found]] == uniq[found]
    if not np.all(found):
        raise KeyError(
            f"surface {tag}: element nodes {uniq[~found][:5].tolist()} "
            f"are not in the model's node table"
        )
    return node_xyz[idx], np.asarray(inverse, dtype=np.int64).reshape(-1, 3)


def capture_geometry(*, source: str, curve_samples: int = CURVE_SAMPLES) -> GeometryCapture:
    """Tessellate the current gmsh model into a :class:`GeometryCapture`.

    Points come from the kernel, curves from ``curve_samples`` parametric
    samples, surfaces from the **mesh currently in the model** and
    volumes from their boundary surfaces.  An entity whose tessellation
    fails is kept with ``ok = 0`` (and the capture is ``partial``), so one
    bad face never loses the rest of the model.
    """
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {_SOURCES}, got {source!r}")
    from apeGmsh.core.Labels import LABEL_PREFIX, is_label_pg

    entities = [(int(d), int(t)) for d, t in gmsh.model.getEntities()]
    ent_dim = np.array([d for d, _ in entities], dtype=np.int64)
    ent_tag = np.array([t for _, t in entities], dtype=np.int64)
    ent_bbox = np.full((len(entities), 6), np.nan, dtype=np.float64)
    ent_ok = np.ones(len(entities), dtype=np.int64)
    for i, (d, t) in enumerate(entities):
        try:
            ent_bbox[i] = gmsh.model.getBoundingBox(d, t)
        except Exception:  # noqa: BLE001 - a degenerate entity has none
            pass

    # points
    p_entity: list[int] = []
    p_xyz: list[np.ndarray] = []
    # curves
    c_entity: list[int] = []
    c_offsets: list[int] = [0]
    c_vertices: list[np.ndarray] = []
    # surfaces
    s_entity: list[int] = []
    s_voffsets: list[int] = [0]
    s_vertices: list[np.ndarray] = []
    s_toffsets: list[int] = [0]
    s_triangles: list[np.ndarray] = []
    surface_row: dict[int, int] = {}
    # volumes
    v_entity: list[int] = []
    v_offsets: list[int] = [0]
    v_faces: list[int] = []

    node_tags, node_xyz = _node_table()
    for i, (d, t) in enumerate(entities):
        if d == 0:
            p_entity.append(i)
            try:
                xyz = np.asarray(gmsh.model.getValue(0, t, []), dtype=np.float64)
                p_xyz.append(xyz.reshape(3))
            except Exception:  # noqa: BLE001 - kept as ok = 0
                ent_ok[i] = 0
                p_xyz.append(np.full(3, np.nan))
        elif d == 1:
            c_entity.append(i)
            try:
                pts = _sample_curve(t, curve_samples)
                c_vertices.append(pts)
                c_offsets.append(c_offsets[-1] + pts.shape[0])
            except Exception:  # noqa: BLE001 - kept as ok = 0
                ent_ok[i] = 0
                c_offsets.append(c_offsets[-1])
        elif d == 2:
            surface_row[t] = len(s_entity)
            s_entity.append(i)
            try:
                verts, tris = _tessellate_surface(t, node_tags, node_xyz)
                s_vertices.append(verts)
                s_triangles.append(tris)
                s_voffsets.append(s_voffsets[-1] + verts.shape[0])
                s_toffsets.append(s_toffsets[-1] + tris.shape[0])
            except Exception:  # noqa: BLE001 - kept as ok = 0
                ent_ok[i] = 0
                s_voffsets.append(s_voffsets[-1])
                s_toffsets.append(s_toffsets[-1])

    for i, (d, t) in enumerate(entities):
        if d != 3:
            continue
        v_entity.append(i)
        try:
            faces = [
                surface_row[abs(int(ft))]
                for _fd, ft in gmsh.model.getBoundary(
                    [(3, t)], combined=False, oriented=False, recursive=False
                )
            ]
            v_faces.extend(faces)
            v_offsets.append(v_offsets[-1] + len(faces))
        except Exception:  # noqa: BLE001 - kept as ok = 0
            ent_ok[i] = 0
            v_offsets.append(v_offsets[-1])

    # memberships: labels are ``_label:``-prefixed physical groups
    m_dim: list[int] = []
    m_tag: list[int] = []
    m_kind: list[str] = []
    m_name: list[str] = []
    m_pg: list[int] = []
    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        if is_label_pg(name):
            kind, clean, pg = "label", name[len(LABEL_PREFIX):], -1
        else:
            kind, clean, pg = "physical_group", name, int(pg_tag)
        for et in gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag):
            m_dim.append(int(pg_dim))
            m_tag.append(int(et))
            m_kind.append(kind)
            m_name.append(clean)
            m_pg.append(pg)

    bbox = _model_bbox()
    return GeometryCapture(
        source=source,
        gmsh_version=str(gmsh.__version__),
        curve_samples=int(curve_samples),
        lod_size=_lod_size(bbox),
        bbox=bbox,
        status="ok" if bool(np.all(ent_ok == 1)) else "partial",
        entities_dim=ent_dim,
        entities_tag=ent_tag,
        entities_bbox=ent_bbox,
        entities_ok=ent_ok,
        points_entity=np.asarray(p_entity, dtype=np.int64),
        points_xyz=(np.vstack(p_xyz) if p_xyz else _empty_f()),
        curves_entity=np.asarray(c_entity, dtype=np.int64),
        curves_vertex_offsets=np.asarray(c_offsets, dtype=np.int64),
        curves_vertices=(np.vstack(c_vertices) if c_vertices else _empty_f()),
        surfaces_entity=np.asarray(s_entity, dtype=np.int64),
        surfaces_vertex_offsets=np.asarray(s_voffsets, dtype=np.int64),
        surfaces_vertices=(np.vstack(s_vertices) if s_vertices else _empty_f()),
        surfaces_triangle_offsets=np.asarray(s_toffsets, dtype=np.int64),
        surfaces_triangles=(
            np.vstack(s_triangles) if s_triangles else _empty_i((0, 3))
        ),
        volumes_entity=np.asarray(v_entity, dtype=np.int64),
        volumes_face_offsets=np.asarray(v_offsets, dtype=np.int64),
        volumes_faces=np.asarray(v_faces, dtype=np.int64),
        memberships_dim=np.asarray(m_dim, dtype=np.int64),
        memberships_tag=np.asarray(m_tag, dtype=np.int64),
        memberships_kind=tuple(m_kind),
        memberships_name=tuple(m_name),
        memberships_pg=np.asarray(m_pg, dtype=np.int64),
    )


_TEMP_MESH_SIZE_OPTIONS = ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax")
#: A display tessellation needs no element quality: plain Delaunay with
#: optimisation and smoothing off meshes a unit box in 29 ms against 66 ms
#: for the default Frontal-Delaunay (measured on #1305), and this runs at
#: the end of every session that never meshed.
_TEMP_MESH_FIXED_OPTIONS = {
    "Mesh.Algorithm": 5,
    "Mesh.Optimize": 0,
    "Mesh.Smoothing": 0,
}


def _has_surface_elements() -> bool:
    _etypes, tags, _nodes = gmsh.model.mesh.getElements(2)
    return any(len(t) for t in tags)


def capture_fallback(*, curve_samples: int = CURVE_SAMPLES) -> GeometryCapture:
    """The ``end()`` route for a session with no capture from ``generate()``.

    If the model already carries 2-D elements (a ``from_msh`` import, a
    mesh made with raw gmsh calls, a ``generate()`` whose capture
    failed), they are captured as ``source = "mesh"`` and the user's mesh
    is left exactly as it is: nothing is generated and nothing cleared.
    Otherwise :func:`capture_temp_mesh` builds and clears a temporary 2-D
    mesh.
    """
    if _has_surface_elements():
        return capture_geometry(source="mesh", curve_samples=curve_samples)
    return capture_temp_mesh(curve_samples=curve_samples)


def capture_temp_mesh(*, curve_samples: int = CURVE_SAMPLES) -> GeometryCapture:
    """Capture a session that never meshed: a 2-D temporary mesh, then clear.

    The surfaces are meshed at ``lod_size`` (``Mesh.MeshSizeMin`` and
    ``Mesh.MeshSizeMax`` are pinned to it, and the algorithm set to plain
    Delaunay without optimisation, for the call; every option is restored
    after), **never in 3-D** (V0 amendment 3).  Whatever the mesher
    leaves is cleared before returning, so the model carries no mesh
    afterwards.  A mesher failure is folded into the capture (the
    unmeshed surfaces are ``ok = 0``, the capture ``partial``) and
    reported through the returned record, never raised: the geometry is
    written even when meshing fails.
    """
    has_surfaces = any(int(d) >= 2 for d, _ in gmsh.model.getEntities())
    if not has_surfaces:
        return capture_geometry(source="temp_mesh", curve_samples=curve_samples)

    lod = _lod_size(_model_bbox())
    saved = {
        k: gmsh.option.getNumber(k)
        for k in (*_TEMP_MESH_SIZE_OPTIONS, *_TEMP_MESH_FIXED_OPTIONS)
    }
    try:
        if lod > 0.0:
            for k in _TEMP_MESH_SIZE_OPTIONS:
                gmsh.option.setNumber(k, lod)
        for k, v in _TEMP_MESH_FIXED_OPTIONS.items():
            gmsh.option.setNumber(k, v)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:  # noqa: BLE001 - captured as partial
            warnings.warn(
                f"geometry sibling: the 2-D temporary mesh failed "
                f"({exc!r}); surfaces are written as not tessellated",
                GeometryArtifactWarning,
                stacklevel=2,
            )
        return capture_geometry(source="temp_mesh", curve_samples=curve_samples)
    finally:
        gmsh.model.mesh.clear()
        for k, v in saved.items():
            gmsh.option.setNumber(k, v)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _as_i4(arr: np.ndarray, what: str) -> np.ndarray:
    """Narrow to ``int32`` or refuse loudly (V0 amendment 2)."""
    a = np.asarray(arr, dtype=np.int64)
    if a.size and (int(a.max()) > _I4_MAX or int(a.min()) < _I4_MIN):
        raise GeometryInt32Overflow(
            f"/geometry/{what}: value outside int32 "
            f"(min {int(a.min())}, max {int(a.max())}); the geometry zone "
            f"stores no int64 and the writer never truncates "
            f"(architecture/h5-schema.md, 'Integer policy')"
        )
    return a.astype(np.int32)


def _as_i1(arr: np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64)
    if a.size and (int(a.max()) > 127 or int(a.min()) < -128):
        raise GeometryInt32Overflow(f"/geometry/{what}: value outside int8")
    return a.astype(np.int8)


def write_geometry_h5(
    path: "str | Path",
    capture: GeometryCapture,
    *,
    session_id: str,
    model_name: str,
    apegmsh_version: str = "",
) -> Path:
    """Write ``capture`` as a ``<stem>.geometry.h5`` sibling.

    ``session_id`` must be the id of the ``model.h5`` written by the same
    session: the pairing rule reads equality (h5-schema.md,
    "/meta/session_id").  Every integer dataset is narrowed and checked
    **before** the file is opened; an overflow raises
    :class:`GeometryInt32Overflow` and leaves no file behind.

    The write is atomic: the payload goes to ``<path>.tmp-<uuid>`` in the
    same directory and is moved into place with ``os.replace`` only once
    complete, so a failure mid-write leaves a previous sibling untouched
    and never a torn file that still pairs with the model.
    """
    import uuid

    from apeGmsh._atomic_io import replace_with_retry
    from .FEMData import _validated_session_id

    sid = _validated_session_id(session_id)
    if capture.source not in _SOURCES:
        raise ValueError(f"capture.source must be one of {_SOURCES}")

    c = capture
    ints = {
        "entities/dim": _as_i1(c.entities_dim, "entities/dim"),
        "entities/tag": _as_i4(c.entities_tag, "entities/tag"),
        "entities/ok": _as_i1(c.entities_ok, "entities/ok"),
        "points/entity": _as_i4(c.points_entity, "points/entity"),
        "curves/entity": _as_i4(c.curves_entity, "curves/entity"),
        "curves/vertex_offsets": _as_i4(c.curves_vertex_offsets, "curves/vertex_offsets"),
        "surfaces/entity": _as_i4(c.surfaces_entity, "surfaces/entity"),
        "surfaces/vertex_offsets": _as_i4(
            c.surfaces_vertex_offsets, "surfaces/vertex_offsets"),
        "surfaces/triangle_offsets": _as_i4(
            c.surfaces_triangle_offsets, "surfaces/triangle_offsets"),
        "surfaces/triangles": _as_i4(c.surfaces_triangles, "surfaces/triangles"),
        "volumes/entity": _as_i4(c.volumes_entity, "volumes/entity"),
        "volumes/face_offsets": _as_i4(c.volumes_face_offsets, "volumes/face_offsets"),
        "volumes/faces": _as_i4(c.volumes_faces, "volumes/faces"),
        "memberships/dim": _as_i1(c.memberships_dim, "memberships/dim"),
        "memberships/tag": _as_i4(c.memberships_tag, "memberships/tag"),
        "memberships/pg": _as_i4(c.memberships_pg, "memberships/pg"),
    }
    floats = {
        "entities/bbox": np.asarray(c.entities_bbox, dtype=np.float64).reshape(-1, 6),
        "points/xyz": np.asarray(c.points_xyz, dtype=np.float64).reshape(-1, 3),
        "curves/vertices": np.asarray(c.curves_vertices, dtype=np.float64).reshape(-1, 3),
        "surfaces/vertices": np.asarray(
            c.surfaces_vertices, dtype=np.float64).reshape(-1, 3),
    }
    ints["surfaces/triangles"] = ints["surfaces/triangles"].reshape(-1, 3)
    if len(c.memberships_kind) != len(c.memberships_name) or \
            len(c.memberships_kind) != ints["memberships/dim"].shape[0]:
        raise ValueError("memberships columns differ in length")
    out = Path(path)
    tmp = out.with_name(f"{out.name}.tmp-{uuid.uuid4().hex}")
    try:
        with h5py.File(str(tmp), "w") as f:
            _write_payload(
                f, c, ints, floats, session_id=sid, model_name=model_name,
                apegmsh_version=apegmsh_version,
            )
        replace_with_retry(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _write_payload(
    f: "h5py.File",
    c: GeometryCapture,
    ints: dict[str, np.ndarray],
    floats: dict[str, np.ndarray],
    *,
    session_id: str,
    model_name: str,
    apegmsh_version: str,
) -> None:
    """The file body, into an open (temporary) HDF5 file."""
    from apeGmsh.opensees._internal.schema_version import (
        GEOMETRY_KEY, GEOMETRY_SCHEMA_VERSION,
    )

    str_t = h5py.string_dtype(encoding="utf-8")
    meta = f.create_group("meta")
    meta.attrs[GEOMETRY_KEY] = GEOMETRY_SCHEMA_VERSION
    meta.attrs["session_id"] = session_id
    meta.attrs["model_name"] = str(model_name)
    meta.attrs["apeGmsh_version"] = str(apegmsh_version)
    meta.attrs["created_iso"] = datetime.now(tz=timezone.utc).isoformat()

    geo = f.create_group("geometry")
    geo.attrs["source"] = c.source
    geo.attrs["gmsh_version"] = str(c.gmsh_version)
    geo.attrs["curve_samples"] = np.int32(c.curve_samples)
    geo.attrs["lod_size"] = np.float64(c.lod_size)
    geo.attrs["bbox"] = np.asarray(c.bbox, dtype=np.float64).reshape(6)
    geo.attrs["status"] = c.status
    for grp in ("entities", "points", "curves", "surfaces", "volumes", "memberships"):
        geo.create_group(grp)
    for key, arr in ints.items():
        geo.create_dataset(key, data=arr)
    for key, arr in floats.items():
        geo.create_dataset(key, data=arr)
    geo.create_dataset(
        "memberships/kind", data=np.asarray(c.memberships_kind, dtype=object), dtype=str_t
    )
    geo.create_dataset(
        "memberships/name", data=np.asarray(c.memberships_name, dtype=object), dtype=str_t
    )
