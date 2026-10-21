"""Regenerate the legacy native files in this directory.

These files are native ``results.h5`` written by apeGmsh's writers BEFORE
the response catalog moved its simplex Gauss tables to gmsh reference
coordinates. Their ``_natural_coords`` hold the old node-0-first
barycentric rows. They must be written by that old code, never by the
current one, so this script runs against an old checkout::

    git worktree add --detach <old> 31ba6a63c
    python tests/fixtures/native_legacy_simplex/generate.py <old> \\
        tests/fixtures/simplex_gp tests/fixtures/native_legacy_simplex

``<old>`` is the checkout root; its ``src`` and its ``tests`` are put first
on ``sys.path``. ``tests/fixtures/simplex_gp`` supplies the engine's own
strains for the recorder transcoder (one solve, fork build 79e062367).
The capture files need openseespy with the Ladruno fork build
(TenNodeTetrahedron returns its 24 strain values only there), and they
need the old checkout for a second reason: the in-process capture that
wrote them was deleted from main in #2191.

Both paths load each element with the quadratic field ``u_i = x_i^2 / 2``,
so the strain an element records at a Gauss point IS that point's world
position: the files carry their own oracle.

Files written:

* ``transcoded_tri.h5`` / ``transcoded_tet.h5`` — the recorder transcoder
  (``RecorderTranscoder``) on ``.out`` files holding the engine's strains.
* ``capture_tri.h5`` / ``capture_tet.h5`` — ``DomainCapture`` of a live
  solve.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

OLD = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(OLD / "src"), str(OLD)]

import numpy as np  # noqa: E402

import apeGmsh  # noqa: E402

assert Path(apeGmsh.__file__).resolve().is_relative_to(OLD / "src"), (
    f"imported {apeGmsh.__file__}, not the old checkout under {OLD}"
)

from apeGmsh.opensees._response_catalog import (  # noqa: E402
    IntRule,
    flatten,
    lookup,
)

SIMPLEX_GP = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3]).resolve()

#: The geometry of ``tests/fixtures/simplex_gp`` (skewed: no node at the
#: reference origin, so a rotated frame cannot hide).
TRI = np.array([[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]])
TET = np.array([[0.1, 0.2, 0.0], [2.3, 0.1, 0.4],
                [0.5, 1.7, 0.2], [0.3, 0.4, 1.9]])

#: (element id in simplex_gp, class, rule) per stem. Tri31 and
#: FourNodeTetrahedron are left out: an ``.out`` recorder gets no
#: element-level strain from them, and their one centroid point does not
#: move between frames.
TRANSCODED = {
    "tri": [(2, "SixNodeTri", IntRule.Triangle_GL_2),
            (4, "LadrunoLST", IntRule.Triangle_GL_2)],
    "tet": [(2, "TenNodeTetrahedron", IntRule.Tet_GL_2),
            (3, "BezierTet10", IntRule.Tet_GL_2)],
}


class _MinimalFem:
    def __init__(self, node_ids: "list[int]", coords: np.ndarray) -> None:
        coords3d = np.zeros((len(node_ids), 3), dtype=np.float64)
        coords3d[:, :coords.shape[1]] = coords
        self.nodes = SimpleNamespace(
            ids=np.asarray(node_ids, dtype=np.int64), coords=coords3d,
        )
        self.elements: list[Any] = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group: Any) -> None:
        from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group, ndf=3)


def _engine_strains(stem: str, cls: str, eid: int, layout: Any) -> Any:
    import warnings

    from apeGmsh.results.readers._mpco import MPCOReader

    per_comp = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MPCOReader(SIMPLEX_GP / f"{stem}.mpco") as r:
            sid = r.stages()[0].id
            for comp in layout.component_layout:
                slab = r.read_gauss(sid, comp, element_ids=np.array([eid]))
                sel = np.asarray(slab.element_index) == eid
                per_comp[comp] = np.asarray(slab.values)[-1][sel][None, None, :]
    return flatten(per_comp, layout)                     # (1, 1, K)


def transcode(stem: str) -> None:
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    from apeGmsh.results.transcoders._recorder import (
        RecorderTranscoder,
        quiet_emit_logical,
    )

    out_dir = OUT / f"_out_{stem}"
    out_dir.mkdir(exist_ok=True)
    fem = _MinimalFem([1], np.zeros((1, 3)))
    records = []
    for eid, cls, rule in TRANSCODED[stem]:
        layout = lookup(cls, rule, "strain")
        records.append(ResolvedRecorderRecord(
            category="gauss", name=cls,
            components=tuple(layout.component_layout),
            dt=None, n_steps=None, element_ids=np.array([eid]),
            element_class_name=cls,
        ))
        (logical,) = quiet_emit_logical(
            records[-1], output_dir=str(out_dir), file_format="out",
            stage_id=None,
        )
        flat = _engine_strains(stem, cls, eid, layout)
        np.savetxt(logical.file_path,
                   np.column_stack([[1.0], flat.reshape(1, -1)]))
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id, records=tuple(records),
    )
    target = OUT / f"transcoded_{stem}.h5"
    target.unlink(missing_ok=True)
    RecorderTranscoder(spec, out_dir, target, fem).run()
    for p in out_dir.iterdir():
        p.unlink()
    out_dir.rmdir()
    print("wrote", target)


def _field(p: Any) -> np.ndarray:
    return 0.5 * np.asarray(p, dtype=float) ** 2


def _bezier_ctrl(corners: np.ndarray, edges: Any) -> list:
    """Bernstein control values of the field on a straight-sided element."""
    return ([_field(p) for p in corners]
            + [2.0 * _field((corners[a] + corners[b]) / 2.0)
               - (_field(corners[a]) + _field(corners[b])) / 2.0
               for a, b in edges])


def capture(stem: str) -> None:
    from apeGmsh.opensees.emitter.live import get_ops
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.results.capture.spec import (
        ResolvedDomainCaptureRecord,
        ResolvedDomainCaptureSpec,
    )

    from tests.fixtures.capture_tags import identity_tags

    ops = get_ops()
    ops.wipe()
    ndm = 2 if stem == "tri" else 3
    ops.model("basic", "-ndm", ndm, "-ndf", ndm)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    node_ids: list[int] = []
    coords: list[np.ndarray] = []
    prescribed: list[tuple[int, np.ndarray]] = []

    def place(start: int, points: Any, values: Any) -> list[int]:
        ids = []
        for i, (p, v) in enumerate(zip(points, values)):
            ops.node(start + i, *map(float, p))
            node_ids.append(start + i)
            coords.append(np.asarray(p, dtype=float))
            prescribed.append((start + i, np.asarray(v, dtype=float)))
            ids.append(start + i)
        return ids

    if stem == "tri":
        e3 = [(0, 1), (1, 2), (2, 0)]
        p6 = list(TRI) + [(TRI[a] + TRI[b]) / 2.0 for a, b in e3]
        n = place(1, TRI, [_field(p) for p in TRI])
        ops.element("tri31", 1, *n, 1.0, "PlaneStress", 1)
        n = place(11, p6, [_field(p) for p in p6])
        ops.element("tri6n", 2, *n, 1.0, "PlaneStress", 1)
        n = place(31, p6, [_field(p) for p in p6])
        ops.element("LadrunoLST", 4, *n, 1)
        eids = [1, 2, 4]
    else:
        e_ten = [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)]
        e_bez = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]
        n = place(1, TET, [_field(p) for p in TET])
        ops.element("FourNodeTetrahedron", 1, *n, 1)
        p10 = list(TET) + [(TET[a] + TET[b]) / 2.0 for a, b in e_ten]
        n = place(11, p10, [_field(p) for p in p10])
        ops.element("TenNodeTetrahedron", 2, *n, 1)
        p10b = list(TET) + [(TET[a] + TET[b]) / 2.0 for a, b in e_bez]
        n = place(21, p10b, _bezier_ctrl(TET, e_bez))
        ops.element("BezierTet10", 3, *n, 1)
        eids = [1, 2, 3]

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid, v in prescribed:
        for d in range(ndm):
            ops.sp(nid, d + 1, float(v[d]))
    ops.constraints("Penalty", 1.0e18, 1.0e18)
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0

    axes = ("xx", "yy", "xy") if ndm == 2 else (
        "xx", "yy", "zz", "xy", "yz", "xz")
    fem = _MinimalFem(node_ids, np.asarray(coords))
    spec = ResolvedDomainCaptureSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedDomainCaptureRecord(
            category="gauss", name="rec",
            components=tuple(f"strain_{a}" for a in axes),
            dt=None, n_steps=None, element_ids=np.array(eids),
        ),),
    )
    target = OUT / f"capture_{stem}.h5"
    target.unlink(missing_ok=True)
    try:
        with DomainCapture(spec, target, fem, ops=ops,
                           tag_map=identity_tags(spec)) as cap:
            cap.begin_stage("static", kind="static")
            cap.step(t=ops.getTime())
            cap.end_stage()
    finally:
        ops.wipe()
    print("wrote", target)


if __name__ == "__main__":
    for s in ("tri", "tet"):
        transcode(s)
    for s in ("tri", "tet"):
        capture(s)
