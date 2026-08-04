"""Regression — element results over a sparsely-renumbered UNCOMPOSED solid.

The ordinary condition the ADR 0043 slice-1.3 compose-provenance gate
missed: gmsh numbers lower-dimensional elements first, so a 3-D solid with
surface physical groups gets volume ``fem_eid``s starting **above 1**
while the bridge allocates ops element tags from 1. The model is NOT
composed — ``fem.composed_from`` is empty — yet ``fem_eid != ops_tag``
for every element.

Pre-fix, ``Results.from_ladruno(..., model_h5=...)`` skipped the
fem_eid↔ops-tag translator for this model, so:

* ``pg=``-filtered gauss reads silently dropped every element whose
  ``fem_eid`` exceeded the max ops tag (COUNT symptom), and
* the returned ``element_index`` leaked ops tags, attributing every
  Gauss value to a *different* element (SPATIAL-SCRAMBLE symptom).

This test builds a real cantilever in bending and asserts both:

1. **Count** — the gauss slab covers every element of the ``pg=`` filter
   (no silent drops).
2. **Spatial correctness** — near the fixed end of a cantilever under a
   downward tip load, ``stress_xx`` MUST correlate strongly and
   positively with the through-thickness coordinate ``z`` (top fibers in
   tension). A scrambled element_index destroys that correlation, so the
   pure count assertion alone would NOT have caught the bug.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

L, B, H = 2.0, 0.5, 0.5   # cantilever along +x, bending about y
P_TOTAL = 1.0e5           # total tip load, applied in -z


def _nodes_on_plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _tet_centroids(fem) -> dict[int, np.ndarray]:
    """fem_eid → centroid, from the snapshot's tet connectivity."""
    coords_by_id = dict(zip(
        (int(i) for i in np.asarray(fem.nodes.ids)),
        np.asarray(fem.nodes.coords, dtype=np.float64),
    ))
    out: dict[int, np.ndarray] = {}
    for group in fem.elements:
        for eid, conn in zip(group.ids, group.connectivity):
            out[int(eid)] = np.mean(
                [coords_by_id[int(n)] for n in conn], axis=0,
            )
    return out


def test_uncomposed_offset_solid_gauss_reads_are_correct(tmp_path) -> None:
    # ------------------------------------------------------------------
    # Mesh — a solid box with surface PGs, the ordinary gmsh condition
    # that renumbers volume elements away from 1.
    # ------------------------------------------------------------------
    with apeGmsh(model_name="tag_offset_cantilever", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, L, B, H)
        g.model.sync()
        g.physical.add(3, [box], name="Body")
        # Surface PG on the fixed face — boundary triangles get meshed
        # (and numbered) before the tets either way; the PG makes this
        # the documented "solid with surface physical groups" case and
        # gives the fixity a name.
        eps = 1e-6
        fix_faces = [
            t for _, t in g.model.queries.entities_in_bounding_box(
                -eps, -eps, -eps, eps, B + eps, H + eps, dim=2,
            )
        ]
        assert fix_faces, "no x=0 face found"
        g.physical.add(2, fix_faces, name="FixFace")
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)

    # Preconditions — this IS the bug's condition: uncomposed, offset ids.
    assert len(fem.composed_from) == 0, "precondition: uncomposed model"
    tet_ids = np.concatenate([np.asarray(gr.ids) for gr in fem.elements])
    n_tets = int(tet_ids.size)
    assert int(tet_ids.min()) > 1, (
        "precondition lost: gmsh no longer numbers boundary elements "
        "before the tets — this regression test needs fem_eid offset > 0"
    )

    # ------------------------------------------------------------------
    # Bridge + live fork run — cantilever bending, .ladruno + model.h5
    # ------------------------------------------------------------------
    base = _nodes_on_plane(fem, 0, 0.0)
    tip = _nodes_on_plane(fem, 0, L)
    assert base and tip

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    ops.fix(nodes=base, dofs=(1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in tip:
            p.load(node=nid, forces=(0.0, 0.0, -P_TOTAL / len(tip)))

    lad = str(tmp_path / "cantilever.ladruno")
    ops.recorder.Ladruno(file=lad, elem_responses=("stresses",))

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    mh5 = str(tmp_path / "model.h5")
    ops.h5(mh5)

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    emitter.ops.remove("recorders")  # flush the .ladruno

    # Sanity that the OFFSET condition really reached the bridge: the
    # persisted element_meta pairs allocator tags (1-based) with the
    # offset fem_eids.
    import h5py
    with h5py.File(mh5, "r") as f:
        meta = f["/opensees/element_meta"]
        pairs = {
            int(t): int(e)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }
    assert set(pairs.values()) == set(int(i) for i in tet_ids)
    assert all(tag != fem_eid for tag, fem_eid in pairs.items()), (
        "precondition lost: allocator tags coincide with fem_eids"
    )

    # ------------------------------------------------------------------
    # Read back through the public factory — the code under test.
    # ------------------------------------------------------------------
    r = Results.from_ladruno(lad, fem=fem, model_h5=mh5)

    slab = r.elements.gauss.get(component="stress_xx", pg="Body")

    # 1) SPATIAL — the assertion that matters (a pure count check would
    #    NOT catch the scramble). Mechanics: near the fixed end of a
    #    cantilever under a downward tip load, sigma_xx grows with z
    #    (top fibers in tension). Correlate the last-step GP values
    #    against the parent element's centroid z, restricted to the
    #    support half of the span where the bending signal dominates.
    centroids = _tet_centroids(fem)
    vals = np.asarray(slab.values[-1], dtype=np.float64)
    eidx = np.asarray(slab.element_index, dtype=np.int64)
    assert vals.shape == eidx.shape
    assert all(int(e) in centroids for e in eidx), (
        "element_index carries ids the FEMData does not know — ops tags "
        "leaked through"
    )

    cen = np.array([centroids[int(e)] for e in eidx])
    near = cen[:, 0] < 0.5 * L
    assert int(near.sum()) > 20, "too few near-support GPs to correlate"
    corr = float(np.corrcoef(vals[near], cen[near, 2])[0, 1])
    assert corr > 0.8, (
        f"corr(stress_xx, z) near the support is {corr:+.3f}; bending "
        "mechanics requires a strong POSITIVE correlation — the gauss "
        "values are attributed to the wrong elements (scrambled "
        "element_index)"
    )

    # 2) COUNT — the pg= filter must cover EVERY element (pre-fix the
    #    fem_eid filter was compared against ops-tag bucket IDs and
    #    elements past the max ops tag silently vanished).
    covered = set(int(e) for e in slab.element_index)
    assert covered == set(int(i) for i in tet_ids), (
        f"gauss slab covers {len(covered)}/{n_tets} elements of pg='Body' "
        "— elements silently dropped (fem_eid filter met ops-tag bucket)"
    )
