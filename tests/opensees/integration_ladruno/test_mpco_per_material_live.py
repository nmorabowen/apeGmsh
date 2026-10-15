"""Fork lane: per-material damage and Tri31 strain through the spec MPCO emit.

Gap G5 of the DomainCapture parity table (#1767, #1945; slice #1952).
DomainCapture reads two responses per Gauss point through
``eleResponse(e, "material", gp, <resp>)``: ``damage`` on a catalogued host,
and ``strain`` on Tri31, which has no element-level ``strains`` branch. This
file proves the spec MPCO emit (``mpco_ops_args``) carries both, on the
module the bridge drives (``get_ops()``, the fork on the ``live-fork`` lane;
pip ``openseespy`` 3.8.0 ships no MPCO recorder):

* **Damage:** two side-by-side ``Brick`` of a damage material
  (``BRICK_MATERIALS``) pulled past cracking. ``damage_tension`` /
  ``damage_compression`` read through ``Results.from_mpco`` equal
  ``eleResponse(e, "material", gp, "damage")`` at every step and GP, and the
  ``material.damage`` bucket is 2 wide per GP (the card's stop condition).
* **Tri31 strain, mixed mesh:** a bridge-built plate, a ``Tri`` group of
  ``Tri31`` beside a ``Quad`` group of ``FourNodeQuad``, of one material
  (``PLATE_MATERIALS``). The record names only the ``Tri`` group. MPCO has no
  per-token element filter, so ``material.strain`` records the quads too
  (ruling on #1952). The read must not hand them back mislabelled: a read by
  ``pg="Tri"`` returns exactly the Tri FEM ids; an unfiltered read returns
  Tri and Quad FEM ids, each row equal to ``eleResponse`` on the OpenSees tag
  the bridge paired with that id (Tri31: ``material 1 strain``; quad:
  ``strains``), and each id carrying its own class's Gauss rule. FEM ids are
  not the OpenSees tags, so a read that labels by tag fails.

Authority: ``eleResponse`` on the live domain after each step, independent of
the emit and the reader under test. Add a material by adding a row to
``BRICK_MATERIALS`` / ``PLATE_MATERIALS``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter, get_ops
from apeGmsh.results import Results
from apeGmsh.results.spec._emit import mpco_ops_args
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord

from tests.conftest import _stub_model_h5_path

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10


def _gauss(name: str, comps: tuple[str, ...], cls: str,
           ids: Any) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="gauss", name=name, components=comps, dt=None, n_steps=None,
        element_ids=np.asarray(ids), element_class_name=cls,
    )


def _close(want: np.ndarray, got: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * float(np.abs(want).max()),
        err_msg=msg,
    )


def _bucket_widths(path: Path, group: str, cls: str) -> list[int]:
    """``META/NUM_COMPONENTS`` of the one ``cls`` bucket under ``group``."""
    with h5py.File(path, "r") as h:
        grp = h[f"MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/{group}"]
        (key,) = [k for k in grp if f"-{cls}[" in k]
        return grp[key]["META/NUM_COMPONENTS"][()].ravel().tolist()


# ---------------------------------------------------------------------------
# Damage: Brick + a damage material
# ---------------------------------------------------------------------------

E, FT = 30e9, 2.0e6
N_PULL = 10
U_TOP = 3.0 * FT / E                   # 1 m tall: 3x the strain at f_t
BRICK_TAGS = (7, 9)

#: ``ops.nDMaterial`` arguments after the tag. Each must crack in tension
#: at ``FT`` and answer ``material <gp> damage`` with ``[d+, d-]``.
BRICK_MATERIALS: dict[str, tuple[Any, ...]] = {
    "ASDConcrete3D": ("ASDConcrete3D", E, 0.2, "-fc", 30e6, "-ft", FT),
}


@pytest.mark.parametrize("material", sorted(BRICK_MATERIALS))
def test_mpco_damage_matches_eleresponse(tmp_path: Path, material: str) -> None:
    rec = _gauss("dmg", ("damage_tension", "damage_compression"), "Brick",
                 BRICK_TAGS)
    args = mpco_ops_args([rec], output_dir=str(tmp_path), filename="b.mpco")
    assert args == ("mpco", f"{tmp_path}/b.mpco", "-E", "material.damage")
    path = tmp_path / "b.mpco"

    ops = get_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)

    def nid(ix: int, iy: int, iz: int) -> int:
        return 100 * iz + 10 * ix + iy + 1

    for ix in range(3):
        for iy in range(2):
            for iz in range(2):
                ops.node(nid(ix, iy, iz), float(ix), float(iy), float(iz))
            # Bottom on rollers, x held on x=0 and y on y=0: free Poisson.
            ops.fix(nid(ix, iy, 0), int(ix == 0), int(iy == 0), 1)
    ops.nDMaterial(BRICK_MATERIALS[material][0], 1,
                   *BRICK_MATERIALS[material][1:])
    # Side by side, not stacked: both see the same strain, no localisation.
    for ix, tag in enumerate(BRICK_TAGS):
        face = ((ix, 0), (ix + 1, 0), (ix + 1, 1), (ix, 1))
        ops.element("stdBrick", tag, *(nid(x, y, 0) for x, y in face),
                    *(nid(x, y, 1) for x, y in face), 1)
    ops.recorder(*args)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for ix in range(3):
        for iy in range(2):
            ops.sp(nid(ix, iy, 1), 3, U_TOP)
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / N_PULL)
    ops.analysis("Static")
    truth = []                                # (step, elem, gp, [t, c])
    for _ in range(N_PULL):
        assert ops.analyze(1) == 0
        truth.append([[list(ops.eleResponse(e, "material", str(gp), "damage"))
                       for gp in range(1, 9)] for e in BRICK_TAGS])
    ops.remove("recorders")
    ops.wipe()
    want = np.asarray(truth, dtype=float)
    assert want.shape == (N_PULL, 2, 8, 2)
    assert want[-1, :, :, 0].max() > 0.0     # cracked in tension
    assert np.all(want[0, :, :, 0] == 0.0)   # first step is elastic

    # Measured layout (the card's stop condition): Brick bucket, 2 per GP.
    assert _bucket_widths(path, "material.damage", "Brick") == [2] * 8

    with Results.from_mpco(str(path), model_h5=_stub_model_h5_path()) as r:
        gauss = r.stage(r.stages[0].id).elements.gauss
        for k, comp in enumerate(("damage_tension", "damage_compression")):
            slab = gauss.get(component=comp)
            assert [int(e) for e in slab.element_index] == [
                t for t in BRICK_TAGS for _ in range(8)]
            got = np.asarray(slab.values, dtype=float)
            assert got.shape == (N_PULL, 16), (comp, got.shape)
            _close(want[..., k].reshape(N_PULL, 16), got, comp)


# ---------------------------------------------------------------------------
# Tri31 strain, in a bridge-built plate that also holds FourNodeQuads
# ---------------------------------------------------------------------------

#: ``eleResponse(e, "material", gp, "strain")`` order on a plane element:
#: spelled out, not read from RESPONSE_CATALOG (the reader's layout source).
STRAIN_2D = ("strain_xx", "strain_yy", "strain_xy")
N_PLATE = 2

#: ``(nDMaterial bridge kwargs, edge load, yields)`` in N, mm, MPa:
#: J2Plasticity never yields with ``sig0 >= 1e8``
#: (``J2PlasticityNoYieldWarning``). ``yields`` is checked: the second
#: step's strain is twice the first's iff the response stayed linear.
PLATE_MATERIALS: dict[str, tuple[dict[str, float], float, bool]] = {
    "ElasticIsotropic": (dict(E=200e3, nu=0.3), 100.0, False),
    "J2Plasticity": (dict(K=166.7e3, G=76.9e3, sig0=250.0, sigInf=350.0,
                          delta=10.0, H=1.0e3), 260.0, True),
}

#: Parent coordinates of each class's Gauss rule, in ``eleResponse`` order:
#: Tri31 one point at the centroid, FourNodeQuad 2x2 counter-clockwise.
_G = 1.0 / np.sqrt(3.0)
GP_COORDS = {
    "Tri": np.array([[1.0 / 3.0, 1.0 / 3.0]]),
    "Quad": np.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]]),
}


def _plate_fem() -> Any:
    """Two unit squares sharing an edge: ``Tri`` (left, triangles) and
    ``Quad`` (right, recombined), 2x2 divisions each; ``Left`` and
    ``Right`` edges."""
    with apeGmsh(model_name="g5_mixed_plate", verbose=False) as g:
        geo = g.model.geometry
        p = [geo.add_point(x, y, 0.0) for x, y in
             ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1))]
        l_bot, l_mid, l_top, l_left = (
            geo.add_line(p[0], p[1]), geo.add_line(p[1], p[4]),
            geo.add_line(p[4], p[5]), geo.add_line(p[5], p[0]))
        r_bot, r_right, r_top = (
            geo.add_line(p[1], p[2]), geo.add_line(p[2], p[3]),
            geo.add_line(p[3], p[4]))
        tri = geo.add_plane_surface(
            geo.add_curve_loop([l_bot, l_mid, l_top, l_left]))
        quad = geo.add_plane_surface(
            geo.add_curve_loop([r_bot, r_right, r_top, -l_mid]))
        g.model.sync()
        for ln in (l_bot, l_mid, l_top, l_left, r_bot, r_right, r_top):
            gmsh.model.mesh.setTransfiniteCurve(ln, 3)
        for s in (tri, quad):
            g.mesh.structured.set_transfinite_surface(s)
        g.mesh.structured.set_recombine(quad)
        g.physical.add(2, [tri], name="Tri")
        g.physical.add(2, [quad], name="Quad")
        g.physical.add(1, [l_left], name="Left")
        g.physical.add(1, [r_right], name="Right")
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=2)


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


@pytest.fixture(scope="module")
def plate_fem() -> Any:
    return _plate_fem()


def _solve_plate(fem: Any, material: str, tmp: Path,
                 records_for: Any) -> dict[str, Any]:
    """Solve the plate; ``records_for(groups)`` gives the spec records."""
    kwargs, load, yields = PLATE_MATERIALS[material]
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = getattr(ops.nDMaterial, material)(**kwargs)
    ops.element.Tri31(pg="Tri", thickness=1.0, material=mat,
                      plane_type="PlaneStress")
    ops.element.FourNodeQuad(pg="Quad", thickness=1.0, material=mat,
                             plane_type="PlaneStress")
    ops.fix(pg="Left", dofs=(1, 1))
    right = [int(n) for n in fem.nodes.select(pg="Right").ids]
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for k, n in enumerate(right):
            p.load(node=n, forces=(load / len(right), 0.2 * load * (k - 1)))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / N_PLATE)
    ops.analysis.Static()
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    o = emitter.ops
    fem_to_ops = _fem_to_ops(model_h5)
    groups = {pg: [int(e) for e in fem.elements.select(pg=pg).ids]
              for pg in ("Tri", "Quad")}
    # FEM ids are not the OpenSees tags: a read labelled by tag must fail.
    assert any(fem_to_ops[e] != e for e in groups["Tri"] + groups["Quad"])
    records = records_for(groups)
    path = tmp / "plate.mpco"
    o.recorder(*mpco_ops_args(records, output_dir=str(tmp),
                              filename=path.name))

    def resp(e: int) -> np.ndarray:
        tag = fem_to_ops[e]
        if e in groups["Tri"]:
            return np.asarray(o.eleResponse(tag, "material", "1", "strain"),
                              dtype=float).reshape(1, 3)
        return np.asarray(o.eleResponse(tag, "strains"),
                          dtype=float).reshape(4, 3)

    live = []
    for _ in range(N_PLATE):
        assert emitter.analyze(steps=1) == 0
        live.append({e: resp(e) for e in groups["Tri"] + groups["Quad"]})
    tri_tag = fem_to_ops[groups["Tri"][0]]
    assert list(o.eleResponse(tri_tag, "strains")) == []   # why it matters
    o.remove("recorders")
    o.wipe()
    for e, v in live[-1].items():
        assert np.abs(v).max() > 0.0, e
    linear = all(np.allclose(live[1][e], 2.0 * live[0][e], rtol=1e-8)
                 for e in live[0])
    assert linear is not yields, (material, linear)
    return dict(path=path, model_h5=model_h5, groups=groups, live=live,
                records=records)


def _rows(fem: Any, s: dict[str, Any], comp: str, k: int,
          **sel: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{fem_id: (values (T, n_gp), natural coords (n_gp, 2))}``."""
    with Results.from_mpco(str(s["path"]), fem=fem,
                           model_h5=s["model_h5"]) as r:
        slab = r.stage(r.stages[0].id).elements.gauss.get(
            component=comp, **sel)
        eidx = np.asarray(slab.element_index, dtype=int)
        vals = np.asarray(slab.values, dtype=float)
        xi = np.asarray(slab.natural_coords, dtype=float)
    assert vals.shape == (N_PLATE, eidx.size), (comp, vals.shape)
    return {int(e): (vals[:, eidx == e], xi[eidx == e])
            for e in np.unique(eidx)}


def _assert_rows(rows: dict[int, Any], s: dict[str, Any], k: int,
                 comp: str) -> None:
    for e, (vals, xi) in rows.items():
        pg = "Tri" if e in s["groups"]["Tri"] else "Quad"
        # The id's class: its Gauss rule, then its live values.
        np.testing.assert_allclose(xi, GP_COORDS[pg], atol=1e-12,
                                   err_msg=f"{comp} rule on {pg} {e}")
        want = np.stack([step[e][:, k] for step in s["live"]])
        _close(want, vals, f"{comp} on {pg} {e}")


@pytest.mark.parametrize("material", sorted(PLATE_MATERIALS))
@pytest.mark.parametrize("with_quad_record", [False, True],
                         ids=["tri-record", "tri+quad-records"])
def test_mpco_tri31_strain_mixed_mesh(
    plate_fem: Any, tmp_path: Path, material: str, with_quad_record: bool,
) -> None:
    def records_for(groups: dict[str, list[int]]) -> list[Any]:
        recs = [_gauss("tri", STRAIN_2D, "Tri31", groups["Tri"])]
        if with_quad_record:
            recs.append(_gauss("quad", STRAIN_2D, "FourNodeQuad",
                               groups["Quad"]))
        return recs

    s = _solve_plate(plate_fem, material, tmp_path, records_for)
    toks = mpco_ops_args(s["records"])[3:]
    assert toks == (("material.strain", "strains") if with_quad_record
                    else ("material.strain",))
    assert _bucket_widths(s["path"], "material.strain", "Tri31") == [3]
    # No per-token filter in MPCO: the quads are in this group too.
    assert _bucket_widths(s["path"], "material.strain",
                          "FourNodeQuad") == [3] * 4

    tri, quad = s["groups"]["Tri"], s["groups"]["Quad"]
    for k, comp in enumerate(STRAIN_2D):
        # The record's own group: exactly its FEM ids, no quad rows.
        rows = _rows(plate_fem, s, comp, k, pg="Tri")
        assert sorted(rows) == sorted(tri), comp
        _assert_rows(rows, s, k, comp)
        # Unfiltered: the over-recorded quads come back under their own
        # FEM ids and rule, with their own values.
        rows = _rows(plate_fem, s, comp, k)
        assert sorted(rows) == sorted(tri + quad), comp
        _assert_rows(rows, s, k, comp)
