"""ADR 0117 P2, link AS2a: general rehydration without a per-row selector.

Oracles, each naming the right answer independently of the code under test:

* parity (INV-4) for a beam frame — transforms of all three types, a beam
  integration, two dampings attached by ``damp=``, an ``Elastic`` section
  and the three beam-column elements with their ``-mass`` / ``-cMass`` /
  ``-iter`` flags. The answer is the deck the source script emits on its
  own FEM, ids shifted back by the closed-form relocation
  ``1_000_000 - source_min``.
* parity for a multi-spec block (two materials on two PGs of one volume).
* rebar carry — two instances of one cage file: each bar cell arrives once,
  as a ``CorotTruss`` on the instance's own ``{inst}.rebar`` material, with
  nodes at the closed-form ``k * 1_000_000 - source_min`` offset.
* INV-5 — the assembly's names are exactly the source's model-kind names,
  prefixed per instance; no time series or pattern name travels.
* refusals raise before registration — args that vary inside a group (an
  orientation-derived transform on a ring) name AS2b (#1542); a damping
  attached by region names the region; ``ops`` stays empty.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh

GRANULE = 1_000_000
E = 200_000.0
MODEL_KINDS = {"uniaxialMaterial", "nDMaterial", "section", "geomTransf",
               "beamIntegration", "damping"}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def frame_fem():
    """A portal frame: PGs ``ColL``, ``ColR`` (z up) and ``Beam`` (along x)."""
    with apeGmsh(model_name="frame", verbose=False) as g:
        geo = g.model.geometry
        p = [geo.add_point(*xyz) for xyz in
             ((0, 0, 0), (0, 0, 3), (4, 0, 3), (4, 0, 0))]
        cl, bm, cr = (geo.add_line(p[0], p[1]), geo.add_line(p[1], p[2]),
                      geo.add_line(p[3], p[2]))
        g.model.sync()
        g.physical.add(1, [cl], name="ColL")
        g.physical.add(1, [cr], name="ColR")
        g.physical.add(1, [bm], name="Beam")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def declare_frame(ops, *, analysis: bool = False) -> None:
    ops.model(ndm=3, ndf=6)
    t_col = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), name="colT")
    t_cor = ops.geomTransf.Corotational(vecxz=(1.0, 0.0, 0.0), name="corT")
    t_bm = ops.geomTransf.PDelta(vecxz=(0.0, 0.0, 1.0), name="beamT")
    sec = ops.section.Elastic(E=E, A=100.0, Iz=1e4, Iy=2e4, G=8e4, J=3e4,
                              name="sec")
    ip = ops.beamIntegration.Lobatto(section=sec, n_ip=5, name="lob")
    uni = ops.damping.uniform(ratio=0.02, freq_lower=1.0, freq_upper=10.0,
                              activate_time=0.5, name="uni")
    stif = ops.damping.sec_stif(beta=0.001, name="stif")
    ops.element.forceBeamColumn(pg="ColL", transf=t_col, integration=ip,
                                mass=0.25, max_iter=10, tol=1e-10, damp=uni)
    ops.element.dispBeamColumn(pg="ColR", transf=t_cor, integration=ip,
                               mass=0.5, c_mass=True, damp=stif)
    ops.element.elasticBeamColumn(pg="Beam", transf=t_bm, A=50.0, E=E,
                                  Iz=3e3, Iy=4e3, G=8e4, J=5e3, mass=0.1)
    if analysis:  # analysis content: must not travel (ADR 0117 D4)
        ts = ops.timeSeries.Linear(name="ramp")
        with ops.pattern.Plain(series=ts, name="push") as pat:
            pat.load(pg="Beam", forces=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def split_block_fem():
    """A 2x2x2 hex8 block split in two PGs, ``Left`` (x<5) and ``Right``."""
    with apeGmsh(model_name="split", verbose=False) as g:
        a = g.model.geometry.add_box(0.0, 0.0, 0.0, 5.0, 10.0, 10.0)
        b = g.model.geometry.add_box(5.0, 0.0, 0.0, 5.0, 10.0, 10.0)
        g.model.boolean.fragment([a], [b])
        g.model.sync()
        vols = sorted(g.model.select(None, dim=3).result().tags())
        g.physical.add(3, [vols[0]], name="Left")
        g.physical.add(3, [vols[1]], name="Right")
        g.physical.add(3, vols, name="Vol")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        return g.mesh.queries.get_fem_data(dim=None)


def declare_split(ops) -> None:
    ops.model(ndm=3, ndf=3)
    soft = ops.nDMaterial.ElasticIsotropic(E=E / 2, nu=0.2, rho=1e-9, name="soft")
    hard = ops.nDMaterial.ElasticIsotropic(E=E, nu=0.3, rho=2e-9, name="hard")
    ops.element.stdBrick(pg="Left", material=soft)
    ops.element.stdBrick(pg="Right", material=hard)


def cage_fem():
    """A tet-meshed column with one conformal bar, ``emit_elements=True``."""
    from apeGmsh._kernel.defs.rebar import Cage

    with apeGmsh(model_name="cage", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                          db=0.0254, material="rebar", name="L1")
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.physical.add_volume("ConcreteVol", name="Conc")
        g.mesh.sizing.set_global_size(0.25)
        g.mesh.generation.generate(dim=3)
        return g.mesh.queries.get_fem_data(dim=3)


def declare_cage(ops) -> None:
    ops.model(ndm=3, ndf=3)
    ops.uniaxialMaterial.ElasticMaterial(E=E, name="rebar")
    conc = ops.nDMaterial.ElasticIsotropic(E=30_000.0, nu=0.2, rho=0.0,
                                           name="conc")
    ops.element.FourNodeTetrahedron(pg="Conc", material=conc)


def ring_fem():
    """A circle of beam elements, PG ``Ring``."""
    with apeGmsh(model_name="ring", verbose=False) as g:
        g.model.geometry.add_circle(0, 0, 0, 5.0, label="ring")
        g.model.sync()
        g.physical.add(1, g.model.select(None, dim=1).result().tags(),
                       name="Ring")
        g.mesh.sizing.set_global_size(2.0)
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def declare_ring(ops) -> None:
    """An orientation-derived transform: one geomTransf per element, so
    the element rows' args vary inside ``Ring``."""
    from apeGmsh.opensees._orientation import Spherical

    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(orientation=Spherical(origin=(0, 0, 0)),
                              name="ringT")
    sec = ops.section.Elastic(E=E, A=1.0, Iz=1.0, Iy=1.0, G=1.0, J=1.0,
                              name="sec")
    ops.element.elasticBeamColumn(pg="Ring", transf=t, A=1.0, E=E, Iz=1.0,
                                  Iy=1.0, G=1.0, J=1.0)
    del sec


def declare_region_damped(ops) -> None:
    declare_split(ops)
    ops.damping.uniform(ratio=0.02, freq_lower=1.0, freq_upper=10.0,
                        on="Left", name="regional")


def write_instance(path: Path, fem, declare, **kw) -> Path:
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    declare(ops, **kw)
    ops.h5(str(path))
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("as2a")
    frame, split = frame_fem(), split_block_fem()
    return {
        "frame": write_instance(d / "frame.h5", frame, declare_frame),
        "frame_analysis": write_instance(
            d / "frame_an.h5", frame, declare_frame, analysis=True),
        "split": write_instance(d / "split.h5", split, declare_split),
        "cage": write_instance(d / "cage.h5", cage_fem(), declare_cage),
        "ring": write_instance(d / "ring.h5", ring_fem(), declare_ring),
        "regional": write_instance(
            d / "regional.h5", split, declare_region_damped),
        "dir": d,
    }


def _deck(ops, path: Path) -> str:
    ops.tcl(str(path), flat=True)
    return path.read_text(encoding="utf-8")


def _source_min(path: Path) -> int:
    from apeGmsh.mesh import FEMData

    src = FEMData.from_h5(str(path))
    ids = [int(np.min(src.nodes.ids))] + [
        int(np.min(g.ids)) for g in src.elements if len(g.ids)]
    return min(ids)


def _bridge(*instances, ndf: int):
    import warnings

    from apeGmsh.assembly import Assembly, AssemblyRankWarning

    asm = Assembly("asm")
    for label, path, kw in instances:
        asm.instance(label, path, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssemblyRankWarning)
        return asm.bridge(ndm=3, ndf=ndf)


# ---------------------------------------------------------------------------
# INV-4 parity: transforms, integrations, dampings, flags; multi-spec block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind, declare, ndf", [
    ("frame", declare_frame, 6),
    ("split", declare_split, 3),
])
def test_one_instance_deck_equals_the_source_deck(files, kind, declare, ndf,
                                                  tmp_path):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees

    src = files[kind]
    ref = apeSees(FEMData.from_h5(str(src)), element_tags="fem")
    declare(ref)
    want = _deck(ref, tmp_path / "ref.tcl")

    got = _deck(_bridge(("inst", src, {}), ndf=ndf), tmp_path / "asm.tcl")
    off = GRANULE - _source_min(src)
    got = re.sub(r"(?<![\w.\-])\d{7,}(?![\w.])",
                 lambda m: str(int(m.group(0)) - off), got)
    assert got == want
    if kind == "frame":
        for line in ("geomTransf Linear", "geomTransf Corotational",
                     "geomTransf PDelta", "beamIntegration Lobatto",
                     "damping Uniform", "damping SecStif", "-iter", "-cMass",
                     "-damp", "-activateTime"):
            assert line in want, line


def test_two_instances_carry_their_own_transforms_and_dampings(files, tmp_path):
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    ops = _bridge(("a", files["frame"], {}),
                  ("b", files["frame"], {"translate": (10.0, 0.0, 0.0)}), ndf=6)
    out = tmp_path / "two.h5"
    ops.h5(str(out))
    names = {(n, k) for n, k, _ in OpenSeesModel.from_h5(str(out)).names()}
    for lab in ("a", "b"):
        assert {(f"{lab}.colT", "geomTransf"), (f"{lab}.lob", "beamIntegration"),
                (f"{lab}.uni", "damping"), (f"{lab}.stif", "damping")} <= names
    deck = _deck(ops, tmp_path / "two.tcl")
    assert deck.count("damping Uniform") == 2
    assert deck.count("geomTransf PDelta") == 2


# ---------------------------------------------------------------------------
# /rebar_elements carry
# ---------------------------------------------------------------------------

def test_rebar_cage_instances_emit_each_bar_once_on_their_own_material(
        files, tmp_path):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    src = FEMData.from_h5(str(files["cage"]))
    (bar,) = src.elements.rebar_elements
    smin = _source_min(files["cage"])

    ops = _bridge(("a", files["cage"], {}),
                  ("b", files["cage"], {"translate": (2.0, 0.0, 0.0)}), ndf=3)
    carried = ops.fem.elements.rebar_elements
    assert [r.material for r in carried] == ["a.rebar", "b.rebar"]
    # The bar PG is prefixed as every PG of the instance is (ADR 0038
    # alternation: a name that already holds one '.' takes '/').
    sep = "/" if "." in bar.pg else "."
    assert [r.pg for r in carried] == [f"a{sep}{bar.pg}", f"b{sep}{bar.pg}"]
    assert [r.area for r in carried] == [bar.area, bar.area]

    out = tmp_path / "cage.h5"
    ops.h5(str(out))
    alias = {(k, t): n for n, k, t in OpenSeesModel.from_h5(str(out)).names()}
    mat_tag = {alias[k]: k[1] for k in alias if k[0] == "uniaxialMaterial"}
    deck = _deck(ops, tmp_path / "cage.tcl")
    trusses = [ln.split() for ln in deck.splitlines()
               if ln.startswith("element CorotTruss")]
    assert len(trusses) == 2 * len(bar.connectivity)   # once each, no replay
    for k, lab in enumerate(("a", "b"), start=1):
        off = k * GRANULE - smin
        want = {(i + off, j + off) for i, j in bar.connectivity}
        got = {(int(t[3]), int(t[4])) for t in trusses
               if int(t[-1]) == mat_tag[f"{lab}.rebar"]}
        assert got == want


# ---------------------------------------------------------------------------
# INV-5 carried-set exclusivity
# ---------------------------------------------------------------------------

def test_assembly_names_are_the_prefixed_model_names_only(files, tmp_path):
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    src_names = OpenSeesModel.from_h5(str(files["frame_analysis"])).names()
    assert {k for _, k, _ in src_names} - MODEL_KINDS, \
        "the source must carry analysis names for the oracle to bite"
    ops = _bridge(("a", files["frame_analysis"], {}),
                  ("b", files["frame_analysis"], {"translate": (10.0, 0, 0)}),
                  ndf=6)
    out = tmp_path / "inv5.h5"
    ops.h5(str(out))
    got = {(n, k) for n, k, _ in OpenSeesModel.from_h5(str(out)).names()}
    want = {(f"{lab}.{n}", k) for lab in ("a", "b")
            for n, k, _ in src_names if k in MODEL_KINDS}
    assert got == want


# ---------------------------------------------------------------------------
# Refusals raise before any registration
# ---------------------------------------------------------------------------

def _empty_after_refusal(path: Path, ndm_ndf: tuple[int, int], tmp_path,
                         match: str) -> None:
    from apeGmsh.assembly import AssemblyError
    from apeGmsh.assembly._rehydrate import rehydrate
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    model = OpenSeesModel.from_h5(str(path))
    ops = apeSees(model.fem)
    ops.model(ndm=ndm_ndf[0], ndf=ndm_ndf[1])
    with pytest.raises(AssemblyError, match=match):
        rehydrate(ops, "p", model)
    deck = _deck(ops, tmp_path / "empty.tcl")
    declared = [ln for ln in deck.splitlines()
                if ln.split()[:1] and ln.split()[0] in
                {"element", "section", "geomTransf", "nDMaterial",
                 "uniaxialMaterial", "beamIntegration", "damping"}]
    assert declared == []


def test_args_varying_inside_a_group_raise_naming_as2b(files, tmp_path):
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    model = OpenSeesModel.from_h5(str(files["ring"]))
    assert len({r.args for r in model.elements()}) == len(model.elements()) > 1
    _empty_after_refusal(files["ring"], (3, 6), tmp_path,
                         r"vary inside a group.*AS2b selector \(#1542\)")


def test_region_attached_damping_raises_before_registration(files, tmp_path):
    _empty_after_refusal(files["regional"], (3, 3), tmp_path,
                         r"damping tags \[1\] are attached by region")
