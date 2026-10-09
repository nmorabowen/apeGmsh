"""ADR 0117 P2, link AS2a: general rehydration without a per-row selector.

Oracles, each naming the right answer independently of the code under test:

* parity (INV-4) for a beam frame — transforms of all three types, a beam
  integration, two dampings attached by ``damp=``, an ``Elastic`` section
  and the three beam-column elements with their ``-mass`` / ``-cMass`` /
  ``-iter`` flags. The answer is the deck the source script emits on its
  own FEM, ids shifted back by the closed-form relocation
  ``1_000_000 - source_min``.
* parity for a multi-spec block (two materials on two PGs of one volume).
* parity for args that vary inside a group (AS2b, #1542): an
  orientation-derived transform on a ring, one transform per element,
  carried as one spec per row on ``{inst}.Ring#<k>``.
* rebar carry — two instances of one cage file: each bar cell arrives once,
  as a ``CorotTruss`` on the instance's own ``{inst}.rebar`` material, with
  nodes at the closed-form ``k * 1_000_000 - source_min`` offset.
* INV-5 — the assembly's names are exactly the source's model-kind names,
  prefixed per instance; no time series or pattern name travels.
* refusals raise before registration — a damping attached by region
  names the region; ``ops`` stays empty. A synthesized row group that
  would shadow a source group raises.
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


def ring_fem(*extra: str):
    """A circle of beam elements, PG ``Ring`` (plus ``extra`` PGs on it)."""
    with apeGmsh(model_name="ring", verbose=False) as g:
        g.model.geometry.add_circle(0, 0, 0, 5.0, label="ring")
        g.model.sync()
        curves = g.model.select(None, dim=1).result().tags()
        for name in ("Ring",) + extra:
            g.physical.add(1, curves, name=name)
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


def declare_urd(ops) -> None:
    """URD / URDbeta dampings, a deactivate window and ``-cMass``."""
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), name="colT")
    tb = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), name="bmT")
    u = ops.damping.urd(points=[(1.0, 0.02), (5.0, 0.03), (10.0, 0.05)],
                        name="u")
    ub = ops.damping.urd_beta(points=[(1.0, 0.001), (10.0, 0.002)],
                              deactivate_time=3.0, name="ub")
    props = {"A": 1.0, "E": E, "Iz": 2.0, "Iy": 3.0, "G": 4.0, "J": 5.0}
    ops.element.elasticBeamColumn(pg="ColL", transf=t, damp=u, c_mass=True,
                                  **props)
    ops.element.elasticBeamColumn(pg="ColR", transf=t, damp=ub, **props)
    ops.element.elasticBeamColumn(pg="Beam", transf=tb, **props)


def frame2d_fem():
    """The portal frame in the x-y plane, for ``ndm=2``."""
    with apeGmsh(model_name="frame2d", verbose=False) as g:
        geo = g.model.geometry
        p = [geo.add_point(*xyz) for xyz in
             ((0, 0, 0), (0, 3, 0), (4, 3, 0), (4, 0, 0))]
        cl, bm, cr = (geo.add_line(p[0], p[1]), geo.add_line(p[1], p[2]),
                      geo.add_line(p[3], p[2]))
        g.model.sync()
        g.physical.add(1, [cl], name="ColL")
        g.physical.add(1, [cr], name="ColR")
        g.physical.add(1, [bm], name="Beam")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def declare_frame2d(ops) -> None:
    """2-D ``section Elastic`` (with G, alphaY) and 2-D ``elasticBeamColumn``."""
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear(name="t2")
    sec = ops.section.Elastic(E=E, A=100.0, Iz=1e4, G=8e4, alphaY=0.83,
                              name="sec2")
    ip = ops.beamIntegration.Legendre(section=sec, n_ip=3, name="leg")
    ops.element.forceBeamColumn(pg="ColL", transf=t, integration=ip)
    ops.element.dispBeamColumn(pg="ColR", transf=t, integration=ip)
    ops.element.elasticBeamColumn(pg="Beam", transf=t, A=50.0, E=E, Iz=3e3,
                                  mass=0.1, c_mass=True)


def declare_both_attached(ops) -> None:
    """One damping attached by ``damp=`` on ColL AND by region on ColR."""
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), name="colT")
    tb = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), name="bmT")
    uni = ops.damping.uniform(ratio=0.02, freq_lower=1.0, freq_upper=10.0,
                              on="ColR", name="uni")
    props = {"A": 1.0, "E": E, "Iz": 1.0, "Iy": 1.0, "G": 1.0, "J": 1.0}
    ops.element.elasticBeamColumn(pg="ColL", transf=t, damp=uni, **props)
    ops.element.elasticBeamColumn(pg="ColR", transf=t, **props)
    ops.element.elasticBeamColumn(pg="Beam", transf=tb, **props)


def declare_factor(ops) -> None:
    """A damping scaled by a time series (``-factor``), attached by damp=."""
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), name="colT")
    tb = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), name="bmT")
    ts = ops.timeSeries.Linear(name="ramp")
    s = ops.damping.sec_stif(beta=0.01, factor=ts, name="s")
    props = {"A": 1.0, "E": E, "Iz": 2.0, "Iy": 3.0, "G": 4.0, "J": 5.0}
    for pg, tr in (("ColL", t), ("ColR", t), ("Beam", tb)):
        ops.element.elasticBeamColumn(pg=pg, transf=tr, damp=s, **props)


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
        "ring_shadow": write_instance(d / "ring_shadow.h5",
                                      ring_fem("Ring#1"), declare_ring),
        "regional": write_instance(
            d / "regional.h5", split, declare_region_damped),
        "urd": write_instance(d / "urd.h5", frame, declare_urd),
        "frame2d": write_instance(d / "frame2d.h5", frame2d_fem(),
                                  declare_frame2d),
        "both": write_instance(d / "both.h5", frame, declare_both_attached),
        "factor": write_instance(d / "factor.h5", frame, declare_factor),
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


def _bridge(*instances, ndf: int, ndm: int = 3):
    from apeGmsh.assembly import Assembly

    asm = Assembly("asm")
    for label, path, kw in instances:
        asm.instance(label, path, **kw)
    return asm.bridge(ndm=ndm, ndf=ndf)


# ---------------------------------------------------------------------------
# INV-4 parity: transforms, integrations, dampings, flags; multi-spec block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind, declare, ndm, ndf", [
    ("frame", declare_frame, 3, 6),
    ("split", declare_split, 3, 3),
    ("urd", declare_urd, 3, 6),
    ("frame2d", declare_frame2d, 2, 3),
    ("ring", declare_ring, 3, 6),
])
def test_one_instance_deck_equals_the_source_deck(files, kind, declare, ndm,
                                                  ndf, tmp_path):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import apeSees

    src = files[kind]
    ref = apeSees(FEMData.from_h5(str(src)), element_tags="fem")
    declare(ref)
    want = _deck(ref, tmp_path / "ref.tcl")

    got = _deck(_bridge(("inst", src, {}), ndf=ndf, ndm=ndm),
                tmp_path / "asm.tcl")
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
    if kind == "urd":
        for line in ("damping URD ", "damping URDbeta ", "-deactivateTime",
                     "-cMass"):
            assert line in want, line
    if kind == "frame2d":
        assert "section Elastic 1 200000.0 100.0 10000.0 80000.0 0.83" in want
        assert re.search(r"element elasticBeamColumn \d+ \d+ \d+ 50\.0 "
                         r"200000\.0 3000\.0 1 -mass 0\.1 -cMass", want)
    if kind == "ring":          # one transform per element: args vary
        assert want.count("geomTransf Linear") == \
            want.count("element elasticBeamColumn") > 1


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


def test_args_varying_inside_a_group_are_carried_per_row(files, tmp_path):
    """AS2b (#1542): one ``{inst}.Ring#<k>`` group per distinct args row,
    each one element, together exactly ``{inst}.Ring``; the groups and the
    per-row rows survive the assembly archive, which re-bridges as an
    instance to the same deck."""
    from apeGmsh.assembly import Assembly
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    src = OpenSeesModel.from_h5(str(files["ring"]))
    n = len(src.elements())
    assert len({r.args for r in src.elements()}) == n > 1

    asm = Assembly("asm")
    asm.instance("a", files["ring"])
    asm.instance("b", files["ring"], translate=(20.0, 0.0, 0.0))
    ops = asm.bridge(ndm=3, ndf=6)
    phys = ops.fem.elements.physical
    every: set[str] = set()
    for lab in ("a", "b"):
        rows = [f"{lab}.Ring#{k}" for k in range(1, n + 1)]
        assert {nm for nm in phys.names() if nm.startswith(f"{lab}.Ring#")} \
            == set(rows)
        ids = [int(e) for nm in rows for e in phys.element_ids(nm)]
        assert len(ids) == n
        assert sorted(ids) == sorted(int(e) for e in phys.element_ids(f"{lab}.Ring"))
        # #k follows tag order: #1 holds the smallest-tag row's element,
        # at the closed-form k * GRANULE - source_min relocation.
        off = (1 + ("a", "b").index(lab)) * GRANULE - _source_min(files["ring"])
        by_tag = [r.fem_eid + off for r in sorted(src.elements(), key=lambda r: r.tag)]
        assert ids == by_tag
        every.update(rows)

    out = tmp_path / "ring_asm.h5"
    asm.h5(out)
    assert [i.label for i in Assembly.from_h5(out).instances] == ["a", "b"]
    back = OpenSeesModel.from_h5(str(out))
    assert len({r.args for r in back.elements()}) == len(back.elements()) == 2 * n
    assert every <= set(FEMData.from_h5(str(out)).elements.physical.names())

    # The archive as an instance: its rows match a.Ring#k exactly.
    want = _deck(ops, tmp_path / "asm.tcl")
    nested = Assembly("outer")
    nested.instance("x", out)
    got = _deck(nested.bridge(ndm=3, ndf=6), tmp_path / "nested.tcl")
    off = GRANULE - _source_min(out)
    got = re.sub(r"(?<![\w.\-])\d{7,}(?![\w.])",
                 lambda m: str(int(m.group(0)) - off), got)
    assert got == want


def test_a_row_group_that_shadows_a_source_group_raises(files):
    """The ``#<k>`` suffix is checked: a source group named ``Ring#1``
    would be shadowed by the synthesized ``p.Ring#1``."""
    from apeGmsh.assembly import AssemblyError

    with pytest.raises(AssemblyError, match=r"'p\.Ring#1' would shadow"):
        _bridge(("p", files["ring_shadow"], {}), ndf=6)


def test_region_attached_damping_raises_before_registration(files, tmp_path):
    _empty_after_refusal(files["regional"], (3, 3), tmp_path,
                         r"damping tags \[1\] are attached by region")


def test_region_attached_damping_with_an_element_attach_raises(files):
    """The region half of a doubly attached damping is not carried, and the
    model reader cannot see it: the bridge refuses before it exists, so no
    deck and no registration can follow."""
    from apeGmsh.assembly import Assembly, AssemblyError
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    rows = OpenSeesModel.from_h5(str(files["both"])).elements()
    assert any("-damp" in r.args for r in rows), "the element half exists"
    asm = Assembly("x").instance("p", files["both"])
    with pytest.raises(AssemblyError,
                       match=r"damping tags \[1\] are attached by region "
                             r"\(\['/opensees/regions/region_\d+'\]\)"):
        asm.bridge(ndm=3, ndf=6)


def test_time_series_factor_raises_before_registration(files, tmp_path):
    _empty_after_refusal(files["factor"], (3, 6), tmp_path,
                         r"'-factor' references a time series")


def _stub_model(bar_area: float, bar_material: str):
    """The members ``_rebar_rows`` / ``_element_specs`` read, for one bar
    cell (1, 2) and one ``fem_eid=-1`` CorotTruss row on material tag 3."""
    from types import SimpleNamespace

    from apeGmsh._kernel.records._rebar import RebarElementRecord
    from apeGmsh.opensees._internal.typed_records import ElementRecord

    bar = RebarElementRecord(pg="L1", element="truss", material=bar_material,
                             area=bar_area, connectivity=((1, 2),))
    row = ElementRecord(type_token="CorotTruss", tag=7, args=(1, 2, 0.5, 3),
                        connectivity=(1, 2), fem_eid=-1)
    fem = SimpleNamespace(elements=SimpleNamespace(rebar_elements=[bar]))
    return SimpleNamespace(fem=fem, elements=lambda: (row,))


@pytest.mark.parametrize("area, material, skipped", [
    (0.5, "rebar", {7}),        # the carried bar: not re-declared
    (0.25, "rebar", set()),     # another area: a foreign row
    (0.5, "other", set()),      # another material: a foreign row
])
def test_rebar_row_skip_needs_nodes_area_and_material(area, material, skipped):
    from apeGmsh.assembly import AssemblyError
    from apeGmsh.assembly._rehydrate import _element_specs, _rebar_rows

    model = _stub_model(area, material)
    skip = _rebar_rows("p", model, {("uniaxialMaterial", 3): "rebar"})
    assert skip == skipped
    if not skipped:             # a row the stream does not emit must raise
        with pytest.raises(AssemblyError, match="not a carried rebar bar"):
            _element_specs("p", model, skip)


def declare_stage_attached(ops) -> None:
    """Element-attached dampings plus one attached by a stage's region."""
    declare_urd(ops)
    with ops.stage(name="shake") as s:
        s.damping.sec_stif(beta=0.002, on="Beam", name="staged")
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-6, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)


def test_stage_attached_damping_raises(tmp_path):
    from apeGmsh.assembly import Assembly, AssemblyError

    src = write_instance(tmp_path / "staged.h5", frame_fem(),
                         declare_stage_attached)
    asm = Assembly("x").instance("p", src)
    with pytest.raises(AssemblyError,
                       match=r"attached by region \(\['/opensees/stages/"):
        asm.bridge(ndm=3, ndf=6)
