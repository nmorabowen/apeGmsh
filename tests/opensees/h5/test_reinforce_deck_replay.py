"""OpenSeesModel.build() deck-replay re-emits g.reinforce ties (ADR 0067 P5.1
"A4 full"; archived from opensees 2.29.0, K2b).

From opensees 2.29.0 the archive stores every ``LadrunoEmbeddedRebar`` tie
the build wrote (``/opensees/constraints/embeddedRebar``: its tag and its
arguments, the bond tag already resolved), and ``_replay_into`` writes those
rows. An archive below 2.29.0 stores none; its replay re-derives the ties
from the neutral-zone ``fem`` (``fem.elements.reinforce_ties``), allocating
tie element tags past the max replayed element tag and resolving a ``-bond
<name>`` via the ``/opensees/names`` sidecar (``test_h5_embedded_rebar_archive``
covers that path and the version gate).

Built on a real non-matching mesh (g.reinforce needs gmsh + the inverse map);
emit/replay only — no fork build needed.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees


def _reinforced_ops(*, bond_by_name: bool) -> apeSees:
    """A reinforced apeSees model: a tet-meshed box host + an embedded rebar
    line, the LadrunoBondSlip bond material declared by name when requested."""
    with apeGmsh(model_name="p5_deck_replay", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        p0 = gmsh.model.occ.addPoint(0.5, 0.5, 0.2)
        p1 = gmsh.model.occ.addPoint(0.5, 0.5, 0.8)
        ln = gmsh.model.occ.addLine(p0, p1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        g.physical.add(1, [ln], name="rebar")
        if bond_by_name:
            g.reinforce(host="concrete", bars="rebar", bond="bond1",
                        bar_diameter=0.02)
        else:
            g.reinforce(host="concrete", bars="rebar", perfect=1.0e12,
                        bar_diameter=0.025)
        fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.5e10, nu=0.2, rho=2400.0)
    ops.element.FourNodeTetrahedron(pg="concrete", material=mat)
    if bond_by_name:
        ops.uniaxialMaterial.LadrunoBondSlip(
            tau_max=10.0e6, s1=0.001, s2=0.003, s3=0.01, tau_f=2.0e6,
            alpha=0.4, name="bond1")
    return ops


def _reinforced_fixed_ops(*, bond_by_name: bool = False,
                          rebar_cells: bool = False) -> apeSees:
    """A reinforced model with fixes whose deck round-trips byte for byte
    (K2b, #1795): an embedded bar in a tet-meshed box, its base fixed.

    Every material value is non-integral and the uniaxial material is
    declared before the nD one, so the replay's integral-float and
    declaration-order differences (the oracle's classes A and B) cannot
    arise. ``rebar_cells`` also places the bar's own ``CorotTruss`` cells
    (``emit_elements=True``): the tag plan numbers them after the ties, so
    the archived cells raise the highest element tag above the ties and a
    replay that re-derived the ties would tag them differently.
    """
    from apeGmsh._kernel.defs.rebar import Cage

    name = f"k2b_fixed_{int(bond_by_name)}{int(rebar_cells)}"
    with apeGmsh(model_name=name, verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="Box")
        g.physical.add_volume(g.model.select(dim=3).tags(), name="Concrete")
        g.physical.add_surface(
            g.model.select(dim=2).in_box(
                (-0.01, -0.01, -0.01), (0.51, 0.51, 0.01)).tags(),
            name="Base")
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                          db=0.0254, material="steel", name="L1")
        law = {"bond": "bond1"} if bond_by_name else {"perfect": 1.5e8}
        g.rebar.place(Cage(bars=(bar,)), into="Concrete", coupling="embedded",
                      emit_elements=rebar_cells, name="cage1", **law)
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=None)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    if bond_by_name:
        ops.uniaxialMaterial.LadrunoBondSlip(
            tau_max=10.5, s1=0.001, s2=0.003, s3=0.01, tau_f=2.5,
            alpha=0.4, name="bond1")
    if rebar_cells:
        ops.uniaxialMaterial.ElasticMaterial(
            E=200000.5, eta=0.05, name="steel")
    conc = ops.nDMaterial.ElasticIsotropic(E=30000.5, nu=0.2, rho=2.5e-6)
    ops.element.FourNodeTetrahedron(pg="Concrete", material=conc)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


def _forward_deck(ops: apeSees, target: str, tmp_path) -> str:
    """The forward apeSees deck text (``tcl``/``py`` write to a file)."""
    out = tmp_path / f"forward.{target}"
    getattr(ops, target)(str(out))
    return out.read_text(encoding="utf-8")


def _count_token(text: str, token: str) -> int:
    return sum(
        1 for ln in text.splitlines()
        if token in ln and not ln.strip().startswith("#")
    )


def _element_tags(text: str) -> list[int]:
    """The 2nd token (the tag) of every Tcl ``element <type> <tag> …`` line."""
    tags: list[int] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("element "):
            parts = s.split()
            if len(parts) >= 3:
                tags.append(int(parts[2]))
    return tags


@pytest.mark.parametrize("bond_by_name", [False, True], ids=["perfect", "bond"])
def test_deck_replay_tcl_reemits_reinforce_ties(tmp_path, bond_by_name):
    ops = _reinforced_ops(bond_by_name=bond_by_name)
    forward = _forward_deck(ops, "tcl", tmp_path)
    n_fwd = _count_token(forward, "LadrunoEmbeddedRebar")
    assert n_fwd >= 2, "fixture should produce multiple ties"

    path = str(tmp_path / "model.h5")
    ops.h5(path)

    om = OpenSeesModel.from_h5(path)
    assert len(om._fem.elements.reinforce_ties) == n_fwd
    replayed = om.build("tcl")
    assert isinstance(replayed, str)
    # The deck-replay deck carries the same number of tie elements as the
    # forward apeSees deck (was 0 before the fix).
    assert _count_token(replayed, "LadrunoEmbeddedRebar") == n_fwd


def test_deck_replay_py_target_reemits_ties(tmp_path):
    ops = _reinforced_ops(bond_by_name=False)
    n_fwd = _count_token(_forward_deck(ops, "py", tmp_path),
                         "LadrunoEmbeddedRebar")
    assert n_fwd >= 2

    path = str(tmp_path / "model.h5")
    ops.h5(path)
    replayed = OpenSeesModel.from_h5(path).build("py")
    assert _count_token(replayed, "LadrunoEmbeddedRebar") == n_fwd


def test_deck_replay_tie_tags_dont_collide(tmp_path):
    # The tie element tags share the element namespace with the hosts; every
    # element tag in the replayed deck must stay unique. On this model (no
    # bar cells) the planned tie tags run contiguously above the hosts.
    ops = _reinforced_ops(bond_by_name=False)
    path = str(tmp_path / "model.h5")
    ops.h5(path)
    replayed = OpenSeesModel.from_h5(path).build("tcl")
    tags = _element_tags(replayed)
    assert len(tags) == len(set(tags)), "duplicate element tags in replayed deck"
    # the tie tags are the highest (planned after the host elements)
    tie_tags = [
        int(ln.split()[2]) for ln in replayed.splitlines()
        if ln.strip().startswith("element LadrunoEmbeddedRebar")
    ]
    host_tags = [t for t in tags if t not in tie_tags]
    assert min(tie_tags) > max(host_tags)


def test_deck_replay_h5_target_persists_ties_via_neutral_zone(tmp_path):
    # build("h5") re-emits to a new archive; reinforce ties survive via the
    # NEUTRAL zone (the H5 re-emit path passes no `fem` to _replay_into, so the
    # deck-zone re-emit is correctly skipped — persistence is the neutral zone).
    ops = _reinforced_ops(bond_by_name=False)
    src = str(tmp_path / "src.h5")
    ops.h5(src)
    om = OpenSeesModel.from_h5(src)
    out = str(tmp_path / "rebuilt.h5")
    om.build("h5", out=out)
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    back = read_fem_h5(out)
    assert len(back.elements.reinforce_ties) == \
        len(om._fem.elements.reinforce_ties)
