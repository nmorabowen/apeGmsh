"""Chain M link M3: the ``mpm`` family on the bridge (ADR 0134 D1, D2, D6, D7).

Oracles:

* the particle set a deck carries is the lattice of the declared grid, counted
  and sized in closed form (a 1.0 x 0.4 block at h = 0.1, ppc = 2: 20 x 8 = 160
  particles of volume (h/2)^2 x 1, half-length h/4), its ``MATERIAL_TAG`` the
  body material's allocated ``nDMaterial`` tag, and the fork's own validator
  (``mpm_contracts.validate_particle_set``, vendored) passes on it;
* the replay from ``model.h5`` seeds the same content under a new ``SET_ID``;
* the grid's tag (a fork load-pattern tag) is never a ``pattern`` tag of the deck;
* every refusal names its cause and leaves no deck, no set and no declaration.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees import mpm as mpm_mod
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.types import Pattern
from apeGmsh.opensees._mpm import (
    MPMDeckError,
    MPMSeedError,
    ParticleSetOverwriteRefused,
    StaleParticleSetWarning,
)
from apeGmsh.opensees.emitter import h5_reader

from ._meshes import grid_nodes, make_fem, quads

H = 0.1
LO, HI = (0.5, 0.0), (1.5, 0.4)
N_PARTICLES = 20 * 8          # (1.0 / (H/2)) x (0.4 / (H/2))


def _fem(extra: dict | None = None):
    nodes, ids = grid_nodes((4, 2), LO, HI)
    bodies = {"Soil": quads(ids)}
    if extra:
        bodies.update(extra)
    return make_fem(nodes, bodies)


def _bridge(*, units: str | None = "kN_m_s", patterns: int = 0,
            fem=None) -> apeSees:
    ops = apeSees(fem if fem is not None else _fem(), units=units,
                  _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.ElasticIsotropic(E=1.0e4, nu=0.3, name="fill")  # tag 1
    ops.nDMaterial.ElasticIsotropic(E=2.0e4, nu=0.25, name="soil")  # tag 2
    ops.timeSeries.Linear(name="ramp")
    for _ in range(patterns):
        ops.pattern.Plain(series="ramp")
    return ops


def _declare(ops: apeSees, **body) -> None:
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    ops.mpm.body("Soil", material="soil", density=1.8, ppc=2, **body)
    ops.mpm.wall("ymin", 0.3)
    ops.mpm.gravity(0.0, -9.81)
    ops.mpm.timeSeries("ramp")
    ops.mpm.analyze(10, 1.0e-4)
    ops.fix(pg="Soil", dofs=[1, 1])


def _particles(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        out = {k: f["PARTICLES"][k][...] for k in f["PARTICLES"]}
        out["attrs"] = dict(f.attrs)
        out["prov"] = dict(f["PROVENANCE"].attrs)
        out["grid"] = dict(f["GRID"].attrs)
    return out


# -- the deck and its set ---------------------------------------------------

@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_the_deck_writes_its_particle_set_beside_it(
    tmp_path: Path, dialect: str, contracts: ModuleType,
) -> None:
    ops = _bridge()
    _declare(ops)
    deck = tmp_path / "runs" / f"case1.{dialect}"
    deck.parent.mkdir()
    getattr(ops, dialect)(str(deck))
    pset = tmp_path / "runs" / "case1.particles.h5"
    assert pset.is_file()
    assert contracts.validate_particle_set(str(pset)) == []
    p = _particles(pset)
    assert p["attrs"]["NUM_PARTICLES"] == N_PARTICLES
    np.testing.assert_allclose(p["VOLUME"], (H / 2) ** 2 * 1.0, rtol=1e-15)
    np.testing.assert_allclose(p["GIMP_HALF_LENGTH"], H / 4, rtol=1e-15)
    np.testing.assert_allclose(p["DENSITY"], 1.8)
    assert set(p["MATERIAL_TAG"].tolist()) == {ops.tag_for(ops._names["soil"])} == {2}
    assert p["attrs"]["PLANE_STRAIN_THICKNESS"] == 1.0
    assert (p["attrs"]["UNIT_LENGTH"], p["attrs"]["UNIT_FORCE"],
            p["attrs"]["UNIT_TIME"], p["attrs"]["UNIT_SYSTEM"]) == ("m", "kN", "s", "kN_m_s")
    assert p["prov"]["session_id"] == ops.fem.session_id
    assert p["prov"]["fem_hash"] == ops.fem.snapshot_id
    assert "model_hash" not in p["prov"]
    text = deck.read_text(encoding="utf-8")
    if dialect == "tcl":
        assert ("mpm particles -file [file join [file dirname [info script]] "
                "{case1.particles.h5}]") in text
    else:
        assert ('ops.mpm(\'particles\', \'-file\', os.path.join(os.path.dirname('
                'os.path.abspath(__file__)), "case1.particles.h5"))') in text
        assert "\nimport os\n" in text


def test_mpm_rows_are_archived_and_the_file_line_is_not(tmp_path: Path) -> None:
    ops = _bridge(patterns=1)
    _declare(ops)
    ops.h5(str(tmp_path / "model.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    rows = m._mpm_rows()
    assert [r[0] for r in rows] == ["grid", "body", "wall", "gravity",
                                    "timeSeries", "analyze"]
    assert rows[1] == ("body", "Soil", "-pg", "Soil", "-mat", 2, "-rho", 1.8, "-ppc", 2)
    assert not any("particles" in map(str, r) for r in rows)


@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_the_replay_reseeds_the_same_set_under_a_new_set_id(
    tmp_path: Path, dialect: str,
) -> None:
    ops = _bridge(patterns=2)
    _declare(ops)
    getattr(ops, dialect)(str(tmp_path / f"orig.{dialect}"))
    ops.h5(str(tmp_path / "model.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    text = m.build(dialect, out=str(tmp_path / f"again.{dialect}"))
    assert "again.particles.h5" in text and "orig.particles.h5" not in text
    a = _particles(tmp_path / "orig.particles.h5")
    b = _particles(tmp_path / "again.particles.h5")
    for key in ("ID", "POSITION", "VOLUME", "DENSITY", "GIMP_HALF_LENGTH",
                "BODY_ID", "MATERIAL_TAG"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert a["grid"].keys() == b["grid"].keys()
    assert a["attrs"]["SET_ID"] != b["attrs"]["SET_ID"]
    # The replayed deck is the bridge's, but for the set's name.
    orig = (tmp_path / f"orig.{dialect}").read_text(encoding="utf-8")
    assert [ln for ln in text.splitlines() if "mpm" in ln and "particles" not in ln] \
        == [ln for ln in orig.splitlines() if "mpm" in ln and "particles" not in ln]


def test_the_replay_refuses_without_a_deck_path_and_on_live(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    ops.h5(str(tmp_path / "model.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    with pytest.raises(ValueError, match="pass out="):
        m.build("tcl")
    with pytest.raises(RuntimeError, match="deck-only"):
        m.build("live")


# -- the grid tag is a load-pattern tag ---------------------------------------

#: ``_declare`` registers six ``mpm`` members: the grid (``mpm`` tag 1), then
#: the body, wall, gravity, timeSeries and analyze (``mpm`` tags 2..6).
N_MEMBERS = 6


@pytest.mark.parametrize("before,after", [
    (0, 0), (1, 0), (3, 0), (0, 2), (2, 3), (8, 0), (4, 5)])
def test_the_grid_tag_is_never_a_pattern_tag(
    tmp_path: Path, before: int, after: int,
) -> None:
    """The deck's grid tag, ``ops.tag_for(grid)`` and the archive's
    declaration row are one tag (review finding 1), and it is no pattern's.

    Closed form: one above every ``pattern`` tag and every other member's
    ``mpm`` tag, ``max(before + after, N_MEMBERS) + 1``.
    """
    ops = _bridge(patterns=before)
    _declare(ops)
    for _ in range(after):
        ops.pattern.Plain(series="ramp")
    ops.tcl(str(tmp_path / "d.tcl"))
    text = (tmp_path / "d.tcl").read_text(encoding="utf-8")
    patterns = [int(t) for t in re.findall(r"^pattern Plain (\d+) ", text, re.M)]
    (grid,) = [int(t) for t in re.findall(r"^mpm grid (\d+) ", text, re.M)]
    assert len(patterns) == before + after
    assert grid not in patterns
    assert grid == max(before + after, N_MEMBERS) + 1
    # The public tag is the deck's, and no pattern's.
    (prim,) = [p for p in ops._primitives if isinstance(p, mpm_mod.Grid)]
    assert ops.tag_for(prim) == grid
    assert ops.tag_for(prim) not in {
        ops.tag_for(p) for p in ops._primitives if isinstance(p, Pattern)}
    # The archive's declaration table joins ("mpm", grid) to the grid's
    # declaration; the grid's bare allocator tag is no row's.
    ops.h5(str(tmp_path / "model.h5"))
    with h5_reader.open(str(tmp_path / "model.h5")) as m:
        table = m.declarations()
    assert table is not None
    assert table.for_tag("mpm", grid).key == ops._decls[id(prim)][1][0]
    mpm_rows = sorted(first for kind, first, _n, _d in table.tags
                      if kind == "mpm")
    assert mpm_rows == [*range(2, N_MEMBERS + 1), grid]


# -- refusals -------------------------------------------------------------------

def _no_files(tmp_path: Path) -> None:
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_foreign_file_at_the_set_path_refuses_both(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    foreign = tmp_path / "case1.particles.h5"
    with h5py.File(foreign, "w") as f:
        f.attrs["FORMAT"] = "something-else"
    before = foreign.read_bytes()
    with pytest.raises(ParticleSetOverwriteRefused, match="refusing to overwrite"):
        ops.tcl(str(tmp_path / "case1.tcl"))
    assert not (tmp_path / "case1.tcl").exists()
    assert foreign.read_bytes() == before


def test_an_apegmsh_set_is_replaced_with_its_deck(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    ops.tcl(str(tmp_path / "case1.tcl"))
    first = _particles(tmp_path / "case1.particles.h5")["attrs"]["SET_ID"]
    ops.tcl(str(tmp_path / "case1.tcl"))
    assert _particles(tmp_path / "case1.particles.h5")["attrs"]["SET_ID"] != first


def test_a_staged_model_refuses() -> None:
    """MPM is one unstaged block; the emit's guard refuses a staged model
    (checked on the guard itself: a stage needs a whole chain to build)."""
    from dataclasses import replace

    from apeGmsh.opensees.emitter.tcl import TclEmitter

    ops = _bridge()
    _declare(ops)
    bm = ops.build()
    staged = replace(bm, stage_records=(object(),))
    with pytest.raises(BridgeError, match="deck-only and unstaged") as exc:
        staged._mpm_block(TclEmitter(), staged.mpm_members())
    # Review finding 2: the refusal promises no route v1 cannot emit (the
    # fork's ``parameter ... loadPattern`` staging is a follow-up).
    assert "parameter" not in str(exc.value)
    assert "loadPattern" not in str(exc.value)


def test_per_rank_refuses(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    with pytest.raises(BridgeError, match="per_rank.*serial"):
        ops.tcl(str(tmp_path / "case1.tcl"), per_rank=True)
    _no_files(tmp_path)


def test_a_partitioned_mesh_refuses_and_a_composed_one_emits_flat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apeGmsh.opensees.apesees as apesees_mod

    ops = _bridge()
    _declare(ops)
    monkeypatch.setattr(apesees_mod, "is_partitioned", lambda fem: True)
    with pytest.raises(BridgeError, match="partitioned mesh"):
        ops.tcl(str(tmp_path / "a.tcl"), flat=True)
    _no_files(tmp_path)
    # A composed model (ADR 0038: auto-partitioned one rank per module) emits
    # when the emit is serial, and refuses the per-rank fan-out.
    monkeypatch.setattr(ops.fem, "composed_from", ("module",))
    with pytest.raises(BridgeError, match="flat=True"):
        ops.tcl(str(tmp_path / "b.tcl"))
    _no_files(tmp_path)
    ops.tcl(str(tmp_path / "c.tcl"), flat=True)
    assert (tmp_path / "c.particles.h5").is_file()


def test_the_live_route_refuses_before_the_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apeGmsh.opensees.emitter.live as live

    def _touched(*a, **k):
        raise AssertionError("the live domain was touched")

    monkeypatch.setattr(live.LiveOpsEmitter, "__init__", _touched)
    ops = _bridge()
    _declare(ops)
    with pytest.raises(RuntimeError, match="deck-only"):
        ops.run()
    # The gate every live route calls before its wipe (28 sites).
    with pytest.raises(RuntimeError, match="deck-only"):
        live.refuse_fork_chain_before_wipe(ops.build())


def test_an_unknown_physical_group_or_material_registers_nothing() -> None:
    ops = _bridge()
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    n = len(ops._primitives)
    with pytest.raises(mpm_mod.MPMError, match="no physical group 'Clay'"):
        ops.mpm.body("Clay", material="soil", density=1.8, ppc=2)
    with pytest.raises(KeyError, match="'peat'"):
        ops.mpm.body("Soil", material="peat", density=1.8, ppc=2)
    with pytest.raises(TypeError, match="nDMaterial|NDMaterial"):
        ops.mpm.body("Soil", material="ramp", density=1.8, ppc=2)
    assert len(ops._primitives) == n


def test_a_body_that_seeds_no_particle_refuses(tmp_path: Path) -> None:
    """A physical group too small to hold a sub-cell centre."""
    far, ids = grid_nodes((1, 1), (1.80, 0.80), (1.81, 0.81), first=100)
    nodes, base = grid_nodes((4, 2), LO, HI)
    nodes.update(far)
    fem = make_fem(nodes, {"Soil": quads(base), "Far": quads(ids)})
    ops = _bridge(fem=fem)
    _declare(ops)
    ops.mpm.body("Far", material="fill", density=1.0, ppc=2)
    with pytest.raises(MPMSeedError, match="'Far'.*holds no lattice point"):
        ops.tcl(str(tmp_path / "case1.tcl"))
    _no_files(tmp_path)


def test_two_bodies_sharing_a_point_still_refuse(tmp_path: Path) -> None:
    """The open owner question (a seed point on the interface of two bodies)
    stays as M2 left it: refused."""
    ops = _bridge()
    _declare(ops)
    ops.mpm.body("Soil", material="fill", density=1.0, ppc=2, name="again")
    with pytest.raises(MPMSeedError, match="share 160 lattice points"):
        ops.tcl(str(tmp_path / "case1.tcl"))
    _no_files(tmp_path)


@pytest.mark.parametrize("case,match", [
    ("no-grid", "exactly one ops.mpm.grid"),
    ("no-body", "at least one ops.mpm.body"),
    ("ppc", "share one ppc"),
    ("ndm", "ndm=3"),
    ("face", "faces"),
    ("gravity", "takes 2 components"),
])
def test_model_wide_refusals(tmp_path: Path, case: str, match: str) -> None:
    ops = _bridge()
    if case == "ndm":
        ops.model(ndm=3, ndf=3)
    if case != "no-grid":
        ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    if case != "no-body":
        ops.mpm.body("Soil", material="soil", density=1.8, ppc=2)
    if case == "ppc":
        ops.mpm.body("Soil", material="soil", density=1.8, ppc=3, name="b")
    if case == "face":
        ops.mpm.fix("zmin", "x")
    if case == "gravity":
        ops.mpm.gravity(0.0, 0.0, -9.81)
    with pytest.raises((BridgeError, mpm_mod.MPMError), match=match):
        ops.tcl(str(tmp_path / "case1.tcl"))
    _no_files(tmp_path)


def test_a_second_grid_or_units_refuses_at_the_call() -> None:
    ops = _bridge()
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    ops.mpm.units("m", "kN", "s")
    n = len(ops._primitives)
    with pytest.raises(mpm_mod.MPMError, match="already has an mpm grid"):
        ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    with pytest.raises(mpm_mod.MPMError, match="already declares mpm units"):
        ops.mpm.units("m", "kN", "s")
    assert len(ops._primitives) == n


def test_units_come_from_the_member_or_the_bridge_and_must_agree(
    tmp_path: Path,
) -> None:
    ops = _bridge(units=None)
    _declare(ops)
    with pytest.raises(MPMDeckError, match="needs unit symbols"):
        ops.tcl(str(tmp_path / "a.tcl"))
    _no_files(tmp_path)
    ops.mpm.units("mm", "N", "s")
    ops.tcl(str(tmp_path / "a.tcl"))
    assert _particles(tmp_path / "a.particles.h5")["attrs"]["UNIT_LENGTH"] == "mm"
    assert "mpm units mm N s" in (tmp_path / "a.tcl").read_text(encoding="utf-8")

    ops = _bridge(units="kN_m_s")
    _declare(ops)
    ops.mpm.units("mm", "N", "s")
    with pytest.raises(MPMDeckError, match="differs from the model's units"):
        ops.tcl(str(tmp_path / "b.tcl"))
    assert not (tmp_path / "b.tcl").exists()


@pytest.mark.parametrize("kwargs,match", [
    ({"origin": (0.0,), "h": 0.1, "cells": (2,)}, "2 or 3"),
    ({"origin": (0.0, 0.0), "h": 0.0, "cells": (2, 2)}, "> 0"),
    ({"origin": (0.0, 0.0), "h": 0.1, "cells": (2, 0)}, ">= 1"),
    ({"origin": (0.0, 0.0), "h": 0.1, "cells": (2, 2, 2)}, "2 counts"),
    ({"origin": (0.0, 0.0), "h": 0.1, "cells": (2, 2), "basis": "quad"}, "basis"),
])
def test_grid_arguments_refuse_by_name(kwargs: dict, match: str) -> None:
    with pytest.raises(mpm_mod.MPMError, match=match):
        mpm_mod.Grid(**kwargs)


@pytest.mark.parametrize("make,match", [
    (lambda: mpm_mod.Fix(face="top", components="x"), "face"),
    (lambda: mpm_mod.Fix(face="xmin", components="xx"), "without repeats"),
    (lambda: mpm_mod.Wall(face="ymin", mu=-0.1), ">= 0"),
    (lambda: mpm_mod.Damping(alpha=1.0), r"\[0, 1\)"),
    (lambda: mpm_mod.Flip(alpha=1.5), r"\[0, 1\]"),
    (lambda: mpm_mod.Safety(s=0.0), r"\(0, 1\]"),
    (lambda: mpm_mod.MassCutoff(c=-1.0), ">= 0"),
    (lambda: mpm_mod.Output(path="a", every=2, dt=0.1), "exclusive"),
    (lambda: mpm_mod.Output(path="a", fields=("STATE_x",)), "STATE"),
    (lambda: mpm_mod.Output(path="a{b}"), "brace"),
    (lambda: mpm_mod.Analyze(steps=-1, dt=0.1), ">= 0"),
    (lambda: mpm_mod.Units(length="k m", force="N", time="s"), "one word"),
    # Review finding 5: every token the deck writes bare is one word.
    (lambda: mpm_mod.Units(length="m", force="N", time="s",
                           system="kN m s"), r"Units\.system .*one word"),
    (lambda: mpm_mod.Units(length="m", force="N", time="s",
                           system="SI	2"), r"Units\.system .*one word"),
    (lambda: mpm_mod.Units(length="m", force="k	N", time="s"),
     r"Units\.force .*one word"),
])
def test_member_arguments_refuse_by_name(make, match: str) -> None:
    with pytest.raises(mpm_mod.MPMError, match=match):
        make()


# -- a composed (partitioned) archive replays the flat deck (review finding 3) --

def _composed_fem():
    """The ``_fem`` block as a composed model (ADR 0038: one rank per module):
    the bottom row of elements (1, 3, 5, 7) on rank 1, the top row on rank 2,
    each node owned by the first rank whose element holds it (nodes 1-10, then
    11-15), so the archive's per-rank rows run in the flat deck's node order."""
    from apeGmsh._kernel.record_sets import ComposeSet
    from apeGmsh._kernel.records._compose import ComposeRecord

    nodes, ids = grid_nodes((4, 2), LO, HI)
    elems = quads(ids)
    owned: set[int] = set()
    parts = {}
    for rank, chunk in ((1, range(0, 8, 2)), (2, range(1, 8, 2))):
        mine = sorted({int(n) for e in chunk for n in elems[e][1]} - owned)
        owned.update(mine)
        parts[rank] = {"node_ids": np.array(mine, dtype=np.int64),
                       "element_ids": np.array([e + 1 for e in chunk],
                                               dtype=np.int64)}
    modules = ComposeSet(tuple(
        ComposeRecord(label=f"m{rank}", source_path=f"m{rank}.h5",
                      source_fem_hash=f"m{rank}",
                      source_neutral_schema_version="2.9.0",
                      translate=(0.0, 0.0, 0.0), partition_rank=rank,
                      composed_at="2026-11-15T00:00:00Z")
        for rank in parts))
    return make_fem(nodes, {"Soil": elems}, partitions=parts,
                    composed_from=modules)


def _deck_lines(path: Path, stem: str) -> list[str]:
    """The deck's lines, the stamp comments dropped (the ``# mpm body`` rows
    kept) and the particle set's file name made the stem-free ``<stem>``."""
    text = path.read_text(encoding="utf-8").replace(stem, "<stem>")
    return [ln for ln in text.splitlines()
            if ln.strip() and (not ln.lstrip().startswith("#")
                               or ln.lstrip().startswith("# mpm"))]


@pytest.mark.parametrize("patterns", [0, 2])
def test_a_composed_archive_replays_the_flat_deck_in_its_order(
    tmp_path: Path, patterns: int,
) -> None:
    """The oracle is ``ops.tcl(flat=True)`` for the same model, whole deck,
    line for line: the mpm block follows the fixes, the patterns and every
    other line, where the bridge writes it, not at the global command slot
    (section 7). The patterns carry a load: a partitioned archive keeps no
    empty pattern, an older gap of its replay that is not this one's."""
    import warnings

    from apeGmsh.opensees._internal.build.ownership import is_partitioned

    ops = _bridge(fem=_composed_fem())
    assert is_partitioned(ops.fem) and ops.fem.composed_from
    _declare(ops)
    for _ in range(patterns):
        with ops.pattern.Plain(series="ramp") as p:
            p.load(pg="Soil", forces=(1.0, 0.0))
    bridge = tmp_path / "bridge.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(bridge), flat=True)
        ops.h5(str(tmp_path / "model.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    assert any(r.method == "partition_open" for r in m.program())
    again = tmp_path / "again.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.build("tcl", out=str(again), flat=True)
    want = _deck_lines(bridge, "bridge")
    got = _deck_lines(again, "again")
    assert got == want
    # Closed form: the mpm block is one run of lines that ends the deck, and
    # every fix and pattern sits above it.
    mpm = [i for i, ln in enumerate(got) if ln.startswith(("mpm ", "# mpm "))]
    assert mpm == list(range(len(got) - len(mpm), len(got)))
    assert len(mpm) == 7  # grid, particles, body, wall, gravity, series, analyze
    pats = [i for i, ln in enumerate(got) if ln.startswith("pattern Plain")]
    fixes = [i for i, ln in enumerate(got) if ln.startswith("fix ")]
    assert len(pats) == patterns and len(fixes) == 15
    assert max(pats + fixes) < mpm[0]
    assert _particles(tmp_path / "again.particles.h5")["attrs"]["NUM_PARTICLES"]         == N_PARTICLES


# -- a deck with no mpm members never keeps a stale set (review finding 4) -----

def _plain(fem=None) -> apeSees:
    """The same mesh and materials with no mpm member."""
    ops = _bridge(fem=fem)
    ops.fix(pg="Soil", dofs=[1, 1])
    return ops


@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_a_deck_without_mpm_removes_the_apegmsh_set_of_its_stem(
    tmp_path: Path, dialect: str,
) -> None:
    """ADR 0134 D1 ("never pair a new deck with a stale set"): the viewer
    pairs ``<stem>.particles.h5`` with the deck of that stem, so re-emitting
    a model without MPM over an MPM deck removes the set an apeGmsh emit
    wrote there, silently (no warning)."""
    import warnings

    ops = _bridge()
    _declare(ops)
    deck = tmp_path / f"case1.{dialect}"
    getattr(ops, dialect)(str(deck))
    pset = tmp_path / "case1.particles.h5"
    assert pset.is_file()
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleParticleSetWarning)
        getattr(_plain(), dialect)(str(deck))
    assert not pset.exists()
    assert "mpm" not in deck.read_text(encoding="utf-8")
    # No set there: nothing to decide, nothing warned.
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleParticleSetWarning)
        getattr(_plain(), dialect)(str(deck))


@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_the_replay_without_mpm_removes_the_apegmsh_set_of_its_stem(
    tmp_path: Path, dialect: str,
) -> None:
    ops = _bridge()
    _declare(ops)
    ops.tcl(str(tmp_path / f"case1.{dialect}"))
    pset = tmp_path / "case1.particles.h5"
    assert pset.is_file()
    plain = _plain()
    plain.h5(str(tmp_path / "plain.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "plain.h5"))
    m.build(dialect, out=str(tmp_path / f"case1.{dialect}"))
    assert not pset.exists()
    # out=None writes no deck, so it decides nothing.
    ops.tcl(str(tmp_path / f"case1.{dialect}"))
    m.build(dialect)
    assert pset.is_file()


@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_a_foreign_file_beside_a_deck_without_mpm_stays_and_warns(
    tmp_path: Path, dialect: str,
) -> None:
    foreign = tmp_path / "case1.particles.h5"
    with h5py.File(foreign, "w") as f:
        f.attrs["FORMAT"] = "ladruno-mpm-particles"
        f.attrs["GENERATOR"] = "someone-else"
    before = foreign.read_bytes()
    with pytest.warns(StaleParticleSetWarning, match=re.escape(repr(str(foreign)))):
        getattr(_plain(), dialect)(str(tmp_path / f"case1.{dialect}"))
    assert foreign.read_bytes() == before
    assert (tmp_path / f"case1.{dialect}").is_file()
    plain = _plain()
    plain.h5(str(tmp_path / "plain.h5"))
    with pytest.warns(StaleParticleSetWarning, match="not a particle set an apeGmsh"):
        OpenSeesModel.from_h5(str(tmp_path / "plain.h5")).build(
            dialect, out=str(tmp_path / f"case1.{dialect}"))
    assert foreign.read_bytes() == before


@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_a_failed_emit_without_mpm_keeps_the_set(
    tmp_path: Path, dialect: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Together or not at all: the set is removed only once the deck emitted
    whole, so an emit that raises leaves the old deck and its set as they
    were."""
    import apeGmsh.opensees.apesees as apesees_mod

    ops = _bridge()
    _declare(ops)
    deck = tmp_path / f"case1.{dialect}"
    getattr(ops, dialect)(str(deck))
    old = deck.read_bytes()

    def _boom(*a, **k):
        raise RuntimeError("emit failed")

    monkeypatch.setattr(apesees_mod, "emit_deck_tail", _boom)
    with pytest.raises(RuntimeError, match="emit failed"):
        getattr(_plain(), dialect)(str(deck))
    assert (tmp_path / "case1.particles.h5").is_file()
    assert deck.read_bytes() == old
