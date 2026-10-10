"""AS5-b (G6): the ``g.constraints.tie`` penalty knobs on ``Assembly.tie``.

v1's ``couple(kind="tie", **options)`` forwarded ``stiffness``,
``stiffness_p``, ``rotational``, ``pressure``, ``control`` and ``outward``
to ``g.constraints.tie``; v2's ``tie`` takes them and passes them to the
same ``TieDef``. Oracles, each independent of the code under test:

* the deck — an explicit ``stiffness`` gives the ``ASDEmbeddedNodeElement``
  lines that ``g.constraints.tie`` gives in one apeGmsh session holding
  the same two stacked blocks, token for token once every node tag is
  replaced by its coordinates (the two routes number the FEM
  differently: the session numbers one mesh, v2 relocates every
  instance);
* pass-through — each knob reaches the ``TieDef`` with the value given;
* refusals — the combinations ``TieDef`` refuses raise at declaration and
  record nothing; a knob of the wrong type raises;
* reload — every knob round-trips through ``h5`` / ``from_h5``; a tie that
  sets none writes the 1.0.0 row (``dofs``, ``enforce``, ``method``,
  ``tolerance`` only); a stored knob at its default, or a foreign key, is
  refused on read.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from tests.assembly.test_two_instances_one_tie import (
    E,
    H,
    NU,
    RHO,
    SIDE,
    TOL,
    block_fem,
    declare_block,
    write_instance,
)

K = 1.0e10


@pytest.fixture(scope="module")
def block(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tie_options")
    return write_instance(d / "block.h5", block_fem(d), declare_block)


def _stack(block):
    from apeGmsh.assembly import Assembly

    return (Assembly("stack")
            .instance("pier_1", block)
            .instance("pier_2", block, translate=(0.0, 0.0, H)))


def _embedded_lines(deck: Path, fem) -> list[tuple[str, ...]]:
    """``ASDEmbeddedNodeElement`` lines with the element tag dropped and
    each node tag replaced by its coordinates."""
    import numpy as np

    xyz = dict(zip((int(i) for i in fem.nodes.ids),
                   np.asarray(fem.nodes.coords, dtype=float)))
    out = []
    for line in deck.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if tok[:2] != ["element", "ASDEmbeddedNodeElement"]:
            continue
        rest = tok[3:]
        k = next(i for i, t in enumerate(rest) if t.startswith("-"))
        nodes = tuple(repr(tuple(round(float(c), 9) for c in xyz[int(t)]))
                      for t in rest[:k])
        out.append(tuple(tok[:2]) + nodes + tuple(rest[k:]))
    return sorted(out)


def _one_session_stack(deck: Path):
    """The two stacked blocks of :func:`_stack` meshed in one session and
    tied with ``g.constraints.tie``; writes ``deck`` and returns the FEM."""
    import gmsh

    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="one_session", verbose=False) as g:
        low = g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, H)
        high = g.model.geometry.add_box(0.0, 0.0, H, SIDE, SIDE, H)
        g.model.sync()
        at_h = set(g.model.select(None, dim=2).in_box(
            (-SIDE, -SIDE, H - TOL), (2 * SIDE, 2 * SIDE, H + TOL),
        ).result().tags())

        def face_at_h(vol: int) -> list[int]:
            return [t for _, t in gmsh.model.getBoundary(
                [(3, vol)], oriented=False) if t in at_h]

        g.physical.add_volume([low, high], name="Vol")
        g.physical.add_surface(face_at_h(low), name="top")
        g.physical.add_surface(face_at_h(high), name="bot")
        g.constraints.tie("top", "bot", dofs=[1, 2, 3], stiffness=K)
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=None)
    ops = apeSees(fem)
    declare_block(ops)
    ops.tcl(str(deck), flat=True)
    return fem


def test_an_explicit_stiffness_emits_the_one_session_tie_deck_lines(
        block, tmp_path):
    v2 = _stack(block).tie("pier_1.top", "pier_2.bot", dofs=[1, 2, 3],
                           stiffness=K).bridge(ndm=3, ndf=3)
    v2.tcl(str(tmp_path / "v2.tcl"), flat=True)
    fem = _one_session_stack(tmp_path / "one.tcl")

    lines_v2 = _embedded_lines(tmp_path / "v2.tcl", v2.fem)
    assert len(lines_v2) == 9                 # the 3x3 interface nodes
    assert {ln[-2:] for ln in lines_v2} == {("-K", repr(K))}
    assert lines_v2 == _embedded_lines(tmp_path / "one.tcl", fem)


def test_each_knob_reaches_the_tie_definition(block):
    from apeGmsh._kernel._coupling_control import CouplingControl

    ctrl = CouplingControl(k=2.0e9, enforce="al")
    asm = (_stack(block)
           .tie("pier_1.top", "pier_2.bot", stiffness=K, stiffness_p=3.0e9,
                pressure=True, name="p")
           .tie("pier_1.bot", "pier_2.top", rotational=True, name="r",
                tolerance=100.0)
           .tie("pier_2.top", "pier_1.bot", enforce="penalty_al",
                control=ctrl, name="al", tolerance=100.0)
           .tie("pier_2.bot", "pier_1.top", enforce="equation",
                method="mortar", outward=(0.0, 0.0, 1.0), name="m"))
    p, r, al, m = (t.definition for t in asm.ties)
    assert (p.stiffness, p.stiffness_p, p.pressure, p.rotational) == (
        K, 3.0e9, True, False)
    assert (r.stiffness, r.rotational) == ("auto", True)
    assert al.control == ctrl and al.enforce == "penalty_al"
    assert m.outward == (0.0, 0.0, 1.0) and m.method == "mortar"
    # The record mirrors its def.
    assert [(t.stiffness, t.control, t.outward) for t in asm.ties] == [
        (K, None, None), ("auto", None, None), ("auto", ctrl, None),
        ("auto", None, (0.0, 0.0, 1.0))]


def test_a_penalty_al_tie_emits_the_control_flags(block, tmp_path):
    from apeGmsh._kernel._coupling_control import CouplingControl

    ops = _stack(block).tie(
        "pier_1.top", "pier_2.bot", enforce="penalty_al",
        control=CouplingControl(k=2.0e9)).bridge(ndm=3, ndf=3)
    ops.tcl(str(tmp_path / "al.tcl"), flat=True)
    lines = [ln for ln in (tmp_path / "al.tcl").read_text(
        encoding="utf-8").splitlines() if ln.startswith("element LadrunoEmbeddedNode")]
    assert len(lines) == 9
    assert all(ln.split()[-2:] == ["-k", repr(2.0e9)] for ln in lines), lines[0]


@pytest.mark.parametrize("kw, match", [
    ({"rotational": True, "pressure": True}, "mutually exclusive"),
    ({"stiffness_p": 1.0e9}, "only meaningful when pressure=True"),
    ({"enforce": "equation", "rotational": True}, "rotational"),
    ({"enforce": "equation", "stiffness_p": 1.0, "pressure": True}, "pressure"),
    ({"control": "al"}, "expected a CouplingControl"),
    ({"outward": (0.0, 0.0, 1.0)}, "only valid with method='mortar'"),
    ({"enforce": "equation", "method": "mortar", "outward": (0, 0, 0)}, "is zero"),
    ({"stiffness": -1.0}, "> 0"),
    ({"stiffness": "AUTO"}, "expected a number"),
    ({"stiffness_p": 0.0, "pressure": True}, "> 0"),
    ({"rotational": 1}, "expected True or False"),
    ({"pressure": "yes"}, "expected True or False"),
])
def test_bad_knobs_raise_at_declaration_and_record_nothing(block, kw, match):
    from apeGmsh.assembly import AssemblyError

    asm = _stack(block)
    with pytest.raises(AssemblyError, match=match):
        asm.tie("pier_1.top", "pier_2.bot", name="t", **kw)
    assert asm.ties == ()
    asm.tie("pier_1.top", "pier_2.bot", name="t")   # the name is still free
    assert [t.name for t in asm.ties] == ["t"]


def test_a_control_needs_the_penalty_al_route(block):
    from apeGmsh._kernel._coupling_control import CouplingControl
    from apeGmsh.assembly import AssemblyError

    with pytest.raises(AssemblyError, match="penalty_al"):
        _stack(block).tie("pier_1.top", "pier_2.bot",
                          control=CouplingControl(k=1.0e9))


@pytest.mark.parametrize("knobs", [
    {"k": "auto", "host": 1_000_001},
    {"k": "auto", "k_alpha": 50.0, "host": 1_000_001},
    {"k": 1.0e9, "bipenalty_wcap": 0.5, "host": 1_000_001},
])
def test_a_tie_control_with_auto_stiffness_or_a_host_is_refused(block, knobs):
    """Review F4 (#1591): a tie's ``control`` gets the RBE refusal. ``host``
    is a raw FEM element id the assembly's relocation does not track, and
    ``k="auto"`` / ``k_alpha`` need it."""
    from apeGmsh._kernel._coupling_control import CouplingControl
    from apeGmsh.assembly import AssemblyError

    asm = _stack(block)
    with pytest.raises(AssemblyError) as info:
        asm.tie("pier_1.top", "pier_2.bot", enforce="penalty_al",
                control=CouplingControl(**knobs))
    msg = str(info.value)
    assert "Assembly requires an explicit k" in msg
    assert "may return later as a label-based host" in msg
    assert asm.ties == ()


def _tie_params(path: Path) -> dict[str, dict]:
    from apeGmsh.assembly._h5 import read_assembly_zone

    return {t.name: json.loads(t.params) for t in read_assembly_zone(path).ties}


def test_every_knob_round_trips_through_the_archive(block, tmp_path):
    from apeGmsh._kernel._coupling_control import CouplingControl
    from apeGmsh.assembly import Assembly

    ctrl = CouplingControl(k=2.0e9, kr=1.0e8, enforce="al", absolute=True)
    asm = (_stack(block)
           .tie("pier_1.top", "pier_2.bot", dofs=[1, 2, 3], name="plain")
           .tie("pier_1.top", "pier_2.bot", stiffness=K, name="k")
           .tie("pier_1.bot", "pier_2.top", rotational=True, tolerance=100.0,
                name="rot")
           .tie("pier_2.top", "pier_1.bot", enforce="penalty_al", control=ctrl,
                tolerance=100.0, name="al")
           .tie("pier_2.bot", "pier_1.top", enforce="equation", method="mortar",
                outward=(0.0, 0.0, -1.0), name="mortar"))
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "knobs.h5"
    asm.h5(out)

    rows = _tie_params(out)
    # A tie that sets no knob writes exactly the 1.0.0 row.
    assert rows["plain"] == {"dofs": [1, 2, 3], "enforce": "penalty",
                             "method": "collocation", "tolerance": 1.0}
    assert rows["k"]["stiffness"] == K and "stiffness_p" not in rows["k"]
    assert rows["rot"]["rotational"] is True
    assert rows["al"]["control"] == {
        "absolute": True, "al_update": None, "bipenalty_dtcr": None,
        "bipenalty_wcap": None, "enforce": "al", "host": None, "k": 2.0e9,
        "k_alpha": None, "kr": 1.0e8}
    assert rows["mortar"]["outward"] == [0.0, 0.0, -1.0]

    back = Assembly.from_h5(out)
    assert back.ties == asm.ties
    for a, b in zip(asm.ties, back.ties):
        assert a.definition == b.definition


def _tampered(src: Path, dst: Path, row: int, params: dict) -> Path:
    dst.write_bytes(src.read_bytes())
    with h5py.File(str(dst), "r+") as f:
        f["assembly/ties/params"][row] = json.dumps(params)
    return dst


@pytest.mark.parametrize("params, match", [
    ({"dofs": None, "enforce": "penalty", "method": "collocation",
      "tolerance": 1.0, "stiffness": "auto"}, r"\['stiffness'\] at the default"),
    ({"dofs": None, "enforce": "penalty", "method": "collocation",
      "tolerance": 1.0, "rotational": False}, r"\['rotational'\] at the default"),
    ({"dofs": None, "enforce": "penalty", "method": "collocation",
      "tolerance": 1.0, "kr": 1.0}, "params carry"),
    ({"dofs": None, "enforce": "penalty", "method": "collocation"},
     "params carry"),
    ({"dofs": None, "enforce": "penalty", "method": "collocation",
      "tolerance": 1.0, "control": {"k": 1.0}}, "expected a CouplingControl"),
    ({"dofs": None, "enforce": "equation", "method": "collocation",
      "tolerance": 1.0, "rotational": True}, "rotational"),
])
def test_a_bad_tie_row_is_refused_on_read(block, tmp_path, params, match):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = _stack(block).tie("pier_1.top", "pier_2.bot", stiffness=K, name="t")
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "t.h5"
    asm.h5(out)
    assert Assembly.from_h5(out).ties == asm.ties
    with pytest.raises(AssemblyError, match=match):
        Assembly.from_h5(_tampered(out, tmp_path / "bad.h5", 0, params))
