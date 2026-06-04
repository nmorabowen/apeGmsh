"""R3 — g.reinforce explicit ergonomics: -enforce al + -bipenalty -dtcr.

The `-enforce al` (augmented Lagrangian) leg already flows from R2 (the
ReinforceDef.enforce field); this module locks its emit and adds the new
`-bipenalty -dtcr` explicit critical-time-step control. The fork live
round-trip is gated `@pytest.mark.live` — the deployed build (605affeb)
predates LadrunoEmbeddedRebar, so it auto-skips here.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.constraints import ReinforceDef
from apeGmsh._kernel.records._constraints import ReinforceTieRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import emit_reinforce_ties
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# --------------------------------------------------------------------------
# Def-level validation (no mesh, no bridge).
# --------------------------------------------------------------------------
def test_bipenalty_requires_dtcr():
    with pytest.raises(ValueError, match="dtcr"):
        ReinforceDef(master_label="h", slave_label="b",
                     perfect=1.0, bipenalty=True)


def test_dtcr_requires_bipenalty():
    with pytest.raises(ValueError, match="bipenalty"):
        ReinforceDef(master_label="h", slave_label="b",
                     perfect=1.0, dtcr=1e-5)


def test_bipenalty_gated_on_penalty_enforcement():
    with pytest.raises(ValueError, match="penalty"):
        ReinforceDef(master_label="h", slave_label="b",
                     perfect=1.0, enforce="al", bipenalty=True, dtcr=1e-5)


def test_dtcr_must_be_positive():
    with pytest.raises(ValueError, match="dtcr"):
        ReinforceDef(master_label="h", slave_label="b",
                     perfect=1.0, bipenalty=True, dtcr=-1.0)


# --------------------------------------------------------------------------
# Emit-unit (hand-built record).
# --------------------------------------------------------------------------
def _tie(**over):
    base = dict(
        kind="reinforce", rebar_node=9,
        host_nodes=[1, 2, 3, 4], weights=np.full(4, 0.25),
        direction=np.array([0.0, 0.0, 1.0]),
        bond_scale=None, bond=None, perfect=1.0e12,
        enforce="penalty",
    )
    base.update(over)
    return ReinforceTieRecord(**base)


class _Fem:
    def __init__(self, ties):
        self.elements = type("E", (), {"reinforce_ties": ties})()


def test_bipenalty_dtcr_emits_flags():
    em = RecordingEmitter()
    emit_reinforce_ties(em, _Fem([_tie(bipenalty=True, dtcr=2.5e-6)]),
                        TagAllocator(), name_to_tag={})
    args = [c for c in em.calls if c[0] == "embedded_rebar"][0][1]
    assert "-bipenalty" in args
    assert "-dtcr" in args
    assert args[args.index("-dtcr") + 1] == 2.5e-6


def test_al_emits_enforce_al():
    em = RecordingEmitter()
    emit_reinforce_ties(em, _Fem([_tie(enforce="al")]),
                        TagAllocator(), name_to_tag={})
    args = [c for c in em.calls if c[0] == "embedded_rebar"][0][1]
    assert "-enforce" in args
    assert args[args.index("-enforce") + 1] == "al"
    assert "-bipenalty" not in args


# --------------------------------------------------------------------------
# Composite end-to-end on a real mesh.
# --------------------------------------------------------------------------
def _build(g, size=0.5):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    p0 = gmsh.model.occ.addPoint(0.5, 0.5, 0.2)
    p1 = gmsh.model.occ.addPoint(0.5, 0.5, 0.8)
    ln = gmsh.model.occ.addLine(p0, p1)
    g.model.sync()
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box], name="concrete")
    g.physical.add(1, [ln], name="rebar")


def _emit(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    path = os.path.join(tempfile.gettempdir(), "apegmsh_reinforce_r3.tcl")
    ops.tcl(path)
    return open(path).read()


def test_al_end_to_end():
    with apeGmsh(model_name="r3_al", verbose=False) as g:
        _build(g)
        g.reinforce(host="concrete", bars="rebar",
                    perfect=1.0e12, enforce="al")
        fem = g.mesh.queries.get_fem_data(dim=3)
        assert all(t.enforce == "al" for t in fem.elements.reinforce_ties)
        lines = [l for l in _emit(fem).splitlines()
                 if "LadrunoEmbeddedRebar" in l]
        assert lines and all(l.rstrip().endswith("-enforce al") for l in lines)


def test_bipenalty_end_to_end():
    with apeGmsh(model_name="r3_bipen", verbose=False) as g:
        _build(g)
        g.reinforce(host="concrete", bars="rebar",
                    perfect=1.0e12, bipenalty=True, dtcr=1.0e-5)
        fem = g.mesh.queries.get_fem_data(dim=3)
        ties = fem.elements.reinforce_ties
        assert ties and all(t.bipenalty and t.dtcr == 1.0e-5 for t in ties)
        lines = [l for l in _emit(fem).splitlines()
                 if "LadrunoEmbeddedRebar" in l]
        assert lines and all("-bipenalty -dtcr 1e-05" in l for l in lines)


# --------------------------------------------------------------------------
# Live fork round-trip (skipped on a build without LadrunoEmbeddedRebar).
# --------------------------------------------------------------------------
def _fork_has_embedded_rebar() -> bool:
    try:
        import openseespy.opensees as ops
    except Exception:
        return False
    try:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 3)
        for i, c in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], 1):
            ops.node(i, *c)
        ops.node(99, 0.25, 0.25, 0.25)
        ops.element("LadrunoEmbeddedRebar", 1, 99, 4, 1, 2, 3, 4,
                    "-shape", 0.25, 0.25, 0.25, 0.25,
                    "-dir", 0.0, 0.0, 1.0, "-perfect", 1e12)
        ok = 1 in (ops.getEleTags() or [])
        ops.wipe()
        return bool(ok)
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(not _fork_has_embedded_rebar(),
                    reason="OpenSees build lacks LadrunoEmbeddedRebar (ELE 33005)")
def test_perfect_bond_live_roundtrip():
    """Drive the emitted perfect-bond reinforced model into a live
    OpenSees domain on the fork build; the LadrunoEmbeddedRebar couplings
    must actually load (appear in getEleTags). Runs only where the fork
    element is compiled in."""
    import openseespy.opensees as ops_py

    with apeGmsh(model_name="r3_live", verbose=False) as g:
        _build(g, size=0.5)
        g.reinforce(host="concrete", bars="rebar", perfect=1.0e12)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_ties = len(fem.elements.reinforce_ties)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # drives a LiveOpsEmitter through the full deck

    ele_tags = ops_py.getEleTags() or []
    # The host tets plus one LadrunoEmbeddedRebar per rebar node loaded.
    assert n_ties > 0
    assert len(ele_tags) >= n_ties
