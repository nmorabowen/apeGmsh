"""``mass_from_model()`` archives to ``model.h5`` (ADR 0112 amendment 5, #1304).

Before #1304 ``apeSees.h5`` raised ``BridgeError`` under ``mass_from_model()``,
so under ADR 0112 D1's unconditional write every such model failed at the end
of its run. Now the H5 emit skips the redundant mass stream (the masses already
persist in the neutral zone) and marks ``/opensees/bcs@mass_from_model``, and
replay streams the neutral masses back.

Oracle: the deck ``apeSees.tcl()`` emits is the right answer. Its masses must
equal, node by node and dof by dof,

(a) the masses in the H5 neutral zone, read back with ``FEMData.from_h5`` and
    trimmed positionally to the 3-D ``ndf=3`` layout (mx, my, mz) with the
    rotational part zero, independent of the bridge's mapping helper; and
(b) the masses ``OpenSeesModel.from_h5(...).build('tcl')`` emits.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees._internal.lineage import read_stored_lineage
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error


@pytest.fixture(scope="module")
def fem():
    with apeGmsh(model_name="mfm_h5", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="b")
        g.physical.add_volume("b", name="B")
        g.masses.volume("B", density=2400.0)
        g.mesh.sizing.set_global_size(4.0)
        g.mesh.generation.generate(dim=3)
        out = g.mesh.queries.get_fem_data(dim=3)
    assert len(out.nodes.masses) > 0
    return out


def _bridge(fem_, *, extra=None):
    ops = apeSees(fem_)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.mass_from_model()
    if extra is not None:
        extra(ops)
    return ops


def _deck_masses(text: str) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for line in text.splitlines():
        tok = line.split()
        if tok and tok[0] == "mass":
            nid = int(tok[1])
            assert nid not in out, f"node {nid} carries two mass lines"
            out[nid] = tuple(float(v) for v in tok[2:])
    return out


@pytest.fixture(scope="module")
def archived(fem, tmp_path_factory):
    d = tmp_path_factory.mktemp("mfm_h5")
    ops = _bridge(fem)
    h5, tcl = d / "m.h5", d / "m.tcl"
    ops.h5(str(h5))
    ops.tcl(str(tcl))
    return h5, _deck_masses(tcl.read_text(encoding="utf-8"))


def test_h5_succeeds_skips_stream_and_marks(archived):
    h5, deck = archived
    assert len(deck) > 0
    with h5py.File(h5, "r") as f:
        bcs = f["opensees/bcs"]
        assert int(bcs.attrs["mass_from_model"]) == 1
        assert "mass" not in bcs  # the redundant stream is skipped


def test_neutral_zone_masses_equal_deck(archived):
    h5, deck = archived
    neutral = {
        int(m.node_id): tuple(float(v) for v in m.mass)
        for m in FEMData.from_h5(str(h5)).nodes.masses
    }
    assert set(neutral) == set(deck)
    for nid, vec in neutral.items():
        assert all(v == 0.0 for v in vec[3:]), nid  # no rotational mass
        assert deck[nid] == vec[:3], nid


def test_replay_streams_the_same_masses(archived):
    h5, deck = archived
    replay = _deck_masses(OpenSeesModel.from_h5(str(h5)).build("tcl"))
    assert replay == deck


def test_to_h5_round_trip_is_a_fixed_point(archived, tmp_path):
    h5, deck = archived
    again = tmp_path / "again.h5"
    OpenSeesModel.from_h5(str(h5)).to_h5(str(again))
    with h5py.File(h5, "r") as fa, h5py.File(again, "r") as fb:
        assert int(fb["opensees/bcs"].attrs["mass_from_model"]) == 1
        assert read_stored_lineage(fa["meta"]) == read_stored_lineage(fb["meta"])
    assert _deck_masses(OpenSeesModel.from_h5(str(again)).build("tcl")) == deck


def test_overlap_with_explicit_mass_still_raises_on_h5(fem, tmp_path):
    shared = int(next(iter(fem.nodes.masses)).node_id)
    ops = _bridge(
        fem, extra=lambda o: o.mass(nodes=[shared], values=(1.0, 1.0, 1.0)),
    )
    with pytest.raises(BridgeError, match="double-count|one mass channel"):
        ops.h5(str(tmp_path / "x.h5"))


@pytest.mark.parametrize(
    "value",
    [2, 0, np.array([1, 1], dtype=np.int8), "1", 1.0],
    ids=["two", "zero", "array", "string", "float"],
)
def test_corrupt_marker_is_refused(archived, tmp_path, value):
    """Anything but the integer scalar 1 is MalformedH5Error, never a
    TypeError from coercing a non-scalar."""
    h5, _ = archived
    bad = tmp_path / "bad.h5"
    bad.write_bytes(Path(h5).read_bytes())
    with h5py.File(bad, "r+") as f:
        f["opensees/bcs"].attrs["mass_from_model"] = value
    with pytest.raises(MalformedH5Error, match="mass_from_model"):
        OpenSeesModel.from_h5(str(bad))
