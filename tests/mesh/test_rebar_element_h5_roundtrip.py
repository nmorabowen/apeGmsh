"""Neutral-zone H5 round-trip for auto-emitted rebar elements (ADR 0067 P5.2 / B1a.2).

`fem.elements.rebar_elements` (the cage's `place(emit_elements=True)` structural
elements) now persist through `FEMData.to_h5` / `from_h5` into a dedicated
`/rebar_elements` group (neutral schema 2.16.0). Previously dropped with a
deferral warning. Built on a real conformal cage placement (no fork build)."""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Cage
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


def _emit_cage_fem():
    """A conformal cage with one interior bar, emit_elements=True."""
    with apeGmsh(model_name="rebar_elem_h5", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
        bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                          db=0.0254, material="rebar", name="L1")
        g.rebar.place(Cage(bars=(bar,)), into="ConcreteVol",
                      coupling="conformal", emit_elements=True)
        g.mesh.sizing.set_global_size(0.2)
        g.mesh.generation.generate(dim=3)
        return g.mesh.queries.get_fem_data(dim=3)


def _plain_fem():
    with apeGmsh(model_name="rebar_elem_h5_plain", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        return g.mesh.queries.get_fem_data(dim=3)


def _roundtrip(fem, tmp_path):
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    p = str(tmp_path / "m.h5")
    fem.to_h5(p)
    return read_fem_h5(p), p


def test_rebar_elements_roundtrip(tmp_path):
    fem = _emit_cage_fem()
    src = fem.elements.rebar_elements
    assert len(src) == 1
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.rebar_elements
    assert len(got) == len(src)
    a, b = src[0], got[0]
    assert b.pg == a.pg and b.element == a.element
    assert b.material == a.material and b.role == a.role
    assert b.area == pytest.approx(a.area)
    # connectivity survives losslessly (as (i, j) tuples)
    assert tuple(tuple(c) for c in b.connectivity) == \
        tuple(tuple(c) for c in a.connectivity)
    assert len(b.connectivity) >= 1


def test_rebar_free_model_omits_group_and_keeps_snapshot(tmp_path):
    import h5py
    fem = _plain_fem()
    assert not fem.elements.rebar_elements
    back, p = _roundtrip(fem, tmp_path)
    with h5py.File(p, "r") as f:
        assert "rebar_elements" not in f                  # group omitted
    assert back.snapshot_id == fem.snapshot_id


def test_to_h5_no_deferral_warning(tmp_path, recwarn):
    fem = _emit_cage_fem()
    fem.to_h5(str(tmp_path / "m.h5"))
    assert not [w for w in recwarn.list
                if "rebar" in str(w.message).lower()
                and ("not yet persisted" in str(w.message).lower()
                     or "missing" in str(w.message).lower())]


def test_writer_stamps_current_neutral_version():
    from tests.fixtures.schema import NEUTRAL_CURRENT
    assert NEUTRAL_SCHEMA_VERSION == NEUTRAL_CURRENT


# ── encode/decode hardening ──────────────────────────────────────────

def _rec(**over):
    from apeGmsh._kernel.records._rebar import RebarElementRecord
    base = dict(pg="bar.L1", element="truss", material="rebar", area=5.0e-4,
                role="longitudinal", connectivity=((1, 2), (2, 3)))
    base.update(over)
    return RebarElementRecord(**base)


def test_encode_rejects_empty_connectivity():
    from apeGmsh.mesh._femdata_h5_io import _encode_rebar_element
    with pytest.raises(ValueError, match="connectivity is empty"):
        _encode_rebar_element(_rec(connectivity=()))


def test_reads_pre_2_16_0_file_within_window(tmp_path):
    # A prior-minor file with no /rebar_elements group. The reader's
    # two-version window must still read it → empty rebar_elements.
    import h5py

    from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
    fem = _emit_cage_fem()
    p = str(tmp_path / "old.h5")
    fem.to_h5(p)
    with h5py.File(p, "r+") as f:
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        if "rebar_elements" in f:
            del f["rebar_elements"]
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    back = read_fem_h5(p)                               # within window → no raise
    assert back.elements.rebar_elements == []          # absent group → none
