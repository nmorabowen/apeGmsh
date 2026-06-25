"""Neutral-zone H5 round-trip for node-to-host embedment ties (ADR 0073).

``fem.elements.embed_ties`` (g.embed ``LadrunoEmbeddedNode`` node-to-host
couplings) now persist through ``FEMData.to_h5`` / ``from_h5`` into a dedicated
``/embed_ties`` group (neutral schema 2.22.0). Previously dropped on the
OpenSees deck zone with an ``H5FeatureDeferredWarning`` and no neutral
persistence. Built on a real point-in-solid mesh — no fork build needed (the
ties are resolved at ``get_fem_data``; only *running* the deck needs the fork).
The isotropic sibling of ``test_reinforce_tie_h5_roundtrip.py``."""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


def _embed_in_box(g, *, x=0.4, y=0.4, z=0.4, size=0.5):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    pt = gmsh.model.occ.addPoint(x, y, z)
    g.model.sync()
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box], name="host")
    g.physical.add(0, [pt], name="probe")


def _embedded_fem(**embed_kw):
    with apeGmsh(model_name="embed_h5", verbose=False) as g:
        _embed_in_box(g)
        g.embed(host="host", nodes="probe", **embed_kw)
        return g.mesh.queries.get_fem_data(dim=3)


def _plain_fem():
    with apeGmsh(model_name="embed_h5_plain", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        return g.mesh.queries.get_fem_data(dim=3)


def _roundtrip(fem, tmp_path):
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    p = str(tmp_path / "m.h5")
    fem.to_h5(p)
    return read_fem_h5(p), p


def _eq(a, b):
    assert a.node == b.node
    assert list(a.host_nodes) == list(b.host_nodes)
    assert a.name == b.name and a.enforce == b.enforce
    assert a.bipenalty == b.bipenalty and a.staged == b.staged
    assert a.in_bounds == b.in_bounds
    for f in ("k", "k_alpha", "dtcr", "excess"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert x == pytest.approx(y), f
    x, y = a.weights, b.weights
    assert (x is None) == (y is None)
    if x is not None:
        assert np.allclose(np.asarray(x), np.asarray(y))


def test_penalty_embed_ties_roundtrip(tmp_path):
    fem = _embedded_fem(k=1.0e12)
    src = sorted(fem.elements.embed_ties, key=lambda t: t.node)
    assert len(src) == 1 and src[0].enforce == "penalty"
    back, _ = _roundtrip(fem, tmp_path)
    got = sorted(back.elements.embed_ties, key=lambda t: t.node)
    assert len(got) == len(src)
    for a, b in zip(got, src):
        _eq(a, b)
        assert a.k == pytest.approx(1.0e12)
        assert a.weights is not None and len(a.weights) == len(a.host_nodes)


def test_al_explicit_embed_ties_roundtrip(tmp_path):
    # enforce="al" + bipenalty/dtcr exercise the non-default flags.
    fem = _embedded_fem(k=1.0e12, enforce="al")
    src = fem.elements.embed_ties[0]
    assert src.enforce == "al"
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.embed_ties[0]
    _eq(got, src)
    assert got.enforce == "al"


def test_embed_free_model_omits_group_and_keeps_snapshot(tmp_path):
    import h5py
    fem = _plain_fem()
    assert not fem.elements.embed_ties
    back, p = _roundtrip(fem, tmp_path)
    with h5py.File(p, "r") as f:
        assert "embed_ties" not in f                    # group omitted
    assert back.snapshot_id == fem.snapshot_id


def test_embedded_snapshot_id_stable_on_roundtrip(tmp_path):
    # snapshot_id excludes the tie overlay (consistent with constraints / ties).
    fem = _embedded_fem(k=1.0e12)
    back, _ = _roundtrip(fem, tmp_path)
    assert back.snapshot_id == fem.snapshot_id
    assert back.elements.embed_ties                      # really present


def test_to_h5_no_deferral_warning(tmp_path, recwarn):
    fem = _embedded_fem(k=1.0e12)
    fem.to_h5(str(tmp_path / "m.h5"))
    assert not [w for w in recwarn.list
                if "not persisted" in str(w.message)
                or "deferred" in str(w.message)]


def test_writer_stamps_current_neutral_version():
    from tests.fixtures.schema import NEUTRAL_CURRENT
    assert NEUTRAL_SCHEMA_VERSION == NEUTRAL_CURRENT


def test_reads_prior_minor_file_without_embed_group_within_window(tmp_path):
    # An in-window prior-minor file with the /embed_ties group stripped must
    # still read → empty ties (absence ⇒ no ties).
    import h5py

    from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
    fem = _embedded_fem(k=1.0e12)
    p = str(tmp_path / "old.h5")
    fem.to_h5(p)
    with h5py.File(p, "r+") as f:
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        if "embed_ties" in f:
            del f["embed_ties"]
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    back = read_fem_h5(p)                               # within window → no raise
    assert back.elements.embed_ties == []              # absent group → no ties


# ── encode-side fail-loud hardening ──────────────────────────────────

def _bad_tie(**over):
    from apeGmsh._kernel.records._constraints import EmbedTieRecord
    base = dict(kind="embed", node=9, host_nodes=[1, 2, 3, 4],
                weights=np.full(4, 0.25), k=1.0e12)
    base.update(over)
    return EmbedTieRecord(**base)


def test_encode_rejects_empty_host_nodes():
    from apeGmsh.mesh._femdata_h5_io import _encode_embed_tie
    with pytest.raises(ValueError, match="host_nodes is empty"):
        _encode_embed_tie(_bad_tie(host_nodes=[], weights=None))


def test_encode_rejects_mismatched_weights():
    from apeGmsh.mesh._femdata_h5_io import _encode_embed_tie
    with pytest.raises(ValueError, match="weights length"):
        _encode_embed_tie(_bad_tie(host_nodes=[1, 2, 3, 4],
                                   weights=np.full(3, 1.0 / 3)))
