"""Reader-conformance / parity suite (ADR 0042 test-net).

The root cause of the composed-file bug class (#440/#441/#442) was *N parallel
readers each resolving the neutral-zone root their own way*, with no cross-check
forcing them to agree:

  * ``OpenSeesModel.from_h5``  → ``_resolve_fem_root_for_read`` auto-detect (#441)
  * ``FEMData.from_h5``        → explicit ``root=`` argument
  * ``h5_reader.H5Model``      → ``meta_path`` + ``self._neutral`` base (#442)
  * ``NativeReader.fem``       → its own resolution
  * ``ViewerData.from_h5``     → ``h5_reader.open`` auto-detect

These tests force every reader to decode the SAME composed bytes and agree on
geometry + lineage. A reader that silently resolves to a different root than its
siblings trips a node-count / coords / snapshot_id mismatch here, instead of
shipping an empty-snapshot crash to a user.
"""
from __future__ import annotations

import numpy as np
import pytest

# Rehydrating the typed model needs the apeSees bridge.
pytest.importorskip("openseespy.opensees", reason="apeSees bridge (readers)")


def _fem_key(fem):
    """(ids, coords, snapshot_id) sorted by node id — order-independent."""
    ids = np.array([int(x) for x in fem.nodes.ids], dtype=np.int64)
    order = np.argsort(ids)
    coords = np.asarray(fem.nodes.coords, dtype=float)[order]
    return ids[order], coords, str(fem.snapshot_id)


# --- INVARIANT 1: every neutral-zone reader of the SAME composed file agrees -
def test_neutral_zone_readers_agree(composed_results_h5):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results.readers._native import NativeReader

    res = str(composed_results_h5)
    keys = [
        _fem_key(OpenSeesModel.from_h5(res).fem),
        _fem_key(FEMData.from_h5(res, root="/model")),
    ]
    with NativeReader(res) as nr:
        keys.append(_fem_key(nr.fem()))

    ref_ids, ref_coords, ref_snap = keys[0]
    for ids, coords, snap in keys[1:]:
        np.testing.assert_array_equal(ids, ref_ids)
        np.testing.assert_allclose(coords, ref_coords)
        assert snap == ref_snap  # lineage must survive every reader path


# --- INVARIANT 2: composed read at /model == standalone read at / -----------
def test_composed_model_zone_equals_standalone(composed_results_h5, composed_model_h5):
    from apeGmsh.mesh import FEMData

    a = _fem_key(FEMData.from_h5(str(composed_results_h5), root="/model"))
    b = _fem_key(FEMData.from_h5(str(composed_model_h5)))  # root="/"
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert a[2] == b[2]  # same make_demo_results run -> identical lineage


# --- INVARIANT 3: ViewerData decode agrees with OpenSeesModel.fem -----------
def test_viewerdata_matches_model_fem(composed_results_h5):
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.viewers.data._viewer_data import ViewerData

    res = str(composed_results_h5)
    vd = ViewerData.from_h5(res)                 # #442: was 0 nodes (root stub)
    fem = OpenSeesModel.from_h5(res).fem
    assert vd.nodes.ids.size == fem.nodes.ids.size
    assert str(vd.snapshot_id) == str(fem.snapshot_id)


# --- INVARIANT 4: from_h5 auto-detect == explicit meta_path="model/meta" -----
def test_from_h5_autodetect_matches_explicit_root(composed_results_h5):
    from apeGmsh.opensees.emitter import h5_reader
    from apeGmsh.viewers.data._viewer_data import ViewerData

    res = str(composed_results_h5)
    with h5_reader.open(res, meta_path="model/meta") as reader:
        explicit = ViewerData.from_reader(reader)
    auto = ViewerData.from_h5(res)               # locks the #442 auto-detect
    assert auto.nodes.ids.size == explicit.nodes.ids.size
    assert str(auto.snapshot_id) == str(explicit.snapshot_id)
