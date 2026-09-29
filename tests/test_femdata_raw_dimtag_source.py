"""A FEMData snapshot answers a raw ``(dim, tag)`` selection only from
the Gmsh model it was extracted from.

``fem.nodes.select(target=[(2, tag)])`` and ``fem.elements.select(...)``
look a raw DimTag up in live Gmsh.  They used to ask whatever model was
current, so a snapshot of model A queried while model B was live
silently returned B's nodes and elements.  The snapshot now records its
producing model (name plus a geometry/mesh fingerprint) and refuses the
lookup when the current model is a different one.

Models A and B share the default name ``"ModelName"`` in most cases on
purpose: that is the common case, and the model name alone cannot tell
them apart.
"""
from __future__ import annotations

import pickle
from contextlib import contextmanager

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData


@contextmanager
def _session(name):
    g = apeGmsh(model_name=name, verbose=False)
    g.begin()
    try:
        yield g
    finally:
        g.end()


def _unit_box(g, n):
    """Transfinite unit cube with PGs ``Body`` (volume) and ``Top`` (z=1).

    ``n`` nodes per edge keeps the mesh deterministic: n=4 gives 16
    top-face nodes and 27 hexes.  Returns ``(volume_tag, top_face_tag)``.
    """
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="box")
    g.physical.add_volume("box", name="Body")
    (top,) = [t for _d, t in g.model.queries.boundary("box", dim=2)
              if np.isclose(gmsh.model.getBoundingBox(2, t)[2], 1.0)]
    g.physical.add_surface([top], name="Top")
    g.mesh.structured.set_transfinite_box("box", n=n)
    g.mesh.generation.generate(dim=3)
    (vol,) = [t for _d, t in gmsh.model.getEntities(3)]
    return vol, top


def _ids(selection):
    return sorted(int(i) for i in selection.ids)


def _assert_refused(fem, vol, top):
    for snap in (fem, pickle.loads(pickle.dumps(fem))):
        with pytest.raises(
            RuntimeError, match="extracted from Gmsh model 'ModelName'",
        ):
            snap.nodes.select(target=[(2, top)])
        with pytest.raises(
            RuntimeError, match="extracted from Gmsh model 'ModelName'",
        ):
            snap.elements.select(target=[(3, vol)])


def test_raw_dimtag_resolves_while_the_producing_model_is_current():
    with _session("ModelName") as a:
        vol, top = _unit_box(a, n=4)
        fem = a.mesh.queries.get_fem_data(dim=3)
        top_pg = _ids(fem.nodes.select(pg="Top"))
        body_pg = _ids(fem.elements.select(pg="Body"))

        assert _ids(fem.nodes.select(target=[(2, top)])) == top_pg
        assert _ids(fem.elements.select(target=[(3, vol)])) == body_pg

        # A pickled copy is the same snapshot: same id, same answers.
        clone = pickle.loads(pickle.dumps(fem))
        assert clone.snapshot_id == fem.snapshot_id
        assert _ids(clone.nodes.select(target=[(2, top)])) == top_pg
        assert _ids(clone.elements.select(target=[(3, vol)])) == body_pg

    assert (len(top_pg), len(body_pg)) == (16, 27)


@pytest.mark.parametrize("b_name", ["ModelName", "Other"])
def test_raw_dimtag_refused_after_the_producing_session_closed(b_name):
    with _session("ModelName") as a:
        vol, top = _unit_box(a, n=4)
        fem = a.mesh.queries.get_fem_data(dim=3)
    with _session(b_name) as b:
        _unit_box(b, n=3)       # same entity tags, a different mesh
        _assert_refused(fem, vol, top)


@pytest.mark.parametrize("b_name", ["ModelName", "Other"])
def test_raw_dimtag_refused_while_another_model_is_current(b_name):
    with _session("ModelName") as a:
        vol, top = _unit_box(a, n=4)
        fem = a.mesh.queries.get_fem_data(dim=3)
        with _session(b_name) as b:
            _unit_box(b, n=3)
            _assert_refused(fem, vol, top)


def test_raw_dimtag_refused_after_the_producing_model_is_remeshed():
    # Same session, same name: only the fingerprint can tell.
    with _session("ModelName") as a:
        vol, top = _unit_box(a, n=4)
        fem = a.mesh.queries.get_fem_data(dim=3)
        gmsh.model.mesh.refine()
        with pytest.raises(RuntimeError, match="different geometry or mesh"):
            fem.nodes.select(target=[(2, top)])
        with pytest.raises(RuntimeError, match="different geometry or mesh"):
            fem.elements.select(target=[(3, vol)])


def test_raw_dimtag_refused_for_a_snapshot_loaded_from_h5(tmp_path):
    # The file has no link to a live model, so it is refused even while
    # the model it was written from is still the current one.
    with _session("ModelName") as a:
        vol, top = _unit_box(a, n=4)
        fem = a.mesh.queries.get_fem_data(dim=3)
        fem.to_h5(str(tmp_path / "model.h5"))
        loaded = FEMData.from_h5(str(tmp_path / "model.h5"))
        with pytest.raises(RuntimeError, match="not extracted from a live"):
            loaded.nodes.select(target=[(2, top)])
        with pytest.raises(RuntimeError, match="not extracted from a live"):
            loaded.elements.select(target=[(3, vol)])


def test_pg_select_answers_from_the_snapshot_with_no_session():
    with _session("ModelName") as a:
        _vol, top = _unit_box(a, n=4)
        fem = a.mesh.queries.get_fem_data(dim=3)
        top_pg = _ids(fem.nodes.select(pg="Top"))
        body_pg = _ids(fem.elements.select(pg="Body"))

    assert _ids(fem.nodes.select(pg="Top")) == top_pg
    assert _ids(fem.elements.select(pg="Body")) == body_pg
    with pytest.raises(RuntimeError, match="could not resolve raw DimTag"):
        fem.nodes.select(target=[(2, top)])
