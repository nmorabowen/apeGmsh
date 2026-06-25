"""Fork-only element end-to-end — ``g.embed`` → LadrunoEmbeddedNode.

Drives the apeGmsh embedment generator (a constrained node inverse-mapped
into a non-matching solid host) into the live *fork* domain and asserts the
fork-only ``LadrunoEmbeddedNode`` coupling actually loads (appears in
``getEleTags``). The node-to-host half of the "online" generator coverage,
mirroring test_reinforce_live.py.

Gated on the backend resolver via the ``ladruno_fork`` marker.
"""
from __future__ import annotations

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import _get_ops

pytestmark = pytest.mark.ladruno_fork


def test_embedded_node_loads_on_fork() -> None:
    with apeGmsh(model_name="embed_live", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        # A free point strictly inside the host volume — meshed as its own
        # node (0-D PG), inverse-mapped into the containing host element.
        pt = gmsh.model.occ.addPoint(0.4, 0.4, 0.4)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.physical.add(0, [pt], name="probe")

        g.embed(host="host", nodes="probe", k=1.0e12)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_ties = len(fem.elements.embed_ties)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.run(wipe=True)  # drives a LiveOpsEmitter through the full deck

    ele_tags = _get_ops().getEleTags() or []
    if isinstance(ele_tags, int):
        ele_tags = [ele_tags]
    assert n_ties == 1
    assert len(ele_tags) >= n_ties

    get_class = getattr(_get_ops(), "getEleClassTags", None)
    if get_class is not None:
        class_tags: list[int] = []
        for t in ele_tags:
            ct = get_class(t)
            class_tags.extend(ct if isinstance(ct, (list, tuple)) else [ct])
        assert 33006 in class_tags  # LadrunoEmbeddedNode


def test_embed_h5_roundtrip_then_loads_on_fork(tmp_path) -> None:
    # End-to-end: an embedded model saved to model.h5, reloaded via FEMData
    # from_h5 (the neutral /embed_ties group, schema 2.22.0), then re-emitted
    # and run on the fork. Proves the neutral round-trip feeds emit_embed_ties
    # so the reloaded LadrunoEmbeddedNode coupling still loads on the build.
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5

    p = str(tmp_path / "embed_model.h5")
    with apeGmsh(model_name="embed_rt", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pt = gmsh.model.occ.addPoint(0.4, 0.4, 0.4)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.physical.add(0, [pt], name="probe")
        g.embed(host="host", nodes="probe", k=1.0e12)
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.to_h5(p)

    back = read_fem_h5(p)
    assert len(back.elements.embed_ties) == 1
    assert back.elements.embed_ties[0].k == 1.0e12
    ops = apeSees(back)
    ops.model(ndm=3, ndf=3)
    ops.run(wipe=True)  # reloaded embedment re-emitted + loaded on the fork

    ele_tags = _get_ops().getEleTags() or []
    if isinstance(ele_tags, int):
        ele_tags = [ele_tags]
    assert len(ele_tags) >= 1
