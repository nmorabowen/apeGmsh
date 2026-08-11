"""Compose carries node-to-host embedment ties (ADR 0073 g.embed under
ADR 0038 compose). A module's ``elements.embed_ties`` arrive on the host
offset into the module's reservation window with namespaced names, the
host's own embed ties survive the merge, and the merged model save/reloads
losslessly. Previously BOTH sides were silently dropped: the bundle never
read the embed-tie stream and the merge rebuilt ``ElementComposite``
without ``embed_ties=`` — the isotropic sibling of the reinforce-tie
carry in ``test_compose_reinforce_ties.py``."""
from __future__ import annotations

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.mesh._compose import (
    ComposeReinforceCrossPartError,
    _guard_reinforce_cross_part,
)


def _embedded_module_h5(path, *, k=1.0e12, name="pin"):
    """One box with a single interior point embedded via g.embed; saved
    to H5.  Returns the source EmbedTieRecord for comparison."""
    with apeGmsh(model_name="embed_mod", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pt = gmsh.model.occ.addPoint(0.4, 0.4, 0.4)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.physical.add(0, [pt], name="probe")
        g.embed(host="host", nodes="probe", k=k, name=name)
        fem = g.mesh.queries.get_fem_data(dim=3)
        fem.to_h5(str(path))
        assert len(fem.elements.embed_ties) == 1
        return fem.elements.embed_ties[0]


def _plain_host_h5(path):
    with apeGmsh(model_name="plain_host", verbose=False) as g:
        box = g.model.geometry.add_box(5, 5, 5, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.mesh.queries.get_fem_data(dim=3).to_h5(str(path))


def _compose_and_reload(host, mod, tmp_path, **compose_kw):
    g = apeGmsh.from_h5(str(host))
    g.compose(str(mod), **compose_kw)
    out = tmp_path / "out.h5"
    g.save(str(out))
    return FEMData.from_h5(str(out))


def test_compose_carries_module_embed_tie_with_offset_tags(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    src = _embedded_module_h5(mod)
    _plain_host_h5(host)

    merged = _compose_and_reload(
        host, mod, tmp_path, label="A", translate=(3.0, 0.0, 0.0))

    assert len(merged.elements.embed_ties) == 1
    got = merged.elements.embed_ties[0]
    assert got.name == "A.pin"                        # namespaced
    assert got.k == pytest.approx(1.0e12)
    assert got.enforce == src.enforce
    assert got.staged == src.staged
    # every node tag shifted by ONE constant positive offset vs the source
    offs = {got.node - src.node}
    offs |= {g - s for g, s in zip(got.host_nodes, src.host_nodes)}
    assert len(offs) == 1 and next(iter(offs)) > 0
    # geometry (weights) untouched by the tag rewrite
    assert np.allclose(np.asarray(got.weights), np.asarray(src.weights))
    # the rewritten tags actually exist on the merged model
    node_ids = set(np.asarray(merged.nodes.ids, dtype=np.int64).tolist())
    assert int(got.node) in node_ids
    assert set(int(n) for n in got.host_nodes) <= node_ids


def test_compose_preserves_host_embed_tie_with_plain_module(tmp_path):
    embed_host = tmp_path / "embed_host.h5"
    plain_mod = tmp_path / "plain_mod.h5"
    src = _embedded_module_h5(embed_host)
    _plain_host_h5(plain_mod)

    merged = _compose_and_reload(
        embed_host, plain_mod, tmp_path, label="B",
        translate=(3.0, 0.0, 0.0))

    assert len(merged.elements.embed_ties) == 1
    got = merged.elements.embed_ties[0]
    assert got.name == "pin"                          # host-owned: unprefixed
    assert got.node == src.node
    assert list(got.host_nodes) == list(src.host_nodes)


def test_compose_both_sides_carry_embed_ties(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _embedded_module_h5(a)
    _embedded_module_h5(b)

    merged = _compose_and_reload(
        a, b, tmp_path, label="M", translate=(4.0, 0.0, 0.0))

    names = sorted(t.name for t in merged.elements.embed_ties)
    assert names == ["M.pin", "pin"]


def test_live_session_compose_keeps_embed_tie_on_reextraction(tmp_path):
    """A LIVE session with a resolved embed tie composes a module, and the
    post-compose ``get_fem_data`` (which re-extracts and re-applies every
    bundle through the merge) must keep the tie."""
    plain = tmp_path / "plain.h5"
    _plain_host_h5(plain)

    with apeGmsh(model_name="live_embed", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        pt = gmsh.model.occ.addPoint(0.4, 0.4, 0.4)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        g.physical.add(0, [pt], name="probe")
        g.embed(host="host", nodes="probe", k=1.0e12)
        fem0 = g.mesh.queries.get_fem_data(dim=3)
        assert len(fem0.elements.embed_ties) == 1

        g.compose(str(plain), label="M", translate=(10.0, 0.0, 0.0))
        fem1 = g.mesh.queries.get_fem_data(dim=3)
        assert len(fem1.elements.embed_ties) == 1

        out = tmp_path / "live_out.h5"
        g.save(str(out))
    assert len(FEMData.from_h5(str(out)).elements.embed_ties) == 1


# ── Cross-Part guard on embed ties (unit) ────────────────────────────

class _StubNodes:
    def __init__(self, part_node_map):
        self._part_node_map = part_node_map


class _StubSource:
    def __init__(self, part_node_map):
        self.nodes = _StubNodes(part_node_map)


def _tie(node, host_nodes):
    from apeGmsh._kernel.records._constraints import EmbedTieRecord
    return EmbedTieRecord(
        kind="embed", node=node, host_nodes=list(host_nodes),
        weights=np.full(len(host_nodes), 1.0 / len(host_nodes)),
        k=1.0e12)


def test_cross_part_embed_tie_raises():
    src = _StubSource({"P1": {1, 2, 3, 4}, "P2": {5, 6, 7, 8, 9}})
    # node 9 (P2) embedded into host nodes 1..4 (P1) → spans two Parts
    with pytest.raises(ComposeReinforceCrossPartError, match="spans Parts"):
        _guard_reinforce_cross_part(
            src, [_tie(9, [1, 2, 3, 4])], label="X",
            node_attr="node", kind="embed")


def test_same_part_embed_tie_passes():
    src = _StubSource({"P1": {1, 2, 3, 4}, "P2": {5, 6, 7, 8, 9}})
    # node 5 + hosts 6,7,8,9 all in P2 → fine
    _guard_reinforce_cross_part(
        src, [_tie(5, [6, 7, 8, 9])], label="X",
        node_attr="node", kind="embed")
