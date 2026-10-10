"""Assembly v2 carries embedded-reinforcement ties + the cross-Part guard
(ADR 0067 P5.1, A2 + A3). Ties round-trip through the neutral H5 (A1), so an
instanced cage keeps its reinforcement with offset tags."""
from __future__ import annotations

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.assembly import Assembly
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees
from apeGmsh.mesh._compose import (
    ComposeReinforceCrossPartError,
    _guard_reinforce_cross_part,
)


def _write_source(fem, path) -> None:
    """A bridgeable source: the FEM plus ``/opensees`` ``model(3, 3)``."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.h5(str(path))


def _reinforced_module_h5(path, *, perfect=1.0e12):
    with apeGmsh(model_name="mod", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        p0 = gmsh.model.occ.addPoint(0.5, 0.5, 0.2)
        p1 = gmsh.model.occ.addPoint(0.5, 0.5, 0.8)
        ln = gmsh.model.occ.addLine(p0, p1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        g.physical.add(1, [ln], name="rebar")
        g.reinforce(host="concrete", bars="rebar",
                    perfect=perfect, bar_diameter=0.025)
        fem = g.mesh.queries.get_fem_data(dim=3)
        _write_source(fem, path)
        return len(fem.elements.reinforce_ties)


def _plain_host_h5(path):
    with apeGmsh(model_name="host", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="host")
        _write_source(g.mesh.queries.get_fem_data(dim=3), path)


def _compose_and_reload(host, mod, tmp_path, *, label, **instance_kw):
    """Instance ``host`` (as ``host.*``) then ``mod`` (as ``{label}.*``),
    bridge, write the archive and read the merged FEM back."""
    asm = Assembly("reinforce")
    asm.instance("host", str(host))
    asm.instance(label, str(mod), **instance_kw)
    asm.bridge(ndm=3, ndf=3)
    out = tmp_path / "out.h5"
    asm.h5(str(out))
    return FEMData.from_h5(str(out))


def test_compose_carries_reinforce_ties(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    n_ties = _reinforced_module_h5(mod)
    assert n_ties >= 2
    _plain_host_h5(host)

    merged = _compose_and_reload(host, mod, tmp_path, label="A",
                                 translate=(2.0, 0.0, 0.0))

    ties = merged.elements.reinforce_ties
    assert len(ties) == n_ties                       # module's ties carried
    for t in ties:
        # weights are a partition of unity, geometry untouched by rewrite
        assert abs(float(np.sum(t.weights)) - 1.0) < 1e-9
        assert len(t.host_nodes) == len(t.weights)
        assert t.perfect == pytest.approx(1.0e12)


def test_compose_offsets_tie_tags(tmp_path):
    mod = tmp_path / "mod.h5"
    host = tmp_path / "host.h5"
    _reinforced_module_h5(mod)
    _plain_host_h5(host)
    # the module's own ties (pre-compose) for comparison
    src_ties = sorted(FEMData.from_h5(str(mod)).elements.reinforce_ties,
                      key=lambda t: t.rebar_node)

    got = sorted(_compose_and_reload(host, mod, tmp_path, label="A")
                 .elements.reinforce_ties,
                 key=lambda t: t.rebar_node)

    assert len(got) == len(src_ties)
    # every composed tie tag is shifted by one constant offset (> 0) vs source
    offs = {g.rebar_node - s.rebar_node for g, s in zip(got, src_ties)}
    assert len(offs) == 1 and next(iter(offs)) > 0
    off = next(iter(offs))
    for gt, st in zip(got, src_ties):
        assert list(gt.host_nodes) == [h + off for h in st.host_nodes]
        assert np.allclose(gt.weights, st.weights)        # geometry preserved
        assert np.allclose(gt.direction, st.direction)


def test_compose_preserves_host_ties_with_plain_module(tmp_path):
    # a reinforced HOST + a plain module → the host's own ties survive merge
    rein_host = tmp_path / "rein_host.h5"
    plain_mod = tmp_path / "plain_mod.h5"
    n = _reinforced_module_h5(rein_host)
    _plain_host_h5(plain_mod)

    merged = _compose_and_reload(rein_host, plain_mod, tmp_path, label="B",
                                 translate=(2.0, 0.0, 0.0))
    assert len(merged.elements.reinforce_ties) == n


# ── A3: cross-Part guard (unit) ──────────────────────────────────────

class _StubNodes:
    def __init__(self, part_node_map):
        self._part_node_map = part_node_map


class _StubSource:
    def __init__(self, part_node_map):
        self.nodes = _StubNodes(part_node_map)


def _tie(rebar_node, host_nodes):
    from apeGmsh._kernel.records._constraints import ReinforceTieRecord
    return ReinforceTieRecord(
        kind="reinforce", rebar_node=rebar_node, host_nodes=list(host_nodes),
        weights=np.full(len(host_nodes), 1.0 / len(host_nodes)),
        direction=np.array([0.0, 0.0, 1.0]), perfect=1.0e12)


def test_cross_part_tie_raises():
    src = _StubSource({"P1": {1, 2, 3, 4}, "P2": {5, 6, 7, 8, 9}})
    # rebar node 9 (P2) tied to host nodes 1..4 (P1) → spans two Parts
    with pytest.raises(ComposeReinforceCrossPartError, match="spans Parts"):
        _guard_reinforce_cross_part(src, [_tie(9, [1, 2, 3, 4])], label="X")


def test_same_part_tie_passes():
    src = _StubSource({"P1": {1, 2, 3, 4}, "P2": {5, 6, 7, 8, 9}})
    # rebar node 5 + hosts 6,7,8,9 all in P2 → fine
    _guard_reinforce_cross_part(src, [_tie(5, [6, 7, 8, 9])], label="X")


def test_guard_noop_without_parts():
    # fewer than two Parts → no cross-Part possible → never raises
    _guard_reinforce_cross_part(_StubSource({}), [_tie(9, [1, 2, 3, 4])],
                                label="X")
    _guard_reinforce_cross_part(_StubSource({"only": {1, 2, 3, 4, 9}}),
                                [_tie(9, [1, 2, 3, 4])], label="X")
