"""Unit battery for the reinforcement resolver core (`resolve_reinforce`).

Proves the apeGmsh-owned crux of `g.reinforce`: a pre-meshed rebar line PG
inverts into a non-matching solid host, producing one LadrunoEmbeddedRebar
tie per rebar node with correct host coupling, bar axis, and bondScale.
Pure arrays — no Gmsh, no bridge, no fork.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.geometry._inverse_map import InverseMapWarning
from apeGmsh._kernel.resolvers._reinforce import (
    node_directions,
    resolve_reinforce,
    tributary_lengths,
)
from apeGmsh.opensees.element.embedded_rebar import embedded_rebar_args


# unit hex host, gmsh/VTK corner order, node tags 1..8
HEX_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
HEX_XYZ = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)

# 3 rebar nodes along x at mid-height, inside the hex; tags 10,11,12
BAR_IDS = [10, 11, 12]
BAR_XYZ = np.array([
    [0.25, 0.5, 0.5],
    [0.50, 0.5, 0.5],
    [0.75, 0.5, 0.5],
], dtype=float)
BAR_SEG = [(10, 11), (11, 12)]


def _coords(ids, xyz):
    return {int(i): np.asarray(xyz[k], float) for k, i in enumerate(ids)}


class TestTributaryAndDirection:
    def test_tributary_lengths(self) -> None:
        L = tributary_lengths(BAR_IDS, BAR_SEG, _coords(BAR_IDS, BAR_XYZ))
        # segments are each 0.25 long; endpoints get half a segment,
        # the interior node gets half of both.
        assert L[10] == pytest.approx(0.125)
        assert L[12] == pytest.approx(0.125)
        assert L[11] == pytest.approx(0.25)

    def test_directions_along_x(self) -> None:
        d = node_directions(BAR_IDS, BAR_SEG, _coords(BAR_IDS, BAR_XYZ))
        for nid in BAR_IDS:
            assert np.allclose(np.abs(d[nid]), [1.0, 0.0, 0.0], atol=1e-12)

    def test_isolated_node_raises(self) -> None:
        with pytest.raises(ValueError, match="no adjacent bar segment"):
            node_directions([99], [], {99: np.zeros(3)})


class TestResolve:
    def _resolve(self, **over):
        kw = dict(
            bar_node_ids=BAR_IDS,
            bar_node_coords=BAR_XYZ,
            bar_segments=BAR_SEG,
            host_node_ids=[HEX_IDS],
            host_node_coords=[HEX_XYZ],
            host_kinds=["hex8"],
            perfect=1.0e8,
        )
        kw.update(over)
        return resolve_reinforce(**kw)

    def test_one_record_per_rebar_node(self) -> None:
        recs = self._resolve()
        assert [r.rebar_node for r in recs] == [10, 11, 12]

    def test_hex_host_couples_all_eight_nodes(self) -> None:
        rec = self._resolve()[1]            # interior node 11 at the centre
        assert rec.host_nodes == HEX_IDS
        assert rec.weights.shape == (8,)
        assert rec.weights.sum() == pytest.approx(1.0)
        # centre of the unit hex -> all weights 1/8
        assert np.allclose(rec.weights, 0.125, atol=1e-9)
        # weights reconstruct the rebar point
        assert np.allclose(rec.weights @ HEX_XYZ, BAR_XYZ[1], atol=1e-9)

    def test_direction_recorded(self) -> None:
        rec = self._resolve()[1]
        assert np.allclose(np.abs(rec.direction), [1, 0, 0], atol=1e-12)

    def test_perfect_has_no_bondscale(self) -> None:
        for r in self._resolve():
            assert r.perfect == 1.0e8
            assert r.bond is None
            assert r.bond_scale is None

    def test_bond_computes_bondscale(self) -> None:
        # bondScale = pi * d * L_trib ; d=0.02, interior L_trib=0.25
        recs = self._resolve(perfect=None, bond="bond_cebfip", diameter=0.02)
        interior = recs[1]
        assert interior.bond == "bond_cebfip"
        assert interior.perfect is None
        assert interior.bond_scale == pytest.approx(np.pi * 0.02 * 0.25)
        # endpoint L_trib = 0.125
        assert recs[0].bond_scale == pytest.approx(np.pi * 0.02 * 0.125)

    def test_in_bounds_true_for_interior(self) -> None:
        assert all(r.in_bounds for r in self._resolve())

    def test_passthrough_params(self) -> None:
        rec = self._resolve(kt=1.0e8, enforce="al")[0]
        assert rec.kt == 1.0e8 and rec.enforce == "al"


class TestPolicies:
    def test_bond_without_diameter_raises(self) -> None:
        with pytest.raises(ValueError, match="needs `diameter`"):
            resolve_reinforce(
                bar_node_ids=BAR_IDS, bar_node_coords=BAR_XYZ,
                bar_segments=BAR_SEG, host_node_ids=[HEX_IDS],
                host_node_coords=[HEX_XYZ], host_kinds=["hex8"],
                bond="b",
            )

    def test_two_axial_laws_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one axial law"):
            resolve_reinforce(
                bar_node_ids=BAR_IDS, bar_node_coords=BAR_XYZ,
                bar_segments=BAR_SEG, host_node_ids=[HEX_IDS],
                host_node_coords=[HEX_XYZ], host_kinds=["hex8"],
                bond="b", perfect=1e8, diameter=0.02,
            )

    def test_rebar_outside_host_raises(self) -> None:
        outside = np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]])
        with pytest.raises(ValueError, match="lies outside every host"):
            resolve_reinforce(
                bar_node_ids=[20, 21], bar_node_coords=outside,
                bar_segments=[(20, 21)], host_node_ids=[HEX_IDS],
                host_node_coords=[HEX_XYZ], host_kinds=["hex8"],
                perfect=1e8,
            )

    def test_rebar_outside_host_snaps(self) -> None:
        # one node inside, one just above the hex top
        coords = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]])
        with pytest.warns(InverseMapWarning):
            recs = resolve_reinforce(
                bar_node_ids=[20, 21], bar_node_coords=coords,
                bar_segments=[(20, 21)], host_node_ids=[HEX_IDS],
                host_node_coords=[HEX_XYZ], host_kinds=["hex8"],
                perfect=1e8, snap=True,
            )
        assert not recs[1].in_bounds


class TestRecordToEmitArgs:
    """The resolved record must drive the R0 emit grammar (the -shape path)."""

    def test_record_feeds_embedded_rebar_args(self) -> None:
        rec = resolve_reinforce(
            bar_node_ids=BAR_IDS, bar_node_coords=BAR_XYZ,
            bar_segments=BAR_SEG, host_node_ids=[HEX_IDS],
            host_node_coords=[HEX_XYZ], host_kinds=["hex8"],
            bond="bond1", diameter=0.02, kt=1.0e8,
        )[1]
        # simulate the bridge build step: bond name -> tag 5
        args = embedded_rebar_args(
            rebar_node=rec.rebar_node,
            host_nodes=rec.host_nodes,
            shape=list(rec.weights),
            direction=list(rec.direction),
            bond=5,
            bond_scale=rec.bond_scale,
            kt=rec.kt,
            enforce=rec.enforce,
        )
        assert args[0] == 11                      # rebar node
        assert args[1] == 8 and args[2:10] == HEX_IDS   # nHost + host nodes
        assert "-shape" in args and "-bond" in args
        assert args[args.index("-bond") + 1] == 5
