"""Emit-time ``stiffness="auto"`` for the penalty tie family (slice B).

``K = AUTO_STIFFNESS_ALPHA · E_host · L_char``: E from the declared
element spec's material (largest modulus among specs whose PG touches
the record's master nodes), L from the largest pairwise distance among
the master nodes.  ``"auto"`` is the new default on
``tie``/``tied_contact``/``embedded`` — replacing the unit-blind 1e18
C++-parity default that stalls Newton in N/mm/MPa models.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh._kernel.defs.constraints import (
    EmbeddedDef,
    TiedContactDef,
    TieDef,
)
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees._internal.build import (
    AUTO_STIFFNESS_ALPHA,
    BridgeError,
    _emit_one_interpolation,
    make_auto_stiffness_resolver,
)
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter

E_HOST = 200_000.0


def _host_fem() -> FEMData:
    """One quad (nodes 1-4, 2x1 in plan) + a slave node 5 above it."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([500], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {"name": "host_body",
                     "node_ids": np.array([1, 2, 3, 4], dtype=np.int64),
                     "node_coords": coords[:4]},
        }),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=5, n_elems=1, bandwidth=1, types=[quad_info])
    return FEMData(nodes=nodes, elements=elements, info=info,
                   composed_from=ComposeSet(()))


class _Mat:
    E = E_HOST


class _Spec:
    pg = "host_body"
    material = _Mat()


def _auto_rec(**kw) -> InterpolationRecord:
    return InterpolationRecord(
        kind=ConstraintKind.TIE, slave_node=5,
        master_nodes=[1, 2, 3, 4],
        weights=np.array([0.25, 0.25, 0.25, 0.25]),
        dofs=[1, 2, 3], stiffness="auto", **kw,
    )


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


def test_def_defaults_are_auto() -> None:
    assert TieDef(master_label="a", slave_label="b").stiffness == "auto"
    assert TiedContactDef(
        master_label="a", slave_label="b").stiffness == "auto"
    assert EmbeddedDef(master_label="a", slave_label="b").stiffness == "auto"
    # The record default stays numeric — pre-2.27.0 h5 decode parity.
    assert InterpolationRecord(
        kind=ConstraintKind.TIE).stiffness == 1.0e18


def test_def_rejects_a_typo_sentinel() -> None:
    with pytest.raises(ValueError, match="'auto'"):
        TieDef(master_label="a", slave_label="b", stiffness="AUTO")


# ---------------------------------------------------------------------
# Resolver math
# ---------------------------------------------------------------------


def test_resolver_computes_alpha_E_lchar() -> None:
    resolver = make_auto_stiffness_resolver(_host_fem(), [_Spec()])
    k = resolver(_auto_rec())
    # L_char = max pairwise distance among quad corners = diag = sqrt(5)
    assert k == pytest.approx(
        AUTO_STIFFNESS_ALPHA * E_HOST * float(np.sqrt(5.0)))


def test_resolver_fails_loud_without_material() -> None:
    resolver = make_auto_stiffness_resolver(_host_fem(), [])
    with pytest.raises(BridgeError, match="E-carrying material"):
        resolver(_auto_rec())


# ---------------------------------------------------------------------
# Emit integration
# ---------------------------------------------------------------------


def test_emit_resolves_auto_to_a_finite_K() -> None:
    resolver = make_auto_stiffness_resolver(_host_fem(), [_Spec()])
    e = RecordingEmitter()
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # no 1e18 warning on auto
        _emit_one_interpolation(
            e, _auto_rec(), TagAllocator(), stiffness_resolver=resolver)
    calls = [c for c in e.calls if c[0] == "embeddedNode"]
    assert len(calls) == 1
    k = calls[0][2]["stiffness"]
    assert isinstance(k, float) and np.isfinite(k)
    assert k == pytest.approx(
        AUTO_STIFFNESS_ALPHA * E_HOST * float(np.sqrt(5.0)))


def test_emit_without_resolver_fails_loud() -> None:
    e = RecordingEmitter()
    with pytest.raises(BridgeError, match="auto"):
        _emit_one_interpolation(e, _auto_rec(), TagAllocator())


def test_equation_route_ignores_auto() -> None:
    rec = _auto_rec(enforce="equation")
    e = RecordingEmitter()
    _emit_one_interpolation(e, rec, TagAllocator())   # no resolver needed
    assert any(c[0] == "equationConstraint" for c in e.calls)


# ---------------------------------------------------------------------
# h5 round-trip (schema 2.27.0)
# ---------------------------------------------------------------------


def test_auto_round_trips_through_h5(tmp_path) -> None:
    fem = _host_fem().with_constraint(_auto_rec())
    numeric = InterpolationRecord(
        kind=ConstraintKind.TIE, slave_node=5, master_nodes=[1, 2, 3],
        weights=np.array([0.4, 0.3, 0.3]), dofs=[1, 2, 3],
        stiffness=1.0e11,
    )
    fem = fem.with_constraint(numeric)
    out = tmp_path / "m.h5"
    fem.to_h5(str(out))
    back = FEMData.from_h5(str(out))
    recs = list(back.elements.constraints.interpolations())
    stiffs = sorted(str(r.stiffness) for r in recs)
    assert stiffs == sorted(["auto", "100000000000.0"])


def test_tied_contact_slave_records_auto_round_trip(tmp_path) -> None:
    sc = SurfaceCouplingRecord(
        kind=ConstraintKind.TIED_CONTACT,
        slave_records=[_auto_rec()],
        master_nodes=[1, 2, 3, 4], slave_nodes=[5], dofs=[1, 2, 3],
    )
    fem = _host_fem().with_constraint(sc)
    out = tmp_path / "m.h5"
    fem.to_h5(str(out))
    back = FEMData.from_h5(str(out))
    scs = [r for r in back.elements.constraints
           if isinstance(r, SurfaceCouplingRecord)]
    assert len(scs) == 1
    assert scs[0].slave_records[0].stiffness == "auto"
