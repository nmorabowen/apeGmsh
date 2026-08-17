"""ADR 0098 S1-B — realize(): the remaining six §4 slots.

Primary oracle is the same as S1-A: RecordingBackend layer assertions,
no GL, in-CI. Covered here —

* every §4 category emits its layer under a stable ``pane:category``
  key (reactions emits two: the force and moment families);
* the `vector` slot routes to ``vector_glyph`` vs ``principal_glyph``
  by where the quantity is RECORDED, never by the token's shape (the
  trap: a tensor token through vector_glyph reads nothing, silently);
* INV-MESH-2 generalized — the grey substrate is suppressed by ANY
  occluding occupant (contour *and* sand), and by nothing else;
* the §5 colour-mapped column: contour/vector/gauss/sand emit a scale,
  line/loads/reactions never do, and two slots of the same field share
  ONE scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    Contour,
    Gauss,
    Loads,
    Reactions,
    Sand,
    Vector,
)
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.session import realize_pane

from tests.conftest import _open_model_from_h5
from tests.viewers.conftest import RecordingBackend

STAGE = "grav"
T = 3


@pytest.fixture
def slot_results(g, tmp_path: Path):
    """A cube recording every family the six slots need.

    * nodal vectors — ``displacement_x/y/z`` (``displacement_z`` is
      ``node_id + 100*t`` so per-step assertions are exact);
    * nodal reactions — ``reaction_force_*`` / ``reaction_moment_*``
      (the names the reactions kind actually reads);
    * a Gauss stress tensor — all six ``stress_**`` components,
      ``element_id + t``, one integration point per element;
    * two nodal load patterns, appended the way the shipped loads
      fixture does.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    element_ids = np.concatenate(
        [np.asarray(grp.ids, dtype=np.int64) for grp in fem.elements]
    )

    nodal: dict[str, np.ndarray] = {}
    for axis_i, axis in enumerate("xyz"):
        nodal[f"displacement_{axis}"] = np.tile(
            coords[:, axis_i] * 0.01, (T, 1)
        )
        nodal[f"reaction_force_{axis}"] = np.tile(
            coords[:, axis_i] * 5.0 + 1.0, (T, 1)
        )
        nodal[f"reaction_moment_{axis}"] = np.tile(
            coords[:, axis_i] * 2.0 + 0.5, (T, 1)
        )
    nodal["displacement_z"] = np.stack(
        [node_ids + t * 100.0 for t in range(T)]
    )

    gauss: dict[str, np.ndarray] = {}
    for comp in (
        "stress_xx", "stress_yy", "stress_zz",
        "stress_xy", "stress_yz", "stress_xz",
    ):
        values = np.zeros((T, element_ids.size, 1), dtype=np.float64)
        for t in range(T):
            values[t, :, 0] = element_ids + t
        gauss[comp] = values

    path = tmp_path / "slots.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(T, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids, components=nodal,
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0",
            class_tag=4, int_rule=1,
            element_index=element_ids,
            natural_coords=np.zeros((1, 3)),
            components=gauss,
        )
        w.end_stage()

    results = Results.from_native(path, model=_open_model_from_h5(path))
    from apeGmsh._kernel.records import NodalLoadRecord

    after = list(results.fem.nodes.ids)
    results.fem.nodes.loads._records.append(NodalLoadRecord(
        pattern="P1", node_id=int(after[0]), force_xyz=(10.0, 0.0, 0.0),
    ))
    results.fem.nodes.loads._records.append(NodalLoadRecord(
        pattern="P2", node_id=int(after[1]), force_xyz=(0.0, -5.0, 0.0),
    ))
    return results


def _realize(results, mutate):
    session = results.session()
    view = session.panes[0]
    mutate(view)
    backend = RecordingBackend()
    return backend, realize_pane(session, view, backend)


def _keys(realized):
    return [layer.key for layer in realized.layers]


# =====================================================================
# Each §4 category emits its layer
# =====================================================================


@pytest.mark.parametrize(
    "mutate, expected_key",
    [
        (lambda v: setattr(v, "vector", Vector("displacement")),
         "mesh-1:vector"),
        (lambda v: setattr(v, "gauss", Gauss("stress_xx")),
         "mesh-1:gauss"),
        (lambda v: setattr(v, "sand", Sand("displacement_z")),
         "mesh-1:sand"),
        (lambda v: setattr(v, "loads", Loads(pattern="P1")),
         "mesh-1:loads"),
    ],
    ids=["vector", "gauss", "sand", "loads"],
)
def test_slot_emits_its_layer(slot_results, mutate, expected_key):
    backend, realized = _realize(slot_results, mutate)
    assert expected_key in _keys(realized)
    assert backend.layers, "the slot emitted no layer at all"


def test_reactions_emits_both_families(slot_results):
    """The multi-layer kind: force and moment are separate glyph
    layers, and BOTH must reach the diff surface or the S2 reconciler
    could never update or remove them."""
    _backend, realized = _realize(
        slot_results, lambda v: setattr(v, "reactions", Reactions()),
    )
    keys = _keys(realized)
    assert "mesh-1:reactions.force" in keys
    assert "mesh-1:reactions.moment" in keys


# =====================================================================
# §4 vector routing — by what is RECORDED, not by the token's shape
# =====================================================================


@pytest.mark.parametrize(
    "quantity, expected_kind",
    [
        ("displacement", "vector_glyph"),      # bare nodal prefix
        ("displacement_z", "vector_glyph"),    # nodal axis token
        ("stress_xx", "principal_glyph"),      # gauss tensor component
        ("stress", "principal_glyph"),         # bare gauss prefix
    ],
)
def test_vector_slot_routes_by_recorded_family(
    slot_results, quantity, expected_kind,
):
    _backend, realized = _realize(
        slot_results, lambda v: setattr(v, "vector", Vector(quantity)),
    )
    (diagram,) = realized.diagrams
    assert diagram.kind == expected_kind


def test_vector_unrecorded_quantity_is_loud(slot_results):
    with pytest.raises(ValueError, match="not recorded"):
        _realize(
            slot_results, lambda v: setattr(v, "vector", Vector("nope")),
        )


# =====================================================================
# INV-MESH-2 generalized — occluding occupants suppress the substrate
# =====================================================================


@pytest.mark.parametrize(
    "mutate, substrate_expected",
    [
        (lambda v: setattr(v, "contour", Contour("displacement_z")), False),
        (lambda v: setattr(v, "sand", Sand("displacement_z")), False),
        (lambda v: setattr(v, "gauss", Gauss("stress_xx")), True),
        (lambda v: setattr(v, "vector", Vector("displacement")), True),
        (lambda v: setattr(v, "reactions", Reactions()), True),
        (lambda v: setattr(v, "loads", Loads(pattern="P1")), True),
    ],
    ids=["contour", "sand", "gauss", "vector", "reactions", "loads"],
)
def test_substrate_only_under_non_occluding_slots(
    slot_results, mutate, substrate_expected,
):
    """One surface: contour and sand ARE the surface; glyph slots sit
    on top of the grey mesh and must not remove it."""
    _backend, realized = _realize(slot_results, mutate)
    has_substrate = "mesh-1:substrate" in _keys(realized)
    assert has_substrate is substrate_expected


def test_sand_and_contour_together_still_one_surface(slot_results):
    def mutate(v):
        v.contour = Contour("displacement_z")
        v.sand = Sand("displacement_z")

    _backend, realized = _realize(slot_results, mutate)
    assert "mesh-1:substrate" not in _keys(realized)


# =====================================================================
# §5 — the colour-mapped column decides who gets a scale
# =====================================================================


@pytest.mark.parametrize(
    "mutate, expected_bars",
    [
        (lambda v: setattr(v, "gauss", Gauss("stress_xx")), ["stress_xx"]),
        (lambda v: setattr(v, "sand", Sand("displacement_z")),
         ["displacement_z"]),
        (lambda v: setattr(v, "vector", Vector("displacement")),
         ["displacement"]),
        (lambda v: setattr(v, "reactions", Reactions()), []),
        (lambda v: setattr(v, "loads", Loads(pattern="P1")), []),
    ],
    ids=["gauss", "sand", "vector", "reactions", "loads"],
)
def test_only_colour_mapped_slots_emit_a_scale(
    slot_results, mutate, expected_bars,
):
    backend, realized = _realize(slot_results, mutate)
    assert list(backend.scalar_bars) == expected_bars
    assert list(realized.scalar_bars) == expected_bars


def test_same_field_on_two_slots_is_one_scale(slot_results):
    """§5: contour of S + Gauss of S → ONE scale."""
    def mutate(v):
        v.contour = Contour("stress_xx")
        v.gauss = Gauss("stress_xx")

    backend, realized = _realize(slot_results, mutate)
    assert list(backend.scalar_bars) == ["stress_xx"]
    assert list(realized.scalar_bars) == ["stress_xx"]


def test_different_fields_are_two_scales(slot_results):
    def mutate(v):
        v.contour = Contour("displacement_z")
        v.gauss = Gauss("stress_xx")

    backend, _realized = _realize(slot_results, mutate)
    assert sorted(backend.scalar_bars) == ["displacement_z", "stress_xx"]


def test_categories_stack(slot_results):
    """Different categories stack (§4) — three occupants, three
    layer families, and the substrate gone under the contour."""
    def mutate(v):
        v.contour = Contour("displacement_z")
        v.gauss = Gauss("stress_xx")
        v.reactions = Reactions()

    _backend, realized = _realize(slot_results, mutate)
    keys = _keys(realized)
    assert "mesh-1:contour" in keys
    assert "mesh-1:gauss" in keys
    assert "mesh-1:reactions.force" in keys
    assert "mesh-1:substrate" not in keys


# =====================================================================
# The loads slot's pattern (§4 — "pattern / stage" inside the slot)
# =====================================================================


def test_loads_pattern_is_honoured(slot_results):
    _backend, realized = _realize(
        slot_results, lambda v: setattr(v, "loads", Loads(pattern="P2")),
    )
    (diagram,) = realized.diagrams
    assert diagram.spec.selector.component == "P2"


def test_loads_unknown_pattern_is_loud(slot_results):
    with pytest.raises(ValueError, match="not in this model"):
        _realize(
            slot_results, lambda v: setattr(v, "loads", Loads(pattern="X")),
        )


def test_loads_ambiguous_default_refuses(slot_results):
    """Two patterns and no choice: drawing one silently would show an
    arbitrary subset of the model's loads."""
    with pytest.raises(ValueError, match="load patterns"):
        _realize(slot_results, lambda v: setattr(v, "loads", Loads()))
