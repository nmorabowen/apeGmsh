"""#2340: a resolved recorder record refuses a component that records nothing.

``ResolvedRecorderRecord(components=("displacement",))`` wrote no recorder
on any route (``.out``, MPCO, the fork's file) and read back empty with no
warning: a resolved record carries canonical names and nothing expands a
shorthand there. It now refuses by name at construction, naming the
``expand_shorthand`` expansion.

The family shorthands that do record as written stay accepted:
``stress`` / ``strain`` / ``plastic_strain`` on a gauss record and
``section_force`` on a line-stations record. Each emits the family's whole
response, and its read-back was checked against the explicit components on
stock pip and on the fork before this change.

The live case pins the explicit form on the classic-recorder route against
the closed form of a bar under an axial load, ``u = P L / (E A)``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh._vocabulary import expand_shorthand
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

#: Every nodal family shorthand: none of them wrote a recorder (#2340).
NODAL_SHORTHANDS = (
    "displacement", "velocity", "acceleration", "displacement_increment",
    "rotation", "angular_velocity", "angular_acceleration",
    "force", "moment", "reaction", "reaction_force", "reaction_moment",
)


def _record(category: str, comps: "tuple[str, ...]") -> ResolvedRecorderRecord:
    ids = {"node_ids": np.array([1])} if category == "nodes" else {
        "element_ids": np.array([1])}
    return ResolvedRecorderRecord(category=category, name="rec",
                                  components=comps, dt=None, n_steps=None,
                                  **ids)


# ---------------------------------------------------------------------------
# Construction: refuse by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shorthand", NODAL_SHORTHANDS)
def test_a_nodal_shorthand_refuses_by_name_with_its_expansion(
    shorthand: str,
) -> None:
    full = ", ".join(expand_shorthand(shorthand))
    with pytest.raises(ValueError) as info:
        _record("nodes", ("displacement_x", shorthand))
    msg = str(info.value)
    assert f"component {shorthand!r} is a shorthand" in msg
    assert f"expand_shorthand({shorthand!r}" in msg
    assert f"({full})" in msg
    assert "'rec'" in msg


@pytest.mark.parametrize(("category", "shorthand"), [
    ("gauss", "section_force"),
    ("gauss", "section_deformation"),
    ("gauss", "shell_resultant"),
    ("gauss", "shell_deformation"),
    ("line_stations", "section_deformation"),
    ("line_stations", "shell_resultant"),
    ("elements", "displacement"),
    ("elements", "stress"),
    ("fibers", "strain"),
    ("layers", "stress"),
])
def test_a_shorthand_that_records_nothing_in_its_category_refuses(
    category: str, shorthand: str,
) -> None:
    with pytest.raises(ValueError, match=f"component {shorthand!r} is a "
                                         "shorthand"):
        _record(category, (shorthand,))


@pytest.mark.parametrize(("category", "name"), [
    ("nodes", "displacment_x"),
    ("gauss", "stresses"),
    ("line_stations", "bending_moment"),
])
def test_a_name_outside_the_vocabulary_refuses(category: str, name: str) -> None:
    with pytest.raises(ValueError, match=f"component {name!r} is not a "
                                         "canonical"):
        _record(category, (name,))


# ---------------------------------------------------------------------------
# Construction: what records today stays accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("category", "comps"), [
    ("gauss", ("stress",)),
    ("gauss", ("strain",)),
    ("gauss", ("plastic_strain",)),
    ("line_stations", ("section_force",)),
    ("nodes", ("displacement_x", "rotation_z", "reaction_force_y")),
    ("gauss", ("stress_xx", "von_mises_stress", "state_variable_3")),
    ("line_stations", ("axial_force", "bending_moment_z")),
    ("elements", ("nodal_resisting_force_x",)),
])
def test_canonical_names_and_recorded_shorthands_are_accepted(
    category: str, comps: "tuple[str, ...]",
) -> None:
    assert _record(category, comps).components == comps


def test_a_modal_record_is_not_checked() -> None:
    rec = ResolvedRecorderRecord(category="modal", name="m", components=(),
                                 dt=None, n_steps=None, n_modes=2)
    assert rec.n_modes == 2


# ---------------------------------------------------------------------------
# Live: the explicit form records as before (closed form)
# ---------------------------------------------------------------------------

P, E_MOD, AREA, LENGTH = 5.0, 1000.0, 1.0, 1.0
LAMBDAS = (0.5, 1.0)


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


def _bar_fem() -> FEMData:
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1,
                          npe=2, count=1)
    group = ElementGroup(element_type=info, ids=np.array([1], dtype=np.int64),
                         connectivity=np.array([[1, 2]], dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.array([1, 2], dtype=np.int64),
                            node_coords=np.array([[0.0, 0, 0],
                                                  [LENGTH, 0, 0]]),
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=2, n_elems=1, bandwidth=2, types=[info]),
        model_name="bar_2340",
    )


def _build_bar(ops: Any) -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, LENGTH, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, E_MOD)
    ops.element("Truss", 1, 1, 2, AREA, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, P, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", LAMBDAS[0])
    ops.analysis("Static")


@pytest.mark.live
def test_explicit_displacement_records_the_closed_form_on_the_recorders_route(
    opspy: Any, tmp_path: Path,
) -> None:
    """The explicit components the refusal points to record ``u = λ P L /
    (E A)`` at the free end, at rtol 1e-12, on the classic-recorder route
    (the stock pip route; :class:`NoMPCO` takes it on any build)."""
    from tests.test_results_live_default_format import NoMPCO

    with pytest.raises(ValueError, match="'displacement' is a shorthand"):
        _record("nodes", ("displacement",))

    _build_bar(opspy)
    fem = _bar_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id, records=(
        ResolvedRecorderRecord(category="nodes", name="u",
                               components=("displacement_x", "displacement_y"),
                               dt=None, n_steps=None,
                               node_ids=np.array([1, 2])),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(tmp_path / "run", model_h5=model_h5,
                               ops=NoMPCO(opspy)) as run:
            for _ in LAMBDAS:
                assert opspy.analyze(1) == 0
    assert run.route == "recorders"
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    res = run.results(cache_root=tmp_path / "cache")
    ux = np.asarray(res.nodes.get(component="displacement_x").values)
    uy = np.asarray(res.nodes.get(component="displacement_y").values)
    tip = np.array([lam * P * LENGTH / (E_MOD * AREA) for lam in LAMBDAS])
    np.testing.assert_allclose(ux[:, 1], tip, rtol=1e-12, atol=0)
    np.testing.assert_allclose(ux[:, 0], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(uy, 0.0, rtol=0, atol=0)
