"""#1711: recorder emits translate FEM ids to OpenSees tags, or refuse.

No solver here. The live parity oracles are in
``test_results_recorder_tag_parity.py``; these pin the three emit routes
(``to_tcl_commands``, ``to_python_commands``, ``emit_recorders``), the
refusals, and that an ``element_tags="fem"`` build emits exactly what the
untranslated spec emitted before the fix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.spec._emit import emit_logical, format_python, format_tcl
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

from tests.fixtures.recorder_models import TagMapModel
from tests.test_results_live_recorders import FakeOps

#: The portal frame's beam under sequential tags (FEM id -> ops tag).
SEQ = {8: 3, 9: 4, 10: 5}
#: The node record's nodes, as 3-D frame nodes (#2018: node DOFs need it).
NODES_NDF = {1: 6, 2: 6}


def _spec(*records: ResolvedRecorderRecord,
          fem_snapshot_id: str = "h") -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(fem_snapshot_id=fem_snapshot_id,
                                records=records)


def _element_record(category: str = "elements",
                    ids: tuple[int, ...] = (8, 9, 10)) -> ResolvedRecorderRecord:
    components = {
        "elements": ("nodal_resisting_force_x",),
        "gauss": ("stress_xx",),
        "line_stations": ("axial_force",),
    }[category]
    return ResolvedRecorderRecord(
        category=category, name="r", components=components,
        dt=None, n_steps=None, element_ids=np.asarray(ids),
    )


def _node_record() -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="nodes", name="n", components=("displacement_x",),
        dt=None, n_steps=None, node_ids=np.array([1, 2]),
    )


def _ele_ids(args: tuple) -> list[int]:
    i = list(args).index("-ele")
    out = []
    for a in args[i + 1:]:
        if not isinstance(a, int):
            break
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Translation, every element-scoped category, every route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["elements", "gauss", "line_stations"])
def test_tcl_and_python_commands_carry_ops_tags(category: str) -> None:
    spec = _spec(_element_record(category))
    model = TagMapModel(SEQ, "h")
    tcl = spec.to_tcl_commands(model=model)
    py = spec.to_python_commands(model=model)
    assert tcl and len(tcl) == len(py)
    for line in tcl:
        assert " -ele 3 4 5 " in line
    for line in py:
        assert "'-ele', 3, 4, 5," in line


@pytest.mark.parametrize("category", ["elements", "gauss", "line_stations"])
def test_live_route_issues_ops_tags(category: str, tmp_path: Path) -> None:
    fake = FakeOps()
    spec = _spec(_node_record(), _element_record(category))
    with spec.emit_recorders(str(tmp_path), ops=fake,
                             model=TagMapModel(SEQ, "h",
                                               nodes_ndf=NODES_NDF)) as live:
        live.begin_stage("s")
        live.end_stage()
    ele_calls = [c for c in fake.recorder_calls if "-ele" in c]
    assert ele_calls
    assert all(_ele_ids(c) == [3, 4, 5] for c in ele_calls)
    node_calls = [c for c in fake.recorder_calls if "-node" in c]
    assert node_calls                       # node ids pass through untranslated
    assert all(c[c.index("-node") + 1:c.index("-node") + 3] == (1, 2)
               for c in node_calls)


def test_read_side_layout_stays_in_fem_ids() -> None:
    """The transcoder and the cache read ``emit_logical``; it keeps FEM
    ids, so transcoded rows are labelled with the FEM ids requested."""
    [lr] = emit_logical(_element_record())
    assert lr.target_ids == (8, 9, 10)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["to_tcl_commands", "to_python_commands"])
def test_command_routes_refuse_element_records_without_model(route: str) -> None:
    spec = _spec(_element_record())
    with pytest.raises(ValueError, match=r"Pass model="):
        getattr(spec, route)()


def test_live_route_refuses_before_issuing_any_recorder(tmp_path: Path) -> None:
    fake = FakeOps()
    spec = _spec(_node_record(), _element_record())
    with pytest.raises(ValueError, match=r"Pass model="):
        with spec.emit_recorders(str(tmp_path), ops=fake) as live:
            live.begin_stage("s")
    assert fake.recorder_calls == []


def test_an_id_the_model_lacks_is_refused(tmp_path: Path) -> None:
    spec = _spec(_element_record(ids=(8, 9, 99)))
    with pytest.raises(ValueError, match=r"\[99\]"):
        spec.to_tcl_commands(model=TagMapModel(SEQ, "h"))
    with pytest.raises(ValueError, match=r"\[99\]"):
        with spec.emit_recorders(str(tmp_path), ops=FakeOps(),
                                 model=TagMapModel(SEQ, "h")):
            pass


def test_node_only_specs_need_a_model(tmp_path: Path) -> None:
    """#2018 reverses #1711's pass-through: a node's ``-dof`` list depends
    on its ndf and the model's ndm, so every route refuses without one."""
    spec = _spec(_node_record())
    with pytest.raises(ValueError, match=r"Pass model="):
        spec.to_tcl_commands()
    with pytest.raises(ValueError, match=r"Pass model="):
        spec.to_python_commands()
    fake = FakeOps()
    with pytest.raises(ValueError, match=r"Pass model="):
        with spec.emit_recorders(str(tmp_path), ops=fake):
            pass
    assert fake.recorder_calls == []
    model = TagMapModel({}, "h", nodes_ndf=NODES_NDF)
    assert spec.to_tcl_commands(model=model)
    assert spec.to_python_commands(model=model)
    with spec.emit_recorders(str(tmp_path), ops=fake, model=model) as live:
        live.begin_stage("s")
        live.end_stage()
    assert fake.recorder_calls


# ---------------------------------------------------------------------------
# Real built models: identity mode is byte-identical, sequential translates
# ---------------------------------------------------------------------------


def _built_model(element_tags: str, tmp_path: Path):
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    from tests.test_results_recorder_tag_parity import _portal_fem

    fem = _portal_fem()
    ops = apeSees(fem, default_orientation=None,
                  element_tags=element_tags)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Beam", transf=tr,
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    ops.element.elasticBeamColumn(pg="Cols", transf=tr,
                                  A=0.03, E=2.0e11, Iz=5.0e-4)
    path = tmp_path / f"{element_tags}.h5"
    ops.h5(str(path))
    beam = tuple(int(e) for e in fem.elements.select(pg="Beam").ids)
    return OpenSeesModel.from_h5(str(path)), beam


def test_fem_tag_build_emits_the_untranslated_commands(tmp_path: Path) -> None:
    model, beam = _built_model("fem", tmp_path)
    rec = _element_record(ids=beam)
    spec = _spec(rec, fem_snapshot_id=model.lineage.fem_hash)
    assert spec.to_tcl_commands(model=model) == [
        format_tcl(lr) for lr in emit_logical(rec)
    ]
    assert spec.to_python_commands(model=model) == [
        format_python(lr) for lr in emit_logical(rec)
    ]


def test_sequential_build_emits_the_built_tags(tmp_path: Path) -> None:
    model, beam = _built_model("sequential", tmp_path)
    pairing = {int(r.fem_eid): int(r.tag) for r in model.elements()}
    assert any(pairing[e] != e for e in beam)         # not vacuous
    [line] = _spec(_element_record(ids=beam),
                   fem_snapshot_id=model.lineage.fem_hash,
                   ).to_tcl_commands(model=model)
    expected = " ".join(str(pairing[e]) for e in beam)
    assert f" -ele {expected} " in line

# ---------------------------------------------------------------------------
# The public bridge path never reaches these routes and already translates
# ---------------------------------------------------------------------------


def test_bridge_declared_recorders_carry_ops_tags(tmp_path: Path) -> None:
    """``ops.recorder.declare`` / ``ops.recorder.Element`` are emitted by
    the bridge build, which translates FEM ids itself and never calls the
    spec routes; under sequential tags their ``-ele`` lists are the beam's
    ops tags, with no ``model=`` anywhere."""
    from tests.test_results_recorder_tag_parity import (
        _portal_bridge,
        _portal_fem,
    )

    fem = _portal_fem()
    ops = _portal_bridge(fem)
    beam = [int(e) for e in fem.elements.select(pg="Beam").ids]
    ops.recorder.declare(elements=("nodal_resisting_force_x",), pg="Beam",
                         name="beamrec", file_root="out")
    ops.recorder.Element(file="out/b.out", response=("globalForce",),
                         pg="Beam")
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    lines = deck.read_text().splitlines()
    tags = {}
    for line in lines:
        parts = line.split()
        if parts[:2] == ["element", "elasticBeamColumn"]:
            tags[(int(parts[3]), int(parts[4]))] = int(parts[2])
    beam_tags = sorted(tags.values())
    assert beam_tags != sorted(beam)                    # not vacuous
    recorders = [ln for ln in lines if ln.startswith("recorder Element")]
    assert len(recorders) == 2
    for ln in recorders:
        assert f" -ele {' '.join(map(str, beam_tags))} globalForce" in ln


# ---------------------------------------------------------------------------
# model= must belong to the spec
# ---------------------------------------------------------------------------


def test_a_model_of_another_fem_is_refused(tmp_path: Path) -> None:
    spec = _spec(_element_record())                   # fem_snapshot_id "h"
    other = TagMapModel(SEQ, "another-mesh")
    with pytest.raises(ValueError, match=r"not built from this spec's FEMData"):
        spec.to_tcl_commands(model=other)
    with pytest.raises(ValueError, match=r"not built from this spec's FEMData"):
        spec.to_python_commands(model=other)
    fake = FakeOps()
    with pytest.raises(ValueError, match=r"not built from this spec's FEMData"):
        spec.emit_recorders(str(tmp_path), ops=fake, model=other)
    assert fake.recorder_calls == []


def test_a_remeshed_model_with_overlapping_ids_is_refused(
    tmp_path: Path,
) -> None:
    """A real model built from a re-meshed portal shares the beam's FEM ids
    yet pairs them with other tags; its lineage gives it away."""
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    from tests.test_results_recorder_tag_parity import _portal_fem

    model, beam = _built_model("sequential", tmp_path)
    fem_b = _portal_fem(n=5)
    ops_b = apeSees(fem_b, default_orientation=None)
    ops_b.model(ndm=2, ndf=3)
    tr = ops_b.geomTransf.Linear()
    for pg in ("Cols", "Beam"):
        ops_b.element.elasticBeamColumn(pg=pg, transf=tr,
                                        A=0.02, E=2.0e11, Iz=3.0e-4)
    ops_b.h5(str(tmp_path / "b.h5"))
    model_b = OpenSeesModel.from_h5(str(tmp_path / "b.h5"))
    pairing_b = {int(r.fem_eid): int(r.tag) for r in model_b.elements()}
    assert all(e in pairing_b for e in beam)          # the ids overlap
    spec = ResolvedRecorderSpec(fem_snapshot_id=model.lineage.fem_hash,
                                records=(_element_record(ids=beam),))
    assert spec.to_tcl_commands(model=model)          # its own model emits
    with pytest.raises(ValueError, match=r"not built from this spec's FEMData"):
        spec.to_tcl_commands(model=model_b)
