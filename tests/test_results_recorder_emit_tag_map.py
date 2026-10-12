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


def _spec(*records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(fem_snapshot_id="h", records=records)


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
    model = TagMapModel(SEQ)
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
                             model=TagMapModel(SEQ)) as live:
        live.begin_stage("s")
        live.end_stage()
    ele_calls = [c for c in fake.recorder_calls if "-ele" in c]
    assert ele_calls
    assert all(_ele_ids(c) == [3, 4, 5] for c in ele_calls)
    node_calls = [c for c in fake.recorder_calls if "-node" in c]
    assert node_calls                                 # nodes pass through


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
        spec.to_tcl_commands(model=TagMapModel(SEQ))
    with pytest.raises(ValueError, match=r"\[99\]"):
        with spec.emit_recorders(str(tmp_path), ops=FakeOps(),
                                 model=TagMapModel(SEQ)):
            pass


def test_node_only_specs_need_no_model(tmp_path: Path) -> None:
    spec = _spec(_node_record())
    assert spec.to_tcl_commands()
    assert spec.to_python_commands()
    fake = FakeOps()
    with spec.emit_recorders(str(tmp_path), ops=fake) as live:
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
    spec = _spec(rec)
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
    [line] = _spec(_element_record(ids=beam)).to_tcl_commands(model=model)
    expected = " ".join(str(pairing[e]) for e in beam)
    assert f" -ele {expected} " in line
