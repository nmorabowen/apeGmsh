"""ADR 0095 S1 — SelectionEnvelope projector, names-first JSON round-trip."""
from __future__ import annotations

from pathlib import Path

from apeGmsh.studio import (
    NameRecord,
    SelectionEnvelope,
    dumps,
    loads,
    project,
    project_state,
    read_envelope,
    write_envelope,
)
from apeGmsh.viewers.core.selection import SelectionState
from apeGmsh.viewers.scene_ir import SelectionTarget, Substrate


def test_labeled_face_projects_names_not_only_dimtags() -> None:
    sel = SelectionState()
    sel.pick((2, 5))
    names = {
        (2, 5): NameRecord(
            labels=("front",),
            physical_groups=("Face",),
            bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        )
    }
    env = project_state(sel, phase="model", names=names)
    assert env.phase == "model"
    assert env.labels == ("front",)
    assert env.physical_groups == ("Face",)
    assert env.unnamed == ()
    assert env.substrate == "model_brep"
    text = dumps(env)
    assert "front" in text
    assert "Face" in text
    assert '"phase": "model"' in text
    # Dimtags are evidence, not the identity payload.
    assert env.evidence[0].dim == 2
    assert env.evidence[0].key == 5
    roundtrip = loads(text)
    assert roundtrip.labels == ("front",)
    assert roundtrip.physical_groups == ("Face",)
    assert roundtrip.phase == "model"
    assert roundtrip.evidence[0].key == 5


def test_unnamed_pick_is_flagged() -> None:
    env = project([(2, 7)], phase="model")
    assert env.labels == ()
    assert env.physical_groups == ()
    assert len(env.unnamed) == 1
    assert env.unnamed[0].dim == 2
    assert env.unnamed[0].key == 7


def test_selection_target_round_trip(tmp_path: Path) -> None:
    target = SelectionTarget(Substrate.MODEL_BREP, 2, 3)
    env = project(
        [target],
        phase="model",
        names={(2, 3): NameRecord(labels=("wall",))},
    )
    path = write_envelope(tmp_path / "selection.json", env)
    loaded = read_envelope(path)
    assert loaded.labels == ("wall",)
    assert loaded.evidence[0].key == 3


def test_empty_selection_is_valid_envelope() -> None:
    sel = SelectionState()
    env = project_state(sel, phase="model")
    assert env.labels == ()
    assert env.evidence == ()
    assert env.substrate == "model_brep"


def test_envelope_is_frozen() -> None:
    env = project([], phase="mesh")
    assert isinstance(env, SelectionEnvelope)
    try:
        env.labels = ("x",)  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("SelectionEnvelope should be frozen")
