"""ADR 0098 S5c — rendering a saved snapshot, out of process.

The primary oracle is the same one S5a shipped and this slice reuses
verbatim: **restored-realize == built-realize** at the scene-IR layer.
What S5c adds is the DOOR, so what has to be proven here is that the
door's own resolution step — find the broker, load the file, bind them —
produces the session that was saved, not merely *a* session.

The still itself is secondary and sits behind a GL skip (ADR 0094:
skip ≠ pass), because a PNG proves the pipeline ran, not that it drew
the right thing.

The discriminating case is again a NON-EMPTY selection: it is realize
output (`<pane>:selection`), so a door that quietly dropped it would
still produce a plausible picture and an equal-looking layer list.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    Contour,
    Deform,
    Gauss,
    Instant,
    MeshStyle,
    save_snapshot,
)
from apeGmsh.results.session.__main__ import _resolve_session, main
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5
from tests.viewers.test_session_snapshot_realize import _scene

STAGE = "grav"
N_STEPS = 4


# =====================================================================
# Fixture — the same shape S5a's oracle uses
# =====================================================================

@pytest.fixture
def render_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    chunks = [np.asarray(gr.ids, dtype=np.int64) for gr in fem.elements]
    elem_ids = np.concatenate(chunks) if chunks else np.zeros(0, np.int64)

    disp = np.zeros((N_STEPS, node_ids.size), dtype=np.float64)
    for t in range(N_STEPS):
        disp[t] = node_ids + t * 1000.0
    sxx = np.zeros((N_STEPS, elem_ids.size, 1), dtype=np.float64)
    for t in range(N_STEPS):
        sxx[t, :, 0] = elem_ids * 10.0 + t

    path = tmp_path / "render_s5c.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.write_gauss_group(
            sid, "partition_0", "group_0", class_tag=4, int_rule=1,
            element_index=elem_ids, natural_coords=np.zeros((1, 3)),
            components={"stress_xx": sxx},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _built(results):
    """A session with something worth drawing — and a live selection."""
    from apeGmsh.viewers.session import select_all

    session = results.session()
    view = session.panes[0]
    view.deform = Deform("displacement", scale=2.0)
    view.contour = Contour("displacement_z", "unaveraged")
    view.gauss = Gauss("stress_xx")
    view.style = MeshStyle(mesh=True, outlines=True, nodes=True, gauss=True)
    session.time = Instant(STAGE, 2)
    select_all(session, "nodes", axis="physical_groups", names=("Body",))
    session.selection.set_nodes(session.selection.nodes[:4])
    return session


def _saved(results, tmp_path: Path) -> tuple[Path, object]:
    built = _built(results)
    path = tmp_path / "shot.h5.session.json"
    save_snapshot(built, path)
    return path, built


# =====================================================================
# The oracle — the door resolves the session that was saved
# =====================================================================

def test_the_door_resolves_a_session_that_realizes_identically(
    render_results, tmp_path: Path,
):
    snapshot, built = _saved(render_results, tmp_path)

    session, notices = _resolve_session(snapshot)

    assert notices == ()
    assert _scene(session, "mesh-1") == _scene(built, "mesh-1")


def test_the_selection_rides_through_the_door(
    render_results, tmp_path: Path,
):
    """Guards the oracle above against passing vacuously — the same
    discriminating case S5a needed, at the door this time."""
    snapshot, built = _saved(render_results, tmp_path)

    session, _ = _resolve_session(snapshot)

    assert session.selection.nodes == built.selection.nodes
    assert "mesh-1:selection" in _scene(session, "mesh-1")["layers"]


def test_a_snapshot_stripped_of_its_selection_renders_differently(
    render_results, tmp_path: Path,
):
    snapshot, built = _saved(render_results, tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["selection"] = {"kind": None}
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    session, _ = _resolve_session(snapshot)

    assert _scene(session, "mesh-1") != _scene(built, "mesh-1")


# =====================================================================
# Finding the broker
# =====================================================================

def test_the_broker_comes_from_the_snapshot_by_default(
    render_results, tmp_path: Path,
):
    """Two arguments is the common case: the snapshot recorded where its
    results live when it was written."""
    snapshot, _ = _saved(render_results, tmp_path)
    recorded = json.loads(snapshot.read_text(encoding="utf-8"))["results_path"]

    session, _ = _resolve_session(snapshot)

    assert Path(recorded).is_file()
    assert Path(session.results._path).resolve() == Path(recorded).resolve()


def test_an_explicit_results_path_overrides_the_recorded_one(
    render_results, tmp_path: Path,
):
    """The pinned-copy case (S5b) and the moved-file case: the recorded
    path is where the results WERE."""
    snapshot, built = _saved(render_results, tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    real = Path(payload["results_path"])
    payload["results_path"] = str(tmp_path / "moved" / "gone.h5")
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    session, _ = _resolve_session(snapshot, real)

    assert _scene(session, "mesh-1") == _scene(built, "mesh-1")


def test_a_missing_results_file_says_which_one_and_why(
    render_results, tmp_path: Path,
):
    snapshot, _ = _saved(render_results, tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["results_path"] = str(tmp_path / "gone.h5")
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        _resolve_session(snapshot)

    message = str(excinfo.value)
    assert "gone.h5" in message
    assert "--results" in message      # tells the caller the way out


def test_a_snapshot_with_no_recorded_results_refuses(tmp_path: Path):
    snapshot = tmp_path / "bare.session.json"
    snapshot.write_text(json.dumps({
        "kind": "apegmsh.results.session", "version": 1,
        "results_path": None, "panes": [],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="--results"):
        _resolve_session(snapshot)


# =====================================================================
# The legacy refusal — and it never renames
# =====================================================================

def test_the_cli_refuses_a_v13_file_and_renames_nothing(
    render_results, tmp_path: Path,
):
    """S5c's own contract: the rename-aside belongs to the human flow.

    A batch door that moved a user's file aside would be doing surgery
    nobody asked for — so this refuses and leaves the directory exactly
    as it found it.

    The legacy file is written BESIDE its real results file on purpose.
    An earlier version pointed at a results path that did not exist, so
    the run died on the missing broker and never reached the legacy gate
    at all — it passed just as happily with the gate flipped to
    rename-on-sight, which is the mutation it was supposed to catch.
    """
    from apeGmsh.viewers.diagrams._session import serialize_session

    results_file = Path(render_results._path)
    legacy = results_file.with_suffix(
        results_file.suffix + ".viewer-session.json",
    )
    legacy.write_text(json.dumps(serialize_session(
        specs=[], results_path=str(results_file), fem_snapshot_id=None,
    )), encoding="utf-8")
    home = legacy.parent
    before = sorted(q.name for q in home.iterdir())

    code = main(["render", str(legacy), str(tmp_path / "x.png"), "--json"])

    assert code == 2
    # Nothing renamed, nothing created, the original still there.
    assert sorted(q.name for q in home.iterdir()) == before
    assert not Path(str(legacy) + ".legacy").exists()
    assert legacy.is_file()


def test_the_cli_reports_restore_notices_rather_than_swallowing_them(
    render_results, tmp_path: Path, capsys,
):
    """A still drawn at a different instant than the file asked for has
    to say so (the S5a degrade rule, surfaced at the door).

    The snapshot names a stage these results no longer have, so restore
    drops the instant and reports it; the picture still draws.
    """
    snapshot, _ = _saved(render_results, tmp_path)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["time"] = {"stage": "a_stage_from_the_old_run", "step": 1}
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    code = main([
        "render", str(snapshot), str(tmp_path / "notice.png"), "--json",
    ])

    assert code == 0
    err = capsys.readouterr().err
    assert "[session]" in err
    assert "a_stage_from_the_old_run" in err


def test_the_cli_reports_a_refusal_as_json_when_asked(
    tmp_path: Path, capsys,
):
    code = main([
        "render", str(tmp_path / "nope.json"), str(tmp_path / "x.png"),
        "--json",
    ])

    assert code == 2
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["written"] == []
    assert "not found" in payload["error"]


def test_the_bare_module_prints_usage(capsys):
    assert main([]) == 2
    assert "render" in capsys.readouterr().err


# =====================================================================
# The still itself — behind a GL skip (skip != pass)
# =====================================================================

def _gl_or_skip():
    if os.environ.get("APEGMSH_SKIP_VIEWER"):
        pytest.skip("APEGMSH_SKIP_VIEWER is set")
    try:
        import pyvista  # noqa: F401
    except Exception:  # pragma: no cover
        pytest.skip("pyvista unavailable")


def test_the_cli_writes_a_png(render_results, tmp_path: Path):
    _gl_or_skip()
    snapshot, _ = _saved(render_results, tmp_path)
    out = tmp_path / "shots" / "pane.png"

    code = main(["render", str(snapshot), str(out), "--json"])

    assert code == 0
    if not out.is_file():  # the ADR 0094 no-GL skip, taken inside render
        pytest.skip("no GL context; render took its documented skip")
    assert out.stat().st_size > 0


def test_the_subprocess_door_really_runs(render_results, tmp_path: Path):
    """The door is a subprocess in production, so run it as one once —
    an in-process call cannot catch an import that only fails under
    ``-m`` (the whole point of keeping a GL crash out of the kernel)."""
    _gl_or_skip()
    snapshot, _ = _saved(render_results, tmp_path)
    out = tmp_path / "sub.png"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(
        Path(__file__).resolve().parents[2] / "src",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "apeGmsh.results.session", "render",
         str(snapshot), str(out), "--json"],
        capture_output=True, text=True, env=env, timeout=300,
    )

    assert proc.returncode == 0, proc.stderr
    line = [
        ln for ln in proc.stdout.strip().splitlines()
        if ln.strip().startswith("{")
    ][-1]
    payload = json.loads(line)
    assert payload["ok"] is True
