"""ADR 0095 Amendment 11 — the ``.apegmsh/progress.json`` sidecar.

Two layers: the writer on its own (``ProgressSidecar``), and the writer
as ``opensees._run.stream_run`` actually drives it. The second layer
runs a **fake solver** — a three-line Python script that prints the
``APEGMSH_PROGRESS`` markers and exits with a chosen code — so the real
parse / tee / exit path is exercised without an OpenSees binary
anywhere near the suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from apeGmsh.opensees._run import stream_run
from apeGmsh.studio._contract import CONTRACT_VERSION, validate
from apeGmsh.studio._paths import DEFAULT_PROGRESS_REL
from apeGmsh.studio._progress import ProgressSidecar, habitat_root

# Three markers, then a warning line AFTER the last one — so the
# in-flight samples carry 0 warnings and only the terminal record can
# report the true tally. That ordering is the test, not decoration.
_FAKE_SOLVER = """\
import sys
for i in (1, 2, 3):
    print("APEGMSH_PROGRESS i=%d n=3 t=%.1f" % (i, i * 0.5))
print("WARNING something drifted")
sys.exit(int(sys.argv[1]))
"""


def _habitat(tmp_path: Path) -> Path:
    (tmp_path / ".apegmsh").mkdir()
    return tmp_path


def _read(root: Path) -> dict:
    return json.loads((root / DEFAULT_PROGRESS_REL).read_text(encoding="utf-8"))


def _run_fake_solver(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exit_code: int
) -> Path:
    """Drive the real ``stream_run`` over a fake solve. Returns the deck path."""
    monkeypatch.setenv("APEGMSH_ROOT", str(root))
    script = tmp_path / "fake_solver.py"
    script.write_text(_FAKE_SOLVER, encoding="utf-8")
    deck = tmp_path / "out" / "model.tcl"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text("# pretend deck\n", encoding="utf-8")
    log = tmp_path / "out" / "model.log"

    def call() -> None:
        stream_run(
            [sys.executable, str(script), str(exit_code)],
            log_path=str(log),
            verbose=False,
            label="analyze 3",
            deck_path=str(deck),
        )

    if exit_code:
        with pytest.raises(RuntimeError):
            call()
    else:
        call()
    return deck


# ---------------------------------------------------------------------
# The writer on its own
# ---------------------------------------------------------------------


def test_a_sample_lands_as_a_contract_valid_snapshot(tmp_path: Path) -> None:
    root = _habitat(tmp_path)
    sidecar = ProgressSidecar(deck=root / "out" / "m.tcl", log=root / "out" / "m.log",
                              root=root)

    sidecar.sample(i=120, n=500, t="1.2", warnings=0)

    payload = _read(root)
    validate("progress", payload)
    assert payload["schema"] == 1
    assert payload["contract_version"] == CONTRACT_VERSION
    assert (payload["i"], payload["n"], payload["t"]) == (120, 500, "1.2")
    assert payload["done"] is False
    assert "ok" not in payload  # only the terminal write claims an outcome


def test_paths_are_root_relative_posix_under_the_habitat(tmp_path: Path) -> None:
    """S5i, the same stance as the ledger's ``script`` / ``cwd``."""
    root = _habitat(tmp_path)
    outside = tmp_path.parent / "elsewhere.log"
    sidecar = ProgressSidecar(deck=root / "out" / "m.tcl", log=outside, root=root)

    sidecar.sample(i=1, n=2, t="0.5")

    payload = _read(root)
    assert payload["deck"] == "out/m.tcl"
    assert payload["log"] == str(outside.resolve())  # absolute outside the root


def test_the_terminal_write_carries_done_and_ok(tmp_path: Path) -> None:
    root = _habitat(tmp_path)
    sidecar = ProgressSidecar(root=root)
    sidecar.sample(i=3, n=3, t="1.5", warnings=0)

    sidecar.finish(ok=True, warnings=2)

    payload = _read(root)
    validate("progress", payload)
    assert payload["done"] is True
    assert payload["ok"] is True
    assert payload["warnings"] == 2  # refreshed by finish, not frozen at sample time
    assert (payload["i"], payload["n"]) == (3, 3)  # last sample is retained


def test_a_run_with_no_marker_writes_nothing(tmp_path: Path) -> None:
    """A deck with no ``analyze`` has no progress to report.

    Publishing ``0/0`` would tell a poller that an analysis finished at
    step zero, which is a different — and false — claim.
    """
    root = _habitat(tmp_path)
    sidecar = ProgressSidecar(root=root)

    sidecar.finish(ok=True)

    assert not (root / DEFAULT_PROGRESS_REL).exists()


def test_no_habitat_means_no_write_and_no_dot_dir(tmp_path: Path) -> None:
    """INV-28. ``resolve_root`` falls back to cwd, so without the
    ``.apegmsh/`` existence check every script run anywhere would
    deposit a dot-dir."""
    assert habitat_root(tmp_path) is None
    sidecar = ProgressSidecar(deck=tmp_path / "m.tcl", root=tmp_path)

    assert sidecar.enabled is False
    assert sidecar.path is None
    sidecar.sample(i=1, n=2, t="0.5")
    sidecar.finish(ok=True)

    assert not (tmp_path / ".apegmsh").exists()


def test_a_failing_write_is_swallowed(tmp_path: Path, monkeypatch) -> None:
    """INV-29. Losing a sample is a cost; killing the solve is not."""
    root = _habitat(tmp_path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("apeGmsh.studio._progress.atomic_write_text", _boom)
    sidecar = ProgressSidecar(root=root)

    sidecar.sample(i=1, n=2, t="0.5")  # must not raise
    sidecar.finish(ok=True)  # must not raise

    assert not (root / DEFAULT_PROGRESS_REL).exists()


def test_every_write_is_atomic_replace(tmp_path: Path) -> None:
    """INV-16: a poller sees one complete sample or the next, never half.

    Asserted through the observable consequence — the payload parses
    after every write, and no temp file is left behind for a reader to
    trip over.
    """
    root = _habitat(tmp_path)
    sidecar = ProgressSidecar(root=root)

    for i in range(1, 6):
        sidecar.sample(i=i, n=5, t=str(i * 0.2))
        assert _read(root)["i"] == i

    leftovers = [p.name for p in (root / ".apegmsh").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ---------------------------------------------------------------------
# As ``stream_run`` drives it
# ---------------------------------------------------------------------


def test_a_fake_solve_leaves_a_finished_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _habitat(tmp_path)

    deck = _run_fake_solver(root, tmp_path, monkeypatch, exit_code=0)

    payload = _read(root)
    validate("progress", payload)
    assert (payload["i"], payload["n"], payload["t"]) == (3, 3, "1.5")
    assert payload["done"] is True
    assert payload["ok"] is True
    # The warning printed after the last marker still reaches the record.
    assert payload["warnings"] == 1
    assert payload["deck"] == "out/model.tcl"
    assert payload["log"] == "out/model.log"
    assert Path(deck).is_file()


def test_a_failed_solve_still_closes_its_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a consumer polls ``done: false`` forever after a crash."""
    root = _habitat(tmp_path)

    _run_fake_solver(root, tmp_path, monkeypatch, exit_code=1)

    payload = _read(root)
    validate("progress", payload)
    assert payload["done"] is True
    assert payload["ok"] is False


def test_a_solve_outside_a_habitat_grows_no_dot_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()

    _run_fake_solver(bare, tmp_path, monkeypatch, exit_code=0)

    assert not (bare / ".apegmsh").exists()


def test_a_broken_sidecar_cannot_change_the_run_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The solve's success and its failure both survive a dead writer."""
    root = _habitat(tmp_path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("apeGmsh.studio._progress.atomic_write_text", _boom)

    _run_fake_solver(root, tmp_path, monkeypatch, exit_code=0)  # no raise
    # And a real failure still raises the run's own error, not the writer's.
    _run_fake_solver(root, tmp_path, monkeypatch, exit_code=2)


def test_an_unimportable_habitat_cannot_stop_a_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_open_progress`` guards the deferred import itself.

    ``opensees`` is usable without ``studio`` today; that must stay true
    even though the tee now reaches for it.
    """
    root = _habitat(tmp_path)
    import apeGmsh.studio._progress as progress_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("habitat unavailable")

    monkeypatch.setattr(progress_mod, "ProgressSidecar", _boom)

    _run_fake_solver(root, tmp_path, monkeypatch, exit_code=0)

    assert not (root / DEFAULT_PROGRESS_REL).exists()
