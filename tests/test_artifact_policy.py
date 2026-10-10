"""V2d part 4a (#1307): the D1 overwrite policy, rules P1 to P4.

The maintainer's ruling of 2026-10-04 on #1307 is the spec.  Oracles:

* **P1.** ``FEMData.model_name`` is the session's name, round-trips
  through ``/meta/model_name``, is inherited by derived copies, defaults
  on an old pickle, and is read by no hash.
* **P2.** With no ``model_name`` the session is named after the
  ``__main__`` script; with no real script nothing is written
  automatically and exactly one warning says so.  An explicit name wins;
  an empty one is refused; ``save_to=<file>`` needs no name.
* **P3.** A file from another run is replaced when its ``/provenance``
  names this run's script or no script, and kept (one warning) otherwise;
  ``save_to`` is exempt.  Only MPI rank 0 writes; a run whose mesh the
  kernel partitioned gets no automatic write and one warning, while a
  composed session writes.
* **P4.** A file from this run holding a zone the write would drop is
  kept silently when its neutral content is what the write would
  produce (``content_hash``: mesh, groups, labels, loads, masses and
  constraints), and with one "stale" warning otherwise; another run's
  such file keeps V2b's warning.  The hash is stable across a reload and
  deterministic.
* A refused model write skips the geometry sibling too, in one warning
  naming both files.

The happy paths run under ``simplefilter("error")``: no warning may fire
where the rule says silence.
"""
from __future__ import annotations

import copy
import os
import pickle
import sys
import types
import warnings
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh
from apeGmsh._artifact_policy import (
    artifact_content_hash,
    artifact_identity,
    artifact_target_is_ours,
    artifact_verdict,
    content_hash,
    main_script,
    mpi_rank,
    provenance_scripts,
)
from apeGmsh._internal import provenance as prov
from apeGmsh._internal.provenance import FileRow, ProvenanceTable
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees._internal.schema_version import NEUTRAL, PROVENANCE

MODEL_ZONES = frozenset({NEUTRAL, PROVENANCE})
_REAL_TABLE_FOR = prov.table_for


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _small_box(g: apeGmsh) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(3)


def _meta(path: Path, key: str) -> str:
    with h5py.File(path, "r") as f:
        return str(f["meta"].attrs[key])


def _fake_main(monkeypatch, script: Path) -> None:
    """Make ``script`` (written as a real file) Python's ``__main__``."""
    script.write_text("# a model script\n")
    main = types.ModuleType("__main__")
    main.__file__ = str(script)
    monkeypatch.setitem(sys.modules, "__main__", main)


def _name_script(monkeypatch, script: Path) -> None:
    """Make every snapshot's provenance name ``script`` as the run's
    script, the row a ``python script.py`` run writes by itself.  The
    declarations in a test have no user ``__main__`` frame around them,
    so without this a snapshot names no script."""

    def patched(session):
        table = _REAL_TABLE_FOR(session)
        row = FileRow(script.resolve().as_posix(), "", "script")
        return ProvenanceTable(table.files + (row,), table.sites, table.records)

    monkeypatch.setattr(prov, "table_for", patched)


def _run(name: str | None = None, **kw) -> apeGmsh:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name=name, **kw) as g:
            _small_box(g)
    return g


def _messages(records) -> list[str]:
    return [str(r.message) for r in records]


# ---------------------------------------------------------------------------
# P1: FEMData.model_name
# ---------------------------------------------------------------------------


def test_model_name_is_stamped_and_round_trips(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="p1") as g:
        _small_box(g)
        fem = g.mesh.queries.get_fem_data()
    assert fem.model_name == "p1"
    assert _meta(tmp_path / "p1.h5", "model_name") == "p1"
    # the broker's own writer defaults to the snapshot's name
    out = tmp_path / "copy.h5"
    fem.to_h5(str(out))
    assert _meta(out, "model_name") == "p1"
    assert FEMData.from_h5(str(out)).model_name == "p1"
    # an explicit name still wins on the writer
    fem.to_h5(str(out), model_name="other")
    assert FEMData.from_h5(str(out)).model_name == "other"


def test_model_name_is_inherited_unhashed_and_defaults_on_an_old_pickle(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="p1") as g:
        _small_box(g)
        fem = g.mesh.queries.get_fem_data()
    derived = fem._replaced(nodes=fem.nodes)
    assert derived.model_name == "p1"
    renamed = copy.copy(fem)
    del renamed._snapshot_id_cache
    renamed.model_name = "renamed"
    assert renamed.snapshot_id == fem.snapshot_id
    state = pickle.loads(pickle.dumps(fem)).__dict__
    del state["model_name"]
    old = FEMData.__new__(FEMData)
    old.__setstate__(state)
    assert old.model_name == ""


def test_unnamed_snapshot_has_no_model_name(monkeypatch, tmp_path: Path) -> None:
    """Under pytest ``__main__`` is a launcher, not a script: no name."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    assert main_script() is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with apeGmsh() as g:
            _small_box(g)
            fem = g.mesh.queries.get_fem_data()
    assert g.name == "" and fem.model_name == ""


# ---------------------------------------------------------------------------
# P2: the default name
# ---------------------------------------------------------------------------


def test_default_name_is_the_script_stem(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    _fake_main(monkeypatch, tmp_path / "frame_model.py")
    assert main_script() == (tmp_path / "frame_model.py").resolve()
    g = _run()
    assert g.name == "frame_model"
    assert (tmp_path / "frame_model.h5").is_file()
    assert (tmp_path / "frame_model.geometry.h5").is_file()
    assert _meta(tmp_path / "frame_model.h5", "model_name") == "frame_model"


def test_main_script_ignores_launchers_and_pseudo_files(monkeypatch, tmp_path: Path) -> None:
    main = types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert main_script() is None                      # a REPL: no __file__
    main.__file__ = "<stdin>"
    assert main_script() is None                      # not a real file
    main.__file__ = str(tmp_path / "gone.py")
    assert main_script() is None                      # not a real file
    main.__file__ = os.__file__
    assert main_script() is None                      # the stdlib
    import pytest as _pytest
    main.__file__ = _pytest.__file__
    assert main_script() is None                      # site-packages


def test_a_console_script_launcher_gives_no_default_name(monkeypatch, tmp_path: Path) -> None:
    """Linux CI runs ``/opt/.../bin/pytest``, a console script in the
    environment's scripts directory, as ``__main__``: a launcher, never the
    user's script, so there is no default name, no write and one warning.
    Simulated so the test does not depend on how pytest itself was launched."""
    import sysconfig

    launcher = os.path.join(sysconfig.get_paths()["scripts"], "pytest")
    assert prov._classify(launcher) == prov._THIRD_PARTY
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda p: p == launcher or real_isfile(p))
    main = types.ModuleType("__main__")
    main.__file__ = launcher
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    assert main_script() is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh() as g:
            _small_box(g)
    assert g.name == ""
    assert len(w) == 1 and "model_name" in str(w[0].message), _messages(w)
    assert list(tmp_path.iterdir()) == []


def test_explicit_name_wins_over_the_script(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    _fake_main(monkeypatch, tmp_path / "frame_model.py")
    g = _run("explicit")
    assert g.name == "explicit"
    assert sorted(p.name for p in tmp_path.glob("*.h5")) == [
        "explicit.geometry.h5", "explicit.h5",
    ]


def test_no_script_and_no_name_writes_nothing_and_warns_once(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    assert main_script() is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh() as g:
            _small_box(g)
    msgs = _messages(w)
    assert len(msgs) == 1, msgs
    assert "model_name" in msgs[0] and "nothing is written" in msgs[0]
    assert list(tmp_path.iterdir()) == []


def test_empty_model_name_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        apeGmsh(model_name="")


def test_no_name_with_save_to_file_writes_without_a_name(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path / "unused"))
    target = tmp_path / "named.h5"
    _run(save_to=target)
    assert target.is_file() and (tmp_path / "named.geometry.h5").is_file()
    assert _meta(target, "model_name") == ""
    assert not (tmp_path / "unused").exists()


def test_no_name_with_save_to_directory_warns_once_and_writes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(save_to=out) as g:
            _small_box(g)
    assert len(w) == 1 and "model_name" in str(w[0].message)
    assert list(out.iterdir()) == []


def test_a_name_that_is_not_a_file_name_leaves_no_partial_state(
    monkeypatch, tmp_path: Path
) -> None:
    """A refusal or failure never leaves a temp file or a half file."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="missing_dir/model") as g:
            _small_box(g)
    msgs = _messages(w)
    assert len(msgs) == 2, msgs
    assert any("autosave" in m and "failed" in m for m in msgs)
    assert any("not written" in m for m in msgs)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# P3: a file from another run
# ---------------------------------------------------------------------------


def test_same_script_replaces_and_another_script_is_kept(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    model = tmp_path / "frame.h5"
    _name_script(monkeypatch, tmp_path / "a.py")
    _run("frame")
    first = _meta(model, "session_id")
    assert artifact_identity(model)[1] == {
        os.path.normcase(str((tmp_path / "a.py").resolve()))
    }
    _run("frame")                                   # same script, another run
    second = _meta(model, "session_id")
    assert second != first
    sibling = tmp_path / "frame.geometry.h5"
    before = (model.read_bytes(), sibling.read_bytes())
    _name_script(monkeypatch, tmp_path / "b.py")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="frame") as g:
            _small_box(g)
    msgs = _messages(w)
    assert len(msgs) == 1, msgs
    assert "another script" in msgs[0] and "a.py" in msgs[0] and "model_name=" in msgs[0]
    # the refusal names the sibling and skips it, so the pair stays paired
    assert str(sibling) in msgs[0] and "pair stays consistent" in msgs[0]
    assert (model.read_bytes(), sibling.read_bytes()) == before
    assert _meta(sibling, "session_id") == second
    assert not list(tmp_path.glob("*.tmp-*"))


def test_a_file_naming_no_script_or_without_provenance_is_replaced(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    model = tmp_path / "frame.h5"
    _run("frame")                                   # a test run: no script row
    assert artifact_identity(model)[1] == frozenset()
    _name_script(monkeypatch, tmp_path / "a.py")
    first = _meta(model, "session_id")
    _run("frame")
    assert _meta(model, "session_id") != first
    assert artifact_identity(model)[1] == {
        os.path.normcase(str((tmp_path / "a.py").resolve()))
    }
    # no /provenance at all (a snapshot without a table), then a run
    # from a script: replaced, and the file now names that script
    monkeypatch.setattr(prov, "table_for", lambda session: None)
    _run("noprov")
    noprov = tmp_path / "noprov.h5"
    with h5py.File(noprov, "r") as f:
        assert "provenance" not in f
    second = _meta(noprov, "session_id")
    _name_script(monkeypatch, tmp_path / "b.py")
    _run("noprov")
    assert _meta(noprov, "session_id") != second
    assert artifact_identity(noprov)[1] == {
        os.path.normcase(str((tmp_path / "b.py").resolve()))
    }


def test_a_notebook_cell_names_no_script(monkeypatch, tmp_path: Path) -> None:
    """A cell's path carries the kernel's pid (``ipykernel_<pid>``), so an
    unchanged cell re-run after a kernel restart must still replace the
    file: the cell names no script on either side, silently."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    model = tmp_path / "nb.h5"
    cell_a = tmp_path / "ipykernel_1111" / "a1b2c3d4.py"
    cell_b = tmp_path / "ipykernel_2222" / "a1b2c3d4.py"
    for cell in (cell_a, cell_b):
        cell.parent.mkdir()
    _name_script(monkeypatch, cell_a)
    _run("nb")
    first = _meta(model, "session_id")
    assert artifact_identity(model)[1] == frozenset()
    with h5py.File(model, "r") as f:                # the row is written, unkeyed
        kinds = f["provenance/files/kind"].asstr()[()].tolist()
        assert "script" in kinds
    _name_script(monkeypatch, cell_b)               # a new kernel, the same cell
    _run("nb")
    assert _meta(model, "session_id") != first
    table = ProvenanceTable(files=(
        FileRow(cell_a.resolve().as_posix(), "", "script"),
        FileRow("<ipython-input-3-9f8e>", "", "script"),
        FileRow((tmp_path / "run.py").resolve().as_posix(), "", "script"),
    ))
    assert provenance_scripts(table) == {
        os.path.normcase(str((tmp_path / "run.py").resolve())),
    }


def test_save_to_is_exempt_from_the_script_rule(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "x.h5"
    _name_script(monkeypatch, tmp_path / "a.py")
    _run("frame", save_to=target)
    first = _meta(target, "session_id")
    _name_script(monkeypatch, tmp_path / "b.py")
    _run("frame", save_to=target)
    assert _meta(target, "session_id") != first


def test_two_sessions_in_one_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    _run("a")
    _run("b")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "a.geometry.h5", "a.h5", "b.geometry.h5", "b.h5",
    ]
    assert _meta(tmp_path / "a.h5", "session_id") != _meta(tmp_path / "b.h5", "session_id")


@pytest.mark.parametrize("var", ["OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"])
def test_only_mpi_rank_zero_writes(monkeypatch, tmp_path: Path, var: str) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv(var, "1")
    assert mpi_rank() == 1
    _run("mpi")                                     # silent, nothing written
    assert list(tmp_path.iterdir()) == []
    # an explicit save_to= is the user's intent: every rank that asks writes
    _run("mpi", save_to=tmp_path / "rank1.h5")
    assert (tmp_path / "rank1.h5").is_file() and (tmp_path / "rank1.geometry.h5").is_file()
    assert not (tmp_path / "mpi.h5").exists()
    monkeypatch.setenv(var, "0")
    assert mpi_rank() == 0
    _run("mpi")
    assert (tmp_path / "mpi.h5").is_file()


def test_mpi_rank_outside_mpi_and_malformed(monkeypatch, tmp_path: Path) -> None:
    from apeGmsh._artifact_policy import MPI_RANK_ENV

    for var in MPI_RANK_ENV:
        monkeypatch.delenv(var, raising=False)
    assert mpi_rank() is None
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv("PMI_RANK", "rank0")
    with pytest.raises(RuntimeError, match="not an integer MPI rank"):
        mpi_rank()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="mpi") as g:
            _small_box(g)
    assert len(w) == 1 and "not an integer MPI rank" in str(w[0].message)
    assert list(tmp_path.iterdir()) == []


def test_partitioned_run_gets_no_automatic_write(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="parts") as g:
            _small_box(g)
            g.mesh.partitioning.partition(2)
            assert len(g.mesh.queries.get_fem_data().partitions) == 2
    assert len(w) == 1, _messages(w)
    assert "partitioned run (2 partitions" in str(w[0].message)
    assert "parts.h5" in str(w[0].message) and "parts.geometry.h5" in str(w[0].message)
    assert list(tmp_path.iterdir()) == []
    # save_to= is the user's intent and still writes
    target = tmp_path / "parts.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="parts", save_to=target) as g:
            _small_box(g)
            g.mesh.partitioning.partition(2)
    assert target.is_file()
    assert len(FEMData.from_h5(str(target)).partitions) == 2


def test_a_composed_model_writes(monkeypatch, tmp_path: Path) -> None:
    """A composed model reports its modules as partitions (ADR 0038's
    rank model) but is not partitioned for D1: it writes, silently.

    v1 composed inside a session (``g.compose``) and checked the session's
    automatic write; v2 has no composed session (the row-15 ruling), so
    the composed model is an assembly and its write is ``asm.h5``.
    """
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees import apeSees

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="module") as g:
            _small_box(g)
            fem = g.mesh.queries.get_fem_data()
        src = apeSees(fem)
        src.model(ndm=3, ndf=3)
        mat = src.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0, name="m")
        src.element.FourNodeTetrahedron(pg="body", material=mat)
        source = tmp_path / "module_src.h5"
        src.h5(str(source))

        asm = (Assembly("host")
               .instance("H", source, partition_rank=0)
               .instance("M", source, translate=(3.0, 0.0, 0.0),
                         partition_rank=1))
        ops = asm.bridge(ndm=3, ndf=3)
        ops.numberer.ParallelPlain()     # declared: the ranked deck's
        ops.system.Mumps()               # auto-emit would warn
        model = tmp_path / "host.h5"
        asm.h5(model)
        # The D1 gate: a session holding the composed model reports its
        # modules as partitions, and its automatic write still happens.
        with apeGmsh.from_h5(model, model_name="reopened") as g:
            fem = g.mesh.queries.get_fem_data()
            assert len(fem.partitions) == 2          # one per module rank
    assert len(ops.fem.partitions) == 2
    reopened = tmp_path / "reopened.h5"
    assert reopened.is_file()
    reloaded = FEMData.from_h5(str(reopened))
    assert len(reloaded.partitions) == 2
    assert sorted(reloaded.composed_from.labels) == ["H", "M"]
    assert reloaded.session_id == fem.session_id


# ---------------------------------------------------------------------------
# P4: a file from this run
# ---------------------------------------------------------------------------


def _bridge_write(g: apeGmsh, target: Path, fem: FEMData | None = None) -> None:
    from apeGmsh.opensees import apeSees

    if fem is None:
        fem = g.mesh.queries.get_fem_data()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="body", material=mat)
    ops.h5(str(target))


def test_this_runs_fuller_file_is_kept_silently(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    target = tmp_path / "bridge.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="bridge") as g:
            _small_box(g)
            _bridge_write(g, target)
            before = target.read_bytes()
    assert target.read_bytes() == before
    with h5py.File(target, "r") as f:
        assert "opensees" in f
    sibling = tmp_path / "bridge.geometry.h5"
    assert sibling.is_file()
    assert _meta(sibling, "session_id") == _meta(target, "session_id")
    assert not list(tmp_path.glob("*.tmp-*"))


def test_this_runs_fuller_file_from_an_earlier_snapshot_warns_stale(
    monkeypatch, tmp_path: Path
) -> None:
    """The bridge wrote a snapshot of this session whose mesh differs
    from the one ``end()`` holds (the ``dim=3`` slice the bridge is often
    fed, against the session's full extraction): one "stale" warning
    naming both files, the bridge's file kept, no sibling written."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    target = tmp_path / "bridge.h5"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="bridge") as g:
            _small_box(g)
            full = g.mesh.queries.get_fem_data()
            solids = g.mesh.queries.get_fem_data(dim=3)
            assert solids.session_id == full.session_id
            _bridge_write(g, target, fem=solids)
            before = target.read_bytes()
            assert artifact_content_hash(target) == content_hash(solids) != content_hash(full)
    msgs = _messages(w)
    assert len(msgs) == 1, msgs
    assert (
        "holds different content than this session would write (a filtered "
        "get_fem_data, or a change after it was written)" in msgs[0]
    )
    assert "not replaced" in msgs[0] and "opensees" in msgs[0]
    assert "bridge.geometry.h5" in msgs[0] and "pair stays consistent" in msgs[0]
    assert target.read_bytes() == before
    assert not (tmp_path / "bridge.geometry.h5").exists()


@pytest.mark.parametrize("declare", ["load", "mass"])
def test_a_declaration_after_the_write_warns_stale(
    monkeypatch, tmp_path: Path, declare: str
) -> None:
    """P4 compares everything the write would put in the file: a load or
    a mass declared after the bridge's write (``snapshot_id`` is blind
    to both) makes the bridge's file stale: one warning, nothing
    written, the pair untouched."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    target = tmp_path / "bridge.h5"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="bridge") as g:
            _small_box(g)
            _bridge_write(g, target)
            before = target.read_bytes()
            written = artifact_content_hash(target)
            if declare == "load":
                g.loads.point.force(pg="body", force=(0.0, 0.0, -1.0))
            else:
                g.masses.point(pg="body", mass=2.5)
            fem = g.mesh.queries.get_fem_data()
            assert fem.snapshot_id == _meta(target, "snapshot_id")
            assert content_hash(fem) != written
    msgs = _messages(w)
    assert len(msgs) == 1 and "different content" in msgs[0], msgs
    assert target.read_bytes() == before
    assert not (tmp_path / "bridge.geometry.h5").exists()


def test_content_hash_is_stable_across_reload_and_deterministic(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    with apeGmsh(model_name="hash") as g:
        _small_box(g)
        g.loads.point.force(pg="body", force=(1.0, 2.0, 3.0))
        g.masses.point(pg="body", mass=0.1)
        fem = g.mesh.queries.get_fem_data()
    model = tmp_path / "hash.h5"
    h = content_hash(fem)
    assert h == content_hash(fem) == artifact_content_hash(model)
    assert h == content_hash(FEMData.from_h5(str(model)))
    assert h == content_hash(pickle.loads(pickle.dumps(fem)))
    copy_path = tmp_path / "copy.h5"
    fem.to_h5(str(copy_path), model_name="other")   # /meta differs, content equal
    assert artifact_content_hash(copy_path) == h
    assert _meta(copy_path, "model_name") != _meta(model, "model_name")


def test_another_runs_fuller_file_keeps_v2b_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    target = tmp_path / "bridge.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with apeGmsh(model_name="bridge") as g:
            _small_box(g)
            _bridge_write(g, target)
    before = target.read_bytes()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with apeGmsh(model_name="bridge") as g:
            _small_box(g)
    msgs = _messages(w)
    assert len(msgs) == 1 and "would drop" in msgs[0] and "opensees" in msgs[0], msgs
    assert target.read_bytes() == before


def test_reload_then_write_again(monkeypatch, tmp_path: Path) -> None:
    """A reloaded snapshot keeps its name and id; writing it back is the
    user's explicit save, and a session replay of it writes nothing."""
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    model = tmp_path / "again.h5"
    _run("again")
    sid, name = _meta(model, "session_id"), _meta(model, "model_name")
    fem = FEMData.from_h5(str(model))
    assert (fem.session_id, fem.model_name) == (sid, "again")
    fem.to_h5(str(model))
    assert (_meta(model, "session_id"), _meta(model, "model_name")) == (sid, name)
    before = sorted(p.name for p in tmp_path.iterdir())
    g2 = apeGmsh.from_h5(model)
    g2.end()                                        # never began: nothing written
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# The rule as a function
# ---------------------------------------------------------------------------


def test_rule_on_a_missing_target_and_overwrite_false(tmp_path: Path) -> None:
    kw = dict(writes=MODEL_ZONES, session_id="s", content=lambda: "h",
              scripts=frozenset(), explicit=False)
    assert artifact_target_is_ours(tmp_path / "none.h5", overwrite=True, **kw)
    p = tmp_path / "x.h5"
    p.write_bytes(b"x")
    with pytest.warns(UserWarning, match="overwrite=False"):
        assert not artifact_target_is_ours(p, overwrite=False, **kw)
    with pytest.warns(UserWarning, match="not an apeGmsh artifact") as rec:
        assert not artifact_target_is_ours(
            p, overwrite=True, skips=(tmp_path / "x.geometry.h5",), **kw)
    assert len(rec) == 1 and "x.geometry.h5" in str(rec[0].message)


def test_rule_decision_table(tmp_path: Path) -> None:
    """Every row of the ruling, on hand-made artifacts."""
    from tests.fixtures.schema import NEUTRAL_CURRENT

    import numpy as np

    def make(name, *, sid, hash_, groups, scripts=()):
        """``hash_`` is the neutral content: a dataset under ``/nodes``."""
        p = tmp_path / name
        with h5py.File(p, "w") as f:
            m = f.create_group("meta")
            m.attrs["neutral_schema_version"] = NEUTRAL_CURRENT
            m.attrs["session_id"] = sid
            for g_ in groups:
                f.create_group(g_)
            f["nodes"].create_dataset("ids", data=np.frombuffer(hash_.encode(), dtype=np.uint8))
            if scripts:
                pr = f.require_group("provenance")
                pr.attrs["base_dir"] = tmp_path.as_posix()
                files = pr.create_group("files")
                dt = h5py.string_dtype("utf-8")
                files.create_dataset("path", data=list(scripts), dtype=dt)
                files.create_dataset("kind", data=["script"] * len(scripts), dtype=dt)
                files.create_dataset("sha256", data=[""] * len(scripts), dtype=dt)
        return p

    same = make("same.h5", sid="S", hash_="H", groups=("nodes",))
    ours = dict(session_id="S", content=lambda: artifact_content_hash(same),
                scripts=frozenset(), explicit=False)
    neutral = frozenset({NEUTRAL})
    # this run, same zones: refreshed
    assert artifact_target_is_ours(
        make("a.h5", sid="S", hash_="H", groups=("nodes",)),
        writes=neutral, overwrite=True, **ours)
    # this run, fuller, same content: kept silently
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert artifact_verdict(
            make("b.h5", sid="S", hash_="H", groups=("nodes", "opensees")),
            writes=neutral, overwrite=True, **ours) == "keep"
    # this run, fuller, content changed since: refused
    with pytest.warns(UserWarning, match="different content"):
        assert artifact_verdict(
            make("c.h5", sid="S", hash_="OLD", groups=("nodes", "opensees")),
            writes=neutral, overwrite=True, **ours) == "refuse"
    # another run, fuller: V2b
    with pytest.warns(UserWarning, match="would drop"):
        assert not artifact_target_is_ours(
            make("d.h5", sid="T", hash_="H", groups=("nodes", "opensees")),
            writes=neutral, overwrite=True, **ours)
    # another run, same zones, no script named: replaced
    assert artifact_target_is_ours(
        make("e.h5", sid="T", hash_="X", groups=("nodes",)),
        writes=neutral, overwrite=True, **ours)
    # another run naming a script this run does not: kept
    key = os.path.normcase(str((tmp_path / "a.py").resolve()))
    f = make("f.h5", sid="T", hash_="X", groups=("nodes",), scripts=("a.py",))
    assert artifact_identity(f) == ("T", frozenset({key}))
    with pytest.warns(UserWarning, match="another script"):
        assert not artifact_target_is_ours(
            f, writes=MODEL_ZONES, overwrite=True, **ours)
    # ... unless this run names it, or the target was named by the user
    assert artifact_target_is_ours(
        f, writes=MODEL_ZONES, overwrite=True,
        **{**ours, "scripts": frozenset({key})})
    assert artifact_target_is_ours(
        f, writes=MODEL_ZONES, overwrite=True, **{**ours, "explicit": True})
    # a pseudo-file script is compared as written
    g_ = make("g.h5", sid="T", hash_="X", groups=("nodes",), scripts=("<string>",))
    assert artifact_identity(g_)[1] == {"<string>"}
    assert artifact_target_is_ours(
        g_, writes=MODEL_ZONES, overwrite=True,
        **{**ours, "scripts": frozenset({"<string>"})})


def test_provenance_scripts_keys_only_script_rows(tmp_path: Path) -> None:
    assert provenance_scripts(None) == frozenset()
    table = ProvenanceTable(files=(
        FileRow((tmp_path / "run.py").resolve().as_posix(), "", "script"),
        FileRow((tmp_path / "lib.py").resolve().as_posix(), "", "module"),
        FileRow("<string>", "", "script"),
    ))
    assert provenance_scripts(table) == {
        os.path.normcase(str((tmp_path / "run.py").resolve())), "<string>",
    }
