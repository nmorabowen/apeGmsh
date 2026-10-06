"""V2d-4b (#1307): the bridge's automatic ``model.h5`` write.

Oracles:

* **No ``.h5()`` call leaves the pair.**  A user script that builds a
  session with no ``model_name`` and emits through ``apeSees`` with no
  ``ops.h5()`` leaves ``<stem>.h5`` holding the neutral zone,
  ``/opensees`` and ``/provenance``, stamped with the snapshot's
  ``session_id``, beside the session's ``<stem>.geometry.h5`` that
  carries the same id.  A ``mass_from_model()`` model passes this path.
* **P5, one write per model state.**  A loop of emits on one model
  writes once; adding a primitive, or a record that is not a primitive
  (a ``fix``), writes again.  Counted by spying ``apeSees.h5``, the one
  composition path the write goes through.
* **Reload then emit** goes through ``artifact_verdict``: a source file
  outside the artifact directory is untouched; the file at the target
  is replaced only when the policy says so (same ``session_id``, no
  dropped zone), and a file another script wrote is refused with one
  warning and left byte-identical.
* **Warn once per bridge, never raise.**  No name (P2), a partitioned
  snapshot (P3), a refused target and a failing write each warn once
  over three emits, and the emit's return value is unchanged.  An MPI
  rank other than 0 is silent.  A stub snapshot and ``_artifacts=False``
  write nothing and warn nothing.
* **P6.**  The library's own constructors (ETABS, STKO, ``strut_tie``)
  pass ``_artifacts=False``.

Every test here passes under ``-W error::UserWarning``: the warnings the
hook issues are caught where they are expected.
"""
from __future__ import annotations

import re
import runpy
import warnings
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.apesees import OpenSeesAutoEmitWarning
from apeGmsh.opensees._internal.artifact_write import (
    BridgeArtifactWarning,
    BridgeArtifactWriter,
)
from tests.opensees.fixtures.fem_stub import make_two_node_beam

SRC = Path(__file__).resolve().parents[3] / "src" / "apeGmsh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_fem(name: str | None, *, partitions: int = 1) -> FEMData:
    """A one-box tet mesh from a session that writes no artifacts of its
    own (``_artifacts=False``), named ``name`` (``None``: the session's
    default, which under pytest is no name at all)."""
    kw = {} if name is None else {"model_name": name}
    with apeGmsh(verbose=False, _artifacts=False, **kw) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        g.physical.add_volume("b", name="B")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        if partitions > 1:
            g.mesh.partitioning.partition(partitions)
        return g.mesh.queries.get_fem_data(dim=3)


def _bridge(fem, **kw) -> apeSees:
    ops = apeSees(fem, **kw)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    return ops


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch) -> Path:
    """This test's own conventional artifact directory (decks go to
    ``tmp_path`` itself, so the directory holds only automatic writes)."""
    out = tmp_path / "artifacts"
    out.mkdir()
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(out))
    return out


@pytest.fixture
def h5_calls(monkeypatch) -> list[str]:
    """Every ``apeSees.h5`` call (the automatic write's composition path),
    as the path written."""
    calls: list[str] = []
    real = apeSees.h5

    def spy(self, path, **kw):
        calls.append(str(path))
        return real(self, path, **kw)

    monkeypatch.setattr(apeSees, "h5", spy)
    return calls


def _zones(path: Path) -> set[str]:
    with h5py.File(path, "r") as f:
        return set(f.keys())


def _meta(path: Path, key: str) -> str:
    with h5py.File(path, "r") as f:
        raw = f["meta"].attrs[key]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _fem_hash(path: Path) -> str:
    with h5py.File(path, "r") as f:
        raw = f["meta/lineage"].attrs["fem_hash"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _bridge_warnings(record) -> list[str]:
    return [str(w.message) for w in record
            if issubclass(w.category, BridgeArtifactWarning)]


# ---------------------------------------------------------------------------
# The oracle: a run with no .h5() call leaves the pair
# ---------------------------------------------------------------------------

USER_SCRIPT = '''\
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(verbose=False) as g:  # no model_name: the script's stem
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
    g.physical.add_volume("b", name="B")
    g.masses.volume("B", density=2400.0)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
ops.element.FourNodeTetrahedron(pg="B", material=mat)
ops.mass_from_model()
ops.tcl(TCL)
'''


def test_script_with_no_h5_call_leaves_the_model_and_geometry_pair(
    tmp_path, monkeypatch,
):
    """The D1 contract for the bridge: ``python frame_auto.py`` leaves
    ``frame_auto.h5`` (neutral + /opensees + /provenance, this run's
    ``session_id``) beside ``frame_auto.geometry.h5`` with the same id,
    and the user never named a file."""
    out = tmp_path / "artifacts"
    out.mkdir()
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(out))
    script = tmp_path / "frame_auto.py"
    script.write_text(USER_SCRIPT, encoding="utf-8")
    deck = tmp_path / "frame_auto.tcl"

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        glb = runpy.run_path(str(script), init_globals={"TCL": str(deck)},
                             run_name="__main__")

    model = out / "frame_auto.h5"
    sibling = out / "frame_auto.geometry.h5"
    assert deck.exists()
    assert sorted(p.name for p in out.iterdir()) == [
        "frame_auto.geometry.h5", "frame_auto.h5"]
    zones = _zones(model)
    assert {"meta", "nodes", "opensees", "provenance"} <= zones
    fem = glb["fem"]
    assert _meta(model, "session_id") == fem.session_id
    assert _meta(sibling, "session_id") == fem.session_id
    assert _meta(model, "model_name") == "frame_auto"
    assert _fem_hash(model) == fem.snapshot_id
    with h5py.File(model, "r") as f:
        # mass_from_model streams the snapshot's masses into the deck and
        # the file carries them in the neutral zone.
        assert "masses" in f
    assert "mass " in deck.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P5: the dirty flag
# ---------------------------------------------------------------------------


def test_a_loop_of_deck_emits_writes_once_until_the_model_changes(
    artifact_dir, h5_calls, tmp_path,
):
    fem = _box_fem("loop")
    ops = _bridge(fem)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for i in range(100):
            assert ops.tcl(str(tmp_path / f"d{i}.tcl")) is None
        assert len(h5_calls) == 1
        assert ops.py(str(tmp_path / "d.py")) is None
        assert len(h5_calls) == 1
        # A new primitive: the second write.
        ops.uniaxialMaterial.ElasticMaterial(E=1.0)
        ops.tcl(str(tmp_path / "e.tcl"))
        assert len(h5_calls) == 2
        # A record that is not a primitive (the gap K1-3d flagged): the third.
        ops.fix(pg="B", dofs=(1, 1, 1))
        ops.tcl(str(tmp_path / "f.tcl"))
        ops.tcl(str(tmp_path / "f2.tcl"))
        assert len(h5_calls) == 3
    target = artifact_dir / "loop.h5"
    assert all(Path(c).parent == artifact_dir for c in h5_calls)
    assert {"opensees", "nodes", "provenance"} <= _zones(target)
    assert _meta(target, "session_id") == fem.session_id
    assert not list(artifact_dir.glob("*.tmp-*"))


@pytest.mark.live
def test_a_loop_of_live_builds_writes_once_until_the_model_changes(
    artifact_dir, h5_calls,
):
    pytest.importorskip("openseespy.opensees")
    with apeGmsh(model_name="live_loop", verbose=False, _artifacts=False) as g:
        G = g.model.geometry
        a = G.add_point(0.0, 0.0, 0.0)
        b = G.add_point(2.0, 0.0, 0.0)
        bar = G.add_line(a, b)
        g.model.sync()
        g.physical.add(1, [bar], name="Bar")
        g.physical.add(0, [a], name="A")
        g.physical.add(0, [b], name="B")
        g.mesh.structured.set_transfinite_curve(bar, 2)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="Bar", A=1e-3, material=steel)
    ops.fix(pg="A", dofs=(1, 1, 1))
    ops.fix(pg="B", dofs=(0, 1, 1))
    ops.mass(pg="B", values=(10.0, 10.0, 10.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="B", forces=(1000.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for _ in range(100):
            assert ops.analyze(steps=1) == 0
        assert len(h5_calls) == 1
        ops.run()
        assert len(h5_calls) == 1
        ops.uniaxialMaterial.ElasticMaterial(E=1.0)
        ops.run()
        assert len(h5_calls) == 2
    assert {"opensees", "nodes"} <= _zones(artifact_dir / "live_loop.h5")


# ---------------------------------------------------------------------------
# Reload then emit: through the verdict
# ---------------------------------------------------------------------------


@pytest.fixture
def verdicts(monkeypatch) -> list[Path]:
    """Every target ``artifact_verdict`` was asked about."""
    import apeGmsh._artifact_policy as policy

    asked: list[Path] = []
    real = policy.artifact_verdict

    def spy(target, **kw):
        asked.append(Path(target))
        return real(target, **kw)

    monkeypatch.setattr(policy, "artifact_verdict", spy)
    return asked


def test_reload_from_outside_the_artifact_dir_leaves_the_source_alone(
    artifact_dir, tmp_path, verdicts,
):
    src = tmp_path / "elsewhere" / "reloaded.h5"
    src.parent.mkdir()
    fem0 = _box_fem("reloaded")
    fem0.to_h5(str(src))
    before = src.read_bytes()
    fem1 = FEMData.from_h5(str(src))
    assert fem1.model_name == "reloaded"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _bridge(fem1).tcl(str(tmp_path / "r.tcl"))
    assert src.read_bytes() == before
    assert verdicts == [artifact_dir / "reloaded.h5"]
    out = artifact_dir / "reloaded.h5"
    assert {"opensees", "nodes"} <= _zones(out)
    assert _meta(out, "session_id") == fem0.session_id


def test_reload_from_the_target_itself_is_this_runs_file_and_is_enriched(
    artifact_dir, tmp_path, verdicts,
):
    """The "script 2 analyses" flow: the source is the artifact at the
    conventional path (same ``session_id``, no zone dropped), so P4 says
    write.  The neutral content is unchanged (``fem_hash`` equal) and
    the file gains ``/opensees``."""
    fem0 = _box_fem("same")
    target = artifact_dir / "same.h5"
    fem0.to_h5(str(target))
    fem_hash_before = _fem_hash(target)
    fem1 = FEMData.from_h5(str(target))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _bridge(fem1).tcl(str(tmp_path / "s.tcl"))
    assert verdicts == [target]
    assert "opensees" in _zones(target)
    assert _fem_hash(target) == fem_hash_before
    assert _meta(target, "session_id") == fem0.session_id


OTHER_SCRIPT = '''\
from apeGmsh import apeGmsh

with apeGmsh(model_name="twin", verbose=False) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
    g.physical.add_volume("b", name="B")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
'''


def test_a_file_another_script_wrote_is_refused_once_and_kept(
    artifact_dir, tmp_path, verdicts, h5_calls,
):
    """P3 through the verdict: the target names another script in its
    ``/provenance``; this bridge's snapshot names none.  One warning
    over three emits, the file byte-identical, nothing else written."""
    script = tmp_path / "other_script.py"
    script.write_text(OTHER_SCRIPT, encoding="utf-8")
    runpy.run_path(str(script), run_name="__main__")
    # The session's own end() asked the verdict for its two files.
    asked_by_session = len(verdicts)
    target = artifact_dir / "twin.h5"
    before = target.read_bytes()
    fem = _box_fem("twin")
    ops = _bridge(fem)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for i in range(3):
            ops.tcl(str(tmp_path / f"t{i}.tcl"))
    refusals = [str(w.message) for w in rec
                if issubclass(w.category, UserWarning)
                and "another script" in str(w.message)]
    assert len(refusals) == 1
    assert "other_script.py" in refusals[0]
    assert verdicts[asked_by_session:] == [target]
    assert h5_calls == []
    assert target.read_bytes() == before
    assert sorted(p.name for p in artifact_dir.iterdir()) == [
        "twin.geometry.h5", "twin.h5"]


# ---------------------------------------------------------------------------
# Warn once per bridge, never raise
# ---------------------------------------------------------------------------


def test_no_name_session_snapshot_is_silent_and_writes_nothing(
    artifact_dir, tmp_path, h5_calls,
):
    """P2 is one warning: under pytest the session has no script and no
    name, and its ``end()`` is where that warning is given (here the
    session opted out, so none at all); the bridge stays silent and
    writes nothing.  No Windows-only path logic: the same holds on
    Linux CI."""
    fem = _box_fem(None)
    assert fem.model_name == "" and fem.provenance is not None
    ops = _bridge(fem)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for i in range(3):
            assert ops.tcl(str(tmp_path / f"n{i}.tcl")) is None
    assert h5_calls == []
    assert list(artifact_dir.glob("*.h5")) == []


def test_no_name_non_session_snapshot_warns_once_and_writes_nothing(
    artifact_dir, tmp_path, h5_calls,
):
    """A snapshot no session extracted (from_msh, an import: no
    provenance table, no name) has had no P2 warning yet: the bridge
    gives it, once per bridge over three emits, and writes nothing."""
    fem = _box_fem(None)
    fem.provenance = None
    ops = _bridge(fem)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for i in range(3):
            assert ops.tcl(str(tmp_path / f"n{i}.tcl")) is None
    msgs = _bridge_warnings(rec)
    assert len(msgs) == 1
    assert "no model name" in msgs[0]
    assert "ops.h5(path)" in msgs[0]
    assert h5_calls == []
    assert list(artifact_dir.glob("*.h5")) == []


def test_partitioned_snapshot_warns_once_and_writes_nothing(
    artifact_dir, tmp_path, h5_calls,
):
    fem = _box_fem("parts", partitions=2)
    assert len(fem.partitions) == 2
    ops = _bridge(fem)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for i in range(3):
            ops.tcl(str(tmp_path / f"p{i}.tcl"))
    msgs = _bridge_warnings(rec)
    assert len(msgs) == 1
    assert "partitioned run (2 partitions)" in msgs[0]
    assert h5_calls == []
    assert list(artifact_dir.glob("*.h5")) == []


def test_a_failing_write_warns_once_and_keeps_the_return_value(
    artifact_dir, tmp_path, monkeypatch,
):
    """Seam 9: an ``OSError`` inside the write is one warning; the deck
    is still written and the emit returns what it always returns; no
    temp file is left behind, and the bridge does not try again."""
    import apeGmsh._atomic_io as atomic

    def deny(src, dest):
        raise PermissionError(13, "read-only", str(dest))

    monkeypatch.setattr(atomic, "replace_with_retry", deny)
    fem = _box_fem("denied")
    ops = _bridge(fem)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for i in range(3):
            deck = tmp_path / f"w{i}.tcl"
            assert ops.tcl(str(deck)) is None
            assert deck.exists()
    msgs = _bridge_warnings(rec)
    assert len(msgs) == 1
    assert "model.h5 not written" in msgs[0]
    assert "PermissionError" in msgs[0]
    assert "denied.h5" in msgs[0]
    assert list(artifact_dir.iterdir()) == []


def test_an_mpi_rank_other_than_zero_is_silent(artifact_dir, tmp_path, monkeypatch):
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK",
                 "MV2_COMM_WORLD_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)
    fem = _box_fem("rank")
    ops = _bridge(fem)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        monkeypatch.setenv("PMI_RANK", "1")
        ops.tcl(str(tmp_path / "r1.tcl"))
        assert list(artifact_dir.glob("*.h5")) == []
        monkeypatch.setenv("PMI_RANK", "0")
        ops.tcl(str(tmp_path / "r0.tcl"))
    assert (artifact_dir / "rank.h5").exists()


def test_opt_out_and_stub_write_nothing_and_warn_nothing(
    artifact_dir, tmp_path, h5_calls,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ops = _bridge(_box_fem("quiet"), _artifacts=False)
        for i in range(2):
            ops.tcl(str(tmp_path / f"q{i}.tcl"))
        stub = apeSees(make_two_node_beam())  # type: ignore[arg-type]
        stub.model(ndm=3, ndf=6)
        stub.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        stub.tcl(str(tmp_path / "stub.tcl"))
    assert h5_calls == []
    assert list(artifact_dir.iterdir()) == []


def test_explicit_h5_is_not_the_hook(artifact_dir, tmp_path, h5_calls):
    """P7: ``ops.h5(path)`` writes where it is told and nothing else; it
    is the one path the spy sees."""
    ops = _bridge(_box_fem("explicit"))
    out = tmp_path / "mine.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ops.h5(str(out))
    assert h5_calls == [str(out)]
    assert out.exists()
    assert list(artifact_dir.iterdir()) == []


def test_writer_warned_flag_is_per_bridge_and_the_memo_is_per_target():
    """Each bridge keeps its own opt-out and warned flag; what was last
    written is remembered per resolved target for the whole process."""
    from apeGmsh.opensees._internal import artifact_write as aw

    a, b = BridgeArtifactWriter(), BridgeArtifactWriter(enabled=False)
    assert a.enabled and not b.enabled
    assert not a._warned and not hasattr(a, "_last_key")
    assert isinstance(aw._LAST_WRITTEN, dict)


# ---------------------------------------------------------------------------
# P5 across bridges, and the digest sees contents, not counts
# ---------------------------------------------------------------------------


def test_fresh_bridges_on_an_unchanged_model_write_once(
    artifact_dir, tmp_path, h5_calls,
):
    """One bridge per ground motion, same model: the first writes, the
    other four find their digest already at the target.  A sixth bridge
    that changes one material value (same primitive count) writes."""
    fem = _box_fem("ida")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for gm in range(5):
            ops = _bridge(fem)
            ops.tcl(str(tmp_path / f"gm{gm}.tcl"))
        assert len(h5_calls) == 1
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=31e9, nu=0.2, rho=2400.0)
        ops.element.FourNodeTetrahedron(pg="B", material=mat)
        ops.tcl(str(tmp_path / "sweep.tcl"))
        assert len(h5_calls) == 2
        # The file gone from under the memo: written again.
        (artifact_dir / "ida.h5").unlink()
        _bridge(fem).tcl(str(tmp_path / "gm_again.tcl"))
        assert len(h5_calls) == 3
    assert (artifact_dir / "ida.h5").exists()


def _load_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and "load" in name.lower():
                out[name] = tuple(obj.shape)
        f.visititems(visit)
    return out


def test_a_second_load_on_a_registered_pattern_writes_again(
    artifact_dir, tmp_path, h5_calls,
):
    """``Plain.load`` appends to the pattern's private list: no new
    primitive, no new record, but a different deck.  The digest sees
    it; the automatic file equals an explicit ``h5()`` of the same
    state (the reviewer's reproducer: 339 loads where 678 were due)."""
    fem = _box_fem("stale")
    ops = _bridge(fem)
    ops.fix(pg="B", dofs=(1, 1, 1))
    p = ops.pattern.Plain(series=ops.timeSeries.Linear())
    p.load(pg="B", forces=(1.0, 0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ops.tcl(str(tmp_path / "a.tcl"))
        before = _load_shapes(artifact_dir / "stale.h5")
        p.load(pg="B", forces=(0.0, 5.0, 0.0))
        ops.tcl(str(tmp_path / "b.tcl"))
        after = _load_shapes(artifact_dir / "stale.h5")
        ops.h5(str(tmp_path / "explicit.h5"))
    assert len(h5_calls) == 3  # two automatic writes and the explicit one
    assert before != after
    assert after == _load_shapes(tmp_path / "explicit.h5")
    a = (tmp_path / "a.tcl").read_text(encoding="utf-8").count("load ")
    b = (tmp_path / "b.tcl").read_text(encoding="utf-8").count("load ")
    assert b == 2 * a > 0


def test_a_composed_model_writes(artifact_dir, tmp_path, h5_calls):
    """A composed model reports its modules as partitions (ADR 0038) and
    is not partitioned for D1 (`_artifact_policy`, the session's rule):
    the bridge writes it, with no warning."""
    for nm in ("hostmod", "mod"):
        with apeGmsh(model_name=nm, verbose=False, _artifacts=False) as g:
            box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            g.model.sync()
            g.mesh.sizing.set_global_size(0.5)
            g.mesh.generation.generate(3)
            g.physical.add(3, [box], name="B")
            g.mesh.queries.get_fem_data(dim=3).to_h5(str(tmp_path / f"{nm}.h5"))
    g = apeGmsh.from_h5(str(tmp_path / "hostmod.h5"))
    g.compose(str(tmp_path / "mod.h5"), label="M", translate=(2.0, 0.0, 0.0))
    g.save(str(tmp_path / "out.h5"))
    fem = FEMData.from_h5(str(tmp_path / "out.h5"))
    assert fem.model_name == "hostmod"
    assert len(fem.partitions) == 2 and fem.composed_from
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # The bridge's advisory on a multi-partition deck (ADR 0027), not
        # this test's subject.
        warnings.simplefilter("ignore", OpenSeesAutoEmitWarning)
        _bridge(fem).tcl(str(tmp_path / "c.tcl"))
    assert len(h5_calls) == 1
    out = artifact_dir / "hostmod.h5"
    assert {"opensees", "nodes", "partitions"} <= _zones(out)


# ---------------------------------------------------------------------------
# A script run twice (P3 ruling of 2026-10-07 on #1307)
# ---------------------------------------------------------------------------

RERUN_SCRIPT = '''\
import sys
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(verbose=False) as g:  # named by the script's stem
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
    g.physical.add_volume("b", name="B")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
ops.element.FourNodeTetrahedron(pg="B", material=mat)
ops.tcl(TCL)
sys.stdout.write("SESSION " + fem.session_id + "\\n")
'''


def _pair_ids(out: Path, stem: str) -> tuple[str, str, set[str]]:
    model, sibling = out / f"{stem}.h5", out / f"{stem}.geometry.h5"
    return _meta(model, "session_id"), _meta(sibling, "session_id"), _zones(model)


def test_a_script_run_twice_in_process_leaves_run_two_s_pair(tmp_path, monkeypatch):
    """Run 2's session ``end()`` finds run 1's full ``model.h5`` (with
    ``/opensees``) whose ``/provenance`` names the same script: it
    replaces it (P3), the bridge writes the fuller file again, and both
    files carry run 2's ``session_id``.  No warning in either run."""
    out = tmp_path / "artifacts"
    out.mkdir()
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(out))
    script = tmp_path / "twice.py"
    script.write_text(RERUN_SCRIPT, encoding="utf-8")
    ids = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for i in range(2):
            glb = runpy.run_path(str(script), run_name="__main__",
                                 init_globals={"TCL": str(tmp_path / f"t{i}.tcl")})
            ids.append(glb["fem"].session_id)
            model_id, sibling_id, zones = _pair_ids(out, "twice")
            assert model_id == sibling_id == ids[-1]
            assert "opensees" in zones
    assert ids[0] != ids[1]


def test_a_script_run_twice_in_subprocesses_leaves_run_two_s_pair(tmp_path):
    """The same, as ``python twice.py`` twice: a fresh process each time,
    so the P5 memo plays no part and the policy alone decides."""
    import os
    import subprocess
    import sys

    import apeGmsh as pkg

    out = tmp_path / "artifacts"
    out.mkdir()
    script = tmp_path / "twice.py"
    script.write_text(RERUN_SCRIPT.replace("ops.tcl(TCL)", "ops.tcl('twice.tcl')"),
                      encoding="utf-8")
    env = {
        **os.environ,
        "APEGMSH_ARTIFACT_DIR": str(out),
        "PYTHONPATH": str(Path(pkg.__file__).resolve().parents[1]),
        "PYTHONWARNINGS": "error::UserWarning",
    }
    ids = []
    for _ in range(2):
        # stdin from the null device: under pytest's capture the child
        # would inherit a console handle gmsh cannot use on Windows.
        proc = subprocess.run(
            [sys.executable, str(script)], cwd=str(tmp_path), env=env,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        ids.append(proc.stdout.strip().split("SESSION ")[-1].split()[0])
        model_id, sibling_id, zones = _pair_ids(out, "twice")
        assert model_id == sibling_id == ids[-1]
        assert "opensees" in zones
    assert ids[0] != ids[1]


def test_a_same_script_fuller_file_is_replaced_and_a_foreign_one_is_refused(
    tmp_path, monkeypatch,
):
    """The rule itself: another run's ``model.h5`` holding ``/opensees``
    that the neutral write would drop is ``"write"`` when its
    ``/provenance`` names this run's script, and the V2b refusal
    otherwise."""
    from apeGmsh._artifact_policy import artifact_identity, artifact_verdict
    from apeGmsh.opensees._internal.schema_version import NEUTRAL, PROVENANCE

    out = tmp_path / "artifacts"
    out.mkdir()
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(out))
    script = tmp_path / "once.py"
    script.write_text(RERUN_SCRIPT, encoding="utf-8")
    runpy.run_path(str(script), run_name="__main__",
                   init_globals={"TCL": str(tmp_path / "o.tcl")})
    target = out / "once.h5"
    assert "opensees" in _zones(target)
    _, file_scripts = artifact_identity(target)
    assert file_scripts
    neutral = frozenset({NEUTRAL, PROVENANCE})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert artifact_verdict(
            target, writes=neutral, overwrite=True, session_id="another-run",
            content=lambda: "", scripts=file_scripts, explicit=False,
        ) == "write"
    with pytest.warns(UserWarning, match="would drop"):
        assert artifact_verdict(
            target, writes=neutral, overwrite=True, session_id="another-run",
            content=lambda: "", scripts=frozenset({"c:/elsewhere/other.py"}),
            explicit=False,
        ) == "refuse"
    assert "opensees" in _zones(target)


def test_a_session_rerun_in_one_process_rewrites_the_file(
    artifact_dir, tmp_path, h5_calls,
):
    """The reviewer's ``rerun.py``: a notebook cell run again builds the
    same model in a new session (new ``session_id``) and a new bridge.
    The digest carries the ``session_id``, so the second cell writes,
    and the file carries the second session's id."""
    def cell(i: int):
        with apeGmsh(model_name="rr", verbose=False) as g:  # end() writes rr.h5
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
            g.physical.add_volume("b", name="B")
            g.mesh.sizing.set_global_size(0.5)
            g.mesh.generation.generate(dim=3)
            fem = g.mesh.queries.get_fem_data(dim=3)
        _bridge(fem).tcl(str(tmp_path / f"rr{i}.tcl"))
        return fem.session_id

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        first = cell(0)
        assert _meta(artifact_dir / "rr.h5", "session_id") == first
        second = cell(1)
    # Under pytest the cells name no script, so cell 2's session end()
    # keeps V2b's refusal of cell 1's fuller file (the notebook case the
    # P3 ruling leaves as it is); the bridge is silent and rewrites.
    assert _bridge_warnings(rec) == []
    assert [m for m in (str(w.message) for w in rec) if "would drop" in m]
    assert first != second
    assert len(h5_calls) == 2
    model_id, sibling_id, zones = _pair_ids(artifact_dir, "rr")
    assert model_id == second and "opensees" in zones
    # The refused session write skipped its sibling as well (the pair
    # rule), so the sibling still carries cell 1's id here; a script run
    # twice, which names itself, ends with one id on both files
    # (``test_a_script_run_twice_in_process_leaves_run_two_s_pair``).
    assert sibling_id == first


# ---------------------------------------------------------------------------
# Deferred archive features: the explicit save's warning, not the hook's
# ---------------------------------------------------------------------------


def _eq_bridge(fem) -> apeSees:
    ops = _bridge(fem)
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    return ops


def test_automatic_write_is_silent_about_deferred_features(
    artifact_dir, tmp_path, h5_calls,
):
    from apeGmsh.opensees.emitter.h5 import H5FeatureDeferredWarning

    ops = _eq_bridge(_box_fem("eq"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", H5FeatureDeferredWarning)
        # The Lagrange-handler advisory (an EQ_Constraint is present) is
        # the bridge's, not this test's subject.
        warnings.simplefilter("ignore", OpenSeesAutoEmitWarning)
        ops.tcl(str(tmp_path / "a.tcl"))
    assert len(h5_calls) == 1
    assert "opensees" in _zones(artifact_dir / "eq.h5")


def test_explicit_h5_still_warns_about_deferred_features(artifact_dir, tmp_path):
    from apeGmsh.opensees.emitter.h5 import H5FeatureDeferredWarning

    ops = _eq_bridge(_box_fem("eq_explicit"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OpenSeesAutoEmitWarning)
        with pytest.warns(H5FeatureDeferredWarning, match="equation_constraint"):
            ops.h5(str(tmp_path / "explicit.h5"))


# ---------------------------------------------------------------------------
# P6: the library's own constructors opt out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", [
    "interop/etabs_import.py",
    "interop/stko/translate.py",
    "interop/strut_tie.py",
])
def test_internal_constructors_pass_artifacts_false(rel: str):
    text = (SRC / rel).read_text(encoding="utf-8")
    calls = re.findall(r"^\s*ops = apeSees\((.*)\)\s*$", text, re.M)
    assert calls, rel
    assert all("_artifacts=False" in c for c in calls), (rel, calls)
