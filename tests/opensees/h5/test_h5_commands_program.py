"""``/opensees/program`` + ``/opensees/commands`` (ADR 0114 R2/R3a, opensees 2.23.0).

K1-4 archives the four verbs ``ops.h5()`` used to drop: global
``rayleigh``, ``eigen`` + ``modal_damping`` (``ops.damping.modal``) and a
stage's ``profiler`` bracket (``s.profile``). The oracles:

- **Round trip, byte for byte.** ``tcl() == from_h5(h5()).build('tcl')``
  on flat, staged and partitioned (``flat=True``) models, and each
  archived line sits at the index the bridge emitted it. The fixtures use
  non-integral parameters: an integral float such as ``1e6`` is stored as
  ``f8`` and re-emitted as ``1000000`` (a pre-existing replay format
  difference that this slice does not touch), which would mask a
  positional mismatch with an unrelated byte diff.
- **Fixed point.** ``from_h5(p).to_h5()`` keeps ``program`` and
  ``commands`` dataset for dataset, and ``model_hash`` with them.
- **Completeness.** The program's runs tile ``[1, @emit_count]``, and the
  method sequence equals what a ``RecordingEmitter`` sees from the same
  ``BuiltModel``.
- **Hash scope.** One perturbed ``program`` cell changes ``model_hash``.
- **Ledger warning.** Once per bridge for each distinct ledgered set.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.lineage import compute_model_hash
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.apesees import OpenSeesAutoEmitWarning
from apeGmsh.opensees.emitter.h5 import (
    H5Emitter,
    H5FeatureDeferredWarning,
    H5LedgerWarning,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.fixtures.schema import OPENSEES_PRIOR_MINOR
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem
from tests.opensees.h5.test_h5_partitioned_staged_capture import (
    _OVERLAPPING_SPLIT,
)
from tests.opensees.h5.test_h5_partitions_roundtrip import (
    build_partitioned_two_quad_fem,
)
from tests.opensees.h5.test_h5_stages_reader import _chain, build_two_quad_fem


_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _frame(*, rayleigh: bool, modal: bool) -> apeSees:
    ops = apeSees(cast("object", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.0125, E=2900.25, Iz=1.5e-4, Iy=2.5e-4, G=1100.75, J=3.5e-4,
    )
    if rayleigh:
        ops.damping.rayleigh(alpha_m=0.125, beta_k=0.0025)
    if modal:
        ops.damping.modal(0.05, modes=2)
    return ops


def _staged(*, partitioned: bool, rayleigh: bool, profile: bool) -> apeSees:
    fem = (
        build_partitioned_two_quad_fem(partitions=_OVERLAPPING_SPLIT)
        if partitioned else build_two_quad_fem()
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1000000.5, nu=0.3, rho=0.25)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.25, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.25, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    if rayleigh:
        ops.damping.rayleigh(alpha_m=0.125, beta_k=0.0025)
    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.fix(pg="FillTop", dofs=(1, 1))
        if profile:
            s.profile(deep=True, memory=True)
        s.analysis(**_chain(ops))
        s.run(n_increments=5)
    with ops.stage(name="loading") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(pg="Fill", forces=(10.5, 0.0))
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)
    return ops


def _bridge_deck(ops: apeSees, tmp_path: Path, *, flat: bool = False) -> str:
    p = tmp_path / "bridge.tcl"
    if flat:
        ops.tcl(str(p), progress=False, flat=True)
    else:
        ops.tcl(str(p), progress=False)
    return p.read_text(encoding="utf-8")


def _archive(ops: apeSees, tmp_path: Path, name: str = "model.h5") -> Path:
    p = tmp_path / name
    ops.h5(str(p))
    return p


def _replay(path: Path) -> str:
    out = OpenSeesModel.from_h5(str(path)).build("tcl")
    assert isinstance(out, str)
    return out


def _indices(deck: str, prefix: str) -> list[int]:
    return [i for i, ln in enumerate(deck.splitlines()) if ln.startswith(prefix)]


# ---------------------------------------------------------------------------
# 1. Round trip, byte for byte, line at the same position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rayleigh", "modal", "lines"),
    [
        (False, False, ()),
        (True, False, ("rayleigh 0.125 0.0025 0.0 0.0",)),
        (False, True, ("eigen -genBandArpack 2", "modalDamping 0.05")),
        (True, True, ("rayleigh 0.125 0.0025 0.0 0.0",
                      "eigen -genBandArpack 2", "modalDamping 0.05")),
    ],
)
def test_flat_round_trip_is_byte_identical(
    tmp_path: Path, rayleigh: bool, modal: bool, lines: tuple[str, ...],
) -> None:
    ops = _frame(rayleigh=rayleigh, modal=modal)
    bridge = _bridge_deck(ops, tmp_path)
    replay = _replay(_archive(ops, tmp_path))
    assert replay == bridge
    for line in lines:
        assert _indices(replay, line) == _indices(bridge, line) != []


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize(
    ("rayleigh", "profile"), [(False, False), (True, False), (False, True),
                              (True, True)],
)
def test_staged_round_trip_is_byte_identical(
    tmp_path: Path, partitioned: bool, rayleigh: bool, profile: bool,
) -> None:
    """Flat and partitioned staged archives replay the bridge's flat deck.

    A partitioned archive replays a flat single-process deck (the
    documented degrade), so the oracle is the bridge's ``flat=True`` deck.
    """
    ops = _staged(partitioned=partitioned, rayleigh=rayleigh, profile=profile)
    bridge = _bridge_deck(ops, tmp_path, flat=True)
    replay = _replay(_archive(ops, tmp_path))
    assert replay == bridge
    expect = (["rayleigh "] if rayleigh else []) + (
        ["profiler start", "profiler stop", "profiler report"] if profile else [])
    for prefix in expect:
        assert _indices(replay, prefix) == _indices(bridge, prefix) != []
    if profile:
        # The bracket sits around the profiled stage's analyze only.
        start = _indices(replay, "profiler start")[0]
        stop = _indices(replay, "profiler stop")[0]
        analyzed = [i for i in _indices(replay, "") if start < i < stop]
        assert analyzed


def test_partitioned_archive_carries_each_command_once(tmp_path: Path) -> None:
    flat = _archive(
        _staged(partitioned=False, rayleigh=True, profile=True), tmp_path,
        "flat.h5")
    part = _archive(
        _staged(partitioned=True, rayleigh=True, profile=True), tmp_path,
        "part.h5")
    with h5_reader.open(str(flat)) as a, h5_reader.open(str(part)) as b:
        rows_a = [(c.method, c.stage, c.args, c.names) for c in a.commands()]
        rows_b = [(c.method, c.stage, c.args, c.names) for c in b.commands()]
    assert rows_a == rows_b == [
        ("rayleigh", -1, (0.125, 0.0025, 0.0, 0.0), ("", "", "", "")),
        ("profiler", 0, ("start", "-deep", "-memory"), ("", "", "")),
        ("profiler", 0, ("stop",), ("",)),
        ("profiler", 0, ("report", "construction.h5"), ("", "")),
    ]


# ---------------------------------------------------------------------------
# 2. Fixed point under from_h5(p).to_h5()
# ---------------------------------------------------------------------------


def _zone(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        ops = f["opensees"]
        prog = ops["program"]
        out["program"] = prog[()].tolist()
        out["program_attrs"] = {k: np.asarray(v).tolist()
                                for k, v in prog.attrs.items()}
        if "commands" in ops:
            # repr() so the NaN of a string slot compares equal to itself.
            out["commands"] = {
                k: [repr(v) for v in np.asarray(ops["commands"][k][()]).tolist()]
                for k in ops["commands"]}
        out["model_hash"] = compute_model_hash("", ops)
    return out


@pytest.mark.parametrize("case", ["flat", "staged", "partitioned"])
def test_rewrite_keeps_program_commands_and_hash(
    tmp_path: Path, case: str,
) -> None:
    ops = (
        _frame(rayleigh=True, modal=True) if case == "flat"
        else _staged(partitioned=case == "partitioned", rayleigh=True,
                     profile=True)
    )
    src = _archive(ops, tmp_path)
    out = tmp_path / "rewrite.h5"
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    a, b = _zone(src), _zone(out)
    assert "commands" in a
    assert a == b
    # And the rewrite is itself a fixed point.
    out2 = tmp_path / "rewrite2.h5"
    OpenSeesModel.from_h5(str(out)).to_h5(str(out2))
    assert _zone(out2) == b


# ---------------------------------------------------------------------------
# 3. Completeness: the runs tile the emit order a RecordingEmitter sees
# ---------------------------------------------------------------------------


def _expand(runs: tuple[h5_reader.ProgramRun, ...]) -> list[str]:
    return [r.method for r in runs for _ in range(r.count)]


@pytest.mark.parametrize("case", ["flat", "staged", "partitioned"])
def test_program_tiles_the_recording_emit_order(
    tmp_path: Path, case: str,
) -> None:
    ops = (
        _frame(rayleigh=True, modal=True) if case == "flat"
        else _staged(partitioned=case == "partitioned", rayleigh=True,
                     profile=True)
    )
    src = _archive(ops, tmp_path)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    with h5_reader.open(str(src)) as m:
        runs = m.program()
        emit_count = int(m.handle["opensees/program"].attrs["emit_count"])
    assert runs[0].first == 1
    for a, b in zip(runs, runs[1:]):
        assert b.first == a.first + a.count
    assert runs[-1].first + runs[-1].count - 1 == emit_count
    assert _expand(runs) == [name for name, _a, _k in rec.calls]
    # Stage order never decreases (K0-6).
    stages = [r.stage for r in runs if r.stage >= 0]
    assert stages == sorted(stages)


def test_a_dropped_note_is_a_gap(tmp_path: Path) -> None:
    """Plant: a call that bypasses the note shows up as a missing index."""
    ops = _frame(rayleigh=True, modal=False)
    emitter = H5Emitter(model_name="m")
    ops.build().emit(emitter)
    noted = emitter._program.count
    H5Emitter.__dict__["rayleigh"].__wrapped__(emitter, 0.5, 0.0, 0.0, 0.0)
    assert emitter._program.count == noted
    assert len(emitter._commands) == 2


# ---------------------------------------------------------------------------
# 4. Reader accessors and hash scope
# ---------------------------------------------------------------------------


def test_emit_index_accessor_agrees_with_program(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh=True, modal=True), tmp_path)
    om = OpenSeesModel.from_h5(str(src))
    with h5_reader.open(str(src)) as m:
        cmds = m.commands()
        assert [c.method for c in cmds] == ["rayleigh", "eigen", "modal_damping"]
        for c in cmds:
            assert m.emit_index(c.method, 0) == c.emit_index
            assert om.emit_index(c.method, 0) == c.emit_index
        # eigen directly follows rayleigh, modal_damping follows eigen.
        assert [c.emit_index for c in cmds] == list(
            range(cmds[0].emit_index, cmds[0].emit_index + 3))
        with pytest.raises(LookupError):
            m.emit_index("rayleigh", 1)
    assert om.commands() == cmds


def test_one_program_cell_changes_the_hash(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh=True, modal=True), tmp_path)
    dst = tmp_path / "perturbed.h5"
    shutil.copy(src, dst)
    with h5py.File(str(dst), "r+") as f:
        data = f["opensees/program"][()]
        data["row"][-1] += 1
        f["opensees/program"][...] = data
    with h5py.File(str(src), "r") as a, h5py.File(str(dst), "r") as b:
        assert compute_model_hash("", a["opensees"]) != compute_model_hash(
            "", b["opensees"])


def test_a_program_that_skips_an_index_is_malformed(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh=True, modal=False), tmp_path)
    with h5py.File(str(src), "r+") as f:
        data = f["opensees/program"][()]
        data["first"][-1] += 1
        f["opensees/program"][...] = data
    with h5_reader.open(str(src)) as m:
        with pytest.raises(h5_reader.MalformedH5Error, match="emit order"):
            m.program()


def test_a_command_row_with_no_replay_slot_raises(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh=True, modal=False), tmp_path)
    with h5py.File(str(src), "r+") as f:
        del f["opensees/commands/method"]
        f["opensees/commands"].create_dataset(
            "method", data=np.array(["profiler"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"))
        prog = f["opensees/program"]
        methods = [str(m.decode() if isinstance(m, bytes) else m)
                   for m in prog.attrs["methods"]]
        prog.attrs.create(
            "methods",
            np.array([("profiler" if m == "rayleigh" else m) for m in methods],
                     dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"))
    with pytest.raises(NotImplementedError, match="no replay slot"):
        OpenSeesModel.from_h5(str(src)).build("tcl")


def test_global_eigen_inside_a_stage_raises() -> None:
    em = H5Emitter(model_name="m")
    em.stage_open("s")
    with pytest.raises(RuntimeError, match="global-only"):
        em.eigen(2)


# ---------------------------------------------------------------------------
# 5. The ledger warning
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _ledgered_frame() -> apeSees:
    ops = _frame(rayleigh=False, modal=False)
    ops.equation_constraint(constrained=(2, 1), retained=[(1, 1, -1.0)])
    return ops


def _of(record: list[warnings.WarningMessage], cat: type) -> list[Any]:
    return [w for w in record if issubclass(w.category, cat)]


def test_no_ledger_no_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", H5LedgerWarning)
        _frame(rayleigh=True, modal=True).h5(str(tmp_path / "m.h5"))


def test_one_dropped_row_is_one_warning(tmp_path: Path) -> None:
    """``equationConstraint`` raises its own deferred warning; the ledger
    warning skips it, so the row warns once across every category."""
    ops = _ledgered_frame()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ops.h5(str(tmp_path / "m.h5"))
    about_archive = [
        w for w in rec
        if issubclass(w.category, (H5LedgerWarning, H5FeatureDeferredWarning))
    ]
    assert [w.category for w in about_archive] == [H5FeatureDeferredWarning], [
        (w.category.__name__, str(w.message)[:80]) for w in rec]
    others = [w for w in rec if w not in about_archive]
    assert all(issubclass(w.category, OpenSeesAutoEmitWarning)
               for w in others), [(w.category.__name__, str(w.message)[:80])
                                  for w in others]


def test_automatic_write_warns_once_about_the_dropped_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The automatic write silences ``H5FeatureDeferredWarning``, so the
    ledger warning must carry ``equationConstraint`` there: one warning,
    not zero."""
    art = tmp_path / "artifacts"
    art.mkdir()
    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(art))
    ops = _ledgered_frame()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ops.tcl(str(tmp_path / "a.tcl"))
    assert (art / "simple_frame.h5").exists()
    about_archive = [
        w for w in rec
        if issubclass(w.category, (H5LedgerWarning, H5FeatureDeferredWarning))
    ]
    assert [w.category for w in about_archive] == [H5LedgerWarning], [
        (w.category.__name__, str(w.message)[:80]) for w in rec]
    assert "equationConstraint x1" in str(about_archive[0].message)
    assert Path(about_archive[0].filename).resolve() == Path(__file__).resolve()
    others = [w for w in rec if w not in about_archive]
    assert all(issubclass(w.category, OpenSeesAutoEmitWarning)
               for w in others), [(w.category.__name__, str(w.message)[:80])
                                  for w in others]


def _with_ledger(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, int],
) -> None:
    """Add ledger calls the fixture model cannot make (contact needs a
    meshed interface); the bridge reads them through ``ledger_counts``."""
    real = H5Emitter.__dict__["ledger_counts"]

    def _counts(self: H5Emitter) -> dict[str, int]:
        return {**real.fget(self), **extra}

    monkeypatch.setattr(H5Emitter, "ledger_counts", property(_counts))


def test_ledger_warns_once_per_bridge_and_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = _frame(rayleigh=False, modal=False)
    _with_ledger(monkeypatch, {"contact": 2})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        # Two emits whose automatic model.h5 write reaches h5(), then h5().
        ops.tcl(str(tmp_path / "a.tcl"))
        ops.tcl(str(tmp_path / "b.tcl"))
        ops.h5(str(tmp_path / "m.h5"))
    (only,) = _of(rec, H5LedgerWarning)
    assert "contact x2" in str(only.message)
    # It names the user's call (the first tcl() here), not the bridge.
    assert Path(only.filename).resolve() == Path(__file__).resolve()

    # A new ledgered verb is a new set: one more warning, then silence.
    _with_ledger(monkeypatch, {"contact": 2, "embedded_node": 1})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ops.h5(str(tmp_path / "m2.h5"))
        ops.h5(str(tmp_path / "m3.h5"))
    (only,) = _of(rec, H5LedgerWarning)
    assert "embedded_node x1" in str(only.message)
    assert Path(only.filename).resolve() == Path(__file__).resolve()


# ---------------------------------------------------------------------------
# 6. Rewrites never write an order the file does not hold (#1573 review)
# ---------------------------------------------------------------------------


def test_upgrading_a_pre_program_file_writes_no_program(tmp_path: Path) -> None:
    """A source below 2.23.0 has no program; the rewrite must not invent
    one from its own category-major replay."""
    src = _archive(
        _staged(partitioned=False, rayleigh=False, profile=False), tmp_path)
    with h5py.File(str(src), "r+") as f:
        del f["opensees/program"]
        f["meta"].attrs["opensees_schema_version"] = OPENSEES_PRIOR_MINOR
    out = tmp_path / "upgraded.h5"
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    with h5py.File(str(out), "r") as f:
        assert "program" not in f["opensees"]
    with h5_reader.open(str(out)) as m:
        assert m.program() == ()
        with pytest.raises(h5_reader.ProgramAbsentError, match="predates"):
            m.emit_index("analyze", 0, stage=0)
    with pytest.raises(h5_reader.ProgramAbsentError, match="predates"):
        OpenSeesModel.from_h5(str(out)).emit_index("analyze", 0, stage=0)


def test_rewrite_unlinks_runs_whose_store_it_dropped(tmp_path: Path) -> None:
    """The rewrite does not carry ``/opensees/regions`` (a main-side gap,
    #1579); the echoed program must not name it."""
    ops = _frame(rayleigh=True, modal=False)
    ops.damping.rayleigh(alpha_m=0.5, beta_k=0.0, on="Cols")
    src = _archive(ops, tmp_path)
    with h5py.File(str(src), "r") as f:
        assert "regions" in f["opensees"]
    out = tmp_path / "rewrite.h5"
    with pytest.warns(H5FeatureDeferredWarning, match="regions") as rec:
        OpenSeesModel.from_h5(str(out.parent / src.name)).to_h5(str(out))
    (dropped_w,) = [w for w in rec if "regions" in str(w.message)]
    assert Path(dropped_w.filename).resolve() == Path(__file__).resolve()
    with h5py.File(str(out), "r") as f:
        assert "regions" not in f["opensees"]
        stores = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in f["opensees/program"].attrs["stores"]]
    assert not [s for s in stores if "regions" in s]
    with h5_reader.open(str(src)) as a, h5_reader.open(str(out)) as b:
        ra, rb = a.program(), b.program()
    # The order survives: same methods at the same emit indices.
    assert _expand(ra) == _expand(rb)
    dropped = [r for r in rb if r.method == "region"]
    assert dropped and all(r.store == "" and r.row == -1 for r in dropped)
