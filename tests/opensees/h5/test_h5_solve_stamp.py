"""``/opensees@will_solve`` / ``@solve_refusals`` / ``@requires`` (ADR 0114 D6, opensees 2.24.0).

R4 (K1-5). The writer side: ``H5Emitter.set_solve_stamp(will_solve=,
solve_refusals=)`` and ``write`` stamps the three attributes together on
``/opensees``, deriving ``@requires`` from the verbs the program holds;
``H5Model.solve_stamp`` reads them back. The bridge side: ``apeSees.h5``
stamps every archive, an archival emit records the solve-time gates that
would refuse instead of raising, and ``OpenSeesModel`` fails closed on
them. Oracles:

- **Round trip.** Every field comes back as it went in, empty tuples
  included; a fork verb on the tape puts ``"fork"`` in ``@requires``.
- **Absent is None.** A writer never handed a stamp writes no attribute,
  and a reader returns ``None`` for it and for every file below 2.24.0
  (the corpus file of the prior minor).
- **Malformed fails loud.** A partial stamp, a non-0/1 ``@will_solve``, an
  unsorted ``@requires`` or an empty token raises ``MalformedH5Error``.
- **Hash scope.** The stamp folds into ``model_hash``.
- **Bridge.** ``will_solve`` is ``staged or any(Analysis)``; a LadrunoUP
  model with a solve and no system archives (where ``ops.tcl`` refuses)
  with ``"ladruno_up_solver"`` recorded, and its replay to a deck refuses;
  ``to_h5`` echoes the stamp hash-stable; ``build('live')`` refuses a
  ``"fork"`` requirement on a stock backend before touching the domain.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees._internal.lineage import compute_model_hash, read_stored_lineage
from apeGmsh.opensees._target import BackendInfo
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.caps import SolveStamp
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
from apeGmsh.opensees.opensees_model import refuse_live_requires

from tests.fixtures.schema import (
    OPENSEES_CURRENT,
    OPENSEES_SOLVE_STAMP_FROM,
)
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

_CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"


def _write(
    path: Path, stamp: "tuple[bool, tuple[str, ...]] | None", *, fork: bool = False,
) -> Path:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.node(1, 0.0, 0.0, 0.0)
    e.fix(1, 1, 1, 1, 1, 1, 1)
    if fork:
        e.contact_surface(1, 1.0)  # a ledger row whose VERBS row requires the fork
    if stamp is not None:
        e.set_solve_stamp(will_solve=stamp[0], solve_refusals=stamp[1])
    e.write(str(path))
    return path


# -- writer / reader ------------------------------------------------------


@pytest.mark.parametrize("will_solve, refusals, fork", [
    (True, ("ladruno_up_solver",), True),
    (False, (), False),
    (True, ("a", "b", "a"), False),
], ids=["full", "empty", "repeated-refusal"])
def test_solve_stamp_round_trips(
    tmp_path: Path, will_solve: bool, refusals: tuple, fork: bool,
) -> None:
    out = _write(tmp_path / "m.h5", (will_solve, refusals), fork=fork)
    want = SolveStamp(will_solve, refusals, ("fork",) if fork else ())
    assert want.solve_mode == "flat" and want.solve_refusals_flat == refusals
    with h5py.File(out, "r") as f:
        attrs = f["opensees"].attrs
        assert f["meta"].attrs["opensees_schema_version"] == OPENSEES_CURRENT
        assert set(attrs) == {"will_solve", "solve_mode", "solve_refusals",
                              "solve_refusals_flat", "requires"}
        assert attrs["will_solve"].dtype == np.int8
        assert int(attrs["will_solve"]) == int(will_solve)
        assert attrs["solve_mode"] == "flat"
        assert attrs["solve_refusals"].shape == (len(refusals),)
        assert attrs["solve_refusals_flat"].shape == (len(refusals),)
        assert attrs["requires"].shape == (len(want.requires),)
    with h5_reader.open(str(out)) as m:
        assert m.solve_stamp() == want


def test_partitioned_stamp_round_trips_both_verdicts(tmp_path: Path) -> None:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.node(1, 0.0, 0.0, 0.0)
    e.set_solve_stamp(will_solve=True, solve_mode="partitioned",
                      solve_refusals=(), solve_refusals_flat=("ladruno_up_solver",))
    e.write(str(tmp_path / "p.h5"))
    with h5_reader.open(str(tmp_path / "p.h5")) as m:
        stamp = m.solve_stamp()
    assert stamp == SolveStamp(True, (), (), "partitioned", ("ladruno_up_solver",))
    assert stamp.refusals_for("partitioned") == ()
    assert stamp.refusals_for("flat") == ("ladruno_up_solver",)
    with pytest.raises(ValueError, match="no verdict for partition mode"):
        stamp.refusals_for("ranked")
    with pytest.raises(ValueError, match="needs its flat verdict"):
        SolveStamp(True, solve_mode="partitioned")
    with pytest.raises(ValueError, match="needs its flat verdict"):
        H5Emitter().set_solve_stamp(will_solve=True, solve_mode="partitioned")


def test_a_writer_never_handed_a_stamp_writes_no_attribute(tmp_path: Path) -> None:
    out = _write(tmp_path / "m.h5", None)
    with h5py.File(out, "r") as f:
        assert set(f["opensees"].attrs) == set()
    with h5_reader.open(str(out)) as m:
        assert m.solve_stamp() is None


def test_prior_minor_corpus_file_has_no_stamp() -> None:
    """The newest corpus file below the stamp's minor carries none, and
    the stamp's own file does: a fixed point of the history, whatever
    the current or prior minor is."""
    def minor_of(text: str) -> tuple[int, int]:
        major, minor = text.split(".")[:2]
        return int(major), int(minor)

    stamp_from = minor_of(OPENSEES_SOLVE_STAMP_FROM)
    plain = re.compile(r"opensees_(\d+\.\d+)\.h5")   # not the variants
    minors = sorted(
        minor_of(match.group(1))
        for p in _CORPUS.glob("opensees_*.h5")
        if (match := plain.fullmatch(p.name)) is not None)
    below = [m for m in minors if m < stamp_from]
    assert below and stamp_from in minors, "rebuild the corpus"
    for minor, stamped in ((below[-1], False), (stamp_from, True)):
        path = _CORPUS / f"opensees_{minor[0]}.{minor[1]}.h5"
        with h5_reader.open(str(path)) as m:
            assert (m.solve_stamp() is not None) is stamped, path.name


def test_set_solve_stamp_refuses_a_second_call_and_bad_arguments() -> None:
    e = H5Emitter()
    e.set_solve_stamp(will_solve=True)
    with pytest.raises(RuntimeError, match="already set"):
        e.set_solve_stamp(will_solve=False)
    with pytest.raises(TypeError, match="must be a bool"):
        H5Emitter().set_solve_stamp(will_solve=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-empty strings"):
        H5Emitter().set_solve_stamp(will_solve=True, solve_refusals="gate")
    with pytest.raises(TypeError, match="non-empty strings"):
        H5Emitter().set_solve_stamp(will_solve=True, solve_refusals=("",))


def test_solve_stamp_accessor_is_none_until_set_then_derives_requires() -> None:
    e = H5Emitter()
    assert e.solve_stamp() is None
    e.model(ndm=3, ndf=6)
    e.contact_surface(1, 1.0)
    e.set_solve_stamp(will_solve=False)
    assert e.solve_stamp() == SolveStamp(False, (), ("fork",))


def _tamper(path: Path, **attrs: object) -> None:
    with h5py.File(path, "r+") as f:
        g = f["opensees"]
        for name, value in attrs.items():
            if value is None:
                del g.attrs[name]
            elif isinstance(value, list):
                g.attrs.create(name, np.array(value, dtype=object),
                               dtype=h5py.string_dtype(encoding="utf-8"))
            else:
                g.attrs[name] = value


@pytest.mark.parametrize("tamper, match", [
    ({"will_solve": np.int8(2)}, "only stamps the int8 0 or 1"),
    ({"will_solve": np.float64(1.0)}, "only stamps the int8 0 or 1"),
    ({"will_solve": np.array([1, 0], dtype=np.int8)}, "only stamps the int8 0 or 1"),
    ({"requires": None}, "@requires is missing"),
    ({"solve_refusals": None}, "@solve_refusals is missing"),
    ({"solve_refusals_flat": None}, "@solve_refusals_flat is missing"),
    ({"solve_mode": None}, "@solve_mode is missing"),
    ({"solve_mode": "ranked"}, "solve_mode must be one of"),
    ({"solve_refusals_flat": []}, "flat stamp's solve_refusals_flat must equal"),
    ({"requires": ["mp", "fork"]}, "sorted and unique"),
    ({"requires": ["fork", "fork"]}, "sorted and unique"),
    ({"solve_refusals": ["ok", ""]}, "empty token"),
    ({"requires": "fork"}, "expected a 1-D array"),
], ids=[
    "will_solve-2", "will_solve-float", "will_solve-array", "no-requires",
    "no-refusals", "no-refusals-flat", "no-mode", "unknown-mode",
    "flat-verdicts-differ", "unsorted", "duplicate", "empty-token",
    "scalar-requires",
])
def test_malformed_stamp_fails_loud(tmp_path: Path, tamper: dict, match: str) -> None:
    out = _write(tmp_path / "m.h5", (True, ("g",)), fork=True)
    _tamper(out, **tamper)
    with h5_reader.open(str(out)) as m, pytest.raises(MalformedH5Error, match=match):
        m.solve_stamp()


def test_stamp_folds_into_model_hash(tmp_path: Path) -> None:
    def digest(path: Path) -> str:
        with h5py.File(path, "r") as f:
            return compute_model_hash("", f["opensees"])

    none = digest(_write(tmp_path / "none.h5", None))
    solve = digest(_write(tmp_path / "solve.h5", (True, ())))
    archive = digest(_write(tmp_path / "archive.h5", (False, ())))
    refused = digest(_write(tmp_path / "refused.h5", (True, ("g",))))
    assert len({none, solve, archive, refused}) == 4
    assert digest(_write(tmp_path / "solve2.h5", (True, ()))) == solve


# -- the bridge stamps every archive ----------------------------------------


def _frame_bridge(*, analysis: bool) -> apeSees:
    ops = apeSees(build_simple_frame_fem())
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.0125, E=2900.25, Iz=1.5e-4, Iy=2.5e-4, G=1100.75, J=3.5e-4,
    )
    ops.fix(nodes=[1], dofs=(1, 1, 1, 1, 1, 1))
    if analysis:
        # ``will_solve`` is ``staged or any(Analysis)``: the Analysis
        # primitive alone makes the archive a solving one.
        ops.system.UmfPack()
        ops.analysis.Static()
    return ops


@pytest.mark.parametrize("analysis", [False, True], ids=["model-only", "solving"])
def test_apesees_h5_stamps_will_solve(tmp_path: Path, analysis: bool) -> None:
    out = tmp_path / "m.h5"
    _frame_bridge(analysis=analysis).h5(str(out))
    model = OpenSeesModel.from_h5(out)
    assert model.solve_stamp == SolveStamp(analysis, (), ())
    # No refusal recorded, so the deck replays.
    assert "elasticBeamColumn" in model.build("tcl")


def test_to_h5_echoes_the_stamp_hash_stable(tmp_path: Path) -> None:
    src, dst = tmp_path / "src.h5", tmp_path / "dst.h5"
    _frame_bridge(analysis=True).h5(str(src))
    OpenSeesModel.from_h5(src).to_h5(dst)
    assert OpenSeesModel.from_h5(dst).solve_stamp == SolveStamp(True, (), ())
    with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
        assert read_stored_lineage(a["meta"])[1] == read_stored_lineage(b["meta"])[1]
        assert dict(a["opensees"].attrs).keys() == dict(b["opensees"].attrs).keys()


def test_replay_fails_closed_on_a_stored_refusal(tmp_path: Path) -> None:
    out = tmp_path / "m.h5"
    _frame_bridge(analysis=True).h5(str(out))
    _tamper(out, solve_refusals=["serial_mumps"], solve_refusals_flat=["serial_mumps"])
    model = OpenSeesModel.from_h5(out)
    assert model.solve_stamp is not None
    assert model.solve_stamp.solve_refusals == ("serial_mumps",)
    for target in ("tcl", "py", "live"):
        with pytest.raises(BridgeError, match="'serial_mumps'.*fails closed"):
            model.build(target)
    # The H5 -> H5 echo is not a solve: it still rewrites the file as it is.
    model.build("h5", out=str(tmp_path / "echo.h5"))
    assert OpenSeesModel.from_h5(tmp_path / "echo.h5").solve_stamp == model.solve_stamp


# -- an archival emit records the gates that would refuse ------------------


def _up_column_bridge(*, partitions: int = 1, system: "str | None" = None,
                      datum: bool = True):
    pytest.importorskip("gmsh")
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="up_stamp", verbose=False) as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 4.0, label="soil")
        g.physical.add_surface("soil", name="Soil")
        # A drained top line gives the pressure region its datum, so the
        # only gate in play is the D4 solver one.
        g.model.select(dim=1).on_plane((0, 4.0, 0), (0, 1, 0), tol=1e-6).to_physical("Top")
        g.mesh.structured.set_recombine("soil", dim=2)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(2)
        g.mesh.structured.recombine()
        if partitions > 1:
            g.mesh.partitioning.partition(partitions)
        g.mesh.partitioning.renumber(base=1)
        fem = g.mesh.queries.get_fem_data(dim=2)
    assert (len(fem.partitions) > 1) == (partitions > 1)
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0)
    ops.element.LadrunoUP(
        pg="Soil", material=mat, Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4),
    )
    if datum:
        ops.fix(pg="Top", dofs=(0, 0, 1))
    if system is not None:
        getattr(ops.system, system)()
    ops.analysis.Static()  # solve-bearing; with no system the D4 gate refuses a deck
    return ops


def test_archival_emit_records_the_refusing_gates(tmp_path: Path) -> None:
    ops = _up_column_bridge()
    with pytest.raises(BridgeError, match="no linear system"):
        ops.tcl(str(tmp_path / "deck.tcl"))
    out = tmp_path / "m.h5"
    ops.h5(str(out))  # the archive is written where the deck is refused
    model = OpenSeesModel.from_h5(out)
    stamp = model.solve_stamp
    assert stamp is not None and stamp.will_solve
    assert stamp.solve_mode == "flat"
    assert stamp.solve_refusals_flat == stamp.solve_refusals == ("ladruno_up_solver",)
    # A fork ELEMENT type rides the generic ``element`` verb, whose VERBS
    # row requires nothing: the requirement is value-dependent, which ADR
    # 0114 D6 keeps out of ``@requires`` until K4 moves typed fork verbs
    # onto the command channel. Only fork *verbs* (contact, embedded,
    # eigen_feast, profiler, ...) reach it today.
    assert stamp.requires == ()
    assert "ladruno_up_solver" in stamp.solve_refusals
    with pytest.raises(BridgeError, match="'ladruno_up_solver'.*fails closed"):
        model.build("tcl")


# -- Ruling A: verdicts per partition mode -----------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_partitioned_archive_without_system_is_refused_on_flat_replay(
    tmp_path: Path,
) -> None:
    """A partitioned LadrunoUP deck with no explicit ``system`` is allowed
    (it rides the ADR 0027 auto-emitted general solver), so the archive's
    own verdict is empty; its flat replay has no auto-emit and would solve
    on ProfileSPD, so the stored flat verdict refuses it."""
    ops = _up_column_bridge(partitions=2)
    ops.tcl(str(tmp_path / "deck.tcl"))  # the partitioned deck is legal
    out = tmp_path / "p.h5"
    ops.h5(str(out))
    model = OpenSeesModel.from_h5(out)
    stamp = model.solve_stamp
    assert stamp is not None and stamp.will_solve
    assert stamp.solve_mode == "partitioned"
    assert stamp.solve_refusals == ()
    assert stamp.solve_refusals_flat == ("ladruno_up_solver",)
    with pytest.raises(BridgeError, match="flat solve.*'ladruno_up_solver'.*fails closed"):
        model.build("tcl")
    with pytest.raises(BridgeError, match="'ladruno_up_solver'"):
        model.build("py")
    # The echo is not a solve.
    model.build("h5", out=str(tmp_path / "echo.h5"))
    assert OpenSeesModel.from_h5(tmp_path / "echo.h5").solve_stamp == stamp


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_valid_partitioned_archive_still_replays(tmp_path: Path) -> None:
    ops = _up_column_bridge(partitions=2, system="UmfPack")
    out = tmp_path / "p.h5"
    ops.h5(str(out))
    model = OpenSeesModel.from_h5(out)
    stamp = model.solve_stamp
    assert stamp is not None
    assert stamp.solve_mode == "partitioned"
    assert stamp.solve_refusals == () and stamp.solve_refusals_flat == ()
    deck = model.build("tcl")
    assert "LadrunoUP" in deck and "UmfPack" in deck


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_partitioned_mumps_archive_replays_flat_as_the_flat_emit_does(
    tmp_path: Path,
) -> None:
    """The flat verdict is what ``ops.tcl(flat=True)`` decides on the same
    model: the serial-Mumps gate keys on the FEM's partition count (ADR 0027
    twin decks), so a partitioned archive with an explicit ``Mumps`` must
    replay flat exactly as its flat emit succeeds (review of #1616, finding 1)."""
    ops = _up_column_bridge(partitions=2, system="Mumps")
    ops.tcl(str(tmp_path / "flat.tcl"), flat=True)  # the flat emit is legal
    out = tmp_path / "p.h5"
    ops.h5(str(out))
    model = OpenSeesModel.from_h5(out)
    stamp = model.solve_stamp
    assert stamp is not None and stamp.solve_mode == "partitioned"
    assert stamp.solve_refusals == () and stamp.solve_refusals_flat == ()
    assert "Mumps" in model.build("tcl")


def test_archival_stamp_records_the_pressure_datum_gate(tmp_path: Path) -> None:
    """A sealed static u-p column: the datum gate refuses the deck and the
    archive records it, under both modes (the gate ignores the mode)."""
    ops = _up_column_bridge(system="UmfPack", datum=False)
    with pytest.raises(BridgeError, match="NO fixed pressure DOF"):
        ops.tcl(str(tmp_path / "deck.tcl"))
    out = tmp_path / "m.h5"
    ops.h5(str(out))
    stamp = OpenSeesModel.from_h5(out).solve_stamp
    assert stamp is not None
    assert stamp.solve_refusals == ("up_pressure_datum",)
    assert stamp.solve_refusals_flat == ("up_pressure_datum",)


def test_archival_stamp_records_serial_mumps_in_the_archives_own_mode(
    tmp_path: Path,
) -> None:
    """An unpartitioned column with an explicit ``Mumps``: the deck is
    refused ("unknown system type") and the flat archive records the
    gate in its own (flat) verdict."""
    ops = _up_column_bridge(system="Mumps")
    with pytest.raises(BridgeError, match="unknown system type"):
        ops.tcl(str(tmp_path / "deck.tcl"))
    out = tmp_path / "m.h5"
    ops.h5(str(out))
    stamp = OpenSeesModel.from_h5(out).solve_stamp
    assert stamp is not None and stamp.solve_mode == "flat"
    assert stamp.solve_refusals == ("serial_mumps",)
    assert stamp.solve_refusals_flat == ("serial_mumps",)


def test_a_partial_stamp_never_reads_as_unstamped(tmp_path: Path) -> None:
    """Any stamp attribute without ``@will_solve`` is a malformed stamp, not
    an absent one (review of #1616, finding 4)."""
    for name in ("solve_mode", "solve_refusals", "solve_refusals_flat", "requires"):
        out = _write(tmp_path / f"{name}.h5", (True, ("g",)), fork=True)
        _tamper(out, will_solve=None)
        with h5py.File(out, "r+") as f:
            for other in ("solve_mode", "solve_refusals", "solve_refusals_flat", "requires"):
                if other != name:
                    del f["opensees"].attrs[other]
        with h5_reader.open(str(out)) as m, pytest.raises(
                MalformedH5Error, match=rf"carries \['{name}'\] without @will_solve"):
            m.solve_stamp()


# -- build('live') refuses a fork requirement on stock -----------------------


def _backend(kind: str) -> BackendInfo:
    return BackendInfo(kind=kind, build=None if kind == "stock" else "a" * 40,
                       version="3.7.1", source="fake.pyd")


def test_refuse_live_requires_verdicts() -> None:
    refuse_live_requires(None, _backend("stock"))
    refuse_live_requires(SolveStamp(True), _backend("stock"))
    refuse_live_requires(SolveStamp(True, requires=("fork",)), _backend("fork"))
    with pytest.raises(BridgeError, match="require 'fork'.*stock build"):
        refuse_live_requires(SolveStamp(True, requires=("fork",)), _backend("stock"))
    with pytest.raises(BridgeError, match="'mp'.*newer bridge"):
        refuse_live_requires(SolveStamp(True, requires=("fork", "mp")), _backend("fork"))


def test_build_live_refuses_before_touching_the_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apeGmsh.opensees.emitter.live as live_mod

    out = tmp_path / "m.h5"
    _frame_bridge(analysis=True).h5(str(out))
    _tamper(out, requires=["fork"])
    model = OpenSeesModel.from_h5(out)
    assert model.solve_stamp is not None and model.solve_stamp.requires == ("fork",)

    constructed: list[object] = []

    class _Live:
        def __init__(self, *, wipe: bool = True) -> None:
            constructed.append(self)
            raise AssertionError("the live emitter must not be constructed")

    monkeypatch.setattr(live_mod, "LiveOpsEmitter", _Live)
    monkeypatch.setattr(live_mod, "get_backend_info", lambda: _backend("stock"))
    with pytest.raises(BridgeError, match="require 'fork'"):
        model.build("live")
    assert constructed == []
