"""``/opensees@will_solve`` / ``@solve_refusals`` / ``@requires`` (ADR 0114 D6, opensees 2.24.0).

The writer side of R4 (K1-5): ``H5Emitter.set_solve_stamp`` takes one
``SolveStamp`` and ``write`` stamps the three attributes together on
``/opensees``; ``H5Model.solve_stamp`` reads them back. Oracles:

- **Round trip.** Every field comes back as it went in, empty tuples
  included.
- **Absent is None.** A writer that was never handed a stamp writes no
  attribute, and a reader returns ``None`` for it and for every file
  below 2.24.0 (the corpus file of the prior minor).
- **Malformed fails loud.** A partial stamp, a non-0/1 ``@will_solve``, an
  unsorted ``@requires`` or an empty token raises ``MalformedH5Error``.
- **Hash scope.** The stamp folds into ``model_hash``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.lineage import compute_model_hash
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.caps import SolveStamp
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

from tests.fixtures.schema import OPENSEES_CURRENT, OPENSEES_PRIOR_MINOR

_CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"


def _write(path: Path, stamp: SolveStamp | None) -> Path:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.node(1, 0.0, 0.0, 0.0)
    e.fix(1, 1, 1, 1, 1, 1, 1)
    if stamp is not None:
        e.set_solve_stamp(stamp)
    e.write(str(path))
    return path


@pytest.mark.parametrize("stamp", [
    SolveStamp(True, ("ladruno_up_solver",), ("fork",)),
    SolveStamp(False, (), ()),
    SolveStamp(True, ("a", "b", "a"), ("fork", "mp")),
], ids=["full", "empty", "repeated-refusal"])
def test_solve_stamp_round_trips(tmp_path: Path, stamp: SolveStamp) -> None:
    out = _write(tmp_path / "m.h5", stamp)
    with h5py.File(out, "r") as f:
        attrs = f["opensees"].attrs
        assert f["meta"].attrs["opensees_schema_version"] == OPENSEES_CURRENT
        assert attrs["will_solve"].dtype == np.int8
        assert int(attrs["will_solve"]) == int(stamp.will_solve)
        assert attrs["solve_refusals"].shape == (len(stamp.solve_refusals),)
        assert attrs["requires"].shape == (len(stamp.requires),)
    with h5_reader.open(str(out)) as m:
        assert m.solve_stamp() == stamp


def test_a_writer_never_handed_a_stamp_writes_no_attribute(tmp_path: Path) -> None:
    out = _write(tmp_path / "m.h5", None)
    with h5py.File(out, "r") as f:
        assert set(f["opensees"].attrs) == set()
    with h5_reader.open(str(out)) as m:
        assert m.solve_stamp() is None


def test_prior_minor_corpus_file_has_no_stamp() -> None:
    minor = ".".join(OPENSEES_PRIOR_MINOR.split(".")[:2])
    path = _CORPUS / f"opensees_{minor}.h5"
    assert path.exists(), f"corpus file {path.name} missing; rebuild the corpus"
    with h5_reader.open(str(path)) as m:
        assert m.solve_stamp() is None


def test_set_solve_stamp_refuses_a_second_call_and_a_foreign_type() -> None:
    e = H5Emitter()
    e.set_solve_stamp(SolveStamp(True))
    with pytest.raises(RuntimeError, match="already set"):
        e.set_solve_stamp(SolveStamp(False))
    with pytest.raises(TypeError, match="expected a SolveStamp"):
        H5Emitter().set_solve_stamp({"will_solve": True})  # type: ignore[arg-type]


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
    ({"requires": ["mp", "fork"]}, "sorted and unique"),
    ({"requires": ["fork", "fork"]}, "sorted and unique"),
    ({"solve_refusals": ["ok", ""]}, "empty token"),
    ({"requires": "fork"}, "expected a 1-D array"),
], ids=[
    "will_solve-2", "will_solve-float", "will_solve-array", "no-requires",
    "no-refusals", "unsorted", "duplicate", "empty-token", "scalar-requires",
])
def test_malformed_stamp_fails_loud(tmp_path: Path, tamper: dict, match: str) -> None:
    out = _write(tmp_path / "m.h5", SolveStamp(True, ("g",), ("fork",)))
    _tamper(out, **tamper)
    with h5_reader.open(str(out)) as m, pytest.raises(MalformedH5Error, match=match):
        m.solve_stamp()


def test_stamp_folds_into_model_hash(tmp_path: Path) -> None:
    def digest(path: Path) -> str:
        with h5py.File(path, "r") as f:
            return compute_model_hash("", f["opensees"])

    none = digest(_write(tmp_path / "none.h5", None))
    solve = digest(_write(tmp_path / "solve.h5", SolveStamp(True)))
    archive = digest(_write(tmp_path / "archive.h5", SolveStamp(False)))
    fork = digest(_write(tmp_path / "fork.h5", SolveStamp(True, requires=("fork",))))
    assert len({none, solve, archive, fork}) == 4
    # Deterministic for the same stamp.
    assert digest(_write(tmp_path / "solve2.h5", SolveStamp(True))) == solve
