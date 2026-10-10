"""Top-level ``/opensees/regions`` survive ``from_h5`` (B4-e, #1579).

Before this slice the writer persisted the flat regions but nothing read
them back: ``OpenSeesModel.from_h5(p).to_h5()`` dropped the group (and
``model_hash`` did not notice, since ``regions`` is hash-excluded) and
``from_h5(p).build('tcl')`` lost every region-scoped ``rayleigh`` line, so
the reloaded structure was under-damped compared with what was declared.

Every top-level region replays verbatim: the archived row is the resolved
OpenSees call, so a plain named region and a region whose K1-6 declaration
is missing (an older archive) replay too. A row missing ``tag`` or
``params`` fails loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.lineage import compute_model_hash
from apeGmsh.opensees.emitter import h5_reader
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _frame(*, rayleigh_on: str | None, named: bool = False) -> apeSees:
    ops = apeSees(cast("object", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.0125, E=2900.25, Iz=1.5e-4, Iy=2.5e-4, G=1100.75, J=3.5e-4,
    )
    if rayleigh_on is not None:
        ops.damping.rayleigh(alpha_m=0.125, beta_k=0.0025, on=rayleigh_on)
    if named:
        ops.region(name="col_nodes", pg="Cols")
    return ops


def _archive(ops: apeSees, tmp_path: Path, name: str = "model.h5") -> Path:
    p = tmp_path / name
    ops.h5(str(p))
    return p


def _bridge_deck(ops: apeSees, tmp_path: Path) -> str:
    p = tmp_path / "bridge.tcl"
    ops.tcl(str(p), progress=False)
    return p.read_text(encoding="utf-8")


def _replay(path: Path) -> str:
    out = OpenSeesModel.from_h5(str(path)).build("tcl")
    assert isinstance(out, str)
    return out


def _rewrite(src: Path, out: Path) -> Path:
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    return out


def _region_lines(deck: str) -> list[str]:
    return [ln for ln in deck.splitlines() if ln.startswith("region ")]


def _regions_zone(path: Path) -> list[tuple[str, int, list[Any]]]:
    """``(group name, tag, params)`` per ``/opensees/regions`` row, raw."""
    out: list[tuple[str, int, list[Any]]] = []
    with h5py.File(str(path), "r") as f:
        if "regions" not in f["opensees"]:
            return out
        grp = f["opensees"]["regions"]
        for name in sorted(grp):
            g = grp[name]
            params = [repr(v) for v in np.asarray(g.attrs["params"]).tolist()]
            strs = [str(v) for v in g.attrs["params_str"]] \
                if "params_str" in g.attrs else []
            out.append((name, int(g.attrs["tag"]), params + strs))
    return out


def _hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return compute_model_hash("", f["opensees"])


# ---------------------------------------------------------------------------
# 1. The two regressions of #1579
# ---------------------------------------------------------------------------


def test_rewrite_preserves_top_level_regions(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh_on="Cols"), tmp_path)
    before = _regions_zone(src)
    assert before, "the bridge writes the region-scoped rayleigh as a region"

    out = _rewrite(src, tmp_path / "rewrite.h5")

    assert _regions_zone(out) == before


def test_replay_deck_keeps_region_scoped_rayleigh(tmp_path: Path) -> None:
    ops = _frame(rayleigh_on="Cols")
    bridge = _region_lines(_bridge_deck(ops, tmp_path))
    assert len(bridge) == 1 and "-rayleigh" in bridge[0]

    replay = _region_lines(_replay(_archive(ops, tmp_path)))

    assert replay == bridge


# ---------------------------------------------------------------------------
# 2. Seams: idempotence, the shared namespace, empty, malformed
# ---------------------------------------------------------------------------


def test_double_round_trip_is_stable(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh_on="Cols", named=True), tmp_path)
    one = _rewrite(src, tmp_path / "one.h5")
    two = _rewrite(one, tmp_path / "two.h5")

    assert _regions_zone(src) == _regions_zone(one) == _regions_zone(two)
    assert _hash(src) == _hash(one) == _hash(two)
    assert _region_lines(_replay(src)) == _region_lines(_replay(one)) \
        == _region_lines(_replay(two))


def test_plain_named_region_replays_too(tmp_path: Path) -> None:
    """A region with no damping (``ops.region``) shares the namespace."""
    ops = _frame(rayleigh_on="Cols", named=True)
    bridge = _region_lines(_bridge_deck(ops, tmp_path))
    assert len(bridge) == 2
    assert any("-node" in ln and "-rayleigh" not in ln for ln in bridge)

    src = _archive(ops, tmp_path)
    assert sorted(_region_lines(_replay(src))) == sorted(bridge)
    assert _regions_zone(_rewrite(src, tmp_path / "rw.h5")) == \
        _regions_zone(src)


def test_region_without_declaration_replays(tmp_path: Path) -> None:
    """An archive older than K1-6 carries no ``/opensees/decls``; the
    region row is the resolved call, so it replays all the same."""
    ops = _frame(rayleigh_on="Cols")
    bridge = _region_lines(_bridge_deck(ops, tmp_path))
    src = _archive(ops, tmp_path)
    with h5py.File(str(src), "a") as f:
        del f["opensees"]["decls"]
    with h5_reader.open(str(src)) as m:
        assert m.declarations() is None

    assert _region_lines(_replay(src)) == bridge
    assert _regions_zone(_rewrite(src, tmp_path / "rw.h5")) == \
        _regions_zone(src)


def test_reader_regions_accessor(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh_on="Cols"), tmp_path)
    with h5_reader.open(str(src)) as m:
        (rec,) = m.regions()
    assert rec.args[0] == "-ele"
    k = rec.args.index("-rayleigh")
    assert rec.args[k + 1:k + 5] == (0.125, 0.0025, 0.0, 0.0)
    assert all(isinstance(v, float) for v in rec.args[k + 1:k + 5])
    (brec,) = OpenSeesModel.from_h5(str(src)).regions()
    assert brec == rec


def test_model_without_regions_is_unchanged(tmp_path: Path) -> None:
    src = _archive(_frame(rayleigh_on=None), tmp_path)
    assert _regions_zone(src) == []
    with h5_reader.open(str(src)) as m:
        assert m.regions() == []
    out = _rewrite(src, tmp_path / "rw.h5")
    assert _regions_zone(out) == []
    assert _hash(out) == _hash(src)
    assert _region_lines(_replay(src)) == []


@pytest.mark.parametrize("missing", ["tag", "params"])
def test_region_row_missing_attr_fails_loud(
    tmp_path: Path, missing: str,
) -> None:
    src = _archive(_frame(rayleigh_on="Cols"), tmp_path)
    with h5py.File(str(src), "a") as f:
        del f["opensees"]["regions"]["region_000"].attrs[missing]
    with pytest.raises(h5_reader.MalformedH5Error, match="region_000"):
        OpenSeesModel.from_h5(str(src))
