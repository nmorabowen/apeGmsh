"""#1290: the documented join from a bar's line cells to its auto-emitted
``CorotTruss`` rows in ``model.h5``, read with raw h5py only (ADR 0112 D2).

``emit_rebar_elements`` mints one ``CorotTruss`` per bar line cell with
``fem_eids = -1`` (the ADR 0049 node-pair sentinel, ADR 0093 S10) and the
true endpoints in ``inline_connectivity``.  ``architecture/h5-schema.md``
(``/opensees/element_meta``, "Bridge-minted rows") names the join key a
reader uses instead of ``fem_eids``:

* the inline ``(i, j)`` pair equals the corner pair of exactly one line cell
  of the bar's physical group, when the extraction kept the dim-1 cells in
  ``/elements``;
* the rows follow ``/rebar_elements/elements`` order, record by record and
  cell by cell, whether or not ``/elements`` holds the cells;
* ``args[:, 1]`` is the uniaxial material tag, which ``/opensees/names``
  maps back to the record's material name.

The oracle is the neutral zone itself: every expected value comes from
``/elements``, ``/physical_groups`` and ``/rebar_elements``, which the bridge
does not write.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.rebar import Cage
from apeGmsh.opensees import apeSees

pytestmark = pytest.mark.filterwarnings(
    "ignore:g.rebar.place. embedded coupling:UserWarning")


def _build_and_emit(path, *, dim):
    """A box host with two embedded truss bars (``emit_elements=True``),
    written to ``path`` through the bridge's H5 emitter."""
    with apeGmsh(model_name=f"rebar_h5_join_{dim}") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0)
        g.physical.add_volume([vol], name="Concrete")
        bars = (
            g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)], db=0.0254,
                        material="rebar", element="truss", name="A"),
            g.rebar.bar([(0.35, 0.35, 0.1), (0.35, 0.35, 1.9)], db=0.0254,
                        material="rebar", element="truss", name="B"),
        )
        g.rebar.place(Cage(bars=bars), into="Concrete", coupling="embedded",
                      perfect=1.0e8, emit_elements=True)
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=dim)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="rebar")
        ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
        ops.element.FourNodeTetrahedron(pg="Concrete", material="conc")
        ops.h5(str(path))


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory):
    """One emitted file per extraction dim (``None`` keeps the dim-1 bar
    cells in ``/elements``; ``3`` drops them, the common case)."""
    paths = {}

    def _get(dim):
        if dim not in paths:
            paths[dim] = tmp_path_factory.mktemp("rebar_h5_join") / "model.h5"
            _build_and_emit(paths[dim], dim=dim)
        return paths[dim]
    return _get


def _str(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _rebar_records(f):
    """``[(pg, material, [(i, j), ...]), ...]`` from ``/rebar_elements``."""
    out = []
    for row in f["rebar_elements/elements"][...]:
        p = row["payload"]
        flat = np.asarray(p["connectivity"], dtype=np.int64).reshape(-1, 2)
        out.append((_str(p["pg"]), _str(p["material"]),
                    [(int(i), int(j)) for i, j in flat]))
    return out


def _truss_rows(f):
    g = f["opensees/element_meta/CorotTruss"]
    return (
        np.asarray(g["fem_eids"][...]),
        [tuple(int(c) for c in r) for r in g["inline_connectivity"][...]],
        np.asarray(g["args"][...]),
    )


@pytest.mark.parametrize("dim", [None, 3])
def test_rebar_truss_rows_follow_rebar_elements_and_name_their_material(
    h5_path, dim,
):
    """Sentinel ``fem_eids``, inline pairs in ``/rebar_elements`` order, and
    a material tag that ``/opensees/names`` maps to the record's name."""
    with h5py.File(h5_path(dim), "r") as f:
        recs = _rebar_records(f)
        fem_eids, inline, args = _truss_rows(f)

        assert [pg for pg, _, _ in recs] == ["rebar0.A", "rebar0.B"]
        expected_pairs = [pair for _, _, conn in recs for pair in conn]
        assert inline == expected_pairs
        assert (fem_eids == -1).all()

        names = f["opensees/names"]
        kind = [_str(k) for k in names["kind"][...]]
        name = [_str(n) for n in names["name"][...]]
        tag = [int(t) for t in names["tag"][...]]
        uniaxial = {t: n for k, n, t in zip(kind, name, tag)
                    if k == "uniaxialMaterial"}
        expected_mats = [mat for _, mat, conn in recs for _ in conn]
        assert [uniaxial[int(t)] for t in args[:, 1]] == expected_mats


def test_rebar_truss_rows_join_bar_cells_by_node_pair(h5_path):
    """With the dim-1 cells in ``/elements``, each bar cell's corner pair
    matches exactly one ``CorotTruss`` row, and each row matches exactly
    one cell of its bar's physical group (a bijection per bar)."""
    with h5py.File(h5_path(None), "r") as f:
        recs = _rebar_records(f)
        _, inline, _ = _truss_rows(f)

        line_cells: dict[int, tuple[int, int]] = {}
        for grp in f["elements"].values():
            if int(grp.attrs["dim"]) != 1:
                continue
            ids = np.asarray(grp["ids"][...])
            conn = np.asarray(grp["connectivity"][...])
            for eid, row in zip(ids, conn):
                line_cells[int(eid)] = (int(row[0]), int(row[1]))

        start = 0
        for pg, _, conn in recs:
            pg_ids = [int(e) for e in
                      f[f"physical_groups/element_side/{pg}/element_ids"][...]]
            assert pg_ids, pg
            cell_pairs = sorted(line_cells[e] for e in pg_ids)
            rows = inline[start:start + len(conn)]
            start += len(conn)
            assert sorted(rows) == cell_pairs
            assert len(set(rows)) == len(rows)
        assert start == len(inline)
