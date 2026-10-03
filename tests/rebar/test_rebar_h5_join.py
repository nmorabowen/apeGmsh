"""#1290: the documented join from a bar's line cells to its auto-emitted
``CorotTruss`` rows in ``model.h5``, read with raw h5py only (ADR 0112 D2).

``emit_rebar_elements`` mints one ``CorotTruss`` per bar line cell with
``fem_eids = -1`` (the ADR 0049 node-pair sentinel, ADR 0093 S10) and the
true endpoints in ``inline_connectivity``.  ``architecture/h5-schema.md``
(``/opensees/element_meta``, "Bridge-minted rows") names the join key a
reader uses instead of ``fem_eids``:

* the rebar rows are the ``CorotTruss`` rows with ``fem_eids == -1``, and
  they form one contiguous block (a user ``ops.element.CorotTruss(pg=...)``
  writes into the same type group with real ``fem_eids`` and empty inline
  rows);
* inside that block the rows follow ``/rebar_elements/elements`` order,
  record by record and cell by cell, whether or not ``/elements`` holds the
  cells;
* each inline ``(i, j)`` pair equals the corner pair of exactly one line
  cell of the bar's physical group, when the extraction kept the dim-1
  cells in ``/elements``;
* ``args[:, 1]`` is the uniaxial material tag, which ``/opensees/names``
  maps back to the record's material name.

The oracle is the neutral zone itself: every expected value comes from
``/elements``, ``/physical_groups`` and ``/rebar_elements``, which the bridge
does not write.  The two bars use different materials so a mutant that
collapses every row onto one material tag fails.
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


def _build_and_emit(path, *, dim, user_truss):
    """A box host with two embedded truss bars (``emit_elements=True``,
    materials ``rebarA`` / ``rebarB``), written to ``path`` through the
    bridge's H5 emitter.  ``user_truss`` adds a free ``Brace`` line carrying
    a user ``ops.element.CorotTruss(pg="Brace")`` (needs ``dim=None``)."""
    with apeGmsh(model_name=f"rebar_h5_join_{dim}_{user_truss}") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0)
        g.physical.add_volume([vol], name="Concrete")
        if user_truss:
            p0 = g.model.geometry.add_point(1.0, 0.0, 0.0)
            p1 = g.model.geometry.add_point(1.0, 0.0, 2.0)
            ln = g.model.geometry.add_line(p0, p1)
            g.physical.add(1, [ln], name="Brace")
        bars = (
            g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)], db=0.0254,
                        material="rebarA", element="truss", name="A"),
            g.rebar.bar([(0.35, 0.35, 0.1), (0.35, 0.35, 1.9)], db=0.0254,
                        material="rebarB", element="truss", name="B"),
        )
        g.rebar.place(Cage(bars=bars), into="Concrete", coupling="embedded",
                      perfect=1.0e8, emit_elements=True)
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=dim)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="rebarA")
        ops.uniaxialMaterial.Steel02(fy=500e6, E=200e9, b=0.02, name="rebarB")
        ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
        ops.element.FourNodeTetrahedron(pg="Concrete", material="conc")
        if user_truss:
            brace = ops.uniaxialMaterial.Steel02(
                fy=350e6, E=200e9, b=0.01, name="brace")
            ops.element.CorotTruss(pg="Brace", A=1.0e-3, material=brace)
        ops.h5(str(path))


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory):
    """One emitted file per ``(dim, user_truss)``: ``dim=None`` keeps the
    dim-1 cells in ``/elements``; ``dim=3`` drops them (the common case)."""
    paths = {}

    def _get(dim, user_truss=False):
        key = (dim, user_truss)
        if key not in paths:
            paths[key] = tmp_path_factory.mktemp("rebar_h5_join") / "model.h5"
            _build_and_emit(paths[key], dim=dim, user_truss=user_truss)
        return paths[key]
    return _get


def _str(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _rebar_records(f):
    """``[(pg, material, [(i, j), ...]), ...]`` from the ``payload`` field of
    the ``/rebar_elements/elements`` symmetric compound."""
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


def _rebar_block(fem_eids):
    """The slice of ``CorotTruss`` rows with ``fem_eids == -1``, asserting
    that they are one contiguous block."""
    idx = np.flatnonzero(fem_eids == -1)
    assert idx.size, "no bridge-minted CorotTruss rows"
    assert (np.diff(idx) == 1).all(), f"rebar rows not contiguous: {idx}"
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _uniaxial_names(f):
    names = f["opensees/names"]
    kind = [_str(k) for k in names["kind"][...]]
    name = [_str(n) for n in names["name"][...]]
    tag = [int(t) for t in names["tag"][...]]
    return {t: n for k, n, t in zip(kind, name, tag)
            if k == "uniaxialMaterial"}


def _line_cells(f):
    """``{eid: (corner_i, corner_j)}`` over every dim-1 ``/elements`` group."""
    cells: dict[int, tuple[int, int]] = {}
    for grp in f["elements"].values():
        if int(grp.attrs["dim"]) != 1:
            continue
        ids = np.asarray(grp["ids"][...])
        conn = np.asarray(grp["connectivity"][...])
        for eid, row in zip(ids, conn):
            cells[int(eid)] = (int(row[0]), int(row[1]))
    return cells


def _assert_rebar_block_matches_records(f):
    recs = _rebar_records(f)
    fem_eids, inline, args = _truss_rows(f)
    block = _rebar_block(fem_eids)

    assert [pg for pg, _, _ in recs] == ["rebar0.A", "rebar0.B"]
    expected_pairs = [pair for _, _, conn in recs for pair in conn]
    assert inline[block] == expected_pairs

    uniaxial = _uniaxial_names(f)
    tag_of = {n: t for t, n in uniaxial.items()}
    assert tag_of["rebarA"] != tag_of["rebarB"]
    expected_mats = [mat for _, mat, conn in recs for _ in conn]
    assert [uniaxial[int(t)] for t in args[block, 1]] == expected_mats
    return recs, fem_eids, inline, args, block


@pytest.mark.parametrize("dim", [None, 3])
def test_rebar_truss_rows_follow_rebar_elements_and_name_their_material(
    h5_path, dim,
):
    """Sentinel ``fem_eids``, inline pairs in ``/rebar_elements`` order, and
    a per-bar material tag that ``/opensees/names`` maps to the record's
    name."""
    with h5py.File(h5_path(dim), "r") as f:
        _, fem_eids, _, _, block = _assert_rebar_block_matches_records(f)
        assert (block.start, block.stop) == (0, len(fem_eids))


def test_rebar_truss_rows_join_bar_cells_by_node_pair(h5_path):
    """With the dim-1 cells in ``/elements``, each bar cell's corner pair
    matches exactly one ``CorotTruss`` row, and each row matches exactly
    one cell of its bar's physical group (a bijection per bar)."""
    with h5py.File(h5_path(None), "r") as f:
        recs, _, inline, _, block = _assert_rebar_block_matches_records(f)
        line_cells = _line_cells(f)

        start = block.start
        for pg, _, conn in recs:
            pg_ids = [int(e) for e in
                      f[f"physical_groups/element_side/{pg}/element_ids"][...]]
            assert pg_ids, pg
            cell_pairs = sorted(line_cells[e] for e in pg_ids)
            rows = inline[start:start + len(conn)]
            start += len(conn)
            assert sorted(rows) == cell_pairs
            assert len(set(rows)) == len(rows)
        assert start == block.stop


def test_user_corottruss_shares_the_type_group_outside_the_rebar_block(
    h5_path,
):
    """A user ``ops.element.CorotTruss(pg="Brace")`` writes into the same
    ``CorotTruss`` group with real ``fem_eids`` (the ``Brace`` cells) and
    empty inline rows; the rebar rows stay one contiguous ``-1`` block that
    still matches ``/rebar_elements``."""
    with h5py.File(h5_path(None, True), "r") as f:
        _, fem_eids, inline, args, block = (
            _assert_rebar_block_matches_records(f))
        brace_ids = sorted(
            int(e) for e in
            f["physical_groups/element_side/Brace/element_ids"][...])
        assert brace_ids

        user = [i for i in range(len(fem_eids))
                if not block.start <= i < block.stop]
        assert sorted(int(fem_eids[i]) for i in user) == brace_ids
        assert all(inline[i] == () for i in user)
        uniaxial = _uniaxial_names(f)
        assert {uniaxial[int(args[i, 1])] for i in user} == {"brace"}
