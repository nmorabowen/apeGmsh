"""ADR 0115 amendment R2-f (#1735, ruling on #1661, option (a)): a record
declared with ``label=`` keeps the label as **provenance only**.

Oracles, each naming the right answer:

* **The label round-trips.** For each entry point (``ops.fix``,
  ``ops.mass``, ``s.fix``, ``s.mass``, ``recorder.Node``,
  ``recorder.Element``), the ``/opensees/decl_params`` row of a
  ``label=`` declaration reads back ``label`` as the tuple of names the
  call was given, beside the resolved ids.
* **No label without ``label=``.** A ``pg=`` or ``nodes=`` record stores
  ``label: null`` (read back ``None``), the documented "empty" choice.
* **An old archive still reads.** A committed corpus archive written
  before the field opens; its fix and mass rows carry no ``label`` key.
* **Replay ignores the label.** An archive whose stored label is
  tampered to name a *different* node set replays (``OpenSeesModel``
  fixes and masses, the rebuilt deck, ``derive_sequence``) exactly as
  the untampered one: the ids are authoritative.
* **Invariant 6, deck bytes.** ``label=`` and ``pg=`` over the same
  nodes emit identical decks on every entry point, and their archives
  hash identically (``decl_params`` is hash-excluded).
* **The stale ``fix_from_model`` message** (review of #1660) names the
  stage's ``s.fix`` / ``s.support``, not a global ``ops.fix``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.records import FixRecord, MassRecord
from apeGmsh.opensees._internal.sequence_h5 import derive_sequence
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.recorder import Element as ElementRec
from apeGmsh.opensees.recorder import Node as NodeRec

from tests.opensees.unit.test_label_selector_fix_merge import (
    _deck,
    build_footing_fem,
    chain,
    footing,
)
from tests.opensees.unit.test_unconsumed_model_definition import (
    _model,
    _ops,
)

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"


@pytest.fixture(scope="module")
def fem() -> Any:
    return build_footing_fem()


def _labelled(fem: Any, sel: str = "label") -> apeSees:
    """Every entry point once, by ``label=`` (``sel="label"``) or by the
    physical group over the same nodes (``sel="pg"``); every declaration
    named so its key is fixed."""
    left = {"label": "left"} if sel == "label" else {"pg": "Left"}
    bottom = {"label": ["bottom"]} if sel == "label" else {"pg": "Bottom"}
    ops = footing(fem)
    ops.fix(dofs=(1, 0), name="sup", **left)
    ops.mass(values=(2.0, 2.0), name="m", **bottom)
    ops.recorder.Node(file="n.out", response="disp", dofs=(1, 2),
                      name="rn", **left)
    if sel == "label":
        ops.recorder.Element(file="e.out", response=("stresses",),
                             label="plate", name="re")
    else:
        ops.recorder.Element(file="e.out", response=("stresses",),
                             pg="Plate", name="re")
    with ops.stage(name="s1") as s:
        s.fix(dofs=(0, 1), name="sfix", **bottom)
        s.mass(values=(1.0, 1.0), name="smass", overwrite=True, **left)
        s.analysis(**chain(ops))
        s.run(n_increments=1)
    return ops


def _table(path: Path) -> h5_reader.DeclarationTable:
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
    assert table is not None
    return table


def _params(table: h5_reader.DeclarationTable, key: str) -> dict[str, Any]:
    (ro,) = table.params_for(key)
    return dict(ro.params)


def _ids(fem: Any, label: str) -> tuple[int, ...]:
    return tuple(int(n) for n in fem.nodes.select(label=label).ids)


# ---------------------------------------------------------------------------
# The record and the archive keep the label
# ---------------------------------------------------------------------------


def test_records_keep_the_label_names_beside_the_ids(fem: Any) -> None:
    ops = _labelled(fem)
    [fix] = ops._fix_records
    assert fix.label == ("left",) and fix.nodes == _ids(fem, "left")
    assert fix.pg is None
    [mass] = ops._mass_records
    assert mass.label == ("bottom",) and mass.nodes == _ids(fem, "bottom")
    [st] = ops._stage_records
    assert st.fix_records[0].label == ("bottom",)
    assert st.mass_records[0].label == ("left",)
    rn, re_ = (p for p in ops._primitives if isinstance(p, (NodeRec, ElementRec)))
    assert rn.label == ("left",) and rn.nodes == _ids(fem, "left")
    assert re_.label == ("plate",) and re_.pg is None


def test_label_round_trips_in_decl_params_for_each_entry_point(
    fem: Any, tmp_path: Path,
) -> None:
    path = tmp_path / "label.h5"
    _labelled(fem).h5(str(path))
    table = _table(path)
    left, bottom = list(_ids(fem, "left")), list(_ids(fem, "bottom"))
    expect = {
        "opensees/fix/sup": (("left",), left),
        "opensees/mass/m": (("bottom",), bottom),
        "opensees/recorder/rn": (("left",), left),
        "opensees/fix/sfix": (("bottom",), bottom),
        "opensees/mass/smass": (("left",), left),
    }
    for key, (label, ids) in expect.items():
        params = _params(table, key)
        assert params["label"] == label, key
        assert list(params["nodes"]) == ids, key
        assert params["pg"] is None, key
    rec = _params(table, "opensees/recorder/re")
    assert rec["label"] == ("plate",) and rec["pg"] is None
    assert len(rec["elements"]) == len(fem.elements.ids)


def test_pg_and_nodes_records_store_a_null_label(
    fem: Any, tmp_path: Path,
) -> None:
    path = tmp_path / "pg.h5"
    ops = _labelled(fem, sel="pg")
    ops.fix(nodes=[int(fem.nodes.ids[-1])], dofs=(1, 0), name="byid")
    ops.h5(str(path))
    table = _table(path)
    for key in ("opensees/fix/sup", "opensees/mass/m", "opensees/recorder/rn",
                "opensees/recorder/re", "opensees/fix/sfix",
                "opensees/mass/smass", "opensees/fix/byid"):
        params = _params(table, key)
        assert "label" in params and params["label"] is None, key
    with h5py.File(str(path), "r") as f:
        rows = [json.loads(r) for r in
                f["opensees"]["decl_params"]["params"].asstr()[()]]
    assert all(r["label"] is None for r in rows if "label" in r)


def test_old_archives_without_the_field_still_read() -> None:
    """Every committed corpus archive with ``decl_params`` opens; one
    written before the field has fix / mass rows without a ``label``."""
    seen_without = 0
    for path in sorted(CORPUS.glob("opensees_*.h5")):
        with h5py.File(str(path), "r") as f:
            if "decl_params" not in f.get("opensees", {}):
                continue  # before the group, or below the reader floor
        table = _table(path)
        for i, d in enumerate(table.decls):
            if d.family not in ("fix", "mass") or i not in table.params:
                continue
            (ro,) = table.params[i]
            if "label" not in ro.params:
                seen_without += 1
                model = OpenSeesModel.from_h5(str(path))
                assert model.fixes() or model.masses()
    assert seen_without > 0


# ---------------------------------------------------------------------------
# Replay never re-resolves the label
# ---------------------------------------------------------------------------


def _retarget_label(path: Path, key: str, label: str) -> None:
    """Rewrite the stored ``label`` of declaration ``key`` to ``label``."""
    table = _table(path)
    index = next(i for i, d in enumerate(table.decls) if d.key == key)
    with h5py.File(str(path), "a") as f:
        g = f["opensees"]["decl_params"]
        rows = list(g["params"].asstr()[()])
        [row] = [r for r, decl in enumerate(g["decl"][()]) if decl == index]
        params = json.loads(rows[row])
        params["label"] = [label]
        rows[row] = json.dumps(params)
        dt = g["params"].dtype
        del g["params"]
        g.create_dataset("params", data=rows, dtype=dt)


def test_replay_ignores_the_label(fem: Any, tmp_path: Path) -> None:
    clean, tampered = tmp_path / "clean.h5", tmp_path / "tampered.h5"
    ops = _labelled(fem)
    ops.h5(str(clean))
    ops.h5(str(tampered))
    # The stored label now names another node set ("bottom", not "left").
    assert set(_ids(fem, "bottom")) != set(_ids(fem, "left"))
    for key in ("opensees/fix/sup", "opensees/mass/smass"):
        _retarget_label(tampered, key, "bottom")
    assert _params(_table(tampered), "opensees/fix/sup")["label"] == ("bottom",)

    a, b = OpenSeesModel.from_h5(str(clean)), OpenSeesModel.from_h5(str(tampered))
    assert sorted(r.tag for r in b.fixes()) == sorted(_ids(fem, "left"))
    assert b.fixes() == a.fixes() and b.masses() == a.masses()
    assert b.stages() == a.stages()
    assert b.build("tcl") == a.build("tcl")
    with h5_reader.open(str(clean)) as ma, h5_reader.open(str(tampered)) as mb:
        assert derive_sequence(mb) == derive_sequence(ma)


# ---------------------------------------------------------------------------
# Invariant 6: the deck never reads the label
# ---------------------------------------------------------------------------


def test_label_and_pg_emit_identical_decks_and_hashes(
    fem: Any, tmp_path: Path,
) -> None:
    assert _deck(_labelled(fem)) == _deck(_labelled(fem, sel="pg"))
    hashes = []
    for sel in ("label", "pg"):
        path = tmp_path / f"{sel}.h5"
        _labelled(fem, sel=sel).h5(str(path))
        with h5py.File(str(path), "r") as f:
            hashes.append(str(f["meta"]["lineage"].attrs["model_hash"]))
    assert hashes[0] == hashes[1]


# ---------------------------------------------------------------------------
# A label is provenance of resolved ids, never a selector of its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [
    lambda: FixRecord(pg="Left", nodes=None, dofs=(1,), label=("left",)),
    lambda: FixRecord(pg=None, nodes=None, dofs=(1,), label=("left",)),
    lambda: FixRecord(pg=None, nodes=(1,), dofs=(1,), label=()),
    lambda: FixRecord(pg=None, nodes=(1,), dofs=(1,), label=(" ",)),
    lambda: FixRecord(pg=None, nodes=(1,), dofs=(1,),
                      label="left"),  # type: ignore[arg-type]
    lambda: MassRecord(pg="Left", nodes=None, values=(1.0,), label=("left",)),
    lambda: NodeRec(file="n", response="disp", pg="Left", dofs=(1,),
                    label=("left",)),
    lambda: ElementRec(file="e", response=("stresses",), pg="Plate",
                       label=("plate",)),
])
def test_a_label_without_resolved_ids_refuses(make: Any) -> None:
    with pytest.raises(ValueError, match="label"):
        make()


# ---------------------------------------------------------------------------
# fix_from_model's clash message (review of #1660)
# ---------------------------------------------------------------------------


def test_fix_from_model_clash_names_the_stage_tier() -> None:
    ops = _ops(_model())
    ops.fix_from_model()
    with ops.stage(name="late") as s:
        s.support(nodes=[5], dofs=(0, 1, 0))
        s.analysis(**chain(ops))
        s.run(n_increments=1, dt=1.0)
    with pytest.raises(BridgeError) as info:
        ops.build()
    msg = str(info.value)
    assert "a stage's s.fix / s.support" in msg
    assert "explicit ops.fix" not in msg
