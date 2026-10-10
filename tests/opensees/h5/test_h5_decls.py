"""``/opensees/decls``: the bridge's declarations in ``model.h5`` (ADR 0114
R5, program slice K1-6 #1463, opensees schema 2.25.0).

Oracles, each naming the right answer:

* **Every declaration has a key.** The decls rows are exactly the bridge's
  ``/provenance`` paths (the declaration keys of ADR 0112 D3); every
  registered primitive's ``(kind, tag)`` resolves to its own key, a
  fan-out element's tag to its spec's key, and every row of every tagless
  store (``bcs/fix``, ``bcs/mass``, ``recorders`` and their stage twins)
  to the declaration that wrote it.
* **Names read back.** ``name=`` on a flat and a staged ``fix``, on
  ``mass`` and on the recorders is what the reader (``h5_reader``) and
  ``OpenSeesModel.declarations`` return for those rows.
* **Labels, not structure (A7).** Two models that differ only in those
  names have the same ``model_hash``, and the deck bytes are equal; the
  ``decls`` group is excluded (the hash recomputed with it deleted is the
  stamped one), and ``/opensees/program``'s hashed ``decl`` column stays
  ``-1``.
* **The ``declare`` name.** ``RecorderDeclaration``'s own ``name``
  (default ``"default"``, the ``.out`` stem) is forwarded as the bridge
  name only when given: two unnamed declarations still coexist, a repeated
  given name raises.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.lineage import (
    MODEL_HASH_EXCLUDED_CHILDREN,
    compute_model_hash,
)
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees._internal.types import Element
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.golden.builder import build_model
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _two_quad_stub() -> FEMStub:
    """Two quads: "Rock" global, "Fill" activated in a stage."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.0),
            ],
            node_pgs={
                "Rock": [1, 2, 3, 4], "Fill": [3, 4, 5, 6],
                "Base": [1, 2], "FillTop": [5, 6],
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),)),
                "Fill": _ElementGroupView(
                    ids=(2,), connectivity=((4, 3, 5, 6),)),
            },
        ),
    )


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _staged(*, named: bool) -> apeSees:
    """A flat fix and mass, a staged fix and mass, a global and a staged
    recorder; every one of them named iff ``named``."""
    def nm(name: str) -> str | None:
        return name if named else None

    ops = apeSees(_two_quad_stub(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1), name=nm("base"))
    ops.mass(pg="Rock", values=(1.0, 1.0), name=nm("rock_mass"))
    ops.recorder.Node(
        file="out/disp.out", response="disp", pg="Rock", dofs=(1, 2),
        name=nm("rock_disp"),
    )
    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.fix(pg="FillTop", dofs=(1, 1), name=nm("fill_top"))
        s.mass(nodes=[5, 6], values=(2.0, 2.0), name=nm("fill_mass"))
        rec = ops.recorder.Node(
            file="out/fill.out", response="disp", pg="FillTop", dofs=(1,),
            name=nm("fill_disp"),
        )
        s.recorder(rec)
        s.analysis(**_chain(ops))
        s.run(n_increments=2)
    return ops


def _flat_frame(*, named: bool) -> apeSees:
    """One column on a real FEMData, so ``OpenSeesModel.from_h5`` reads it."""
    def nm(name: str) -> str | None:
        return name if named else None

    ops = apeSees(cast("Any", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(nodes=[1], dofs=(1,) * 6, name=nm("base"))
    ops.mass(nodes=[2], values=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
             name=nm("top_mass"))
    ops.recorder.Node(
        file="out/top.out", response="disp", nodes=(2,), dofs=(1, 2),
        name=nm("top_disp"),
    )
    return ops


def _decls(path: Path) -> h5_reader.DeclarationTable:
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
    assert table is not None
    return table


def _stored_model_hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


# ---------------------------------------------------------------------------
# Every declaration has a key
# ---------------------------------------------------------------------------


def _assert_every_declaration_keyed(ops: apeSees, path: Path) -> None:
    table = _decls(path)
    keys = [d.key for d in table.decls]
    assert len(keys) == len(set(keys))
    # The decls rows are the bridge's provenance records, one for one,
    # beside the ``@k`` keys of objects registered inside a recorded call.
    assert {k for k in keys if "/@" not in k} == {
        r.path for r in ops._provenance.snapshot().records}
    bm = ops.build()
    from apeGmsh.opensees.apesees import _kind_of

    own = {id(o): d[0] for o, d in ops._decls.values()}
    for prim in bm.primitives:
        if isinstance(prim, Element):
            continue
        got = table.for_tag(_kind_of(prim), bm.tag_for[id(prim)])
        assert got.key == own[id(prim)], prim
    # Every tagless row of every store this archive wrote is claimed.
    with h5py.File(str(path), "r") as f:
        ops_grp = f["opensees"]
        stores = [s for s in ("bcs/fix", "bcs/mass", "recorders")
                  if s in ops_grp]
        if "stages" in ops_grp:
            for stage in ops_grp["stages"]:
                stores += [f"stages/{stage}/{s}"
                           for s in ("bcs/fix", "bcs/mass", "recorders")
                           if s in ops_grp["stages"][stage]]
        for store in stores:
            assert len(table.rows[store]) == len(ops_grp[store]), store
    # And /opensees/program's hashed decl column never carries one.
    with h5py.File(str(path), "r") as f:
        assert set(f["opensees"]["program"]["decl"][()].tolist()) == {-1}


def test_every_declaration_has_a_key_staged(tmp_path: Path) -> None:
    ops = _staged(named=False)
    p = tmp_path / "staged.h5"
    ops.h5(str(p))
    _assert_every_declaration_keyed(ops, p)
    table = _decls(p)
    # Unnamed, the keys are the families' ordinals.
    assert table.for_row("bcs/fix", 0).key == "opensees/fix/#1"
    assert table.for_row("stages/stage_000/bcs/fix", 0).key == "opensees/fix/#2"
    assert table.for_row("bcs/mass", 0).key == "opensees/mass/#1"
    assert table.for_row("stages/stage_000/bcs/mass", 1).key == "opensees/mass/#2"


@pytest.mark.parametrize("mode", [
    "flat", "partitioned", "staged", "staged_partitioned",
])
def test_every_declaration_has_a_key_golden(tmp_path: Path, mode: str) -> None:
    """The golden recording models, the partitioned ones included: a
    partition replica's row carries its first capture's declaration."""
    ops = build_model("two_column_frame_partitioned", mode, "recording")
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    _assert_every_declaration_keyed(ops, p)


def test_fan_out_elements_inherit_the_spec_key(tmp_path: Path) -> None:
    ops = _staged(named=False)
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _decls(p)
    with h5py.File(str(p), "r") as f:
        ids = f["opensees"]["element_meta"]["quad"]["ids"][()].tolist()
    keys = [table.for_tag("element", int(t)).key for t in ids]
    assert keys == ["opensees/element/#1", "opensees/element/#2"]


def test_orientation_fan_out_transforms_inherit_the_spec_key(
    tmp_path: Path,
) -> None:
    ops = build_model("arch_with_orientation_fan_out", "flat", "tcl")
    p = tmp_path / "arch.h5"
    ops.h5(str(p))
    table = _decls(p)
    with h5py.File(str(p), "r") as f:
        groups = list(f["opensees"]["transforms"])
        tags = [int(f["opensees"]["transforms"][g].attrs["tag"])
                for g in groups]
    assert len(tags) > 1   # one geomTransf per distinct vecxz
    assert {table.for_tag("geomTransf", t).key for t in tags} == {
        "opensees/geomTransf/#1"}


# ---------------------------------------------------------------------------
# Names read back
# ---------------------------------------------------------------------------


def test_names_read_back_staged_via_reader(tmp_path: Path) -> None:
    p = tmp_path / "named.h5"
    _staged(named=True).h5(str(p))
    table = _decls(p)
    assert table.for_row("bcs/fix", 0).name == "base"
    assert table.for_row("bcs/fix", 1).key == "opensees/fix/base"
    assert table.for_row("stages/stage_000/bcs/fix", 0).name == "fill_top"
    assert table.for_row("stages/stage_000/bcs/fix", 1).name == "fill_top"
    assert {table.for_row("bcs/mass", i).name for i in range(4)} == {
        "rock_mass"}
    assert table.for_row("stages/stage_000/bcs/mass", 0).key == (
        "opensees/mass/fill_mass")
    assert table.for_row("recorders", 0).name == "rock_disp"
    assert table.for_row("stages/stage_000/recorders", 0).key == (
        "opensees/recorder/fill_disp")
    # A recorder is tagged too: its tag joins to the same declaration.
    assert table.by_key("opensees/recorder/rock_disp").family == "recorder"


def test_names_read_back_flat_via_opensees_model(tmp_path: Path) -> None:
    p = tmp_path / "named.h5"
    _flat_frame(named=True).h5(str(p))
    om = OpenSeesModel.from_h5(str(p))
    table = om.declarations
    assert table is not None
    assert table.for_row("bcs/fix", 0).name == "base"
    assert table.for_row("bcs/mass", 0).name == "top_mass"
    assert table.for_row("recorders", 0).name == "top_disp"
    # A recorder's name labels its declaration only; nothing resolves a
    # recorder by name, so it is no bridge-wide alias (/opensees/names).
    assert om.tag_for_name("top_disp") is None


def test_rewrite_echoes_the_declarations(tmp_path: Path) -> None:
    p = tmp_path / "src.h5"
    q = tmp_path / "out.h5"
    _flat_frame(named=True).h5(str(p))
    OpenSeesModel.from_h5(str(p)).to_h5(str(q))
    assert _decls(q) == _decls(p)


# ---------------------------------------------------------------------------
# Labels, not structure (A7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_staged, _flat_frame])
def test_names_leave_model_hash_and_deck_unchanged(
    tmp_path: Path, build: Any,
) -> None:
    named, plain = tmp_path / "named.h5", tmp_path / "plain.h5"
    build(named=True).h5(str(named))
    build(named=False).h5(str(plain))
    assert _decls(named) != _decls(plain)
    assert _stored_model_hash(named) == _stored_model_hash(plain)
    build(named=True).tcl(str(tmp_path / "n.tcl"))
    build(named=False).tcl(str(tmp_path / "p.tcl"))
    assert (tmp_path / "n.tcl").read_bytes() == (
        tmp_path / "p.tcl").read_bytes()


def test_decls_group_is_hash_excluded(tmp_path: Path) -> None:
    p = tmp_path / "named.h5"
    _flat_frame(named=True).h5(str(p))
    assert "decls" in MODEL_HASH_EXCLUDED_CHILDREN
    stripped = tmp_path / "stripped.h5"
    shutil.copy(p, stripped)
    with h5py.File(str(stripped), "a") as f:
        fem_hash = str(f["meta"]["lineage"].attrs["fem_hash"])
        del f["opensees"]["decls"]
        assert compute_model_hash(fem_hash, f["opensees"]) == (
            _stored_model_hash(p))


# ---------------------------------------------------------------------------
# Name rules
# ---------------------------------------------------------------------------


def test_fix_names_are_unique_across_flat_and_staged() -> None:
    ops = apeSees(_two_quad_stub(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    ops.fix(pg="Base", dofs=(1, 1), name="pin")
    with pytest.raises(ValueError, match="opensees/fix/pin"):
        ops.fix(pg="FillTop", dofs=(1, 1), name="pin")
    assert len(ops._fix_records) == 1
    with pytest.raises(ValueError, match="opensees/fix/pin"):
        with ops.stage(name="s") as s:
            s.fix(pg="FillTop", dofs=(1, 1), name="pin")
    # A fix and a mass may share a name: one key space per family.
    ops.mass(pg="Base", values=(1.0, 1.0), name="pin")


def test_declare_name_is_the_bridge_name_only_when_given(
    tmp_path: Path,
) -> None:
    """RecorderDeclaration's own ``name="default"`` never collides: the
    unnamed declarations stay unnamed on the bridge (``#k``)."""
    ops = _flat_frame(named=False)
    a = ops.recorder.declare(nodes="displacement_x", pg="Cols")
    b = ops.recorder.declare(nodes="displacement_x", pg="Cols")
    assert a.name == b.name == "default"
    ops.recorder.declare(nodes="displacement_x", pg="Cols", name="drift")
    # The refusal names the duplicate and both call sites.
    with pytest.raises(
        ValueError,
        match=r"'drift'.*test_h5_decls\.py:\d+.*test_h5_decls\.py:\d+",
    ):
        ops.recorder.declare(nodes="displacement_x", pg="Cols", name="drift")
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _decls(p)
    names = [table.for_row("recorders", i).name
             for i in range(len(table.rows["recorders"]))]
    assert names[0] == ""                       # the typed Node recorder
    assert "drift" in names
    assert {d.key for d in table.decls if d.family == "recorder"} == {
        "opensees/recorder/#1", "opensees/recorder/#2",
        "opensees/recorder/#3", "opensees/recorder/drift",
    }


# ---------------------------------------------------------------------------
# The archive side channel fails closed
# ---------------------------------------------------------------------------


def test_tagless_row_without_a_declaration_refuses(tmp_path: Path) -> None:
    """A fix written while no declaration is open is a row no declaration
    claims; the writer refuses it rather than archive it unclaimed."""
    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.set_declarations([("opensees/fix/#1", "fix", "", False)], [])
    e.set_declaration(0)
    e.fix(1, 1, 1)
    e.set_declaration(-1)
    e.fix(2, 1, 1)
    with pytest.raises(RuntimeError, match="bcs/fix row 1"):
        e.write(str(tmp_path / "x.h5"))


@pytest.mark.parametrize("staged", [False, True])
def test_replica_gives_a_ghost_first_row_its_declaration(
    tmp_path: Path, staged: bool,
) -> None:
    """A ghost replay captured before its owner's fix (rank 0 mirrors the
    node rank 1 owns) is written with no declaration open; the owner's
    replica, which the dedupe drops, gives that row its declaration."""
    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.set_declarations([("opensees/fix/base", "fix", "base", False)], [])
    if staged:
        e.stage_open("s")
    e.partition_open(0)
    e.set_declaration(-1)          # the ghost replay: no declaration
    e.fix(3, 1, 1)
    e.partition_close()
    e.partition_open(1)
    e.set_declaration(0)           # the owner's fix loop
    e.fix(3, 1, 1)
    e.set_declaration(-1)
    e.partition_close()
    blk = e._stage_current if staged else None
    column = blk.fix_decls if blk is not None else e._fix_decls
    assert column == [0]


def test_object_registered_inside_a_recorded_call_gets_its_own_key() -> None:
    """A user call whose record another store already holds (a session
    verb that builds bridge primitives, e.g. an interop translator) gives
    the bridge no provenance path: each such unnamed object is keyed
    ``@k`` in its family, flagged ``synth``, never sharing a stranger's
    key and never colliding with the ``#k`` counter."""
    ops = apeSees(_two_quad_stub(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    first = ops.nDMaterial.ElasticIsotropic(E=1.0, nu=0.3, rho=0.0)
    a, b = object(), object()
    ops._note_decl(a, "nDMaterial", None, None, synth=False)
    ops._note_decl(b, "nDMaterial", None, None, synth=False)
    assert ops._decls[id(first)][1][0] == "opensees/nDMaterial/#1"
    assert ops._decls[id(a)][1] == (
        "opensees/nDMaterial/@1", "nDMaterial", "", True)
    assert ops._decls[id(b)][1][0] == "opensees/nDMaterial/@2"


def test_stage_ghost_replaying_a_global_fix_is_claimed(tmp_path: Path) -> None:
    """A ghost declared in a stage replays its owner's whole SP stream,
    the global ``ops.fix`` included (review of #1623 @ 4bad1292): its
    row is claimed by the global record, not left to refuse the write."""
    from tests.opensees.integration.test_emit_partitioned_staged_mp_constraints import (  # noqa: E501
        _frame_ops,
        _frame_with_cross_rank_tie,
        _two_stages,
    )

    ops = _frame_ops(_frame_with_cross_rank_tie())
    ops.fix(pg="Base", dofs=(1,) * 6, name="base")
    _two_stages(ops, lambda s: s.equal_dof(name="x_tie"))
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _decls(p)
    staged = [store for store in table.rows if store.startswith("stages/")]
    assert staged, "the reproducer writes no stage fix row"
    for store in staged:
        if store.endswith("bcs/fix"):
            assert {table.decls[d].key for d in table.rows[store]} <= {
                "opensees/fix/base"}


def test_set_declarations_refuses_a_repeated_key_and_a_shared_tag() -> None:
    with pytest.raises(ValueError, match="repeated"):
        H5Emitter().set_declarations(
            [("k", "fix", "", False), ("k", "fix", "", False)], [])
    with pytest.raises(ValueError, match="claim"):
        H5Emitter().set_declarations(
            [("a", "element", "", False), ("b", "element", "", False)],
            [("element", 1, 3, 0), ("element", 3, 1, 1)],
        )


def test_reader_refuses_a_rows_column_of_the_wrong_length(
    tmp_path: Path,
) -> None:
    p = tmp_path / "m.h5"
    _flat_frame(named=True).h5(str(p))
    with h5py.File(str(p), "a") as f:
        rows = f["opensees"]["decls"]["rows"]["bcs"]
        del rows["fix"]
        rows.create_dataset("fix", data=np.array([0, 0], dtype=np.int64))
    with h5_reader.open(str(p)) as m:
        with pytest.raises(h5_reader.MalformedH5Error, match="bcs/fix"):
            m.declarations()


def test_names_are_unique_per_family_only(tmp_path: Path) -> None:
    """fix, mass and declare names live in their own family's key space:
    one name across families is fine (the primitive + declare pair passes
    on main), and none of them enters the bridge-wide alias table."""
    ops = _flat_frame(named=False)
    ops.nDMaterial.ElasticIsotropic(E=1.0, nu=0.3, rho=0.0, name="Rock")
    ops.recorder.declare(nodes="displacement_x", pg="Cols", name="Rock")
    ops.fix(nodes=[2], dofs=(0, 0, 1, 0, 0, 0), name="Rock")
    ops.mass(nodes=[1], values=(1.0,) * 6, name="Rock")
    assert set(ops._names) == {"Rock"}               # the material only
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    keys = {d.key for d in _decls(p).decls}
    assert {"opensees/nDMaterial/Rock", "opensees/recorder/Rock",
            "opensees/fix/Rock", "opensees/mass/Rock"} <= keys


@pytest.mark.parametrize("declare", [
    lambda ops: ops.fix(nodes=[1], dofs=(1,) * 6, name="@1"),
    lambda ops: ops.mass(nodes=[1], values=(1.0,) * 6, name="@2"),
    lambda ops: ops.timeSeries.Linear(name="@3"),
    lambda ops: ops.region(name="@4", nodes=[1]),
])
def test_names_of_the_internal_at_k_form_are_refused(declare: Any) -> None:
    ops = _flat_frame(named=False)
    with pytest.raises(ValueError, match="'@<k>'"):
        declare(ops)


# ---------------------------------------------------------------------------
# The oracle: every declaration verb of the registry has its key
# ---------------------------------------------------------------------------

#: How each archived verb of ``VERBS`` reaches ``/opensees/decls``. A new
#: verb fails ``test_every_registry_verb_is_classified`` until it is put
#: in one of these, so no declaration verb ships unkeyed.
#:
#: ``tagged``  its record carries its own tag, joined through ``tags``;
#: ``rows``    a tagless declaration: its store has a ``rows`` column;
#: ``part``    a row inside a ``tagged`` verb's record (a pattern's loads,
#:             a section's fibers, a recorder declaration context);
#: ``neutral`` rows the FEM snapshot declares (keyed in the neutral zone);
#: ``control`` model, analysis-chain and stage attributes, no row (the
#:             chain primitives themselves are ``tagged`` declarations);
#: ``blocked`` declared through ``_internal/ns/damping.py``, which records
#:             no declaration yet (outside K1-6's files; reported on #1463).
CLASS: dict[str, str] = {
    **{v: "tagged" for v in (
        "uniaxialMaterial", "nDMaterial", "section", "section_open",
        "geomTransf", "beamIntegration", "element", "timeSeries",
        "pattern_open", "damping", "region")},
    **{v: "rows" for v in (
        "fix", "mass", "recorder", "sp_hold", "remove_sp", "remove_element",
        "update_material_stage", "addToParameter", "step_hook_ramp",
        "flip_element_stage")},
    **{v: "part" for v in (
        "section_close", "patch", "fiber", "layer", "pattern_close", "load",
        "eleLoad", "sp", "recorder_declaration_begin",
        "recorder_declaration_end", "mp_constraint_comment")},
    **{v: "neutral" for v in (
        "node", "equalDOF", "rigidLink", "rigidDiaphragm", "embeddedNode")},
    **{v: "control" for v in (
        "model", "constraints", "numberer", "system", "test", "algorithm",
        "integrator", "analysis", "analyze", "stage_open", "stage_close",
        "domain_change", "set_time", "set_creep", "reset", "partition_open",
        "partition_close", "parallel_runtime_fallback_numberer",
        "parallel_runtime_fallback_system")},
    **{v: "blocked" for v in (
        "rayleigh", "modal_damping", "eigen", "profiler", "command")},
}

#: The store, below a scope, holding each ``rows`` verb's records.
ROWS_STORE: dict[str, str] = {
    "fix": "bcs/fix", "mass": "bcs/mass", "recorder": "recorders",
    "sp_hold": "sp_holds", "remove_sp": "remove_sp",
    "remove_element": "remove_element",
    "update_material_stage": "update_material_stage",
    "addToParameter": "initial_stress", "step_hook_ramp": "initial_stress",
    "flip_element_stage": "activate_absorbing",
}

#: ``tagged`` stores: group path below a scope -> allocator kind.
TAG_STORE: dict[str, str] = {
    "materials/uniaxial": "uniaxialMaterial", "materials/nd": "nDMaterial",
    "sections": "section", "transforms": "geomTransf",
    "beam_integration": "beamIntegration", "time_series": "timeSeries",
    "patterns": "pattern", "dampings": "damping", "regions": "region",
}


def test_every_registry_verb_is_classified() -> None:
    from apeGmsh.opensees.emitter.verbs import VERBS

    archived = {v for v, row in VERBS.items() if row.h5 == "archive"}
    assert set(CLASS) == archived, (
        f"unclassified: {sorted(archived - set(CLASS))}; "
        f"stale: {sorted(set(CLASS) - archived)}")
    # Every verb the registry marks as opening a declaration is keyed.
    for verb, row in VERBS.items():
        if row.decl and row.h5 == "archive":
            assert CLASS[verb] in ("tagged", "rows", "neutral"), verb
    assert set(ROWS_STORE) == {v for v, c in CLASS.items() if c == "rows"}


def _all_verb_models() -> list[apeSees]:
    from tests.opensees.h5.test_h5_stages_writer import (
        _build_kitchen_sink_bridge,
        _build_two_stage_bridge,
        _make_two_quad_fem_stub,
        build_material_stage_bridge,
    )

    flat = apeSees(_two_quad_stub(), default_orientation=None)
    flat.model(ndm=2, ndf=2)
    mat = flat.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    flat.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    flat.fix(pg="Base", dofs=(1, 1))
    flat.region(name="top", pg="FillTop")
    flat.region(name="top", nodes=[4])
    flat.initial_stress(name="geo", pg="Rock", sigma_xx=-1.0,
                        sigma_yy=-1.0, sigma_zz=-1.0, ramp_steps=2)
    flat.equation_constraint(constrained=(3, 1), retained=[(4, 1, -1.0)])
    return [
        flat, _staged(named=True), _build_kitchen_sink_bridge(),
        _build_two_stage_bridge(),
        build_material_stage_bridge(_make_two_quad_fem_stub()),
    ]


def test_every_declaration_verb_has_its_key(tmp_path: Path) -> None:
    """Driven by ``CLASS`` (the registry), not by provenance: every row
    of every ``rows`` verb's store and every tag of every ``tagged``
    store resolves to a declaration, in models that exercise each one."""
    seen: set[str] = set()
    for i, ops in enumerate(_all_verb_models()):
        p = tmp_path / f"m{i}.h5"
        ops.h5(str(p))
        table = _decls(p)
        with h5py.File(str(p), "r") as f:
            root = f["opensees"]
            scopes = [("", root)] + [
                (f"stages/{s}/", root["stages"][s])
                for s in (root["stages"] if "stages" in root else ())]
            for prefix, grp in scopes:
                for verb, leaf in ROWS_STORE.items():
                    paths = [leaf]
                    if leaf == "sp_holds":
                        pats = grp["patterns"] if "patterns" in grp else {}
                        paths = [f"patterns/{n}/sp_holds" for n in pats
                                 if "sp_holds" in pats[n]]
                    for path in paths:
                        if path not in grp or not len(grp[path]):
                            continue
                        seen.add(verb)
                        store = prefix + path
                        assert store in table.rows, store
                        assert len(table.rows[store]) == len(grp[path])
                for path, kind in TAG_STORE.items():
                    if path not in grp:
                        continue
                    for name in grp[path]:
                        g = grp[path][name]
                        if kind == "region" and g.attrs.get("kind") == "rayleigh":
                            continue      # blocked: see CLASS
                        tag = (int(g.attrs["tag"]) if "tag" in g.attrs
                               else int(name.rsplit("_", 1)[1]))
                        table.for_tag(kind, tag)
                        seen.add(kind)
            if "element_meta" in root:
                for token in root["element_meta"]:
                    for tag in root["element_meta"][token]["ids"][()]:
                        table.for_tag("element", int(tag))
        if ops._equation_constraint_records:
            assert any(d.family == "equation_constraint"
                       for d in table.decls)
    # Each ``rows`` verb and the region store were exercised.
    assert set(ROWS_STORE) <= seen, sorted(set(ROWS_STORE) - seen)
    assert "region" in seen
