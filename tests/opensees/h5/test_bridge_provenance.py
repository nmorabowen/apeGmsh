"""Bridge provenance: ``apeSees._register`` capture, the ``/provenance``
composer and the replay writers (ADR 0112 D3, program slice V2d, #1307).

Oracles, each naming the right answer:

* **The helper-call oracle.**  A user script declares its bridge
  primitives inside a helper function.  Each record's ``site`` is the
  helper's line and ``script`` the script line that called it, read
  off ``# MARK:`` comments in the script itself.  The session's own
  records (V2c) sit in the same table, before the bridge's, and the
  one script file is one ``files`` row (the merge deduplicates).
* **Hash invariance.**  ``model_hash`` and ``fem_hash`` recomputed on
  a copy of the file with ``/provenance`` deleted equal the stamped
  ones: no hash reads the zone.
* **Replay (V0 Q7).**  ``OpenSeesModel.from_h5(...).to_h5(...)`` and
  ``ModelData.from_h5(...).write(...)`` into the source's directory
  carry ``/provenance`` byte-equal and the same ``session_id``.  Into
  another directory the decoded table is equal and ``@base_dir`` is
  the new file's own directory (the schema's definition of it).
* **The mass skip (V2a).**  A ``mass_from_model()`` model writes the
  file with the zone, and the neutral-zone masses equal the deck's,
  node by node (the ``test_mass_from_model_h5`` oracle).
"""
from __future__ import annotations

import hashlib
import os
import runpy
import shutil
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh
from apeGmsh._internal.provenance import (
    FileRow,
    ProvenanceOverflowError,
    ProvenanceTable,
    RecordRow,
    SiteRow,
)
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import ModelData, OpenSeesModel, apeSees
from apeGmsh.opensees._internal.compose import _merge_provenance
from apeGmsh.opensees._internal.lineage import (
    compute_fem_hash,
    compute_model_hash,
    read_stored_lineage,
)
from apeGmsh.opensees._internal.schema_version import PROVENANCE_KEY
from tests.fixtures.schema import PROVENANCE_CURRENT
from tests.opensees.fixtures.fem_stub import make_two_node_beam

USER_SCRIPT = '''\
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees


def declare_bridge(ops):
    ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="steel")  # MARK:helper_mat
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)  # MARK:helper_nd
    ops.element.FourNodeTetrahedron(pg="B", material=mat)  # MARK:helper_elem


with apeGmsh(model_name="prov_bridge", verbose=False) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
    g.physical.add_volume("b", name="B")  # MARK:session_pg
    g.masses.volume("B", density=2400.0)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
declare_bridge(ops)  # MARK:script_call
ops.mass_from_model()
ops.h5(OUT)
ops.tcl(TCL)
'''

BRIDGE_PATHS = (
    "opensees/uniaxialMaterial/steel",
    "opensees/nDMaterial/#1",
    "opensees/element/#1",
)


def _line(marker: str) -> int:
    for i, text in enumerate(USER_SCRIPT.splitlines(), start=1):
        if f"# MARK:{marker}" in text:
            return i
    raise AssertionError(marker)


def _deck_masses(text: str) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for line in text.splitlines():
        tok = line.split()
        if tok and tok[0] == "mass":
            out[int(tok[1])] = tuple(float(v) for v in tok[2:])
    return out


def _zone(path: Path) -> dict[str, object]:
    """Every dataset of ``/provenance`` plus ``@base_dir``, as raw arrays."""
    out: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        grp = f["provenance"]
        out["@base_dir"] = grp.attrs["base_dir"]
        for table in grp:
            for col in grp[table]:
                ds = grp[table][col]
                out[f"{table}/{col}"] = (str(ds.dtype), ds[()].tolist())
        out["@key"] = f["meta"].attrs[PROVENANCE_KEY]
        out["@session_id"] = f["meta"].attrs["session_id"]
    return out


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    d = tmp_path_factory.mktemp("prov_bridge")
    script = d / "user_script.py"
    script.write_text(USER_SCRIPT, encoding="utf-8")
    out, tcl = d / "model.h5", d / "model.tcl"
    runpy.run_path(str(script),
                   init_globals={"OUT": str(out), "TCL": str(tcl)},
                   run_name="__main__")
    return script, out, tcl


# ---------------------------------------------------------------------------
# The helper-call oracle
# ---------------------------------------------------------------------------


def test_bridge_declaration_points_at_the_helper_line(oracle):
    script, out, _ = oracle
    table = FEMData.from_h5(str(out)).provenance
    assert table is not None
    want_path = Path(os.path.abspath(str(script))).as_posix()
    want_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    for path, marker in zip(BRIDGE_PATHS,
                            ("helper_mat", "helper_nd", "helper_elem")):
        rec = table.record(path)
        site, top = table.location(rec.site), table.location(rec.script)
        assert (site.path, site.line, site.function) == (
            want_path, _line(marker), "declare_bridge"), path
        assert (top.path, top.line, top.function) == (
            want_path, _line("script_call"), "<module>"), path
        assert site.sha256 == top.sha256 == want_sha


def test_session_and_bridge_records_share_one_table(oracle):
    script, out, _ = oracle
    table = FEMData.from_h5(str(out)).provenance
    assert table is not None
    paths = [r.path for r in table.records]
    assert len(paths) == len(set(paths))
    # The session's records come first, the bridge's after, in
    # declaration order, with ``seq`` the row index of the merged table.
    assert "neutral/physical_groups/B" in paths
    assert paths[-3:] == list(BRIDGE_PATHS)
    assert [r.seq for r in table.records] == list(range(len(paths)))
    pg = table.location(table.record("neutral/physical_groups/B").site)
    assert pg.line == _line("session_pg")
    # Both stores saw the same script, so the merge keeps one file row.
    assert [f.kind for f in table.files] == ["script"]
    assert table.files[0].path == Path(os.path.abspath(str(script))).as_posix()


def test_zone_layout_and_key(oracle):
    script, out, _ = oracle
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs[PROVENANCE_KEY] == PROVENANCE_CURRENT
        grp = f["provenance"]
        assert grp.attrs["base_dir"] == Path(
            os.path.abspath(str(out.parent))).as_posix()
        assert [p.decode() for p in grp["files/path"][()]] == [script.name]
        for table, col in (("sites", "file"), ("sites", "line"),
                           ("records", "site"), ("records", "script"),
                           ("records", "seq")):
            assert grp[table][col].dtype == "int32", (table, col)
        # Written after /opensees: the bridge zone is complete beside it.
        assert "opensees" in f and int(
            f["opensees/bcs"].attrs["mass_from_model"]) == 1


# ---------------------------------------------------------------------------
# Hash invariance
# ---------------------------------------------------------------------------


def test_model_hash_is_equal_with_and_without_provenance(oracle, tmp_path):
    _, out, _ = oracle
    stripped = tmp_path / "stripped.h5"
    shutil.copyfile(out, stripped)
    with h5py.File(stripped, "a") as f:
        del f["provenance"]
        del f["meta"].attrs[PROVENANCE_KEY]
    with h5py.File(out, "r") as f:
        stamped_fem, stamped_model, _ = read_stored_lineage(f["meta"])
        assert stamped_fem and stamped_model
    with h5py.File(stripped, "r") as f:
        fem_hash = compute_fem_hash(f)
        model_hash = compute_model_hash(fem_hash, f["opensees"])
    assert (fem_hash, model_hash) == (stamped_fem, stamped_model)


# ---------------------------------------------------------------------------
# Replay writers (V0 Q7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("replay", ["opensees_model", "model_data"])
def test_replay_into_the_same_directory_is_byte_equal(oracle, replay):
    _, out, _ = oracle
    again = out.parent / f"again_{replay}.h5"
    if replay == "opensees_model":
        OpenSeesModel.from_h5(str(out)).to_h5(str(again))
    else:
        ModelData.from_h5(str(out)).write(str(again))
    assert _zone(again) == _zone(out)


def test_replay_elsewhere_rebases_paths_and_keeps_the_table(oracle, tmp_path):
    _, out, _ = oracle
    elsewhere = tmp_path / "elsewhere" / "again.h5"
    elsewhere.parent.mkdir()
    OpenSeesModel.from_h5(str(out)).to_h5(str(elsewhere))
    src, dst = FEMData.from_h5(str(out)), FEMData.from_h5(str(elsewhere))
    assert dst.provenance == src.provenance
    assert dst.session_id == src.session_id
    with h5py.File(elsewhere, "r") as f:
        grp = f["provenance"]
        assert grp.attrs["base_dir"] == Path(
            os.path.abspath(str(elsewhere.parent))).as_posix()
        # The script is not under the new base_dir, so it is absolute.
        [p] = [p.decode() for p in grp["files/path"][()]]
    assert p == src.provenance.files[0].path and os.path.isabs(p)


# ---------------------------------------------------------------------------
# The mass skip (V2a): a mass_from_model() model passes this path
# ---------------------------------------------------------------------------


def test_mass_from_model_file_masses_equal_the_deck(oracle):
    _, out, tcl = oracle
    deck = _deck_masses(tcl.read_text(encoding="utf-8"))
    assert deck
    fem = FEMData.from_h5(str(out))
    assert fem.provenance is not None
    neutral = {int(m.node_id): tuple(float(v) for v in m.mass)
               for m in fem.nodes.masses}
    assert set(neutral) == set(deck)
    for nid, vec in neutral.items():
        assert deck[nid] == vec[:3], nid


# ---------------------------------------------------------------------------
# Capture rules at _register
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fem():
    with apeGmsh(model_name="prov_bridge_unit", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        g.physical.add_volume("b", name="B")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        return g.mesh.queries.get_fem_data(dim=3)


def test_one_record_per_user_call_and_unnamed_order_keys(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ts = ops.timeSeries.Linear()
    ops.timeSeries.Linear(name="ramp")
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=1, forces=(1.0, 0.0, 0.0))
    paths = [r.path for r in ops._provenance.snapshot().records]
    assert paths == ["opensees/timeSeries/#1", "opensees/timeSeries/ramp",
                     "opensees/pattern/#1"]


# ---------------------------------------------------------------------------
# Synthesised objects (maintainer ruling on #1378, finding 2)
# ---------------------------------------------------------------------------


def _lineno() -> int:
    """The caller's line number."""
    import inspect

    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    return frame.f_back.f_lineno


def _chain(ops):
    return {
        "test": ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def test_support_synthesises_hold_series_and_pattern_under_own_keys(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    with ops.stage(name="s1") as s:
        here = _lineno()
        s.support(pg="B", dofs=(1, 1, 1))  # the verb call: site of both
        s.support(pg="B", dofs=(1, 0, 0))  # same stage: nothing new
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="s2") as s:
        s.support(pg="B", dofs=(0, 0, 1))  # new stage pattern, shared HOLD
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    ramp = ops.timeSeries.Linear()  # the user's first unnamed series
    assert ops.tag_for(ramp) is not None
    table = ops._provenance.snapshot()
    by_path = {r.path: r for r in table.records}
    hold = by_path["opensees/timeSeries/support:s1/hold"]
    pat1 = by_path["opensees/pattern/support:s1"]
    pat2 = by_path["opensees/pattern/support:s2"]
    assert hold.origin == pat1.origin == pat2.origin == "synthesised"
    site = table.location(hold.site)
    assert (site.line, site.function) == (
        here + 1, "test_support_synthesises_hold_series_and_pattern_under_own_keys")
    assert table.location(pat1.site).line == here + 1
    assert "opensees/timeSeries/support:s2/hold" not in by_path
    # The synthesised objects never took a ``#k``: the user's series is #1.
    assert by_path["opensees/timeSeries/#1"].origin == "user"
    assert not any(p.startswith("opensees/pattern/#") for p in by_path)


def test_imposed_displacement_synthesises_series_and_pattern(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    here = _lineno()
    ops.imposed_displacement(nodes=[1], ux=0.01)
    named = ops.imposed_displacement(nodes=[2], uy=0.01, name="push")
    ts = ops.timeSeries.Linear()  # the user's first unnamed series
    ops.imposed_displacement(nodes=[3], uz=0.01, series=ts)
    table = ops._provenance.snapshot()
    by_path = {r.path: r for r in table.records}
    assert [p for p in by_path if "imposed_displacement" in p] == [
        "opensees/timeSeries/imposed_displacement:#1",
        "opensees/pattern/imposed_displacement:#1",
        "opensees/timeSeries/imposed_displacement:push",
        "opensees/pattern/imposed_displacement:push",
        "opensees/pattern/imposed_displacement:#3",
    ]
    assert {by_path[p].origin for p in by_path if "imposed" in p} == {
        "synthesised"}
    assert table.location(
        by_path["opensees/timeSeries/imposed_displacement:#1"].site).line == here + 1
    assert table.location(
        by_path["opensees/pattern/imposed_displacement:push"].site).line == here + 2
    assert by_path["opensees/timeSeries/#1"].origin == "user"
    assert ops._resolve("push") is named


# ---------------------------------------------------------------------------
# Review round 2 (#1378 at 6a323db8)
# ---------------------------------------------------------------------------


def test_imposed_displacement_with_a_named_series_emits(fem, tmp_path):
    """Round 2, finding 1: ``series="<name>"`` resolves through the alias
    table (it was passed to Plain as a string and the emit raised)."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ramp = ops.timeSeries.Linear(factor=2.0, name="ramp")
    plain = ops.imposed_displacement(nodes=[1], ux=0.01, series="ramp",
                                     name="pat")
    deck = tmp_path / "named_series.tcl"
    ops.tcl(str(deck))
    text = deck.read_text(encoding="utf-8")
    assert f"pattern Plain {ops.tag_for(plain)} {ops.tag_for(ramp)}" in text
    assert "sp 1 1 0.01" in text
    # Only the pattern was synthesised; the series is the user's.
    paths = [r.path for r in ops._provenance.snapshot().records]
    assert paths == ["opensees/nDMaterial/#1", "opensees/element/#1",
                     "opensees/timeSeries/ramp",
                     "opensees/pattern/imposed_displacement:pat"]
    with pytest.raises(KeyError, match="no primitive registered"):
        ops.imposed_displacement(nodes=[1], ux=0.01, series="nope")
    with pytest.raises(TypeError, match="TimeSeries is required"):
        ops.imposed_displacement(nodes=[1], ux=0.01, series="pat")
    # Neither refusal registered anything.
    assert [r.path for r in ops._provenance.snapshot().records] == paths
    assert len(ops._primitives) == 4


def test_imposed_displacement_ordinal_key_cannot_collide_with_a_name(fem):
    """Round 2, finding 2a: an unnamed call keys ``#<k>``; a user name may
    not start with ``#``; four objects give four records."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.imposed_displacement(nodes=[1], ux=0.01)
    ops.imposed_displacement(nodes=[2], ux=0.01, name="1")
    paths = [r.path for r in ops._provenance.snapshot().records]
    assert paths == [
        "opensees/timeSeries/imposed_displacement:#1",
        "opensees/pattern/imposed_displacement:#1",
        "opensees/timeSeries/imposed_displacement:1",
        "opensees/pattern/imposed_displacement:1",
    ]
    assert len(ops._primitives) == 4
    before = (len(ops._primitives), len(ops._provenance), dict(ops._names))
    with pytest.raises(ValueError, match="may not start with '#'"):
        ops.imposed_displacement(nodes=[3], ux=0.01, name="#2")
    assert (len(ops._primitives), len(ops._provenance), dict(ops._names)) == before


def test_support_in_two_stages_of_the_same_name_keys_the_second_at_2(fem):
    """Round 2, finding 2b: the bridge allows a repeated stage name, so
    the second stage keys ``support:s@2``; both records exist."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    for _ in range(2):
        with ops.stage(name="s") as s:
            s.support(pg="B", dofs=(1, 1, 1))
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    by_path = {r.path: r for r in ops._provenance.snapshot().records}
    assert "opensees/timeSeries/support:s/hold" in by_path
    assert "opensees/pattern/support:s" in by_path
    assert "opensees/pattern/support:s@2" in by_path
    assert "opensees/timeSeries/support:s@2/hold" not in by_path  # shared HOLD
    assert [r.name for r in ops._stage_records] == ["s", "s"]


def test_capture_synthesised_refuses_a_collision():
    """Round 2, finding 2c: a repeated key raises; no record is dropped."""
    from apeGmsh._internal.provenance import ProvenanceStore

    store = ProvenanceStore()
    assert store.capture_synthesised("opensees", "pattern", "support:s") == (
        "opensees/pattern/support:s")
    with pytest.raises(ValueError, match="already has a record"):
        store.capture_synthesised("opensees", "pattern", "support:s")
    assert len(store) == 1


def test_imposed_displacement_refused_name_leaves_nothing_behind(fem):
    """Round 2, finding 3: a taken ``name=`` raises before any
    registration, so no Linear, Plain, alias or record remains."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.timeSeries.Linear(name="push")
    before = (len(ops._primitives), len(ops._provenance), dict(ops._names),
              ops._imposed_displacement_calls)
    with pytest.raises(ValueError, match="already registered"):
        ops.imposed_displacement(nodes=[1], ux=0.01, name="push")
    assert (len(ops._primitives), len(ops._provenance), dict(ops._names),
            ops._imposed_displacement_calls) == before
    # The next unnamed call is still #1: the refused call took no ordinal.
    ops.imposed_displacement(nodes=[1], ux=0.01)
    assert "opensees/pattern/imposed_displacement:#1" in {
        r.path for r in ops._provenance.snapshot().records}


def _state(ops):
    return (len(ops._primitives), len(ops._provenance), dict(ops._names),
            ops._hold_series, len(ops._stage_records))


# ---------------------------------------------------------------------------
# Review round 3 (#1378 at c314e1c7): one key space, collisions fail
# loud before allocation
# ---------------------------------------------------------------------------


def test_user_name_equal_to_a_synthesised_key_is_refused(fem):
    """Round 3, item 1: the user's Linear was silently unrecorded."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.imposed_displacement(nodes=[1], ux=0.01, name="foo")
    before = _state(ops)
    with pytest.raises(ValueError, match="collides with the existing provenance"):
        ops.timeSeries.Linear(name="imposed_displacement:foo")
    assert _state(ops) == before
    assert before[0] == before[1] == 2
    # The other way round: a user name the verb would synthesise refuses
    # the verb before it registers its series.
    ts = ops.timeSeries.Linear()
    ops.pattern.Plain(series=ts, name="imposed_displacement:bar")
    before = _state(ops)
    with pytest.raises(ValueError, match="already has a record"):
        ops.imposed_displacement(nodes=[1], ux=0.01, name="bar")
    assert _state(ops) == before


def test_taken_hold_key_refuses_support_before_allocation(fem):
    """Round 3, item 2: the refusal left an orphan tagged Constant."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.timeSeries.Linear(name="support:x/hold")
    before = _state(ops)
    with pytest.raises(ValueError, match="support:x/hold"):
        with ops.stage(name="x") as s:
            s.support(pg="B", dofs=(1, 1, 1))
    assert _state(ops) == before
    assert before[0] == before[1] == 1 and before[3] is None
    assert ops._open_stage_builder is None


def test_stage_names_s_s_s2_all_get_distinct_keys(fem):
    """Round 3, item 3: ``@<n>`` is the smallest free ordinal."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    for stage_name in ("s", "s", "s@2"):
        with ops.stage(name=stage_name) as s:
            s.support(pg="B", dofs=(1, 1, 1))
            s.analysis(**_chain(ops))
            s.run(n_increments=1)
    keys = [r.path for r in ops._provenance.snapshot().records
            if r.path.startswith("opensees/pattern/support:")]
    assert keys == ["opensees/pattern/support:s", "opensees/pattern/support:s@2",
                    "opensees/pattern/support:s@2@2"]
    assert [r.name for r in ops._stage_records] == ["s", "s", "s@2"]
    assert len({ops.tag_for(r.support_pattern) for r in ops._stage_records}) == 3


def test_unnamed_key_taken_by_a_user_name_is_refused(fem):
    """Round 3, the same seam on the ``#k`` side: a user name ``#1`` and
    an unnamed declaration used to overwrite the record silently."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.timeSeries.Linear(name="#1")
    before = _state(ops)
    with pytest.raises(ValueError, match="unnamed key '#1'"):
        ops.timeSeries.Linear()
    assert _state(ops) == before
    assert [r.path for r in ops._provenance.snapshot().records] == [
        "opensees/timeSeries/#1"]


def test_a_1_1_file_without_origin_is_malformed(fem, tmp_path):
    """Round 2, finding 4: the column may be absent only below 1.1.0."""
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
    out = tmp_path / "no_origin.h5"
    ops.h5(str(out))
    with h5py.File(out, "a") as f:
        assert f["meta"].attrs[PROVENANCE_KEY] == PROVENANCE_CURRENT
        del f["provenance/records/origin"]
    with pytest.raises(MalformedH5Error, match="records/origin is missing"):
        FEMData.from_h5(str(out))


def test_origin_column_round_trips_and_a_1_0_file_reads_as_user(fem, tmp_path):
    from tests.fixtures.schema import PROVENANCE_PRIOR_MINOR

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.imposed_displacement(pg="B", ux=0.001, name="push")
    out = tmp_path / "origin.h5"
    ops.h5(str(out))
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs[PROVENANCE_KEY] == PROVENANCE_CURRENT
        origins = [o.decode() for o in f["provenance/records/origin"][()]]
        paths = [p.decode() for p in f["provenance/records/path"][()]]
    by_path = dict(zip(paths, origins))
    assert by_path["opensees/nDMaterial/conc"] == "user"
    assert by_path["opensees/pattern/imposed_displacement:push"] == "synthesised"
    back = FEMData.from_h5(str(out)).provenance
    assert back is not None
    assert {r.path: r.origin for r in back.records} == by_path
    # A 1.0.0 file has no origin column: every record reads as "user".
    old = tmp_path / "old.h5"
    shutil.copyfile(out, old)
    with h5py.File(old, "a") as f:
        del f["provenance/records/origin"]
        f["meta"].attrs[PROVENANCE_KEY] = PROVENANCE_PRIOR_MINOR
    older = FEMData.from_h5(str(old)).provenance
    assert older is not None
    assert {r.origin for r in older.records} == {"user"}
    assert [r.path for r in older.records] == [r.path for r in back.records]


# ---------------------------------------------------------------------------
# A reloaded snapshot through a second bridge (the "analyse" script)
# ---------------------------------------------------------------------------


def test_reloaded_snapshot_then_h5_redeclaring_the_same_names(oracle, tmp_path):
    """FEMData.from_h5 -> apeSees -> h5 with the same names raised at
    cfaaaf80 (the snapshot carried the source's opensees/ records)."""
    script, out, _ = oracle
    fem = FEMData.from_h5(str(out))
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="steel")
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.mass_from_model()
    again = tmp_path / "again.h5"
    ops.h5(str(again))
    table = FEMData.from_h5(str(again)).provenance
    assert table is not None
    paths = [r.path for r in table.records]
    assert len(paths) == len(set(paths))
    assert paths[-3:] == list(BRIDGE_PATHS)
    # The records are this bridge's: their site is this test, not the script.
    site = table.location(table.record("opensees/uniaxialMaterial/steel").site)
    assert site.path.endswith("test_bridge_provenance.py")
    assert site.function == "test_reloaded_snapshot_then_h5_redeclaring_the_same_names"
    assert script.as_posix() not in {table.location(r.site).path
                                     for r in table.records
                                     if r.path.startswith("opensees/")}


def test_reloaded_snapshot_then_h5_with_other_declarations_keeps_no_stale_record(
        oracle, tmp_path):
    _, out, _ = oracle
    source = FEMData.from_h5(str(out)).provenance
    assert source is not None
    session_paths = [r.path for r in source.records
                     if not r.path.startswith("opensees/")]
    fem = FEMData.from_h5(str(out))
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc2")
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.mass_from_model()
    again = tmp_path / "other.h5"
    ops.h5(str(again))
    table = FEMData.from_h5(str(again)).provenance
    assert table is not None
    paths = [r.path for r in table.records]
    assert paths == session_paths + ["opensees/nDMaterial/conc2",
                                     "opensees/element/#1"]
    # The source's steel and its unnamed nDMaterial are gone; the element
    # key is this bridge's own #1, not the source's (its site is here).
    assert "opensees/uniaxialMaterial/steel" not in paths
    assert "opensees/nDMaterial/#1" not in paths
    assert table.location(table.record("opensees/element/#1").site).path.endswith(
        "test_bridge_provenance.py")
    assert [r.seq for r in table.records] == list(range(len(paths)))
    # The session's own records are untouched: same sites, same lines.
    for p in session_paths:
        assert table.location(table.record(p).site) == source.location(
            source.record(p).site)
    # The dropped records' file row went with them: the script is still
    # referenced by the session records, so it stays; nothing dangles.
    referenced = {s.file for s in table.sites} | set()
    assert referenced == set(range(len(table.files)))
    used_sites = {r.site for r in table.records} | {r.script for r in table.records}
    assert used_sites - {-1} == set(range(len(table.sites)))


def test_register_of_a_standalone_primitive_records_it(fem):
    from apeGmsh.opensees.material.uniaxial import ElasticMaterial

    ops = apeSees(fem)
    ops.register(ElasticMaterial(E=1.0))
    assert [r.path for r in ops._provenance.snapshot().records] == [
        "opensees/uniaxialMaterial/#1"]


def test_stub_fem_file_carries_no_zone(tmp_path):
    """The bridge-only fallback (a hand-rolled stub, no neutral zone)
    belongs to no run: its records would hold test paths, and the
    golden corpus h5dumps would never be stable.  The bridge still
    captured them."""
    ops = apeSees(make_two_node_beam())  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), name="cols")
    assert [r.path for r in ops._provenance.snapshot().records] == [
        "opensees/geomTransf/cols"]
    out = tmp_path / "stub.h5"
    ops.h5(str(out))
    with h5py.File(out, "r") as f:
        assert "provenance" not in f
        assert PROVENANCE_KEY not in f["meta"].attrs
        assert "opensees" in f


def test_int32_overflow_refuses_before_writing(fem, tmp_path):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.nDMaterial.ElasticIsotropic(E=1.0, nu=0.2)
    fem.provenance = ProvenanceTable(sites=(SiteRow(0, 2**31, "f"),))
    try:
        target = tmp_path / "overflow.h5"
        with pytest.raises(ProvenanceOverflowError, match="int32"):
            ops.h5(str(target))
        assert not target.exists()
    finally:
        fem.provenance = None


# ---------------------------------------------------------------------------
# _merge_provenance
# ---------------------------------------------------------------------------

_F = FileRow("/u/a.py", "aa", "script")
_G = FileRow("/u/lib.py", "bb", "module")


def test_merge_dedupes_files_and_sites_and_continues_seq():
    base = ProvenanceTable(
        files=(_F,), sites=(SiteRow(0, 3, "<module>"),),
        records=(RecordRow("neutral/labels/x", 0, 0, 0),))
    extra = ProvenanceTable(
        files=(_G, _F),
        sites=(SiteRow(0, 7, "helper"), SiteRow(1, 3, "<module>")),
        records=(RecordRow("opensees/element/#1", 0, 1, 0),
                 RecordRow("opensees/element/#2", -1, -1, 1)))
    merged = _merge_provenance(base, extra)
    assert merged is not None
    assert merged.files == (_F, _G)
    assert merged.sites == (SiteRow(0, 3, "<module>"), SiteRow(1, 7, "helper"))
    assert merged.records == (
        RecordRow("neutral/labels/x", 0, 0, 0),
        RecordRow("opensees/element/#1", 1, 0, 1),
        RecordRow("opensees/element/#2", -1, -1, 2))


def test_drop_zone_compacts_files_sites_and_seq():
    from apeGmsh.opensees._internal.compose import _drop_zone

    t = ProvenanceTable(
        files=(_F, _G),
        sites=(SiteRow(0, 3, "<module>"), SiteRow(1, 7, "helper")),
        records=(RecordRow("opensees/element/#1", 1, 0, 0),
                 RecordRow("neutral/labels/x", 0, 0, 1),
                 RecordRow("opensees/pattern/support:s1", 1, -1, 2,
                           "synthesised")))
    dropped = _drop_zone(t, "opensees")
    assert dropped == ProvenanceTable(
        files=(_F,), sites=(SiteRow(0, 3, "<module>"),),
        records=(RecordRow("neutral/labels/x", 0, 0, 0),))
    assert _drop_zone(dropped, "opensees") is dropped


def test_merge_carries_origin():
    base = ProvenanceTable(records=(RecordRow("neutral/labels/x", -1, -1, 0),))
    extra = ProvenanceTable(records=(
        RecordRow("opensees/pattern/support:s1", -1, -1, 0, "synthesised"),))
    merged = _merge_provenance(base, extra)
    assert merged is not None
    assert [r.origin for r in merged.records] == ["user", "synthesised"]


def test_merge_passes_through_when_one_side_is_missing():
    t = ProvenanceTable(records=(RecordRow("opensees/element/#1", -1, -1, 0),))
    assert _merge_provenance(None, None) is None
    assert _merge_provenance(None, t) is t
    assert _merge_provenance(t, None) is t
    assert _merge_provenance(t, ProvenanceTable()) is t


def test_merge_refuses_a_path_recorded_on_both_sides():
    t = ProvenanceTable(records=(RecordRow("opensees/element/#1", -1, -1, 0),))
    with pytest.raises(ValueError, match="both the snapshot and the bridge"):
        _merge_provenance(t, t)


def test_merge_is_what_the_file_carries(oracle):
    """The merged table round-trips: re-encoding the decoded table
    against the file's own base_dir reproduces the stored columns."""
    _, out, _ = oracle
    from apeGmsh._internal.provenance import encode_columns

    table = FEMData.from_h5(str(out)).provenance
    assert table is not None
    with h5py.File(out, "r") as f:
        base_dir = f["provenance"].attrs["base_dir"]
        stored = {
            f"{t}/{c}": f["provenance"][t][c][()].tolist()
            for t in f["provenance"] for c in f["provenance"][t]}
    for t, cols in encode_columns(table, base_dir).items():
        for c, values in cols.items():
            got = stored[f"{t}/{c}"]
            got = [v.decode() if isinstance(v, bytes) else int(v) for v in got]
            assert got == values, (t, c)
