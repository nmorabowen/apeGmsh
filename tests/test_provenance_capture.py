"""V2c (#1306): ADR 0112 D3 declaration provenance on the session side.

Oracles:

* **Helper call.**  A user script calls a helper of its own that declares
  a label and a load.  Each record's ``site`` is the helper's line, its
  ``script`` is the script line that called the helper, and the file's
  ``sha256`` is the digest of the script's bytes.  The expected lines are
  read from marker comments in the script, not from the code under test.
* ``#k``: unnamed declarations are keyed by their 1-based order in their
  family; named ones by the name.
* One record per user call: a geometry call that also creates its label
  records once; three calls on one line record three times and share one
  ``sites`` row.
* Hash invariance: ``snapshot_id`` (and the stored ``fem_hash``) are equal
  with and without provenance.
* An ``int32`` overflow refuses before the file is created.
* The overhead invariant: at most 50 ms per 1,000 registrations (V0 Q6).
"""
from __future__ import annotations

import copy
import hashlib
import os
import runpy
import time
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh
from apeGmsh._internal import provenance as prov
from apeGmsh._internal.provenance import (
    ProvenanceOverflowError,
    ProvenanceTable,
    RecordRow,
    SiteRow,
)
from apeGmsh.core._declarations import _PROVENANCE_FAMILY
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.mesh._femdata_h5_io import write_fem_h5
from apeGmsh.opensees._internal.schema_version import PROVENANCE_KEY
from tests.fixtures.schema import PROVENANCE_CURRENT

USER_SCRIPT = '''\
from apeGmsh import apeGmsh


def declare(g, vol):
    g.labels.add(3, [vol], name="blk")  # MARK:helper_label
    g.loads.gravity("blk", g=(0, 0, -9.81), density=2400.0)  # MARK:helper_load


with apeGmsh(model_name="oracle", verbose=False) as g:
    vol = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    declare(g, vol)  # MARK:script_call
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.queries.get_fem_data().to_h5(OUT)
'''


def _line(marker: str) -> int:
    for i, text in enumerate(USER_SCRIPT.splitlines(), start=1):
        if f"# MARK:{marker}" in text:
            return i
    raise AssertionError(marker)


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    d = tmp_path_factory.mktemp("prov")
    script = d / "user_script.py"
    script.write_text(USER_SCRIPT, encoding="utf-8")
    out = d / "model.h5"
    runpy.run_path(str(script), init_globals={"OUT": str(out)},
                   run_name="__main__")
    return script, out


# ---------------------------------------------------------------------------
# The helper-call oracle
# ---------------------------------------------------------------------------


def test_helper_call_records_helper_site_and_script_line(oracle):
    script, out = oracle
    table = FEMData.from_h5(str(out)).provenance
    assert table is not None
    want_path = Path(os.path.abspath(str(script))).as_posix()
    want_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    for path, marker in (("neutral/labels/blk", "helper_label"),
                         ("neutral/loads/#1", "helper_load")):
        rec = table.record(path)
        site, top = table.location(rec.site), table.location(rec.script)
        assert (site.path, site.line, site.function) == (
            want_path, _line(marker), "declare")
        assert (top.path, top.line, top.function) == (
            want_path, _line("script_call"), "<module>")
        assert site.sha256 == top.sha256 == want_sha
    assert [f.kind for f in table.files] == ["script"]


def test_session_written_file_carries_the_zone(oracle):
    script, out = oracle
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs[PROVENANCE_KEY] == PROVENANCE_CURRENT
        grp = f["provenance"]
        assert grp.attrs["base_dir"] == Path(
            os.path.abspath(str(out.parent))).as_posix()
        # The script lies under @base_dir, so its path is stored relative.
        assert [p.decode() for p in grp["files/path"][()]] == [script.name]
        for table, col in (("sites", "file"), ("sites", "line"),
                           ("records", "site"), ("records", "script"),
                           ("records", "seq")):
            assert grp[table][col].dtype == "int32", (table, col)
        paths = [p.decode() for p in grp["records/path"][()]]
    assert paths == ["geometry/box/#1", "neutral/labels/blk",
                     "neutral/loads/#1"]


def test_round_trip_is_exact(oracle, tmp_path):
    _, out = oracle
    fem = FEMData.from_h5(str(out))
    again = tmp_path / "elsewhere" / "copy.h5"
    again.parent.mkdir()
    fem.to_h5(str(again))
    assert FEMData.from_h5(str(again)).provenance == fem.provenance


# ---------------------------------------------------------------------------
# Keys and the one-record-per-user-call rule
# ---------------------------------------------------------------------------


@pytest.fixture
def g():
    with apeGmsh(model_name="prov", verbose=False) as session:
        yield session


def test_unnamed_declarations_get_order_keys(g):
    v = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.physical.add_volume([v])
    g.physical.add_volume([v], name="Named")
    g.physical.add_volume([v])
    paths = [r.path for r in prov.table_for(g).records]
    assert paths == ["geometry/box/#1", "neutral/physical_groups/#1",
                     "neutral/physical_groups/Named",
                     "neutral/physical_groups/#2"]


def test_one_record_per_user_call(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="slab")
    for i in range(3): g.model.geometry.add_box(2 * i + 2, 0, 0, 1, 1, 1)  # noqa: E701
    table = prov.table_for(g)
    assert [r.path for r in table.records] == [
        "geometry/box/slab", "geometry/box/#1", "geometry/box/#2",
        "geometry/box/#3"]
    loop = table.records[1:]
    assert len({r.site for r in loop}) == 1
    assert len(table.sites) == 2


def test_a_named_path_keeps_its_first_record(g):
    v = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    w = g.model.geometry.add_box(2, 0, 0, 1, 1, 1)
    g.physical.add_volume([v], name="P")
    g.physical.add_volume([w], name="P")  # appends to P
    recs = [r for r in prov.table_for(g).records
            if r.path == "neutral/physical_groups/P"]
    assert len(recs) == 1


def test_every_declaration_store_has_a_family():
    from apeGmsh.core._declarations import _DeclarationsMixin

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    import apeGmsh.core  # noqa: F401  (registers the composites)
    stores = {a for c in walk(_DeclarationsMixin)
              for a in c._DECLARATION_STORES}
    assert stores, "no declaration composites found"
    assert stores <= set(_PROVENANCE_FAMILY), stores - set(_PROVENANCE_FAMILY)


def test_a_non_session_owner_records_nothing():
    assert prov.capture(object(), "neutral", "loads", None) is None
    with pytest.raises(TypeError, match="session"):
        prov.store_for(object())


def test_bad_segment_is_refused():
    with pytest.raises(ValueError, match="family"):
        prov.ProvenanceStore().capture("neutral", "a/b", None)


# ---------------------------------------------------------------------------
# Hash invariance and the int32 refusal
# ---------------------------------------------------------------------------


def test_snapshot_id_ignores_provenance(oracle, tmp_path):
    _, out = oracle
    fem = FEMData.from_h5(str(out))
    assert fem.provenance is not None and fem.provenance.records
    bare = copy.copy(fem)
    bare.provenance = None
    bare.__dict__.pop("_snapshot_id_cache", None)
    assert bare.snapshot_id == fem.snapshot_id
    bare.to_h5(str(tmp_path / "bare.h5"))
    with h5py.File(out, "r") as a, h5py.File(tmp_path / "bare.h5", "r") as b:
        assert "provenance" not in b and PROVENANCE_KEY not in b["meta"].attrs
        assert a["meta"].attrs["snapshot_id"] == b["meta"].attrs["snapshot_id"]
        assert (a["meta/lineage"].attrs["fem_hash"]
                == b["meta/lineage"].attrs["fem_hash"])


@pytest.mark.parametrize("bad", [
    ProvenanceTable(sites=(SiteRow(0, 2**31, "f"),)),
    ProvenanceTable(records=(RecordRow("neutral/loads/#1", 0, 0, 2**31),)),
])
def test_int32_overflow_refuses_before_writing(oracle, tmp_path, bad):
    _, out = oracle
    fem = FEMData.from_h5(str(out))
    fem.provenance = bad
    target = tmp_path / "overflow.h5"
    with pytest.raises(ProvenanceOverflowError, match="int32"):
        write_fem_h5(fem, str(target))
    assert not target.exists()


# ---------------------------------------------------------------------------
# Overhead (V0 Q6): no opt-out, <= 50 ms per 1,000 registrations
# ---------------------------------------------------------------------------


def test_overhead_per_thousand_registrations(g):
    best = float("inf")
    for rep in range(3):
        t0 = time.perf_counter()
        for i in range(1000):
            prov.capture(g, "neutral", "bench", f"r{rep}_{i}")
        best = min(best, time.perf_counter() - t0)
    assert len(prov.store_for(g)) == 3000
    assert best <= 0.050, f"{best * 1e3:.1f} ms per 1,000 registrations"


# ---------------------------------------------------------------------------
# Fable review of b64272ce (#1319): each test fails on that head
# ---------------------------------------------------------------------------


def _run_script(path: Path, text: str) -> ProvenanceTable:
    """Run ``text`` as ``__main__`` from ``path``; it leaves ``TABLE``."""
    path.write_text(text, encoding="utf-8")
    return runpy.run_path(str(path), run_name="__main__")["TABLE"]


def _marked(text: str, marker: str) -> int:
    for i, line in enumerate(text.splitlines(), start=1):
        if f"# MARK:{marker}" in line:
            return i
    raise AssertionError(marker)


def test_promote_to_physical_and_rename_record(g):
    """Finding 1: both create a name without passing g.physical.add /
    g.labels.add, so each captures its own path."""
    v = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.labels.add(3, [v], name="old")
    g.labels.promote_to_physical("old", pg_name="P")
    g.labels.rename("old", "new")
    paths = [r.path for r in prov.table_for(g).records]
    assert "neutral/physical_groups/P" in paths
    assert "neutral/labels/new" in paths


CM_SCRIPT = '''\
from contextlib import contextmanager

from apeGmsh import apeGmsh
from apeGmsh._internal.provenance import table_for


@contextmanager
def building(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cm")  # MARK:cm_site
    yield


with apeGmsh(model_name="cm", verbose=False) as g:
    with building(g):  # MARK:cm_with
        pass
    TABLE = table_for(g)
'''


def test_contextmanager_helper_script_is_the_with_line(tmp_path):
    """Finding 4: the script walk passes through stdlib ``contextlib``."""
    script = tmp_path / "cm_script.py"
    table = _run_script(script, CM_SCRIPT)
    rec = table.record("geometry/box/cm")
    site, top = table.location(rec.site), table.location(rec.script)
    assert (site.line, site.function) == (
        _marked(CM_SCRIPT, "cm_site"), "building")
    assert (top.line, top.function) == (
        _marked(CM_SCRIPT, "cm_with"), "<module>")


IN_PACKAGE_SCRIPT = '''\
from apeGmsh import apeGmsh
from apeGmsh._internal.provenance import table_for

with apeGmsh(model_name="inpkg", verbose=False) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="slab")  # MARK:call
    TABLE = table_for(g)
'''


def test_a_main_script_under_the_package_is_the_users(tmp_path, monkeypatch):
    """Finding 2: a ``__main__`` script that lives under the apeGmsh tree
    (simulated by classifying its file as apeGmsh) still gets a real site
    and script, and exactly one record for the call."""
    script = tmp_path / "in_package.py"
    monkeypatch.setitem(prov._CLASS_CACHE, str(script), prov._APEGMSH)
    table = _run_script(script, IN_PACKAGE_SCRIPT)
    assert [r.path for r in table.records] == ["geometry/box/slab"]
    rec = table.records[0]
    want = (Path(os.path.abspath(str(script))).as_posix(),
            _marked(IN_PACKAGE_SCRIPT, "call"))
    for row in (rec.site, rec.script):
        loc = table.location(row)
        assert (loc.path, loc.line) == want


STDLIB_LAUNCHER = '''\
import runpy

TABLE = runpy.run_path(SCRIPT, run_name="__main__")["TABLE"]
'''


def test_a_stdlib_launcher_main_never_claims_script(tmp_path, monkeypatch):
    """Fable round 2: ``python -m cProfile|pdb|trace script.py`` runs a stdlib
    module as ``__main__`` below the user's script.  Simulated by a launcher
    classified as stdlib that runs the script; ``script`` must stay the
    user's ``with`` line, never the launcher."""
    script = tmp_path / "cm_script.py"
    script.write_text(CM_SCRIPT, encoding="utf-8")
    launcher = tmp_path / "launcher.py"
    launcher.write_text(STDLIB_LAUNCHER, encoding="utf-8")
    monkeypatch.setitem(prov._CLASS_CACHE, str(launcher), prov._STDLIB)
    table = runpy.run_path(
        str(launcher), init_globals={"SCRIPT": str(script)},
        run_name="__main__")["TABLE"]
    rec = table.record("geometry/box/cm")
    top = table.location(rec.script)
    assert (top.path, top.line, top.function) == (
        Path(os.path.abspath(str(script))).as_posix(),
        _marked(CM_SCRIPT, "cm_with"), "<module>")


PDB_LAUNCHER = '''\
G = {"__name__": "__main__", "__builtins__": __builtins__, "SCRIPT": SCRIPT}
exec(compile(
    "exec(compile(open(SCRIPT).read(), SCRIPT, 'exec'))", "<string>", "exec"),
    G)
TABLE = G["TABLE"]
'''


def test_a_pdb_trampoline_never_claims_script(tmp_path, monkeypatch):
    """Fable round 3: ``python -m pdb script.py`` runs the script through a
    ``<string>`` trampoline that ``Pdb.run`` execs in the script's own
    ``__main__`` globals.  Simulated by a stdlib launcher that does the same;
    the pseudo-file frame is outside the script's real-file ``__main__``
    frame and must not override it."""
    script = tmp_path / "cm_script.py"
    script.write_text(CM_SCRIPT, encoding="utf-8")
    launcher = tmp_path / "pdb_launcher.py"
    launcher.write_text(PDB_LAUNCHER, encoding="utf-8")
    monkeypatch.setitem(prov._CLASS_CACHE, str(launcher), prov._STDLIB)
    table = runpy.run_path(
        str(launcher), init_globals={"SCRIPT": str(script)},
        run_name="__main__")["TABLE"]
    top = table.location(table.record("geometry/box/cm").script)
    assert (top.path, top.line, top.function) == (
        Path(os.path.abspath(str(script))).as_posix(),
        _marked(CM_SCRIPT, "cm_with"), "<module>")


def test_dash_c_still_records_string():
    """``python -c`` has no real file: its ``<string>`` ``__main__`` frame
    is the script."""
    glb = {"__name__": "__main__"}
    exec(compile(CM_SCRIPT, "<string>", "exec"), glb)
    top = glb["TABLE"].location(glb["TABLE"].record("geometry/box/cm").script)
    assert (top.path, top.line, top.function) == (
        "<string>", _marked(CM_SCRIPT, "cm_with"), "<module>")


def test_parts_add_records_the_instance_label():
    """Finding 3: ``g.parts.add(part, label='b1')`` records ``b1`` itself,
    not the first synthesised sidecar label ``b1.core``.  Declarations
    made inside the Part's own session stay in the Part's store."""
    from apeGmsh import Part

    part = Part("beam")
    with part:
        part.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="core")
    try:
        with apeGmsh(model_name="asm", verbose=False) as asm:
            asm.parts.add(part, label="b1")
            paths = [r.path for r in prov.table_for(asm).records]
    finally:
        part.cleanup()
    assert paths == ["neutral/labels/b1"]


def test_missing_base_dir_is_malformed(oracle, tmp_path):
    """Finding 5: a missing ``@base_dir`` is MalformedH5Error, not KeyError."""
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    _, out = oracle
    broken = tmp_path / "no_base.h5"
    broken.write_bytes(out.read_bytes())
    with h5py.File(broken, "r+") as f:
        del f["provenance"].attrs["base_dir"]
    with pytest.raises(MalformedH5Error, match="base_dir"):
        FEMData.from_h5(str(broken))
