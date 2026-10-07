"""ADR 0117 P3 (AS3): the ``/assembly`` zone, ``Assembly.h5`` and ``from_h5``.

Oracles, each naming the right answer independently of the code under test:

* INV-2 — two runs of one assembly script give the same archive, attribute
  for attribute and byte for byte in every dataset, except the run's
  timestamps and session id; and the same deck. Checked in one process and
  in two fresh interpreters under different hash seeds (``subprocess``).
* INV-10 — ``/assembly`` and its ``/meta`` key are the only difference
  from the plain ``ops.h5`` of the same bridge; the neutral, opensees and
  provenance stamps are the fixture constants; every model reader returns
  the same semantic dump for both files; a plain file opens as before and
  ``Assembly.from_h5`` refuses it; ``from_h5`` round-trips the declared
  instances and ties.
* closed forms — instance ``k`` of the stack occupies the FEM-id window
  ``k * 1_000_000`` (ADR 0038) of the source's span; each tie of two
  2x2-meshed faces resolves ``3 * 3 = 9`` records; the source hashes are
  the source file's own ``snapshot_id`` and ``model_hash``; each
  provenance record points at the line that declared it.
* seams — zero ties write an empty table; a rewrite replaces the zone; an
  invalid row or a failure part-way leaves no zone; stale or unbridged
  assemblies and archive-read assemblies refuse to write; the reader
  refuses a newer stamp, a missing key, an unknown kind and ragged columns.
* corpus (ADR 0113 D8) — the committed ``assembly_1.0.h5`` opens through
  today's readers with the dump its writer recorded, without its instance
  file beside it, and opening it changes no byte.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from tests.fixtures.schema import (
    NEUTRAL_CURRENT,
    OPENSEES_CURRENT,
    PROVENANCE_CURRENT,
)

# Units: N, mm, MPa.
SIDE = 10.0
H = 10.0
TOL = 0.01
#: ADR 0038 reservation granularity: instance k (1-based) starts at k * GRANULE.
GRANULE = 1_000_000
#: Nodes on one face of the 2x2x2 block: (2 + 1) ** 2.
FACE_NODES = 9
#: The attributes INV-2 lets differ between two runs, by exact path: the
#: archive's write time and session id (ADR 0117 INV-2), and each
#: /composed_from entry's compose time (ADR 0038 ``composed_at``), a
#: timestamp of the same kind that the ADR's list omits (#1550). The same
#: attribute name anywhere else is compared.
RUN_STAMPS: frozenset[tuple[str, str]] = frozenset({
    ("meta", "created_iso"), ("meta", "session_id"),
    ("composed_from/*", "composed_at"),
})


def is_run_stamp(path: str, attr: str) -> bool:
    """Whether ``path@attr`` is one of :data:`RUN_STAMPS`; ``*`` matches
    exactly one path segment."""
    for pattern, name in RUN_STAMPS:
        if attr != name:
            continue
        want, got = pattern.split("/"), path.split("/")
        if len(want) == len(got) and all(w in ("*", g) for w, g in zip(want, got)):
            return True
    return False

_ROOT = Path(__file__).resolve().parents[2]
CORPUS = _ROOT / "tests" / "fixtures" / "schema_corpus"
#: sha256 of the committed corpus file; a regenerated file must say so here.
CORPUS_H5_SHA256 = "abd01c3c7608b84eef4a7c401d28d0e3cc1ce9c659a14cbad1a5a4115c2e01b1"


def _here() -> int:
    """The caller's current line."""
    return sys._getframe(1).f_lineno


def _faces_at_z(g, z: float) -> list[int]:
    lo = (-SIDE, -SIDE, z - TOL)
    hi = (2 * SIDE, 2 * SIDE, z + TOL)
    return g.model.select(None, dim=2).in_box(lo, hi).result().tags()


def write_block(workdir: Path) -> Path:
    """A 2x2x2 hex8 block with PGs ``Vol``, ``bot``, ``top``, as ``block.h5``."""
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="block", save_to=str(workdir / "block_mesh.h5"),
                 overwrite=True) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, H, label="v")
        g.physical.add_volume("v", name="Vol")
        g.physical.add_surface(_faces_at_z(g, 0.0), name="bot")
        g.physical.add_surface(_faces_at_z(g, H), name="top")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        fem = g.mesh.queries.get_fem_data(dim=None)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=200_000.0, nu=0.3, rho=7.85e-9, name="steel")
    ops.element.stdBrick(pg="Vol", material=steel)
    path = workdir / "block.h5"
    ops.h5(str(path))
    return path


def declare_stack(block: Path) -> tuple[Any, dict[str, int]]:
    """Three stacked instances (the third turned half a turn about z), one
    named and one unnamed tie. Returns the assembly and each declaration's
    line."""
    from apeGmsh.assembly import Assembly

    lines: dict[str, int] = {}
    asm = Assembly("stack")
    lines["pier_1"] = _here() + 1
    asm.instance("pier_1", block)
    lines["pier_2"] = _here() + 1
    asm.instance("pier_2", block, translate=(0.0, 0.0, H))
    lines["pier_3"] = _here() + 1
    asm.instance("pier_3", block, translate=(SIDE, SIDE, 2 * H), rotate=((0.0, 0.0, 1.0), math.pi))
    lines["t1"] = _here() + 1
    asm.tie("pier_1.top", "pier_2.bot", enforce="equation", dofs=[1, 2, 3], name="t1")
    lines["#1"] = _here() + 1
    asm.tie("pier_2.top", "pier_3.bot", enforce="equation", dofs=[1, 2, 3])
    return asm, lines


def _bridge(asm: Any) -> Any:
    from apeGmsh.assembly import AssemblyRankWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssemblyRankWarning)
        ops = asm.bridge(ndm=3, ndf=3)
    ops.fix(pg="pier_1.bot", dofs=[1, 1, 1])
    return ops


def build_archive(workdir: Path) -> tuple[Path, Path]:
    """The INV-2 script: block file, stack, bridge, archive and deck."""
    workdir.mkdir(parents=True, exist_ok=True)
    block = write_block(workdir)
    asm, _ = declare_stack(block)
    ops = _bridge(asm)
    archive, deck = workdir / "stack.h5", workdir / "stack.tcl"
    asm.h5(archive, model_name="stack")
    ops.tcl(str(deck), flat=True)
    return archive, deck


def h5dump(path: Path, *, skip_stamps: bool = True) -> dict[str, Any]:
    """Every group, dataset and attribute of ``path`` as comparable values:
    names, dtypes, shapes and the raw bytes of numeric data. With
    ``skip_stamps`` the :data:`RUN_STAMPS` attributes are left out."""

    def value(x: Any) -> Any:
        a = np.asarray(x)
        if a.dtype == object or a.dtype.kind in "SU":
            return [v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in a.reshape(-1).tolist()] + [list(a.shape)]
        if a.dtype.fields is not None:
            # A compound row's raw bytes hold pointers for its vlen fields.
            return [str(a.dtype), list(a.shape), repr(a.tolist())]
        return [a.dtype.str, list(a.shape), hashlib.sha256(
            np.ascontiguousarray(a).tobytes()).hexdigest()]

    def attrs(name: str, obj: Any) -> dict[str, Any]:
        return {k: value(obj.attrs[k]) for k in sorted(obj.attrs)
                if not (skip_stamps and is_run_stamp(name, k))}

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        out["/"] = {"attrs": attrs("", f)}

        def visit(name: str, obj: Any) -> None:
            entry: dict[str, Any] = {"attrs": attrs(name, obj)}
            if isinstance(obj, h5py.Dataset):
                entry["dtype"] = str(obj.dtype)
                entry["data"] = value(obj[()])
            out[name] = entry

        f.visititems(visit)
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, Any]:
    d = tmp_path_factory.mktemp("as3")
    block = write_block(d)
    asm, lines = declare_stack(block)
    ops = _bridge(asm)
    archive, plain = d / "archive.h5", d / "plain.h5"
    asm.h5(archive, model_name="stack")
    ops.h5(str(plain), model_name="stack")
    return {"dir": d, "block": block, "asm": asm, "ops": ops, "lines": lines,
            "archive": archive, "plain": plain}


def _root_and_meta(path: Path) -> tuple[set[str], set[str]]:
    with h5py.File(path, "r") as f:
        return set(f.keys()), set(f["meta"].attrs.keys())


# ---------------------------------------------------------------------------
# INV-2 — two runs, one archive and one deck
# ---------------------------------------------------------------------------

def test_inv2_two_runs_write_the_same_archive_and_deck(tmp_path):
    first_h5, first_tcl = build_archive(tmp_path / "run")
    dump_1, deck_1 = h5dump(first_h5), first_tcl.read_bytes()
    second_h5, second_tcl = build_archive(tmp_path / "run")
    dump_2, deck_2 = h5dump(second_h5), second_tcl.read_bytes()

    assert "assembly/instances/label" in dump_1, "the archive carries /assembly"
    assert "assembly/ties/n_records" in dump_1
    assert dump_1.keys() == dump_2.keys()
    diffs = [k for k in dump_1 if dump_1[k] != dump_2[k]]
    assert not diffs, f"the two runs differ at {diffs}"
    assert deck_1 == deck_2
    # The exclusions are the run stamps, and only those: they are present.
    with h5py.File(second_h5, "r") as f:
        assert {"created_iso", "session_id"} <= set(f["meta"].attrs)
        assert "composed_at" in f["composed_from/pier_1"].attrs


_CHILD = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "from tests.assembly.test_assembly_h5 import build_archive, h5dump\n"
    "h5, tcl = build_archive(Path(sys.argv[1]))\n"
    "json.dump({'h5': h5dump(h5), 'tcl': tcl.read_text(encoding='utf-8')},\n"
    "          open(sys.argv[2], 'w', encoding='utf-8'))\n"
)


@pytest.mark.subprocess
def test_inv2_fresh_interpreters_under_two_hash_seeds(tmp_path):
    runs = []
    for seed in ("1", "2"):
        env = dict(os.environ)
        # The worktree's sources, never the editable install's checkout.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_ROOT / "src"), str(_ROOT), env.get("PYTHONPATH", "")])
        env["PYTHONHASHSEED"] = seed
        out = tmp_path / f"dump_{seed}.json"
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, str(tmp_path / "run"), str(out)],
            cwd=str(_ROOT), env=env, capture_output=True, text=True,
            timeout=300, check=False,
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        runs.append(json.loads(out.read_text(encoding="utf-8")))
    assert "assembly/instances/label" in runs[0]["h5"]
    diffs = [k for k in runs[0]["h5"] if runs[0]["h5"][k] != runs[1]["h5"].get(k)]
    assert not diffs and runs[0]["h5"].keys() == runs[1]["h5"].keys(), diffs
    assert runs[0]["tcl"] == runs[1]["tcl"]


# ---------------------------------------------------------------------------
# INV-10 — one new zone, unchanged versions, every reader, round-trip
# ---------------------------------------------------------------------------

def test_inv10_assembly_is_the_only_new_zone(built):
    from apeGmsh.opensees._internal.schema_version import ASSEMBLY_KEY

    a_root, a_meta = _root_and_meta(built["archive"])
    p_root, p_meta = _root_and_meta(built["plain"])
    assert a_root - p_root == {"assembly"} and p_root <= a_root
    assert a_meta - p_meta == {ASSEMBLY_KEY} and p_meta <= a_meta
    assert "composed_from" in a_root, "/composed_from is still written"
    with h5py.File(built["archive"], "r") as f:
        meta = f["meta"].attrs
        assert meta["neutral_schema_version"] == NEUTRAL_CURRENT
        assert meta["opensees_schema_version"] == OPENSEES_CURRENT
        assert meta["provenance_schema_version"] == PROVENANCE_CURRENT
    # Outside /assembly and its key, the archive is the plain file.
    a_dump, p_dump = h5dump(built["archive"]), h5dump(built["plain"])
    a_dump["meta"]["attrs"].pop(ASSEMBLY_KEY)
    shared = {k: v for k, v in a_dump.items() if not k.startswith("assembly")}
    assert shared == p_dump


@pytest.mark.filterwarnings(
    "ignore::apeGmsh.opensees._internal.compose.ReplaySkippedStreamWarning")
def test_inv10_every_reader_opens_the_archive_as_the_plain_file(built):
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from tests.fixtures.schema_corpus._semantic_dump import dump_fem, dump_model

    for path in (built["archive"], built["plain"]):
        assert FEMData.from_h5(str(path)).snapshot_id
    assert dump_fem(FEMData.from_h5(str(built["archive"]))) == dump_fem(
        FEMData.from_h5(str(built["plain"])))
    a_model = OpenSeesModel.from_h5(str(built["archive"]))
    p_model = OpenSeesModel.from_h5(str(built["plain"]))
    assert dump_model(a_model) == dump_model(p_model)
    assert a_model.build("tcl") == p_model.build("tcl")


def test_inv10_plain_file_has_no_assembly_and_from_h5_refuses_it(built):
    from apeGmsh.assembly import Assembly, AssemblyError

    with pytest.raises(AssemblyError, match="no /assembly zone"):
        Assembly.from_h5(built["plain"])
    with pytest.raises(AssemblyError, match="no /assembly zone"):
        Assembly.from_h5(built["block"])


def test_inv10_from_h5_round_trips_the_declared_instances_and_ties(built):
    from apeGmsh.assembly import Assembly

    asm = built["asm"]
    back = Assembly.from_h5(built["archive"])
    assert back.name == asm.name == "stack"
    assert back.instances == asm.instances
    assert back.ties == asm.ties
    assert [t.name for t in back.ties] == ["t1", None]
    assert back.instances[2].rotate == ((0.0, 0.0, 1.0), math.pi)
    assert back.instances[0].rotate is None
    for got, want in zip(back.ties, asm.ties):
        assert got.definition.enforce == want.definition.enforce
        assert got.definition.dofs == want.definition.dofs


def test_from_h5_never_opens_an_instance_file(built, tmp_path):
    from apeGmsh.assembly import Assembly

    moved = tmp_path / "alone.h5"
    shutil.copy(built["archive"], moved)
    back = Assembly.from_h5(moved)
    assert back.instances == built["asm"].instances
    assert not (tmp_path / "block.h5").exists()


# ---------------------------------------------------------------------------
# Closed forms: the rows
# ---------------------------------------------------------------------------

def test_instance_rows_hold_the_fem_id_windows_and_source_hashes(built):
    from apeGmsh.assembly._h5 import read_assembly_zone
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    src = FEMData.from_h5(str(built["block"]))
    ids = list(src.nodes.ids) + [i for g in src.elements for i in g.ids]
    span = int(max(ids)) - int(min(ids)) + 1
    zone = read_assembly_zone(built["archive"])
    assert [r.label for r in zone.instances] == ["pier_1", "pier_2", "pier_3"]
    assert [r.fem_id_base for r in zone.instances] == [GRANULE, 2 * GRANULE, 3 * GRANULE]
    assert [r.fem_id_span for r in zone.instances] == [span] * 3
    assert {r.source_fem_hash for r in zone.instances} == {src.snapshot_id}
    model_hash = OpenSeesModel.from_h5(str(built["block"])).lineage.model_hash
    assert model_hash and {r.source_opensees_hash for r in zone.instances} == {model_hash}
    assert {r.source_path for r in zone.instances} == {built["block"].as_posix()}
    assert [r.partition_rank for r in zone.instances] == [-1, -1, -1]
    assert zone.instances[1].translate == (0.0, 0.0, H)
    assert zone.instances[1].rotate == (0.0, 0.0, 0.0, 0.0)
    assert zone.instances[2].rotate == (0.0, 0.0, 1.0, math.pi)


def test_tie_rows_hold_ports_params_and_record_counts(built):
    from apeGmsh.assembly._h5 import read_assembly_zone

    zone = read_assembly_zone(built["archive"])
    assert [(t.name, t.kind, t.master, t.slave) for t in zone.ties] == [
        ("t1", "tie", "pier_1.top", "pier_2.bot"),
        ("", "tie", "pier_2.top", "pier_3.bot"),
    ]
    assert [t.n_records for t in zone.ties] == [FACE_NODES, FACE_NODES]
    assert json.loads(zone.ties[0].params) == {
        "dofs": [1, 2, 3], "enforce": "equation", "method": "collocation",
        "tolerance": 1.0}
    # The two counts are the assembly's whole tie set: the bridge FEM holds
    # exactly that many records, every one named after its tie.
    fem = built["asm"]._bridged.fem
    recs = list(fem.nodes.constraints) + list(fem.elements.constraints)
    assert len(recs) == 2 * FACE_NODES
    assert sorted(str(r.name) for r in recs) == ["None"] * FACE_NODES + ["t1"] * FACE_NODES


def test_provenance_records_point_at_the_declaring_lines(built):
    from apeGmsh.mesh import FEMData

    table = FEMData.from_h5(str(built["archive"])).provenance
    for key, line in built["lines"].items():
        family = "ties" if key in ("t1", "#1") else "instances"
        rec = table.record(f"assembly/{family}/{key}")
        assert rec.origin == "user"
        loc = table.location(rec.site)
        assert Path(loc.path).name == Path(__file__).name
        assert (loc.line, loc.function) == (line, "declare_stack")


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------

def test_zero_ties_write_an_empty_ties_table(built, tmp_path):
    from apeGmsh.assembly import Assembly

    asm = Assembly("single").instance("solo", built["block"])
    _bridge_plain(asm)
    out = tmp_path / "single.h5"
    asm.h5(out)
    with h5py.File(out, "r") as f:
        ties = f["assembly/ties"]
        assert set(ties) == {"name", "kind", "master", "slave", "params", "n_records"}
        assert all(ties[c].shape == (0,) for c in ties)
    assert Assembly.from_h5(out).ties == ()


def _bridge_plain(asm: Any) -> Any:
    from apeGmsh.assembly import AssemblyRankWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssemblyRankWarning)
        return asm.bridge(ndm=3, ndf=3)


def test_rewriting_replaces_the_zone(built, tmp_path):
    from apeGmsh.assembly._h5 import read_assembly_zone, write_assembly_zone

    out = tmp_path / "again.h5"
    shutil.copy(built["archive"], out)
    zone = read_assembly_zone(out)
    write_assembly_zone(out, zone.name, zone.instances, zone.ties)
    write_assembly_zone(out, zone.name, zone.instances, zone.ties)
    assert read_assembly_zone(out) == zone
    with h5py.File(out, "r") as f:
        assert [k for k in f if k == "assembly"] == ["assembly"]
        assert f["assembly/instances/label"].shape == (3,)
    built["asm"].h5(out, model_name="stack")
    assert read_assembly_zone(out) == zone


def test_an_invalid_row_raises_before_the_file_changes(built, tmp_path):
    import dataclasses

    from apeGmsh.assembly import AssemblyError
    from apeGmsh.assembly._h5 import read_assembly_zone, write_assembly_zone

    out = tmp_path / "bad.h5"
    shutil.copy(built["archive"], out)
    before = out.read_bytes()
    zone = read_assembly_zone(out)
    bad_tie = dataclasses.replace(zone.ties[0], n_records=0)
    with pytest.raises(AssemblyError, match="n_records=0"):
        write_assembly_zone(out, zone.name, zone.instances, (bad_tie,))
    unknown = dataclasses.replace(zone.ties[0], kind="contact")
    with pytest.raises(AssemblyError, match="kind 'contact'"):
        write_assembly_zone(out, zone.name, zone.instances, (unknown,))
    twice = (zone.instances[0], zone.instances[0])
    with pytest.raises(AssemblyError, match="repeat"):
        write_assembly_zone(out, zone.name, twice, ())
    assert out.read_bytes() == before


def test_a_failure_part_way_leaves_no_zone(built, tmp_path, monkeypatch):
    from apeGmsh.assembly import _h5
    from apeGmsh.opensees._internal.schema_version import ASSEMBLY_KEY

    out = tmp_path / "half.h5"
    shutil.copy(built["archive"], out)
    zone = _h5.read_assembly_zone(out)
    real = _h5._write_columns

    def half(f, name, columns):
        real(f, name, {"instances": columns["instances"]})
        raise OSError("disk full")

    monkeypatch.setattr(_h5, "_write_columns", half)
    with pytest.raises(OSError, match="disk full"):
        _h5.write_assembly_zone(out, zone.name, zone.instances, zone.ties)
    with h5py.File(out, "r") as f:
        assert "assembly" not in f
        assert ASSEMBLY_KEY not in f["meta"].attrs


def test_h5_refuses_without_a_bridge_or_after_new_declarations(built, tmp_path):
    from apeGmsh.assembly import Assembly, AssemblyError

    asm = Assembly("late").instance("a", built["block"])
    out = tmp_path / "late.h5"
    with pytest.raises(AssemblyError, match="call bridge"):
        asm.h5(out)
    _bridge_plain(asm)
    asm.instance("b", built["block"], translate=(0.0, 0.0, H))
    with pytest.raises(AssemblyError, match="declared after bridge"):
        asm.h5(out)
    assert not out.exists()


def test_an_archive_read_assembly_cannot_bridge_or_write(built, tmp_path):
    from apeGmsh.assembly import Assembly, AssemblyError

    back = Assembly.from_h5(built["archive"])
    with pytest.raises(AssemblyError, match="never re-fetched"):
        back.bridge(ndm=3, ndf=3)
    with pytest.raises(AssemblyError, match="re-lists"):
        back.h5(tmp_path / "x.h5")


def _tampered(built, tmp_path, name: str, edit) -> Path:
    out = tmp_path / name
    shutil.copy(built["archive"], out)
    with h5py.File(out, "r+") as f:
        edit(f)
    return out


def test_reader_refuses_a_newer_stamp_and_a_missing_key(built, tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees._internal.schema_version import (
        ASSEMBLY,
        ASSEMBLY_KEY,
        SchemaVersionError,
        reader_version,
    )
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    v = reader_version(ASSEMBLY)
    newer = _tampered(built, tmp_path, "newer.h5", lambda f: f["meta"].attrs.__setitem__(
        ASSEMBLY_KEY, f"{v.major}.{v.minor + 1}.0"))
    with pytest.raises(SchemaVersionError, match="newer than this reader"):
        Assembly.from_h5(newer)
    unkeyed = _tampered(built, tmp_path, "unkeyed.h5",
                        lambda f: f["meta"].attrs.__delitem__(ASSEMBLY_KEY))
    with pytest.raises(MalformedH5Error, match="no assembly_schema_version"):
        Assembly.from_h5(unkeyed)


def test_reader_refuses_an_unknown_kind_and_ragged_columns(built, tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    def kind(f):
        f["assembly/ties/kind"][0] = "contact"

    with pytest.raises(MalformedH5Error, match="kind 'contact'"):
        Assembly.from_h5(_tampered(built, tmp_path, "kind.h5", kind))

    def ragged(f):
        del f["assembly/instances/fem_id_span"]
        f["assembly/instances"].create_dataset("fem_id_span", data=np.array([1, 2]))

    with pytest.raises(MalformedH5Error, match="differ in length"):
        Assembly.from_h5(_tampered(built, tmp_path, "ragged.h5", ragged))

    def missing(f):
        del f["assembly/ties/params"]

    with pytest.raises(MalformedH5Error, match="ties/params is missing"):
        Assembly.from_h5(_tampered(built, tmp_path, "missing.h5", missing))


def test_a_refused_instance_records_no_provenance_and_can_be_retried(built, tmp_path):
    """Review F1: every check runs before the provenance capture."""
    from apeGmsh.assembly import Assembly, AssemblyError
    from apeGmsh.mesh import FEMData

    asm = Assembly("retry")
    with pytest.raises(AssemblyError, match="axis is zero"):
        asm.instance("A", built["block"], rotate=((0.0, 0.0, 0.0), 1.0))
    with pytest.raises(AssemblyError, match="finite"):
        asm.instance("A", built["block"], translate=(0.0, math.nan, 0.0))
    asm.instance("A", built["block"])
    _bridge_plain(asm)
    out = tmp_path / "retry.h5"
    asm.h5(out)
    paths = [r.path for r in FEMData.from_h5(str(out)).provenance.records
             if r.path.startswith("assembly/")]
    assert paths == ["assembly/instances/A"]


def test_a_refused_h5_leaves_an_existing_archive_byte_identical(built, tmp_path, monkeypatch):
    """Review F2: the rows are validated before ops.h5 overwrites the target."""
    import dataclasses

    from apeGmsh.assembly import AssemblyError, _assembly

    out = tmp_path / "keep.h5"
    shutil.copy(built["archive"], out)
    before = out.read_bytes()
    real = _assembly._tie_rows

    def zero_records(b):
        return [dataclasses.replace(t, n_records=0) for t in real(b)]

    monkeypatch.setattr(_assembly, "_tie_rows", zero_records)
    with pytest.raises(AssemblyError, match="n_records=0"):
        built["asm"].h5(out, model_name="stack")
    assert out.read_bytes() == before


def test_reader_refuses_a_zero_axis_with_a_nonzero_angle(built, tmp_path):
    """Review F3: only the all-zero rotate row means 'not rotated'."""
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    def tilt(f):
        f["assembly/instances/rotate"][0, 3] = 0.5

    with pytest.raises(MalformedH5Error, match="zero axis with a nonzero angle"):
        Assembly.from_h5(_tampered(built, tmp_path, "tilt.h5", tilt))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_translate_or_rotate_is_refused(built, bad):
    """Review F5: nan and inf are refused before anything is recorded."""
    from apeGmsh.assembly import Assembly, AssemblyError
    from apeGmsh.assembly._instances import check_rotate, check_translate

    with pytest.raises(AssemblyError, match="finite"):
        check_translate((0.0, bad, 0.0))
    with pytest.raises(AssemblyError, match="finite"):
        check_rotate(((0.0, 0.0, bad), 1.0))
    with pytest.raises(AssemblyError, match="finite"):
        check_rotate(((0.0, 0.0, 1.0), bad))
    asm = Assembly("x")
    with pytest.raises(AssemblyError, match="finite"):
        asm.instance("a", built["block"], translate=(bad, 0.0, 0.0))
    assert asm.instances == ()


def test_inv2_skips_the_run_stamps_only_at_their_own_paths(built, tmp_path):
    """Review F4: a run-stamp name anywhere else is still compared."""
    assert is_run_stamp("meta", "created_iso")
    assert is_run_stamp("composed_from/pier_1", "composed_at")
    assert not is_run_stamp("opensees", "created_iso")
    assert not is_run_stamp("composed_from/pier_1/properties", "composed_at")
    copies = []
    for k, stamp in enumerate(("a", "b")):
        out = tmp_path / f"stamped_{k}.h5"
        shutil.copy(built["archive"], out)
        with h5py.File(out, "r+") as f:
            f["meta"].attrs["created_iso"] = stamp  # skipped
            f["assembly"].attrs["session_id"] = stamp  # compared
        copies.append(h5dump(out))
    assert copies[0]["meta"] == copies[1]["meta"]
    assert copies[0]["assembly"] != copies[1]["assembly"]


# ---------------------------------------------------------------------------
# Corpus (ADR 0113 D8): the committed 1.0 file
# ---------------------------------------------------------------------------

def test_corpus_assembly_file_opens_with_its_recorded_dump(tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.assembly._h5 import read_assembly_zone
    from apeGmsh.mesh import FEMData
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from tests.fixtures.schema_corpus._semantic_dump import (
        dump_assembly,
        dump_fem,
        dump_model,
        dump_stamps,
    )

    src = CORPUS / "assembly_1.0.h5"
    era = json.loads((CORPUS / "assembly_1.0.dump.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(src.read_bytes()).hexdigest() == CORPUS_H5_SHA256
    h5 = tmp_path / src.name  # alone: its block.h5 is not beside it
    shutil.copy(src, h5)
    before = h5.read_bytes()
    today = {
        "fem": dump_fem(FEMData.from_h5(str(h5))),
        "meta": dump_stamps(str(h5)),
        "model": dump_model(OpenSeesModel.from_h5(str(h5))),
        "assembly": dump_assembly(Assembly.from_h5(h5), read_assembly_zone(h5)),
    }
    for key in ("fem", "meta", "model", "assembly"):
        assert today[key] == era[key], key
    assert h5.read_bytes() == before
    with h5py.File(h5, "r") as f:
        assert f["meta"].attrs["assembly_schema_version"] == str(
            read_assembly_zone(h5).version)
