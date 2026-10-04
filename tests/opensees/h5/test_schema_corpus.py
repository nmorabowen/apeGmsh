"""The schema corpus: real old files open through today's readers (ADR 0113 D8).

``tests/fixtures/schema_corpus/`` holds one ``model.h5`` per schema minor,
written by that minor's frozen writer checked out of git
(``scripts/build_schema_corpus.py``), with the semantic dump the era's
**own** reader recorded beside it. ``MANIFEST.json`` names every minor
from each zone's floor to its current minor, ``ok`` or ``gap``, keeps the
real files of the eras *below* a floor as the evidence that raised it
(``below_floor``), and holds one **variant** file per shim-ledger entry
(ADR 0113 D4): ``sp_cases`` before 2.26.1 and ``frame2d`` before 2.34.0.

Invariants held here (ADR 0113):

* INV 5, corpus opens: every ``ok`` file opens through ``FEMData.from_h5``
  (and ``OpenSeesModel.from_h5`` when it has ``/opensees``), and today's
  dump equals the era's except for the fields the shim ledger names;
* INV 6, corpus complete: one entry per minor from floor to current, each
  ``ok`` with its commit, or a ``gap`` with its reason; an entry below the
  floor is a ``below_floor`` gap with a real file;
* INV 7, bytes unchanged: opening a file never changes its bytes;
* INV 10, tamper check kept: a tampered copy fails the snapshot_id check;
* INV 12, paper floor fails: a floor set to the current minor turns the
  corpus check red;
* D3, evidence-gated floors: a floor stands only if no runnable era at or
  above it is a gap, and it sits one minor above the last era whose file
  no reader opens (opensees 2.11 stamped neutral 2.7.0, below the neutral
  floor, so the opensees floor is 2.12.0: decided 2026-10-04 on #1303);
* the below-floor files refuse through the zone's own reader, naming the
  floor; the ledger variants read as the ledger says.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.mesh import _femdata_h5_io
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees._internal.schema_version import (
    NEUTRAL,
    OPENSEES,
    SchemaVersion,
    SchemaVersionError,
    reader_floor,
    reader_version,
)
from apeGmsh.opensees.emitter import h5 as _h5_writer
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5_reader import (
    META_NDM_IS_SPATIAL_FROM,
    MalformedH5Error,
)
from apeGmsh.opensees.opensees_model import OpenSeesModel
from tests.fixtures.schema import NEUTRAL_CURRENT, OPENSEES_CURRENT
from tests.fixtures.schema_corpus._semantic_dump import (
    DUMP_FORMAT,
    dump_fem,
    dump_model,
    dump_stamps,
)

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"
BUILDER = Path(__file__).resolve().parents[3] / "scripts" / "build_schema_corpus.py"
MANIFEST: dict[str, Any] = json.loads((CORPUS / "MANIFEST.json").read_text(encoding="utf-8"))
ENTRIES: list[dict[str, Any]] = MANIFEST["entries"]
ZONES = {"neutral": NEUTRAL, "opensees": OPENSEES}
STAMP_KEYS = {"neutral": "neutral_schema_version", "opensees": "opensees_schema_version"}

#: The ops.model ndm the plain opensees generator declares
#: (``_era_generator.opensees_frame``): the closed-form answer for ``ndm``.
DECLARED_NDM = 3

#: Pre-2.26.1 SP loads read as one ``default`` case (ADR 0113 D4 ledger, Q5).
SP_PER_CASE_FROM = (2, 26, 1)

#: The case names the ``sp_cases`` variant authored (``_era_generator.SP_CASES``;
#: not imported: the generator imports gmsh and binds ``_semantic_dump`` under a
#: bare name, which the shared pytest process must not do). The era recorded
#: them in its dump's ``generator_notes``, and the test holds them to this.
SP_CASES = ["PushA", "PushB"]

#: The shim ledger of ADR 0113 D4, as (neutral version the field is trusted
#: from, dump paths excluded from the era comparison below it).
SHIM_LEDGER: tuple[tuple[tuple[int, int, int], tuple[str, ...]], ...] = (
    (META_NDM_IS_SPATIAL_FROM, ("model.ndm",)),
    (SP_PER_CASE_FROM, ("fem.loads.sp_patterns", "model.fem.loads.sp_patterns")),
)

#: The ``frame2d`` variant declared ``ops.model(ndm=2, ndf=3)``; its
#: pre-2.34.0 writer stamped the mesh dimension (1) in ``/meta/ndm``. The
#: file still says 2-D: a ``(1, 0)`` vecxz exists only in 2-D. Since #1358
#: ``read_spatial_ndm`` salvages 2 from that signature (before it, the
#: stamp, 1, stood and ``build()`` dropped every y coordinate).
FRAME2D_DECLARED_NDM = 2

#: The ``(dof, value)`` multiset the ``sp_cases`` variant authored: 9 base
#: nodes, dofs [1, 1, 1], values (0, 0, -0.01) under PushA and (0.01, 0, 0)
#: under PushB. Both cases' values must survive the flattening.
SP_RECORDS = {(1, 0.01): 9, (3, -0.01): 9, (1, 0.0): 9, (2, 0.0): 18, (3, 0.0): 9}


def _v(s: str) -> tuple[int, int, int]:
    p = SchemaVersion.parse(s)
    return (p.major, p.minor, p.patch)


def _minor_of(minor: str) -> int:
    return int(minor.split(".")[1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain() -> list[dict[str, Any]]:
    return [e for e in ENTRIES if "variant" not in e]


def _variant(name: str) -> dict[str, Any]:
    found = [e for e in ENTRIES if e.get("variant") == name]
    assert len(found) == 1, f"the manifest must hold exactly one {name!r} variant"
    return found[0]


def _with_files() -> list[dict[str, Any]]:
    return [e for e in ENTRIES if "files" in e]


def _openable() -> list[dict[str, Any]]:
    return [e for e in ENTRIES if e["status"] == "ok"]


def _id(e: dict[str, Any]) -> str:
    stem = f"{e['zone']}-{e['minor']}"
    return f"{stem}-{e['variant']}" if "variant" in e else stem


def _excluded(entry: dict[str, Any]) -> set[str]:
    neutral = _v(entry["stamps"]["neutral"])
    return {p for since, paths in SHIM_LEDGER if neutral < since for p in paths}


def _era_dump(entry: dict[str, Any]) -> dict[str, Any]:
    return json.loads((CORPUS / entry["files"]["dump"]["name"]).read_text(encoding="utf-8"))


def _builder() -> Any:
    """``scripts/build_schema_corpus.py`` as a module (it runs no git on import)."""
    spec = importlib.util.spec_from_file_location("build_schema_corpus", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: its dataclass resolves postponed annotations
    # through sys.modules. Removed after, so no bare name outlives the test.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _diff(era: Any, today: Any, path: str, skip: set[str]) -> list[str]:
    """Paths where ``today`` does not reproduce what ``era`` recorded.

    Every key the era's reader produced must come back with an equal value.
    A key only today's reader knows is an additive field the era had no name
    for, so it is not compared.
    """
    if path in skip:
        return []
    if isinstance(era, dict):
        if not isinstance(today, dict):
            return [f"{path}: era dict, today {type(today).__name__}"]
        out: list[str] = []
        for k, v in era.items():
            sub = f"{path}.{k}" if path else k
            if sub in skip:
                continue
            if k not in today:
                out.append(f"{sub}: missing today")
            else:
                out += _diff(v, today[k], sub, skip)
        return out
    if isinstance(era, list):
        if not isinstance(today, list) or len(era) != len(today):
            return [f"{path}: era {era!r} != today {today!r}"]
        return [d for i, (a, b) in enumerate(zip(era, today))
                for d in _diff(a, b, f"{path}[{i}]", skip)]
    return [] if era == today else [f"{path}: era {era!r} != today {today!r}"]


def check_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Open one corpus file through today's readers and compare the dumps.

    Returns today's dump for the variant tests to look into.
    """
    h5 = CORPUS / entry["files"]["h5"]["name"]
    era = _era_dump(entry)
    with h5py.File(h5, "r") as f:
        has_bridge = "opensees" in f
    today: dict[str, Any] = {
        "fem": dump_fem(FEMData.from_h5(str(h5))),
        "meta": dump_stamps(str(h5)),
    }
    if has_bridge:
        model = OpenSeesModel.from_h5(str(h5))
        today["model"] = dump_model(model)
        want = FRAME2D_DECLARED_NDM if entry.get("variant") == "frame2d" else DECLARED_NDM
        assert model.ndm == want, (
            f"{h5.name}: today's reader resolves ndm={model.ndm}, "
            f"the generator declared {want}"
        )
    assert era["fem"]["dump_format"] == DUMP_FORMAT, (
        f"{h5.name}: the era dump is format {era['fem']['dump_format']}, the "
        f"oracle is {DUMP_FORMAT}; rebuild the corpus with scripts/build_schema_corpus.py"
    )
    assert ("model" in era) == has_bridge
    diffs = _diff({k: era[k] for k in ("fem", "model", "meta") if k in era}, today, "",
                  _excluded(entry))
    assert not diffs, f"{h5.name}: today's reader departs from its era:\n" + "\n".join(diffs)
    return today


# ---------------------------------------------------------------------------
# INV 6 — one entry per minor from floor to current; below the floor, evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zone", sorted(ZONES))
def test_manifest_floors_match_the_constants(zone: str) -> None:
    """The corpus was built against the floors the readers hold today."""
    assert MANIFEST["floors"][zone] == str(reader_floor(ZONES[zone]))
    assert MANIFEST["current"][zone] == str(reader_version(ZONES[zone]))


@pytest.mark.parametrize("zone", sorted(ZONES))
def test_corpus_covers_every_minor_from_floor_to_current(zone: str) -> None:
    floor, current = reader_floor(ZONES[zone]), reader_version(ZONES[zone])
    have = {e["minor"]: e for e in _plain() if e["zone"] == zone}
    want = [f"{floor.major}.{m}" for m in range(floor.minor, current.minor + 1)]
    in_range = sorted((m for m in have if _minor_of(m) >= floor.minor), key=_minor_of)
    assert in_range == want, (
        f"{zone}: the manifest must hold exactly the minors {want[0]}..{want[-1]}; "
        "rerun scripts/build_schema_corpus.py after a bump or a floor change"
    )
    # The current minor is never a gap of any kind: a bump PR adds the
    # outgoing minor's file (ADR 0113 D8), and the builder refuses to plan a
    # current minor that no base commit stamps (check_current_is_written).
    assert have[want[-1]]["status"] == "ok", (
        f"{zone} {want[-1]}: the current minor has no file ({have[want[-1]]}); "
        "commit the bump and rebuild with --base HEAD"
    )
    for minor, e in have.items():
        if _minor_of(minor) < floor.minor:
            # Kept as the evidence that raised the floor (ADR 0113 D3): a
            # real file no reader opens, never a file-less placeholder.
            assert e["status"] == "gap" and e.get("gap_kind") == "below_floor", (
                f"{zone} {minor}: below the floor {floor}, so it must be a "
                f"below_floor gap, not {e['status']!r}/{e.get('gap_kind')!r}"
            )
            assert e.get("sha") and e.get("files"), f"{zone} {minor}: evidence without a file"
            continue
        if e["status"] == "ok":
            assert e.get("sha") and e.get("files"), f"{zone} {minor}: ok without a file"
            assert _v(e["stamps"][zone])[:2] == _v(f"{minor}.0")[:2]
        else:
            assert e["status"] == "gap" and e.get("reason"), f"{zone} {minor}: {e}"
            assert e.get("gap_kind") in {"unwritten", "unrunnable", "below_floor"}


def test_current_minor_files_match_the_test_fixture_constants() -> None:
    assert MANIFEST["current"]["neutral"] == NEUTRAL_CURRENT
    assert MANIFEST["current"]["opensees"] == OPENSEES_CURRENT
    for zone in ZONES:
        assert any(e["zone"] == zone for e in _openable()), f"no openable {zone} file"


def test_variants_are_the_shim_ledger() -> None:
    """One variant per ledger row, each at an era below its row's version."""
    assert sorted(e["variant"] for e in ENTRIES if "variant" in e) == ["frame2d", "sp_cases"]
    sp, frame = _variant("sp_cases"), _variant("frame2d")
    assert sp["zone"] == "neutral" and sp["status"] == "ok"
    assert _v(sp["stamps"]["neutral"]) < SP_PER_CASE_FROM
    assert frame["zone"] == "opensees" and frame["status"] == "ok"
    assert _v(frame["stamps"]["neutral"]) < META_NDM_IS_SPATIAL_FROM
    variants = _builder().VARIANTS
    for e in (sp, frame):
        zone, minor, why = variants[e["variant"]]
        assert (e["zone"], e["minor"]) == (zone, minor)
        assert e["generator"].endswith(f"--variant {e['variant']}")
        assert e["why"] == why, f"{e['variant']}: the manifest's why text lags the builder's"


def test_builder_refuses_a_head_that_does_not_stamp_the_current_minor() -> None:
    """A bump PR run without ``--base HEAD`` (or with the bump uncommitted)
    cannot record its own minor as an ``unwritten`` gap (which INV-6
    accepts) and land without the file."""
    builder = _builder()
    head = "b" * 40
    builder.check_head_writes_current("opensees", (2, 22), head, "2.22.0")
    builder.check_head_writes_current("opensees", (2, 22), head, "2.22.3")
    with pytest.raises(RuntimeError, match="2.23.x but bbbbbbbbbb stamps 2.22.0"):
        builder.check_head_writes_current("opensees", (2, 23), head, "2.22.0")
    with pytest.raises(RuntimeError, match="--base HEAD"):
        builder.check_head_writes_current("opensees", (2, 23), head, "2.22.0")


# ---------------------------------------------------------------------------
# #1365 — every non-current era SHA is on main. The builder takes the history
# and every non-current era from the merge-base with origin/main; only the
# current minor is written by --base (HEAD in a bump PR), so a squash merge
# can orphan nothing the manifest keeps.
# ---------------------------------------------------------------------------

REPO = BUILDER.parents[1]

#: The two writer files the builder reads (``build_schema_corpus.ZONES``),
#: with a one-line body that its ``_const`` regex finds.
_WRITERS = {
    "neutral": ("src/apeGmsh/mesh/_femdata_h5_io.py", "NEUTRAL_SCHEMA"),
    "opensees": ("src/apeGmsh/opensees/emitter/h5.py", "SCHEMA"),
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _is_ancestor(root: Path, sha: str, of: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", sha, of],
        capture_output=True,
    ).returncode == 0


def _is_current(entry: dict[str, Any]) -> bool:
    return _minor_of(entry["minor"]) == _v(MANIFEST["current"][entry["zone"]])[1]


def test_manifest_non_current_era_shas_are_on_main() -> None:
    """Every non-current era's commit (and the bump commits it records) is an
    ancestor of ``origin/main``: it was resolved on main's first-parent line,
    not on a PR branch a squash merge orphans. The current minor is written
    by ``--base`` and is re-anchored by the next bump, so it is not held.

    Needs a full clone with ``origin/main``: a shallow CI checkout skips, and
    says so, rather than passing on nothing.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH: cannot check origin/main ancestry")
    if subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "-q",
                       "refs/remotes/origin/main"], capture_output=True).returncode != 0:
        pytest.skip("origin/main is absent (a shallow CI clone): MANIFEST era "
                    "SHAs cannot be checked for main ancestry here; run this test "
                    "locally in a full clone")
    if _git(REPO, "rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow repository: main's history is cut, so MANIFEST era "
                    "SHAs cannot be checked for ancestry; run locally in a full clone")
    checked, off_main = [], []
    for e in ENTRIES:
        if "sha" not in e or _is_current(e):
            continue
        for sha in (e["sha"], *e.get("bump_commits", [])):
            checked.append(sha)
            if not _is_ancestor(REPO, sha, "origin/main"):
                off_main.append(f"{_id(e)}: {sha}")
    assert checked, "the manifest holds no non-current era with a commit"
    assert not off_main, (
        "non-current era SHAs that are not on origin/main (a squash merge orphaned "
        "a PR-branch SHA, #1365); rebuild those eras on main with "
        "scripts/build_schema_corpus.py --zone <Z> --minor <M>:\n" + "\n".join(off_main)
    )


def _write_writer(repo: Path, zone: str, version: str, *, trailer: str = "") -> None:
    path, prefix = _WRITERS[zone]
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(
        f'{prefix}_FLOOR: str = "2.1.0"\n{prefix}_VERSION: str = "{version}"\n{trailer}',
        encoding="utf-8",
    )


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_builder_anchors_non_current_eras_on_the_merge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A multi-commit bump PR on top of main, the shape #1365 reproduced.

    main: A stamps 2.1, B stamps 2.2, C touches nothing of the writer. The
    branch: D1 edits the writer without a bump, D2 stamps 2.3, D3 edits it
    again. Under ``--base HEAD`` the 2.2 era must be C (main's last 2.2
    writer, the merge-base), never D1 (the bump's first parent on the
    branch, which the pre-fix builder recorded); 2.1 stays A; and only
    the current 2.3 is written by HEAD, D3.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write_writer(repo, "neutral", "2.1.0")
    _write_writer(repo, "opensees", "2.1.0")
    a = _commit(repo, "opensees 2.1.0")
    _write_writer(repo, "opensees", "2.2.0")
    b = _commit(repo, "opensees 2.2.0")
    (repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    c = _commit(repo, "unrelated change on main")
    _git(repo, "update-ref", "refs/remotes/origin/main", c)
    _git(repo, "checkout", "-q", "-b", "bump-pr")
    _write_writer(repo, "opensees", "2.2.0", trailer="# branch-only writer tweak\n")
    d1 = _commit(repo, "writer tweak on the branch")
    _write_writer(repo, "opensees", "2.3.0", trailer="# branch-only writer tweak\n")
    d2 = _commit(repo, "opensees 2.3.0")
    _write_writer(repo, "opensees", "2.3.0", trailer="# two branch-only tweaks\n")
    d3 = _commit(repo, "second tweak on the branch")

    builder = _builder()
    monkeypatch.setattr(builder, "ROOT", repo)
    monkeypatch.setattr(builder, "MANIFEST", repo / "MANIFEST.json")  # absent
    monkeypatch.setattr(builder, "VARIANTS", {})  # the real variants name eras this repo lacks

    # The CLI the bridge guide names, run from the PR head.
    assert builder.main(["--list", "--zone", "opensees", "--base", "HEAD"]) == 0
    rows = {ln.split()[1]: ln.split()[2] for ln in capsys.readouterr().out.splitlines()}
    assert rows == {"2.1": a[:10], "2.2": c[:10], "2.3": d3[:10]}, (
        f"--list resolved {rows}; the pre-fix builder gave 2.2 the branch commit "
        f"{d1[:10]} (the bump's first parent), which a squash merge orphans"
    )

    # The plan in full: every non-current era on origin/main, the current one at HEAD.
    eras = {builder._mstr(e.minor): e for e in builder.plan("opensees", c, d3)}
    assert (eras["2.1"].sha, eras["2.1"].version, eras["2.1"].bumps) == (a, "2.1.0", [a])
    assert (eras["2.2"].sha, eras["2.2"].version, eras["2.2"].bumps) == (c, "2.2.0", [b])
    assert (eras["2.3"].sha, eras["2.3"].version, eras["2.3"].bumps) == (d3, "2.3.0", [d2])
    for minor, era in eras.items():
        if minor != "2.3":
            assert _is_ancestor(repo, era.sha, "origin/main"), f"{minor}: {era.sha} is off main"
            assert all(_is_ancestor(repo, s, "origin/main") for s in era.bumps)
    assert not _is_ancestor(repo, eras["2.3"].sha, "origin/main")

    # Without --base HEAD the merge-base writes 2.2, not the tree's 2.3: refused.
    with pytest.raises(RuntimeError, match=r"2\.3\.x but .* stamps 2\.2\.0.*--base HEAD"):
        builder.main(["--list", "--zone", "opensees"])

    # On main itself (HEAD is the merge-base, tree at 2.2) the plan is self-contained.
    _git(repo, "checkout", "-q", "main")
    assert builder.main(["--list", "--zone", "opensees"]) == 0
    rows = {ln.split()[1]: ln.split()[2] for ln in capsys.readouterr().out.splitlines()}
    assert rows == {"2.1": a[:10], "2.2": c[:10]}


# ---------------------------------------------------------------------------
# INV 5 + INV 7 — every ok file opens, matches its era, bytes unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _openable(), ids=_id)
def test_corpus_file_opens_and_matches_its_era(entry: dict[str, Any]) -> None:
    h5 = CORPUS / entry["files"]["h5"]["name"]
    before = _sha(h5)
    assert before == entry["files"]["h5"]["sha256"], (
        f"{h5.name}: committed bytes differ from the manifest; corpus files are "
        "frozen, rebuild with scripts/build_schema_corpus.py"
    )
    check_entry(entry)
    assert _sha(h5) == before, f"{h5.name}: opening the file changed its bytes"


# ---------------------------------------------------------------------------
# The ledger variants read as the ledger says (ADR 0113 D4)
# ---------------------------------------------------------------------------


def test_sp_cases_before_2_26_1_read_as_one_default_case() -> None:
    """Two authored ``g.displacements.case`` names, one ``default`` on read.

    The era's writer flattened every SP record into ``/loads/sp/default``;
    the records all survive, their case names do not, and neither reader
    can tell the cases apart (ledger row "SP loads before 2.26.1", Q5).
    """
    entry = _variant("sp_cases")
    h5 = CORPUS / entry["files"]["h5"]["name"]
    era = _era_dump(entry)
    assert era["generator_notes"]["sp_cases"] == SP_CASES
    with h5py.File(h5, "r") as f:
        # The era's layout: every record under the one ``default`` dataset.
        assert list(f["loads/sp"].keys()) == ["default"]
    today = check_entry(entry)
    fem = FEMData.from_h5(str(h5))
    n_base_nodes = len(fem.nodes.physical.node_ids("Base", dim=2))
    n_dofs = 3
    assert n_base_nodes == 9
    assert today["fem"]["loads"]["sp"] == n_base_nodes * n_dofs * len(SP_CASES)
    assert today["fem"]["loads"]["sp_patterns"] == ["default"]
    assert era["fem"]["loads"]["sp_patterns"] == ["default"]
    assert sorted({r.pattern for r in fem.nodes.sp}) == ["default"]
    # The values of both cases survive the flattening, record by record;
    # only the case binding is lost.
    got = Counter((int(r.dof), float(r.value)) for r in fem.nodes.sp)
    assert dict(got) == SP_RECORDS
    assert sum(SP_RECORDS.values()) == n_base_nodes * n_dofs * len(SP_CASES)
    assert all(int(r.node_id) in set(fem.nodes.physical.node_ids("Base", dim=2))
               for r in fem.nodes.sp)
    # The plain box of the same era carries no SP load: the records are the
    # variant's, not an artefact of the era's reader.
    plain = next(e for e in _plain() if e["zone"] == "neutral" and e["minor"] == entry["minor"])
    assert _era_dump(plain)["fem"]["loads"]["sp"] == 0


def test_frame2d_before_2_34_0_takes_the_ndm_shim() -> None:
    """A 2-D frame written before neutral 2.34.0 stamps the mesh dimension;
    ``read_spatial_ndm`` is the branch that reads it (ledger row
    ``META_NDM_IS_SPATIAL_FROM``, #1300). The file's own 2-D evidence is
    recorded here; what the reader makes of it is the next test."""
    entry = _variant("frame2d")
    h5 = CORPUS / entry["files"]["h5"]["name"]
    era = _era_dump(entry)
    assert era["generator_notes"]["declared_ndm"] == FRAME2D_DECLARED_NDM
    assert era["model"]["ndf"] == 3
    # The writer's own stamp is the mesh dimension of a line mesh, not 2,
    # and the era's own reader read that stamp back.
    assert era["meta"] == {"ndm": 1, "ndf": 3} and dump_stamps(str(h5))["ndm"] == 1
    assert era["model"]["ndm"] == 1
    # The 2-D evidence the file carries: a zero-width vecxz exists only in
    # 2-D (the plain 3-D frame of the same era writes (1, 3)).
    with h5py.File(h5, "r") as f:
        shapes = {f[f"opensees/transforms/{t}/per_element_vecxz"].shape
                  for t in f["opensees/transforms"]}
    assert shapes == {(1, 0)}
    # The salvage is keyed on the neutral stamp: the same bytes at or above
    # META_NDM_IS_SPATIAL_FROM would be read as-is, so the file sits below it.
    assert _v(entry["stamps"]["neutral"]) < META_NDM_IS_SPATIAL_FROM
    # Era equality outside the ledger (model.ndm is excluded below 2.34.0).
    today = check_entry(entry)
    assert today["model"]["ndf"] == 3
    # The plain frame of the same era is lifted to 3 by its 3-wide vecxz.
    plain = next(e for e in _plain() if e["zone"] == "opensees" and e["minor"] == entry["minor"])
    assert _era_dump(plain)["meta"]["ndm"] == 1
    plain_h5 = CORPUS / plain["files"]["h5"]["name"]
    with h5py.File(plain_h5, "r") as f:
        assert {f[f"opensees/transforms/{t}/per_element_vecxz"].shape
                for t in f["opensees/transforms"]} == {(1, 3)}
    assert OpenSeesModel.from_h5(str(plain_h5)).ndm == DECLARED_NDM


def test_frame2d_reads_its_declared_ndm() -> None:
    """The right answer for the ``frame2d`` file is the declared 2 (#1358),
    and the rebuilt deck carries every node's (x, y) from the file."""
    entry = _variant("frame2d")
    h5 = CORPUS / entry["files"]["h5"]["name"]
    model = OpenSeesModel.from_h5(str(h5))
    assert model.ndm == FRAME2D_DECLARED_NDM
    deck = model.build("tcl").splitlines()
    assert "model BasicBuilder -ndm 2 -ndf 3" in deck
    with h5py.File(h5, "r") as f:
        want = {
            int(n): (float(c[0]), float(c[1]))
            for n, c in zip(f["nodes/ids"][()], f["nodes/coords"][()])
        }
    got = {
        int(p[1]): tuple(float(v) for v in p[2:])
        for p in (line.split() for line in deck) if p and p[0] == "node"
    }
    assert got == want


# ---------------------------------------------------------------------------
# D3 — a below-floor era keeps its real file, and the zone's reader refuses
# it naming the floor
# ---------------------------------------------------------------------------


def _below_floor_zones(entry: dict[str, Any]) -> dict[str, SchemaVersion]:
    return {
        z: SchemaVersion.parse(s) for z, s in entry["stamps"].items()
        if z in ZONES and SchemaVersion.parse(s).minor < reader_floor(ZONES[z]).minor
    }


def _open_zone(zone: str, h5: Path) -> None:
    """Open ``h5`` through the reader that validates ``zone``."""
    if zone == "neutral":
        FEMData.from_h5(str(h5))
    else:
        h5_reader.open(str(h5)).close()


@pytest.mark.parametrize(
    "entry", [e for e in _with_files() if e["status"] == "gap"], ids=_id,
)
def test_below_floor_gap_is_refused_naming_the_floor(entry: dict[str, Any]) -> None:
    assert entry["gap_kind"] == "below_floor"
    low = _below_floor_zones(entry)
    assert low, f"{_id(entry)}: recorded below the floor but no stamp is"
    h5 = CORPUS / entry["files"]["h5"]["name"]
    before = _sha(h5)
    for zone, stamp in low.items():
        floor, reader = reader_floor(ZONES[zone]), reader_version(ZONES[zone])
        with pytest.raises(SchemaVersionError) as exc:
            _open_zone(zone, h5)
        msg = str(exc.value)
        assert f"{STAMP_KEYS[zone]}={stamp}: too old" in msg, msg
        assert (
            f"supports {floor.major}.{floor.minor}.x–"
            f"{reader.major}.{reader.minor}.x" in msg
        ), msg
    with pytest.raises(SchemaVersionError, match="too old"):
        check_entry(entry)
    assert _sha(h5) == before


def test_opensees_2_11_is_the_evidence_below_the_floor() -> None:
    """The era the floor rose past (#1329, decision 2026-10-04): its file is
    kept, and it is below the floor of both zones."""
    floor = reader_floor(OPENSEES)
    below = [e for e in _plain() if e["zone"] == "opensees" and _minor_of(e["minor"]) < floor.minor]
    assert [e["minor"] for e in below] == ["2.11"]
    (entry,) = below
    assert entry["gap_kind"] == "below_floor" and (CORPUS / entry["files"]["h5"]["name"]).is_file()
    assert set(_below_floor_zones(entry)) == {"neutral", "opensees"}
    assert SchemaVersion.parse(entry["stamps"]["neutral"]).minor < reader_floor(NEUTRAL).minor


# ---------------------------------------------------------------------------
# INV 10 — a tampered copy fails the snapshot_id check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _openable(), ids=_id)
def test_tampered_copy_fails_snapshot_check(entry: dict[str, Any], tmp_path: Path) -> None:
    src = CORPUS / entry["files"]["h5"]["name"]
    copy = tmp_path / src.name
    shutil.copyfile(src, copy)
    with h5py.File(copy, "r+") as f:
        coords = np.asarray(f["nodes/coords"][...])
        coords[-1, 0] += 0.25
        f["nodes/coords"][...] = coords
    with pytest.raises(MalformedH5Error, match="snapshot_id mismatch"):
        FEMData.from_h5(str(copy))


# ---------------------------------------------------------------------------
# D3 — the floor constants are proven by the corpus
# ---------------------------------------------------------------------------


def _unproven_minors(zone: str) -> list[str]:
    floor = reader_floor(ZONES[zone])
    return [
        e["minor"] for e in _plain()
        if e["zone"] == zone and e["status"] == "gap"
        and e["gap_kind"] != "unwritten"
        and _minor_of(e["minor"]) >= floor.minor
    ]


@pytest.mark.parametrize("zone", sorted(ZONES))
def test_floor_is_proven_by_the_corpus(zone: str) -> None:
    """A floor stands only where no writable era at or above it is a gap.

    An ``unwritten`` minor (no main commit ever stamped it) holds no files in
    the wild and does not count against the floor.
    """
    gaps = _unproven_minors(zone)
    assert not gaps, (
        f"{zone} floor {reader_floor(ZONES[zone])} is not proven: gaps at {gaps}. "
        "ADR 0113 D3: the floor must rise past them (a maintainer decision)."
    )


@pytest.mark.parametrize("zone", sorted(ZONES))
def test_floor_sits_just_above_its_evidence(zone: str) -> None:
    """Where below-floor eras exist, the floor is the first era after them
    (ADR 0113 D3: the floor rises *past* the gap, no further), and that
    era's own file opens."""
    floor = reader_floor(ZONES[zone])
    plain = {e["minor"]: e for e in _plain() if e["zone"] == zone}
    below = [_minor_of(m) for m in plain if _minor_of(m) < floor.minor]
    if below:
        assert max(below) + 1 == floor.minor
    assert plain[f"{floor.major}.{floor.minor}"]["status"] == "ok"


# ---------------------------------------------------------------------------
# INV 12 — a paper floor turns the corpus check red
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zone,module,const", [
    ("neutral", _femdata_h5_io, "NEUTRAL_SCHEMA_FLOOR"),
    ("opensees", _h5_writer, "SCHEMA_FLOOR"),
])
def test_paper_floor_fails_the_corpus(
    zone: str, module: Any, const: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = reader_version(ZONES[zone])
    monkeypatch.setattr(module, const, str(current))
    assert reader_floor(ZONES[zone]) == current
    refused = []
    for entry in _openable():
        if entry["zone"] != zone or _v(entry["stamps"][zone])[1] == current.minor:
            continue
        with pytest.raises(SchemaVersionError, match="supports"):
            check_entry(entry)
        refused.append(_id(entry))
    assert refused, f"a paper {zone} floor at {current} refused no corpus file"
