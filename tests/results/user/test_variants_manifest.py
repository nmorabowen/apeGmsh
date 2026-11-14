"""Reading and writing ``<study>.variants.json`` (ADR 0132 D8; h5-schema section).

Every example under ``architecture/schemas/examples/`` reads with the verdict its
name states, and the JSON Schema (where ``jsonschema`` is installed) agrees with
Python on the form of every example and of a list of crafted paths: two engines,
one answer. The rules the schema cannot express (NFC, unique labels and paths,
confinement after links resolve, the file exists) are tested on their own.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path

import pytest

from apeGmsh.results.user import (
    VariantDef,
    VariantsRefused,
    VariantsWriteRefused,
    _variants,
    read_variants,
    write_variants,
)

from tests.results.user._user_helpers import REPO

SCHEMAS = REPO / "architecture" / "schemas"
EXAMPLES = SCHEMAS / "examples"
FILES = ("slab_uncracked.h5", "slab_cracked.h5", "bischoff/slab_pass2.h5")

#: What each example's name states: a file refusal, or the one row that refuses.
EXPECTED = {
    "variants.valid.json": None,
    "variants.newer-minor.json": ("file", "newer"),
    "invalid/variants.param-null.json": (1, "bad-param"),
    "invalid/variants.param-overflow.json": (0, "bad-param"),
    "invalid/variants.path-absolute.json": (0, "bad-path"),
    "invalid/variants.path-empty-stem.json": (0, "bad-path"),
    "invalid/variants.path-empty-stem-subfolder.json": (2, "bad-path"),
    "invalid/variants.path-folder-trailing-dot.json": (2, "bad-path"),
    "invalid/variants.path-folder-trailing-space.json": (2, "bad-path"),
    "invalid/variants.path-leaves-folder.json": (2, "bad-path"),
    "invalid/variants.path-not-model.json": (1, "bad-path"),
    "invalid/variants.path-particles-sibling.json": (1, "bad-path"),
    "invalid/variants.path-user-sibling-uppercase.json": (1, "bad-path"),
    "invalid/variants.path-workspace-sibling.json": (1, "bad-path"),
    "invalid/variants.version-number.json": ("file", "bad-version"),
    "invalid/variants.wrong-list-key.json": ("file", "bad-envelope"),
}


def _study(folder: Path, manifest: "Path | dict", name: str = "study.variants.json") -> Path:
    """The manifest copied into ``folder`` beside the three model files it names."""
    for rel in FILES:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"model")
    target = folder / name
    if isinstance(manifest, Path):
        shutil.copy(manifest, target)
    else:
        target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def _doc(**rows) -> dict:
    return {"format": "apegmsh-variants", "version": "1.0.0", "variants": list(rows.values())}


def test_every_example_is_named_in_the_expectations() -> None:
    found = {p.relative_to(EXAMPLES).as_posix() for p in EXAMPLES.rglob("*.json")}
    assert found == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_example_reads_with_the_verdict_its_name_states(tmp_path: Path, name: str) -> None:
    target = _study(tmp_path, EXAMPLES / name)
    expected = EXPECTED[name]
    if expected is not None and expected[0] == "file":
        with pytest.raises(VariantsRefused) as err:
            read_variants(target)
        assert err.value.reason == expected[1]
        return
    manifest = read_variants(target)
    if expected is None:
        assert not manifest.refused
        assert [v.label for v in manifest.variants] == [
            "Uncracked", "Cracked", "Bischoff D+SD+0.25L"]
        assert manifest.variants[2].resolved == (tmp_path / FILES[2]).resolve()
        assert manifest.variants[2].params == {
            "state": "cracked", "bischoff": "D+SD+0.25L", "pass": 2, "converged": False}
        return
    row, reason = expected
    assert [(r.row, r.reason) for r in manifest.refused] == [(row, reason)]
    assert len(manifest.variants) == 2, "a bad row refuses alone; the manifest still opens"


def _schema_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / "variants.schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema), schema


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_json_schema_agrees_with_python_on_every_example(tmp_path: Path, name: str) -> None:
    validator, _ = _schema_validator()
    doc = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    schema_ok = validator.is_valid(doc)
    python_ok = EXPECTED[name] is None
    assert schema_ok == python_ok


PATHS = [
    "slab.h5", "a/b/slab.H5", "x.y.h5", "..h5", "user.h5", "slab.user.h5x.h5",
    ".h5", "x/.h5", "/slab.h5", "C:/slab.h5", "a\\b.h5", "a//b.h5", "./a.h5", "a/../b.h5",
    "a./b.h5", "a /b.h5", "slab.ladruno", "slab.part-3.ladruno", "slab.USER.h5",
    "slab.Workspace.H5", "slab.geometry.h5", "slab.results.h5", "slab.h5/", "a/b",
    "tab\there.h5", "", "slab.h5 ", " slab.h5", "caf\u00e9.h5",
]


@pytest.mark.parametrize("path", PATHS)
def test_the_path_form_agrees_with_the_schema(path: str) -> None:
    validator, schema = _schema_validator()
    sub = {"$schema": schema["$schema"], **schema["$defs"]["path"]}
    import jsonschema

    assert jsonschema.Draft202012Validator(sub).is_valid(path) == (
        _variants.path_problem(path) is None), _variants.path_problem(path)


def test_a_label_or_param_key_not_nfc_refuses_empty_name(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "Fisur\u00e9")
    assert decomposed != "Fisur\u00e9"
    target = _study(tmp_path, _doc(
        a={"label": decomposed, "path": FILES[0], "params": {}},
        b={"label": "B", "path": FILES[1], "params": {decomposed: 1}},
        c={"label": "C", "path": FILES[2], "params": {"ok": "Fisur\u00e9"}}))
    manifest = read_variants(target)
    assert [(r.row, r.reason) for r in manifest.refused] == [(0, "empty-name"), (1, "empty-name")]
    assert [v.label for v in manifest.variants] == ["C"]


def test_duplicate_labels_refuse_both_rows(tmp_path: Path) -> None:
    target = _study(tmp_path, _doc(
        a={"label": "A", "path": FILES[0], "params": {}},
        b={"label": "A", "path": FILES[1], "params": {}},
        c={"label": "C", "path": FILES[2], "params": {}}))
    assert [(r.row, r.reason) for r in read_variants(target).refused] == [
        (0, "duplicate"), (1, "duplicate")]


def test_duplicate_paths_refuse_both_rows(tmp_path: Path) -> None:
    other = "bischoff/../slab_uncracked.h5"  # refused by form: '..'
    target = _study(tmp_path, _doc(
        a={"label": "A", "path": FILES[0], "params": {}},
        b={"label": "B", "path": FILES[0], "params": {"pass": 2}},
        c={"label": "C", "path": other, "params": {}}))
    got = [(r.row, r.reason) for r in read_variants(target).refused]
    assert got == [(0, "duplicate-path"), (1, "duplicate-path"), (2, "bad-path")]


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="case-folded only on Windows/macOS")
def test_duplicate_paths_compare_case_folded(tmp_path: Path) -> None:
    target = _study(tmp_path, _doc(
        a={"label": "A", "path": "slab_cracked.h5", "params": {}},
        b={"label": "B", "path": "SLAB_CRACKED.h5", "params": {}}))
    assert [r.reason for r in read_variants(target).refused] == ["duplicate-path"] * 2


def test_a_missing_file_refuses_missing(tmp_path: Path) -> None:
    target = _study(tmp_path, _doc(
        a={"label": "A", "path": "not_run_yet.h5", "params": {}},
        b={"label": "B", "path": FILES[0], "params": {}}))
    (tmp_path / "folder.h5").mkdir()
    assert [(r.row, r.reason) for r in read_variants(target).refused] == [(0, "missing")]
    target.write_text(json.dumps(_doc(a={"label": "A", "path": "folder.h5", "params": {}})))
    assert [r.reason for r in read_variants(target).refused] == ["missing"]


def _link_folder(link: Path, to: Path) -> None:
    """A junction on Windows (no privilege needed), a symlink elsewhere."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(to), str(link))
    else:
        os.symlink(to, link, target_is_directory=True)


def test_a_path_through_a_link_that_leaves_the_folder_refuses_escapes(tmp_path: Path) -> None:
    study, outside = tmp_path / "study", tmp_path / "elsewhere"
    study.mkdir()
    outside.mkdir()
    (outside / "slab_far.h5").write_bytes(b"model")
    inside = study / "inside"
    inside.mkdir()
    (inside / "slab_near.h5").write_bytes(b"model")
    _link_folder(study / "far", outside)
    _link_folder(study / "near", inside)
    try:
        target = _study(study, _doc(
            a={"label": "A", "path": "far/slab_far.h5", "params": {}},
            b={"label": "B", "path": "near/slab_near.h5", "params": {}}))
        manifest = read_variants(target)
        assert [(r.row, r.reason) for r in manifest.refused] == [(0, "escapes")]
        assert manifest.by_label("B").resolved == (inside / "slab_near.h5").resolve()
    finally:
        for link in (study / "far", study / "near"):
            os.unlink(link) if os.name != "nt" else os.rmdir(link)  # the link, never its target
    assert (outside / "slab_far.h5").exists()


@pytest.mark.parametrize("text, reason", [
    ('{"format": "apegmsh-variants", "version": "1.0.0", "variants": [{"label": "A", '
     '"path": "slab_cracked.h5", "params": {"x": NaN}}]}', "bad-json"),
    ('{"format": "apegmsh-variants", "format": "apegmsh-variants", "version": "1.0.0", '
     '"variants": []}', "bad-json"),
    ('[]', "not-a-manifest"),
    ('{"format": "apegmsh-sets", "version": "1.0.0", "variants": []}', "not-a-manifest"),
    ('{"format": "apegmsh-variants", "version": "1.0.0", "variants": []}', "bad-envelope"),
    ('{"format": "apegmsh-variants", "version": "2.0.0", "variants": []}', "major"),
    ('{"format": "apegmsh-variants", "version": "1.0", "variants": []}', "bad-version"),
])
def test_file_refusals(tmp_path: Path, text: str, reason: str) -> None:
    target = tmp_path / "study.variants.json"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(VariantsRefused) as err:
        read_variants(target)
    assert err.value.reason == reason


def test_a_newer_minor_refuses_before_any_validation(tmp_path: Path) -> None:
    """``1.1.0`` with an unknown top key and a broken row: ``newer``, nothing else."""
    target = tmp_path / "study.variants.json"
    target.write_text(json.dumps({"format": "apegmsh-variants", "version": "1.1.0",
                                  "variants": "not even a list", "extra": 1}))
    with pytest.raises(VariantsRefused) as err:
        read_variants(target)
    assert err.value.reason == "newer"


def test_a_byte_order_mark_is_stripped(tmp_path: Path) -> None:
    target = _study(tmp_path, EXAMPLES / "variants.valid.json")
    target.write_bytes(b"\xef\xbb\xbf" + target.read_bytes())
    assert len(read_variants(target).variants) == 3


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def test_a_written_manifest_reads_back_and_carries_no_bom(tmp_path: Path) -> None:
    _study(tmp_path, EXAMPLES / "variants.valid.json")
    rows = [VariantDef("Uncracked", FILES[0], {"state": "uncracked", "Ie/Ig": 1.0}),
            VariantDef("Bischoff", FILES[2], {"pass": 2, "converged": False})]
    target = write_variants(tmp_path / "fresh.variants.json", rows)
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    doc = json.loads(raw)
    assert (doc["format"], doc["version"]) == ("apegmsh-variants", "1.0.0")
    manifest = read_variants(target)
    assert [(v.label, v.path, dict(v.params)) for v in manifest.variants] == [
        (r.label, r.path, dict(r.params)) for r in rows]
    assert not list(tmp_path.glob("*.partial"))
    validator, _ = _schema_validator()
    assert validator.is_valid(doc)


@pytest.mark.parametrize("row, reason", [
    (VariantDef("A", "../slab.h5"), "bad-path"),
    (VariantDef("A", "slab.User.h5"), "bad-path"),
    (VariantDef(" A", FILES[0]), "empty-name"),
    (VariantDef("A", FILES[0], {"x": None}), "bad-param"),
    (VariantDef("A", FILES[0], {"x": float("nan")}), "bad-param"),
])
def test_the_writer_never_writes_a_row_that_would_refuse(tmp_path: Path, row, reason) -> None:
    _study(tmp_path, EXAMPLES / "variants.valid.json")
    target = tmp_path / "new.variants.json"
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(target, [row])
    assert err.value.reason == reason
    assert not target.exists() and not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("version, reason", [("1.1.0", "newer"), ("2.0.0", "major")])
def test_the_writer_never_writes_over_a_newer_manifest(tmp_path: Path, version, reason) -> None:
    target = _study(tmp_path, EXAMPLES / "variants.newer-minor.json")
    doc = json.loads(target.read_text(encoding="utf-8"))
    doc["version"] = version
    target.write_text(json.dumps(doc), encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(target, [VariantDef("A", FILES[0])])
    assert err.value.reason == reason
    assert target.read_bytes() == before


def test_the_writer_never_overwrites_a_manifest_it_cannot_read(tmp_path: Path) -> None:
    target = _study(tmp_path, EXAMPLES / "variants.valid.json")
    target.write_text('{"format": "apegmsh-variants", "version": 1', encoding="utf-8")
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(target, [VariantDef("A", FILES[0])])
    assert err.value.reason == "unreadable"


def test_a_newer_manifest_landing_during_the_write_is_never_overwritten(
    tmp_path: Path, monkeypatch,
) -> None:
    """The bytes are read once: a 1.1.0 file that lands after that read refuses the write."""
    target = _study(tmp_path, EXAMPLES / "variants.valid.json")
    newer = (EXAMPLES / "variants.newer-minor.json").read_bytes()
    real = _variants._fsync

    def lands_meanwhile(path: Path) -> None:
        real(path)
        target.write_bytes(newer)

    monkeypatch.setattr(_variants, "_fsync", lands_meanwhile)
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(target, [VariantDef("A", "slab_new.h5")])
    assert err.value.reason == "revision-moved"
    assert target.read_bytes() == newer
    assert not list(tmp_path.glob("*.partial"))


def test_a_read_and_write_back_is_identical_missing_row_included(tmp_path: Path) -> None:
    """Existence is the reader's rule: the writer lists a variant not yet run, and
    writing the standing rows back carries the ``missing`` one through unchanged."""
    _study(tmp_path, EXAMPLES / "variants.valid.json")
    target = write_variants(tmp_path / "fresh.variants.json", [
        VariantDef("Uncracked", FILES[0], {"state": "uncracked", "Ie/Ig": 1.0}),
        VariantDef("Pass 3", "bischoff/slab_pass3.h5", {"pass": 3}),
        VariantDef("Bischoff", FILES[2], {"pass": 2, "converged": False})])
    before = target.read_bytes()
    manifest = read_variants(target)
    assert [(r.row, r.name, r.reason) for r in manifest.refused] == [(1, "Pass 3", "missing")]
    write_variants(target, [VariantDef(v.label, v.path, v.params) for v in manifest.variants])
    assert target.read_bytes() == before


def test_refused_rows_are_carried_verbatim(tmp_path: Path) -> None:
    target = _study(tmp_path, _doc())
    target.write_text(
        '{"format": "apegmsh-variants", "version": "1.0.0", "variants": [\n'
        ' {"label": "Huge", "path": "slab_cracked.h5", "params": {"Ie/Ig": 1e400, "n": 7}},\n'
        ' {"label": "Twin", "path": "slab_uncracked.h5", "params": {}},\n'
        ' {"label": "Twin", "path": "bischoff/slab_pass2.h5", "params": {}},\n'
        ' {"label": "Colour", "path": "slab_cracked.h5", "params": {}, "colour": "red"}]}\n',
        encoding="utf-8")
    old = json.loads(target.read_text(encoding="utf-8").replace("1e400", '"1e400"'))
    write_variants(target, [VariantDef("New", "slab_new.h5")])
    text = target.read_text(encoding="utf-8")
    assert '"Ie/Ig": 1e400' in text, "the overflow token was not carried as written"
    rows = json.loads(text.replace("1e400", '"1e400"'))["variants"]
    assert rows[:4] == old["variants"] and rows[4]["label"] == "New"
    assert [(r.row, r.reason) for r in read_variants(target).refused] == [
        (0, "bad-param"), (1, "duplicate"), (2, "duplicate"), (3, "bad-param"),
        (4, "missing")]
    for label in ("Twin",):
        with pytest.raises(VariantsWriteRefused) as err:
            write_variants(target, [VariantDef(label, "slab_new.h5")])
        assert err.value.reason == "duplicate"
        with pytest.raises(VariantsWriteRefused) as err:
            write_variants(target, remove=[label])
        assert err.value.reason == "duplicate"


def test_the_writer_never_breaks_a_standing_row(tmp_path: Path) -> None:
    target = _study(tmp_path, EXAMPLES / "variants.valid.json")
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(target, [VariantDef("Again", FILES[0])])
    assert err.value.reason == "duplicate-path"
    write_variants(target, [VariantDef("Cracked", "slab_cracked_v2.h5")], remove=["Uncracked"])
    assert [v.label for v in read_variants(target).variants] == ["Bischoff D+SD+0.25L"]
    assert [r.reason for r in read_variants(target).refused] == ["missing"]


def _short_name(path: Path) -> str:
    import ctypes
    from ctypes import wintypes

    get = ctypes.windll.kernel32.GetShortPathNameW
    get.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    buf = ctypes.create_unicode_buffer(1024)
    if not get(str(path), buf, 1024):
        return ""
    return Path(buf.value).name


@pytest.mark.skipif(os.name != "nt", reason="8.3 short names exist on Windows only")
def test_an_8_3_short_name_of_a_sibling_file_refuses(tmp_path: Path) -> None:
    """``SLABUS~1.H5`` passes the path form but names ``slab.user.h5``: the resolved
    name is judged again, so the row refuses ``bad-path``."""
    sibling = tmp_path / "slab.user.h5"
    sibling.write_bytes(b"user")
    short = _short_name(sibling)
    if not short or short.lower() == sibling.name.lower():
        pytest.skip("this volume does not generate 8.3 short names")
    assert _variants.path_problem(short) is None, "the short name must pass the form"
    target = _study(tmp_path, _doc(a={"label": "A", "path": short, "params": {}},
                                   b={"label": "B", "path": FILES[0], "params": {}}))
    assert [(r.row, r.reason) for r in read_variants(target).refused] == [(0, "bad-path")]


def test_a_manifest_is_named_study_variants_json(tmp_path: Path) -> None:
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(tmp_path / "study.json", [VariantDef("A", "a.h5")])
    assert err.value.reason == "bad-name"


def test_a_newer_manifest_landing_between_parse_and_signature_is_never_overwritten(
    tmp_path: Path, monkeypatch,
) -> None:
    """Parsed and signed from one read: the signature is of the bytes parsed, so a
    file that lands right after the parse still refuses the write."""
    target = _study(tmp_path, EXAMPLES / "variants.valid.json")
    newer = (EXAMPLES / "variants.newer-minor.json").read_bytes()
    real = _variants._envelope
    landed: list[bool] = []

    def lands_after_parse(doc, where):
        out = real(doc, where)
        if not landed and where == target:
            landed.append(True)
            target.write_bytes(newer)
        return out

    monkeypatch.setattr(_variants, "_envelope", lands_after_parse)
    with pytest.raises(VariantsWriteRefused) as err:
        write_variants(target, [VariantDef("A", "slab_new.h5")])
    assert err.value.reason == "revision-moved"
    assert target.read_bytes() == newer
