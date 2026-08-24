"""Emitter-sync lane for the published neutral-zone note.

``docs/design/model-h5-neutral-zone.md`` publishes the ``model.h5``
neutral zone as a contract an external consumer implements against —
the apeWorkbench browser reader (h5wasm + TypeScript, no h5py, no
apeGmsh import) is the first.  A wrong dtype, rank, attribute name or
index base on that page becomes a wrong reader downstream, and the
failure is silent: the page keeps saying what the emitter used to do.

So the page is not the source of truth.
``tests/fixtures/neutral_zone/inventory.py`` is, and this module holds
it against reality from both sides:

* **emitter → inventory.**  The committed golden ``box.h5`` *and* a
  freshly emitted file must match the inventory exactly — every dataset
  a known path with the documented dtype and rank, every group's attrs
  documented, no undocumented root group.  When the emitter drifts,
  these fail.
* **inventory → page.**  Every path in the inventory must appear
  verbatim on the page, and every neutral-zone path the page mentions
  must be in the inventory.  When the page drifts, these fail.

The pinned semantics (``test_connectivity_*``, ``test_coords_*``,
``test_element_side_pg_ids_may_dangle``) are the claims a consumer will
actually code against; they are pinned here so they cannot quietly stop
being true.

Regenerate the golden with
``python tests/fixtures/neutral_zone/_generate_fixtures.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
from apeGmsh.opensees._internal.schema_version import (
    NEUTRAL,
    SchemaVersion,
    reader_version,
    validate_zone_version,
)

from tests.fixtures.neutral_zone.inventory import (
    ATTR_INT,
    ATTR_STR,
    DATASETS,
    GROUPS,
    KNOWN_UNPUBLISHED_ROOT_GROUPS,
    PUBLISHED_ROOT_GROUPS,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _REPO_ROOT / "tests" / "fixtures" / "neutral_zone" / "box.h5"
_PAGE = _REPO_ROOT / "docs" / "design" / "model-h5-neutral-zone.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pattern(path: str) -> str:
    """Collapse a concrete HDF5 path to its inventory pattern.

    Only ever ONE segment of a published path is variable: the element
    type alias under ``/elements``, or the sanitized entry name under a
    named-index side / ``/mesh_selections``.
    """
    parts = path.strip("/").split("/")
    if not parts:
        return "/"
    root = parts[0]
    if root == "elements" and len(parts) >= 2:
        parts[1] = "{type}"
    elif root in ("physical_groups", "labels") and len(parts) >= 3:
        parts[2] = "{name}"
    elif root == "mesh_selections" and len(parts) >= 2:
        parts[1] = "{name}"
    return "/" + "/".join(parts)


def _dtype_token(dset: h5py.Dataset) -> str:
    """Canonical inventory token for a dataset's on-disk type."""
    string_info = h5py.check_string_dtype(dset.dtype)
    if string_info is not None:
        if string_info.length is None:
            enc = string_info.encoding
            return f"vlen_{enc.replace('-', '')}" if enc != "utf-8" \
                else "vlen_utf8"
        return f"fixed_{string_info.length}"
    return str(dset.dtype)


def _attr_token(value: object) -> str:
    if isinstance(value, str):
        return ATTR_STR
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, (np.integer, int)):
        return ATTR_INT
    if isinstance(value, np.ndarray) and value.shape == ():
        return _attr_token(value.item())
    return type(value).__name__


def _walk(f: h5py.File) -> tuple[list[str], list[str]]:
    """Return ``(group_paths, dataset_paths)``, both absolute."""
    groups: list[str] = []
    datasets: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Group):
            groups.append("/" + name)
        else:
            datasets.append("/" + name)

    f.visititems(visit)
    return groups, datasets


def _published(path: str) -> bool:
    """True when ``path`` sits under a root group this note publishes."""
    root = path.strip("/").split("/")[0]
    return root in PUBLISHED_ROOT_GROUPS


@pytest.fixture(scope="module")
def golden() -> h5py.File:
    assert _GOLDEN.exists(), (
        f"missing golden {_GOLDEN}. Regenerate it with "
        f"`python tests/fixtures/neutral_zone/_generate_fixtures.py`."
    )
    with h5py.File(_GOLDEN, "r") as f:
        yield f


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> h5py.File:
    """A file emitted right now by the same generator as the golden."""
    gmsh = pytest.importorskip("gmsh")  # noqa: F841
    from tests.fixtures.neutral_zone._generate_fixtures import _box

    path = tmp_path_factory.mktemp("neutral_zone") / "fresh.h5"
    _box(str(path))
    with h5py.File(path, "r") as f:
        yield f


# ---------------------------------------------------------------------------
# emitter -> inventory
# ---------------------------------------------------------------------------


def test_golden_datasets_match_inventory(golden: h5py.File) -> None:
    """Every published dataset is documented, with the right dtype+rank."""
    _, datasets = _walk(golden)
    unknown: list[str] = []
    wrong: list[str] = []
    for path in datasets:
        if not _published(path):
            continue
        pat = _pattern(path)
        spec = DATASETS.get(pat)
        if spec is None:
            unknown.append(f"{path}  (pattern {pat})")
            continue
        dset = golden[path]
        token = _dtype_token(dset)
        if token != spec.dtype:
            wrong.append(f"{pat}: dtype {token} != documented {spec.dtype}")
        if len(dset.shape) != spec.rank:
            wrong.append(
                f"{pat}: rank {len(dset.shape)} != documented {spec.rank}"
            )
    assert not unknown, (
        "the emitter writes datasets the neutral-zone note does not "
        "document:\n  " + "\n  ".join(unknown)
        + "\nAdd them to tests/fixtures/neutral_zone/inventory.py AND to "
          "docs/design/model-h5-neutral-zone.md."
    )
    assert not wrong, (
        "documented dtype/rank no longer matches the emitter:\n  "
        + "\n  ".join(wrong)
    )


def test_golden_groups_match_inventory(golden: h5py.File) -> None:
    """Every published group is documented, with the right attrs."""
    groups, _ = _walk(golden)
    unknown: list[str] = []
    attr_problems: list[str] = []
    for path in groups:
        if not _published(path):
            continue
        pat = _pattern(path)
        spec = GROUPS.get(pat)
        if spec is None:
            unknown.append(f"{path}  (pattern {pat})")
            continue
        actual = dict(golden[path].attrs)
        for name, want in spec.attrs.items():
            if name not in actual:
                attr_problems.append(f"{pat}: missing documented attr {name!r}")
                continue
            got = _attr_token(actual[name])
            if got != want:
                attr_problems.append(
                    f"{pat}@{name}: type {got} != documented {want}"
                )
        documented = set(spec.attrs) | set(spec.optional_attrs)
        for name in actual:
            if name not in documented:
                attr_problems.append(
                    f"{pat}@{name}: written by the emitter, undocumented"
                )
    assert not unknown, (
        "the emitter writes groups the neutral-zone note does not "
        "document:\n  " + "\n  ".join(unknown)
    )
    assert not attr_problems, (
        "group attributes drifted from the note:\n  "
        + "\n  ".join(attr_problems)
    )


def test_required_inventory_entries_present_in_golden(
    golden: h5py.File,
) -> None:
    """The golden actually exercises everything marked required."""
    groups, datasets = _walk(golden)
    seen = {_pattern(p) for p in groups if _published(p)}
    seen |= {_pattern(p) for p in datasets if _published(p)}
    missing = [
        pat for pat, spec in list(GROUPS.items()) + list(DATASETS.items())
        if spec.required and pat not in seen
    ]
    assert not missing, (
        "the golden no longer covers required parts of the contract: "
        f"{missing}. Regenerate it, or relax `required` with a reason."
    )


def test_golden_root_groups_are_published_or_known(
    golden: h5py.File,
) -> None:
    """A brand-new neutral-zone root group forces a documentation call."""
    roots = set(golden.keys())
    stray = roots - PUBLISHED_ROOT_GROUPS - KNOWN_UNPUBLISHED_ROOT_GROUPS
    assert not stray, (
        f"unrecognised neutral-zone root group(s): {sorted(stray)}. Either "
        "publish them in docs/design/model-h5-neutral-zone.md (and add them "
        "to PUBLISHED_ROOT_GROUPS) or record them in "
        "KNOWN_UNPUBLISHED_ROOT_GROUPS."
    )


def test_fresh_emit_matches_the_golden_inventory(
    golden: h5py.File, fresh: h5py.File,
) -> None:
    """The emitter today writes the same shape the golden captured.

    This is the drift alarm: the golden is bytes on disk and can go
    stale silently, so the same model is emitted fresh and the two
    published inventories are compared pattern-for-pattern.
    """
    def inventory_of(f: h5py.File) -> set[tuple[str, str, int]]:
        _, datasets = _walk(f)
        return {
            (_pattern(p), _dtype_token(f[p]), len(f[p].shape))
            for p in datasets if _published(p)
        }

    stale = inventory_of(golden) - inventory_of(fresh)
    added = inventory_of(fresh) - inventory_of(golden)
    assert not stale and not added, (
        "the emitter's neutral zone no longer matches the committed "
        f"golden.\n  only in golden (stale): {sorted(stale)}"
        f"\n  only in fresh (new):    {sorted(added)}"
        "\nRegenerate: python tests/fixtures/neutral_zone/"
        "_generate_fixtures.py — then update the note."
    )


def test_fresh_emit_stamps_the_writer_constant(fresh: h5py.File) -> None:
    """A file written now carries exactly the writer's current version."""
    assert fresh["meta"].attrs["neutral_schema_version"] == (
        NEUTRAL_SCHEMA_VERSION
    )


def test_golden_version_is_inside_the_reader_window(
    golden: h5py.File,
) -> None:
    """The golden stays readable under the ADR 0023 two-version window."""
    stamped = SchemaVersion.parse(
        golden["meta"].attrs["neutral_schema_version"]
    )
    validate_zone_version(stamped, reader_version(NEUTRAL), zone=NEUTRAL)


# ---------------------------------------------------------------------------
# pinned consumer-facing semantics
# ---------------------------------------------------------------------------


def test_connectivity_holds_node_ids_not_row_indices(
    golden: h5py.File,
) -> None:
    """The claim a consumer's id->row map depends on."""
    node_ids = golden["nodes/ids"][:]
    conn = golden["elements/hex8/connectivity"][:]
    assert set(np.unique(conn)).issubset(set(node_ids.tolist()))
    # ...and the ids are 1-based, so a 0-based positional read would be
    # off-by-one at best and out of range at worst.
    assert node_ids.min() == 1


def test_element_ids_are_not_per_type_one_based(golden: h5py.File) -> None:
    """Element ids share one gmsh numbering space across all types."""
    assert golden["elements/hex8/ids"][0] != 1


def test_coords_are_always_three_columns(golden: h5py.File) -> None:
    assert golden["nodes/coords"].shape[1] == 3


def test_element_side_pg_ids_may_dangle(golden: h5py.File) -> None:
    """PG element_ids can name elements no /elements group carries.

    The golden's ``Base`` PG is a face of the cube: ``get_fem_data(dim=3)``
    keeps the hexahedra and drops the quads, but the PG still records the
    quad ids.  A consumer that assumes every PG element id resolves will
    throw on a perfectly ordinary file.
    """
    known = set(golden["elements/hex8/ids"][:].tolist())
    base = set(golden["physical_groups/element_side/Base/element_ids"][:]
               .tolist())
    assert base and not (base & known)


def test_module_label_reads_back_as_bytes(golden: h5py.File) -> None:
    """Variable-length UTF-8: h5py hands back bytes, not str."""
    assert isinstance(golden["nodes/module_label"][0], bytes)


# ---------------------------------------------------------------------------
# inventory <-> page
# ---------------------------------------------------------------------------


def _page_text() -> str:
    assert _PAGE.exists(), f"missing published page {_PAGE}"
    return _PAGE.read_text(encoding="utf-8")


def test_page_documents_every_inventory_path() -> None:
    """Nothing in the contract is missing from the page."""
    text = _page_text()
    missing = [
        pat for pat in list(GROUPS) + list(DATASETS) if pat not in text
    ]
    assert not missing, (
        "these documented paths never appear on "
        f"docs/design/model-h5-neutral-zone.md: {missing}"
    )


def test_page_mentions_no_undocumented_neutral_path() -> None:
    """Nothing on the page is absent from the contract."""
    text = _page_text()
    known = set(GROUPS) | set(DATASETS)
    roots = "|".join(sorted(PUBLISHED_ROOT_GROUPS))
    found = set(re.findall(rf"`(/(?:{roots})[A-Za-z0-9_{{}}/]*)`", text))
    stray = {
        p for p in found
        if p not in known and p.rstrip("/") not in known
    }
    assert not stray, (
        "the page documents neutral-zone paths that are not in "
        f"tests/fixtures/neutral_zone/inventory.py: {sorted(stray)}"
    )


def test_page_states_the_current_schema_version() -> None:
    """The page's version rule quotes the version the writer stamps."""
    assert NEUTRAL_SCHEMA_VERSION in _page_text()
