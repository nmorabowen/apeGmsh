"""HDF5 I/O for ``SectionCutDef`` / ``SectionSweepDef`` persistence.

Persistence layer for v4 of the apeGmsh.cuts roadmap. Cuts and sweeps
land under ``/opensees/cuts/`` and ``/opensees/sweeps/`` in
``model.h5``; schema bump ``2.4.0 → 2.5.0`` (2.3.0 was the recorder
unification in Phase 9 commit 6 and 2.4.0 was the mesh_selection
neutral-zone addition in Phase 8.7 commit 2; cuts are the next
additive minor).

This module is lazy on ``h5py`` — the public entry points import it
only inside their bodies so ``apeGmsh.cuts`` stays importable without
the dependency at import time.

See ``apeGmsh/cuts/ARCHITECTURE.md`` ("## v4 — Cuts persisted in
``model.h5``") for the locked design decisions H1–H17 and the on-disk
shape spec.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ._defs import SectionCutDef
from ._sweeps import SectionSweepDef

if TYPE_CHECKING:
    import h5py


__all__ = [
    "persist_to_h5",
    "read_cuts_and_sweeps",
    "write_cuts_into",
]


#: Schema version stamped when ``persist_to_h5`` bumps a pre-v4 file.
#: Files already at or above this version are left untouched.
V4_SCHEMA_VERSION: str = "2.5.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_attr(value: Any) -> Any:
    """Decode an HDF5 attr value to a plain Python value.

    Mirrors :func:`apeGmsh.opensees.emitter.h5_reader._decode_bytes` —
    kept inline so the cuts subpackage doesn't pull h5_reader at
    import time.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.dtype.kind in ("O", "S"):
        if value.shape == ():
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="replace")
            return item
    return value


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a semver string into a comparable ``(major, minor, patch)`` tuple.

    Tolerates short forms by zero-padding (``"2"`` → ``(2, 0, 0)``).
    Non-integer parts are treated as ``0`` so a malformed string never
    raises here — strict-major checking lives in
    :func:`apeGmsh.opensees.emitter.h5_reader.open`.
    """
    parts: list[int] = []
    for chunk in version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _index_from_name(name: str, prefix: str) -> int:
    """Extract the integer index from a positional name like ``cut_7``.

    Returns ``-1`` for names that don't follow the ``prefix_<int>``
    shape; those sort first under :func:`sorted` with this key, which
    surfaces oddities rather than silently scrambling them with the
    well-formed entries.
    """
    expected = f"{prefix}_"
    if not name.startswith(expected):
        return -1
    try:
        return int(name[len(expected):])
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_cut_into_group(
    parent: "h5py.Group", name: str, cut: SectionCutDef,
) -> None:
    """Write one :class:`SectionCutDef` as a sub-group of ``parent``.

    Shape per ARCHITECTURE.md "## v4" on-disk spec:

    * attrs: ``plane_point`` (3,)f64, ``plane_normal`` (3,)f64,
      ``side`` utf-8, ``label`` utf-8 (``""`` if None),
      ``has_label`` i8, ``has_bounding`` i8
    * datasets: ``element_ids`` (Ne,)i64,
      ``bounding_polygon`` (Mb, 3)f64 [optional, present iff
      ``has_bounding == 1``]
    """
    import h5py

    g = parent.create_group(name)
    g.attrs["plane_point"] = np.asarray(cut.plane_point, dtype=np.float64)
    g.attrs["plane_normal"] = np.asarray(cut.plane_normal, dtype=np.float64)
    g.attrs.create(
        "side", cut.side, dtype=h5py.string_dtype(encoding="utf-8"),
    )
    g.attrs["has_label"] = np.int8(0 if cut.label is None else 1)
    g.attrs.create(
        "label", cut.label or "",
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    g.attrs["has_bounding"] = np.int8(
        0 if cut.bounding_polygon is None else 1
    )
    g.create_dataset(
        "element_ids",
        data=np.asarray(cut.element_ids, dtype=np.int64),
    )
    if cut.bounding_polygon is not None:
        g.create_dataset(
            "bounding_polygon",
            data=np.asarray(cut.bounding_polygon, dtype=np.float64),
        )


def _write_sweep_into_group(
    parent: "h5py.Group", name: str, sweep: SectionSweepDef,
) -> None:
    """Write one :class:`SectionSweepDef` as a sub-group of ``parent``.

    Shape per ARCHITECTURE.md "## v4":

    * attrs: ``count`` i64, ``order`` vlen utf-8
      (``["cut_0", "cut_1", ...]`` in sweep order)
    * ``cuts/`` sub-group with one ``cut_N`` per sweep member, in
      sweep order — names are positional so :func:`_read_sweep_from_group`
      can walk the explicit ``order`` attr instead of relying on
      alphabetic group iteration.
    """
    import h5py

    g = parent.create_group(name)
    g.attrs["count"] = np.int64(len(sweep))
    order_names = [f"cut_{i}" for i in range(len(sweep))]
    g.attrs.create(
        "order", order_names,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    cuts_group = g.create_group("cuts")
    for i, cut in enumerate(sweep):
        _write_cut_into_group(cuts_group, f"cut_{i}", cut)


def write_cuts_into(
    f: "h5py.File",
    *,
    cuts: Sequence[SectionCutDef] = (),
    sweeps: Sequence[SectionSweepDef] = (),
) -> None:
    """Write ``/opensees/cuts/`` and ``/opensees/sweeps/`` into an open file.

    Primitive used by both :meth:`apeGmsh.opensees.apeSees.h5` (primary
    path, model + cuts in one shot — v4-2) and
    :func:`apeGmsh.cuts.persist_to_h5` (append — v4-3).

    The ``/opensees`` group is created lazily if absent. The
    ``/opensees/cuts`` and ``/opensees/sweeps`` groups are created
    fresh — :class:`ValueError` is raised if either already exists.
    Callers that want clean-append semantics must delete them first
    (:func:`persist_to_h5` does this for the user).

    Empty input (no ``cuts`` and no ``sweeps``) is a no-op — neither
    group is created and the file is left unchanged.
    """
    if not cuts and not sweeps:
        return

    if "opensees" not in f:
        f.create_group("opensees")
    ops = f["opensees"]

    if cuts:
        if "cuts" in ops:
            raise ValueError(
                "/opensees/cuts already exists; caller must clear it "
                "before re-writing."
            )
        cuts_group = ops.create_group("cuts")
        cuts_group.attrs["count"] = np.int64(len(cuts))
        for i, cut in enumerate(cuts):
            _write_cut_into_group(cuts_group, f"cut_{i}", cut)

    if sweeps:
        if "sweeps" in ops:
            raise ValueError(
                "/opensees/sweeps already exists; caller must clear it "
                "before re-writing."
            )
        sweeps_group = ops.create_group("sweeps")
        sweeps_group.attrs["count"] = np.int64(len(sweeps))
        for i, sweep in enumerate(sweeps):
            _write_sweep_into_group(sweeps_group, f"sweep_{i}", sweep)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _read_cut_from_group(g: "h5py.Group") -> SectionCutDef:
    """Reconstruct one :class:`SectionCutDef` from an h5 group.

    Routes through the public constructor so ``__post_init__``
    validation runs on every read.
    """
    plane_point = tuple(float(v) for v in g.attrs["plane_point"])
    plane_normal = tuple(float(v) for v in g.attrs["plane_normal"])
    side = str(_decode_attr(g.attrs["side"]))

    has_label = int(g.attrs.get("has_label", 0))
    raw_label = str(_decode_attr(g.attrs.get("label", "")))
    label = raw_label if has_label else None

    element_ids = tuple(int(x) for x in g["element_ids"][:])

    has_bounding = int(g.attrs.get("has_bounding", 0))
    bounding_polygon: tuple[tuple[float, float, float], ...] | None = None
    if has_bounding:
        bp_arr = g["bounding_polygon"][:]
        bounding_polygon = tuple(
            (float(row[0]), float(row[1]), float(row[2]))
            for row in bp_arr
        )

    return SectionCutDef(
        plane_point=plane_point,  # type: ignore[arg-type]
        plane_normal=plane_normal,  # type: ignore[arg-type]
        element_ids=element_ids,
        side=side,  # type: ignore[arg-type]
        label=label,
        bounding_polygon=bounding_polygon,
    )


def _read_sweep_from_group(g: "h5py.Group") -> SectionSweepDef:
    """Reconstruct one :class:`SectionSweepDef` from an h5 group.

    Walks the explicit ``order`` attr so the sweep tuple is rebuilt
    in the right sequence regardless of HDF5's alphabetic group
    iteration.
    """
    order_raw = g.attrs["order"]
    order_names = [str(_decode_attr(n)) for n in order_raw]
    cuts_group = g["cuts"]
    cuts = tuple(
        _read_cut_from_group(cuts_group[name]) for name in order_names
    )
    return SectionSweepDef(cuts=cuts)


def read_cuts_and_sweeps(
    path: str | Path,
) -> tuple[tuple[SectionCutDef, ...], tuple[SectionSweepDef, ...]]:
    """Read ``/opensees/cuts/`` and ``/opensees/sweeps/`` from ``model.h5``.

    Returns ``(cuts, sweeps)`` tuples. Missing groups → empty tuples,
    so pre-2.3.0 files just produce no cuts (no exception). Standalone
    cuts are returned in writer order; each sweep's contained cuts
    follow the sweep's ``order`` attr.

    Reconstructs every :class:`SectionCutDef` and
    :class:`SectionSweepDef` through their public constructors, so
    ``__post_init__`` validation runs on every read.

    Raises
    ------
    SchemaVersionError
        If ``/meta/schema_version`` major != 2 (propagated from the
        underlying reference reader).
    FileNotFoundError, MalformedH5Error
        Propagated from
        :func:`apeGmsh.opensees.emitter.h5_reader.open`.
    """
    from apeGmsh.opensees.emitter import h5_reader

    cuts: tuple[SectionCutDef, ...] = ()
    sweeps: tuple[SectionSweepDef, ...] = ()

    with h5_reader.open(str(path)) as model:
        f = model.handle
        cuts_group = f.get("opensees/cuts")
        if cuts_group is not None:
            names = sorted(
                cuts_group.keys(),
                key=lambda n: _index_from_name(n, "cut"),
            )
            cuts = tuple(
                _read_cut_from_group(cuts_group[name]) for name in names
            )
        sweeps_group = f.get("opensees/sweeps")
        if sweeps_group is not None:
            names = sorted(
                sweeps_group.keys(),
                key=lambda n: _index_from_name(n, "sweep"),
            )
            sweeps = tuple(
                _read_sweep_from_group(sweeps_group[name])
                for name in names
            )
    return cuts, sweeps


# ---------------------------------------------------------------------------
# Append helper — write cuts to an existing model.h5
# ---------------------------------------------------------------------------

def _maybe_bump_schema_version(
    f: "h5py.File", *, min_version: str = V4_SCHEMA_VERSION,
) -> None:
    """Bump ``/meta/schema_version`` to ``min_version`` if it was lower.

    No-op when ``/meta`` is absent or when the current version is at
    or above ``min_version``. Comparison is on the ``(major, minor,
    patch)`` tuple — so a future ``2.4.x`` file is left alone, while
    a ``2.2.0`` file gets bumped.
    """
    import h5py

    if "meta" not in f:
        return
    current_raw = f["meta"].attrs.get("schema_version", "")
    current = str(_decode_attr(current_raw))
    if not current:
        return
    if _version_tuple(current) >= _version_tuple(min_version):
        return
    # Existing attr may have a different vlen-string dtype; delete and
    # recreate with the canonical encoding rather than relying on
    # h5py.Attrs.__setitem__ to coerce in-place.
    del f["meta"].attrs["schema_version"]
    f["meta"].attrs.create(
        "schema_version", min_version,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def persist_to_h5(
    path: str | Path,
    *,
    cuts: Sequence[SectionCutDef] = (),
    sweeps: Sequence[SectionSweepDef] = (),
) -> None:
    """Append ``/opensees/cuts/`` / ``/opensees/sweeps/`` to an existing file.

    Opens ``path`` in ``r+`` mode (the file must already exist —
    typically produced by an earlier :meth:`apeGmsh.opensees.apeSees.h5`
    call). For each non-empty kwarg, deletes the corresponding
    existing group (if any) before writing the new content. Empty
    kwargs leave the corresponding group untouched — so
    ``persist_to_h5(path, cuts=[c])`` replaces ``/opensees/cuts/``
    but never touches ``/opensees/sweeps/``.

    Bumps ``/meta/schema_version`` to ``"2.5.0"`` if it was lower
    within major 2. Files already at 2.5.0 or higher keep their
    version — that preserves future ``2.6.x`` files unmodified.

    Empty input (no ``cuts`` and no ``sweeps``) is a no-op — the
    file is not opened.

    Raises
    ------
    SchemaVersionError
        Propagated from
        :func:`apeGmsh.opensees.emitter.h5_reader.open` when the file's
        ``/meta/schema_version`` major is not 2.
    FileNotFoundError, MalformedH5Error
        Propagated from the same reader.
    """
    if not cuts and not sweeps:
        return

    import h5py

    from apeGmsh.opensees.emitter import h5_reader

    # Schema-major + /meta validity check via the reference reader.
    # The reader opens read-only and closes when its context exits;
    # we then re-open in r+ for the mutations. The double-open is
    # cheap on a local file and keeps the schema enforcement in one
    # place rather than duplicating the parsing logic here.
    with h5_reader.open(str(path)) as _model:
        _ = _model.schema_version

    with h5py.File(str(path), "r+") as f:
        if cuts and "opensees/cuts" in f:
            del f["opensees/cuts"]
        if sweeps and "opensees/sweeps" in f:
            del f["opensees/sweeps"]
        write_cuts_into(f, cuts=cuts, sweeps=sweeps)
        _maybe_bump_schema_version(f)
