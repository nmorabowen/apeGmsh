"""
Lineage — git-style content-hash chain (ADR 0021).

Three deterministic hashes, chained one-directionally, recorded under
``/meta/lineage`` in every emitted file::

    fem_hash      = blake2b-128(canonical_neutral_zone_bytes)
    model_hash    = blake2b-128(fem_hash || canonical_opensees_zone_bytes)
    results_hash  = blake2b-128(model_hash || canonical_run_zone_bytes)

Each layer's hash includes its parent's hash; the chain is
tamper-evident.  Mismatches on read surface in :attr:`Lineage.warnings`
as plain strings prefixed with ``"[lineage] "`` — they **never raise
from a constructor or loader** (INV-2).  Users who want loud-fail
opt in via :meth:`Lineage.assert_clean`.

Canonical bytes (INV-5)
-----------------------

The canonical-bytes walk is the load-bearing determinism contract.
:func:`canonical_bytes` walks an :class:`h5py.Group` in **name-sorted**
order at every level, hashing:

1. Every dataset's raw bytes (``np.ascontiguousarray(...).tobytes()``).
2. Every attribute, names sorted, values serialised stably (UTF-8 for
   strings, IEEE-754 for floats, raw bytes for integer / float arrays).
3. Recurse into subgroups in name-sorted order.

Two ``to_h5`` calls of the same model — possibly with different chunk
layouts or write orderings — must produce identical canonical bytes.
The tests in ``tests/test_lineage_chain.py`` enforce this across the
two-version reader window of ADR 0023.

See also
========

- :doc:`/architecture/decisions/0021-lineage-chain-replaces-snapshot-id`
  — the full contract.
- :doc:`/architecture/decisions/0023-per-zone-schema-versioning` — the
  schema-version window within which canonical bytes are stable.
- :mod:`apeGmsh.mesh._femdata_hash` — today's ``snapshot_id``;
  :func:`compute_fem_hash` is byte-identical (INV-1).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

if TYPE_CHECKING:
    import h5py

    from apeGmsh.mesh.FEMData import FEMData


__all__ = [
    "Lineage",
    "LineageError",
    "LINEAGE_GROUP",
    "WARNING_PREFIX",
    "canonical_bytes",
    "compose_hash",
    "compute_fem_hash",
    "compute_model_hash",
    "compute_results_hash",
    "read_stored_lineage",
    "write_lineage_attrs",
    "MODEL_HASH_EXCLUDED_CHILDREN",
]


#: HDF5 sub-group of ``/meta`` where the lineage attrs land.
LINEAGE_GROUP: str = "lineage"

#: Prefix every lineage warning starts with.  Tested by
#: ``tests/test_lineage_chain.py::test_lineage_warning_prefix``.
WARNING_PREFIX: str = "[lineage] "

#: Children of ``/opensees/`` excluded from :func:`compute_model_hash`
#: (INV-4).  Cuts and sweeps are user-attached post-hoc artifacts; a
#: cut-edit workflow must not produce lineage warnings on every
#: viewer open.  ``regions`` are derived from ``nodes_pg`` /
#: ``elements_pg`` on every MPCO emit and not loaded by the broker,
#: so ``from_h5 → to_h5`` cycles produce files without regions; we
#: must elide them to keep ``model_hash`` stable across the round-trip.
#: ``names`` is the bridge-side alias sidecar (name→kind+tag): a label,
#: not structure — relabelling a primitive must not invalidate results
#: lineage, so it is excluded on the same grounds.
MODEL_HASH_EXCLUDED_CHILDREN: frozenset[str] = frozenset(
    {"cuts", "sweeps", "regions", "names"}
)

#: blake2b digest size — matches today's ``snapshot_id`` (16 bytes ⇒
#: 32 hex chars).
_DIGEST_SIZE: int = 16


# =====================================================================
# Public surface — exception + dataclass
# =====================================================================


class LineageError(ValueError):
    """Raised by :meth:`Lineage.assert_clean` when warnings are present.

    Lineage mismatches NEVER raise from a constructor or loader (INV-2);
    this exception is only used for the explicit, opt-in
    :meth:`Lineage.assert_clean` check.  Phase 8 deletes the inert
    :class:`apeGmsh.results._bind.BindError`; this is its replacement
    for users who want loud-fail semantics.
    """


@dataclass(frozen=True, slots=True)
class Lineage:
    """Three-link content-hash chain identifying a model's provenance.

    Per ADR 0021::

        fem_hash      = blake2b-128(canonical_neutral_zone_bytes)
        model_hash    = blake2b-128(fem_hash || canonical_opensees_zone_bytes)
        results_hash  = blake2b-128(model_hash || canonical_run_zone_bytes)

    Attributes
    ----------
    fem_hash
        Hex digest over the neutral zone (``/nodes``, ``/elements``,
        ``/physical_groups``, etc.).  Byte-identical to
        :attr:`FEMData.snapshot_id` (INV-1).  Empty string when the
        underlying source carried no FEMData.
    model_hash
        Hex digest chaining :attr:`fem_hash` with the bridge zone
        (``/opensees/...`` minus cuts and sweeps per INV-4).  ``None``
        when no bridge zone is present.
    results_hash
        Hex digest chaining :attr:`model_hash` with the analysis output
        (``/stages/...``).  ``None`` for standalone ``model.h5`` files
        and for results files lacking the ``/opensees/`` zone.
    warnings
        Tuple of human-readable strings describing lineage drift —
        empty when the chain recomputes cleanly.  Every entry starts
        with :data:`WARNING_PREFIX` so callers can filter results-level
        warnings from other diagnostics.  **Never raises**: mismatches
        appear here, not as exceptions.  Call :meth:`assert_clean` for
        opt-in loud-fail.
    """

    fem_hash: str = ""
    model_hash: Optional[str] = None
    results_hash: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def assert_clean(self) -> None:
        """Raise :class:`LineageError` when :attr:`warnings` is non-empty.

        Opt-in loud-fail for users who want a hard stop on lineage
        mismatch::

            results = Results.from_native("run.h5")
            results.lineage.assert_clean()   # raises if drifted

        No-op when :attr:`warnings` is empty.
        """
        if self.warnings:
            joined = "\n".join(self.warnings)
            raise LineageError(
                f"Lineage drift detected ({len(self.warnings)} warning"
                f"{'s' if len(self.warnings) != 1 else ''}):\n{joined}"
            )


# =====================================================================
# Canonical-bytes walk — INV-5 stability surface
# =====================================================================


def canonical_bytes(
    group: "h5py.Group",
    *,
    exclude_children: Iterable[str] = (),
) -> bytes:
    """Return a deterministic byte string for ``group``.

    Walks ``group`` recursively in **name-sorted** order at every
    level.  For each member encountered:

    * Datasets contribute their dtype tag, shape, and the contiguous
      raw bytes of their values.
    * Subgroups contribute their attributes (sorted by name) followed
      by their recursive canonical bytes.
    * Attributes contribute UTF-8 bytes of the name, the stable
      serialisation of the value, and a terminator.

    The serialisation is constructed so chunked vs contiguous storage
    of the same array yields identical bytes (INV-5): ``np.asarray(ds)``
    materialises the dataset as a contiguous numpy array; calling
    ``.tobytes()`` on the contiguous result is layout-independent.

    Parameters
    ----------
    group
        Either an :class:`h5py.Group` or :class:`h5py.File` (which is a
        Group).  The walk starts at this node.
    exclude_children
        Names of direct children to skip.  Used by
        :func:`compute_model_hash` to elide ``cuts`` / ``sweeps``
        (INV-4).  Sub-children of the excluded names are also skipped.

    Returns
    -------
    bytes
        A deterministic byte sequence suitable for feeding into a
        cryptographic hash.  Not human-readable; do not parse.
    """
    excluded = frozenset(exclude_children)

    h = hashlib.blake2b(digest_size=64)
    _canonical_walk(group, h, excluded=excluded, is_root=True)
    return h.digest()


def _canonical_walk(
    node: "h5py.Group | h5py.Dataset",
    h: "hashlib._Hash",
    *,
    excluded: frozenset[str],
    is_root: bool,
) -> None:
    """Inner walk: feed ``node`` deterministically into ``h``."""
    import h5py

    # Walk attributes first (sorted by name) so structural changes to
    # attrs are detectable.
    _hash_attrs(node.attrs, h)

    if isinstance(node, h5py.Dataset):
        h.update(b"D|")
        # dtype tag — guards against silent type drift across writes.
        h.update(_dtype_tag(node.dtype))
        # shape tuple as a fixed-width byte sequence.
        shape = tuple(int(x) for x in node.shape)
        h.update(np.asarray(shape, dtype=np.int64).tobytes())
        # value bytes — np.asarray materialises a contiguous view so
        # chunked vs contiguous storage hits the same bytes here.
        arr = np.asarray(node[()])
        h.update(_array_bytes(arr))
        return

    # Group (or File).  Walk children in name-sorted order.
    h.update(b"G|")
    names = sorted(node.keys())
    for name in names:
        if is_root and name in excluded:
            continue
        # Member separator: name length, name bytes, then recurse.
        name_bytes = name.encode("utf-8")
        h.update(b"M|")
        h.update(np.int64(len(name_bytes)).tobytes())
        h.update(name_bytes)
        child = node[name]
        _canonical_walk(child, h, excluded=excluded, is_root=False)
    # End-of-group marker disambiguates empty groups from absent
    # members.
    h.update(b"E|")


def _hash_attrs(attrs: "h5py.AttributeManager", h: "hashlib._Hash") -> None:
    """Feed sorted ``(name, value)`` attrs into ``h``."""
    names = sorted(attrs.keys())
    h.update(b"A|")
    h.update(np.int64(len(names)).tobytes())
    for name in names:
        name_bytes = name.encode("utf-8")
        h.update(np.int64(len(name_bytes)).tobytes())
        h.update(name_bytes)
        value = attrs[name]
        h.update(b":")
        h.update(_attr_value_bytes(value))
    h.update(b"a|")


def _attr_value_bytes(value: object) -> bytes:
    """Serialise an HDF5 attribute value to stable bytes."""
    if value is None:
        return b"N|"
    if isinstance(value, bool):
        return b"b|" + (b"\x01" if value else b"\x00")
    if isinstance(value, (bytes, bytearray)):
        return b"s|" + bytes(value)
    if isinstance(value, str):
        return b"S|" + value.encode("utf-8")
    if isinstance(value, (int, np.integer)):
        return b"i|" + np.int64(int(value)).tobytes()
    if isinstance(value, (float, np.floating)):
        return b"f|" + np.float64(float(value)).tobytes()
    if isinstance(value, np.ndarray):
        return b"A|" + _dtype_tag(value.dtype) + _array_bytes(value)
    # Fallback: stringify so the canonical walk never crashes on an
    # exotic type the schema added without us noticing.
    return b"?|" + repr(value).encode("utf-8")


def _array_bytes(arr: np.ndarray) -> bytes:
    """Stable bytes for a numpy array.

    Variable-length string arrays (h5py vlen), object dtypes, and
    structured dtypes containing object fields need explicit
    serialisation: ``.tobytes()`` on those would hit the Python object
    pointers (memory addresses) rather than the underlying string
    bytes, producing non-deterministic output across writes (INV-5
    violation).  Plain numeric dtypes are layout-stable via
    ``np.ascontiguousarray(...).tobytes()``.
    """
    # Object / string dtype: encode each element as UTF-8.
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        return _encode_object_array(arr)

    # Structured dtype: check whether any field is object-typed; if so
    # walk field-by-field so the object fields get the UTF-8 path.
    if arr.dtype.names is not None and _has_object_field(arr.dtype):
        return _encode_structured_array(arr)

    # Plain numeric / structured-numeric — tobytes() is layout-stable
    # on a contiguous view.
    contiguous = np.ascontiguousarray(arr)
    return (
        np.asarray(contiguous.shape, dtype=np.int64).tobytes()
        + contiguous.tobytes()
    )


def _has_object_field(dtype: np.dtype) -> bool:
    """Return ``True`` when ``dtype`` is structured with any object field."""
    if dtype.names is None:
        return False
    for name in dtype.names:
        sub = dtype.fields[name][0]
        if sub.kind == "O":
            return True
    return False


def _encode_object_array(arr: np.ndarray) -> bytes:
    """Encode a plain object / string array element-by-element."""
    out = bytearray()
    out += np.asarray(arr.shape, dtype=np.int64).tobytes()
    flat = arr.ravel()
    for el in flat:
        out += _encode_object_element(el)
    return bytes(out)


def _encode_structured_array(arr: np.ndarray) -> bytes:
    """Encode a structured array with one or more object fields.

    Walks fields in declaration order — ``dtype.names`` is part of the
    dtype identity and is byte-stable across writes.  Object fields go
    through the UTF-8 path; numeric fields use ``.tobytes()``.
    """
    out = bytearray()
    out += np.asarray(arr.shape, dtype=np.int64).tobytes()
    flat = arr.ravel()
    names = arr.dtype.names
    for el in flat:
        for name in names:
            sub = arr.dtype.fields[name][0]
            value = el[name]
            if sub.kind == "O":
                out += _encode_object_element(value)
            else:
                sub_arr = np.asarray(value, dtype=sub)
                out += np.ascontiguousarray(sub_arr).tobytes()
    return bytes(out)


def _encode_object_element(el: object) -> bytes:
    """Encode a single object-dtype element to stable bytes."""
    if isinstance(el, bytes):
        payload = el
    elif el is None:
        payload = b""
    else:
        payload = str(el).encode("utf-8")
    return np.int64(len(payload)).tobytes() + payload


def _dtype_tag(dtype: np.dtype) -> bytes:
    """Return a stable byte tag for a numpy dtype."""
    return dtype.str.encode("utf-8")


# =====================================================================
# Hash chain helpers
# =====================================================================


def compute_fem_hash(neutral_group: "h5py.Group") -> str:
    """Return ``fem_hash`` for the neutral zone at ``neutral_group``.

    INV-1: byte-identical to today's :attr:`FEMData.snapshot_id` for
    the same FEMData.  The implementation rehydrates the neutral zone
    into a :class:`FEMData` and recomputes the snapshot hash via the
    existing :func:`apeGmsh.mesh._femdata_hash.compute_snapshot_id`
    code path — guaranteeing equality with the value stamped at write
    time.  Both paths converge on the same 32-char hex digest.

    The canonical-bytes walk over the HDF5 zone is not used here for
    the FEM layer because INV-1 demands exact byte-equality with the
    existing hash semantics over ``(nodes, elements, PGs)``.  The
    walk **is** used for the subsequent ``model_hash`` and
    ``results_hash`` layers, where there's no prior fixed semantics
    to preserve.

    Parameters
    ----------
    neutral_group
        ``h5py.File`` for a standalone ``model.h5`` (with ``/nodes``,
        ``/elements``, ... at root), OR the ``/model/`` group inside a
        composed results file.

    Returns
    -------
    str
        32-character hex digest equal to the source FEMData's
        ``snapshot_id``.
    """
    # Lazy import — the lineage module sits in apeGmsh.opensees but
    # the FEM hash semantics live in apeGmsh.mesh.  Per ADR 0019
    # INV-4 the apeGmsh.opensees import-time graph stays free of
    # apeGmsh.mesh; the edge is bound here at call time only.
    from apeGmsh.mesh._femdata_h5_io import read_neutral_zone_from_group

    fem = read_neutral_zone_from_group(neutral_group, label="<lineage>")
    # Route through the compose-aware wrapper (ADR 0038 §"Lineage
    # chain extension" / D4) so the H5-side rehydrate path picks up
    # any future evolution of the per-module hash composition without
    # needing a parallel edit here.
    return compose_hash(fem)


def compose_hash(fem: "FEMData") -> str:
    """Return the compose-aware ``fem_hash`` for ``fem``.

    Canonical compose-aware entry point for the FEM-layer hash per
    ADR 0038 §"Lineage chain extension" (D4 — compose-aware hash
    wrapper).  Routing all ``fem_hash`` callsites through this
    wrapper future-proofs the chain against ``canonical_bytes()`` /
    ``compute_snapshot_id`` implementation drift: the wrapper owns
    the "sort by ``module_label`` before hashing" contract that
    locks INV-1's compose-order invariance structurally rather than
    relying on the internal iteration order of any one helper.

    Today's implementation delegates to
    :attr:`FEMData.snapshot_id`, which folds ``composed_from`` per
    :func:`apeGmsh.mesh._femdata_hash._hash_composed_from` —
    records sorted by ``label`` so ``compose(A) → compose(B)`` and
    ``compose(B) → compose(A)`` produce the same digest.  On an
    uncomposed FEMData (``fem.composed_from == ()``) the
    ``COMPOSED|`` block is skipped entirely, so the digest is
    **byte-identical to the pre-2.9.0 snapshot_id** — pre-existing
    pin tests and bind-contract round-trips stay green by
    construction.

    The wrapper is the stable name external callers should use; if
    a future refactor introduces per-module ``canonical_bytes()``
    methods (per ADR 0038 §"Lineage chain extension" example), the
    internals of :func:`compose_hash` change but the API surface
    does not.

    Parameters
    ----------
    fem
        Any object satisfying the :class:`FEMData` ``snapshot_id``
        contract.  Test doubles that implement ``snapshot_id`` as a
        property also work — the wrapper is not type-narrowed beyond
        the duck-typed protocol.

    Returns
    -------
    str
        32-character hex digest (blake2b-128) — same width and
        algorithm as :attr:`FEMData.snapshot_id`.

    See also
    --------
    compute_fem_hash : H5-side counterpart that rehydrates a
        neutral zone into a :class:`FEMData` before calling this
        wrapper's underlying snapshot path.
    """
    return str(fem.snapshot_id)


def compute_model_hash(
    fem_hash: str,
    opensees_group: "h5py.Group",
) -> str:
    """Return ``model_hash`` chaining ``fem_hash`` with the bridge zone.

    Per INV-4, cuts and sweeps are excluded from the canonical bytes —
    they're user-attached post-hoc artifacts and must not perturb the
    model identity.

    Parameters
    ----------
    fem_hash
        The FEM-layer hex digest the bridge zone sits on top of.
        Empty string is permitted (bridge-only files lacking a FEMData
        neutral zone).
    opensees_group
        The ``/opensees`` group (read from a standalone ``model.h5``
        at root or a composed results file).

    Returns
    -------
    str
        32-character hex digest.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    h.update(fem_hash.encode("utf-8"))
    h.update(b"|")
    h.update(canonical_bytes(
        opensees_group, exclude_children=MODEL_HASH_EXCLUDED_CHILDREN,
    ))
    return h.hexdigest()


def compute_results_hash(
    model_hash: str,
    run_group: "h5py.Group",
) -> str:
    """Return ``results_hash`` chaining ``model_hash`` with the run zone.

    Parameters
    ----------
    model_hash
        The model-layer hex digest the run sits on top of.  Empty
        string is permitted (results files lacking a bridge zone).
    run_group
        The ``/stages`` group from a native results file.

    Returns
    -------
    str
        32-character hex digest.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    h.update(model_hash.encode("utf-8"))
    h.update(b"|")
    h.update(canonical_bytes(run_group))
    return h.hexdigest()


def _shrink_to_hex(digest: bytes) -> str:
    """Compress a wide blake2b digest to a 32-char hex string.

    ``canonical_bytes`` uses a 64-byte blake2b to spread state during
    the walk; the stamped attrs are 16 bytes (32 hex chars) — same
    width as today's ``snapshot_id`` for INV-1.
    """
    return hashlib.blake2b(digest, digest_size=_DIGEST_SIZE).hexdigest()


# =====================================================================
# Stamp + read helpers
# =====================================================================


def write_lineage_attrs(
    meta_group: "h5py.Group",
    *,
    fem_hash: str | None = None,
    model_hash: str | None = None,
    results_hash: str | None = None,
) -> None:
    """Stamp ``/meta/lineage/{fem_hash, model_hash, results_hash}``.

    Creates the ``lineage`` sub-group under ``meta_group`` (overwriting
    pre-existing attrs).  Only the hash fields the caller supplies are
    written; the others stay absent.

    Per ADR 0023, ``/meta/lineage`` is an additive sub-group; readers
    at older schema minors silently lack it and produce a
    "lineage absent" warning rather than raising.
    """
    if LINEAGE_GROUP in meta_group:
        lin = meta_group[LINEAGE_GROUP]
    else:
        lin = meta_group.create_group(LINEAGE_GROUP)
    if fem_hash is not None:
        lin.attrs["fem_hash"] = str(fem_hash)
    if model_hash is not None:
        lin.attrs["model_hash"] = str(model_hash)
    if results_hash is not None:
        lin.attrs["results_hash"] = str(results_hash)


def read_stored_lineage(
    meta_group: "h5py.Group",
) -> "tuple[str | None, str | None, str | None]":
    """Read ``/meta/lineage/...`` attrs.

    Returns ``(fem_hash, model_hash, results_hash)``.  Any missing
    field becomes ``None``; an absent ``lineage`` sub-group returns
    ``(None, None, None)`` — caller decides whether to surface that as
    a "lineage absent — legacy file" warning.

    Uses ``name in group`` probes per the repository's h5py optional-
    child convention; never calls ``Group.get`` on optional children.
    """
    if LINEAGE_GROUP not in meta_group:
        return None, None, None
    lin = meta_group[LINEAGE_GROUP]
    return (
        _attr_string_or_none(lin, "fem_hash"),
        _attr_string_or_none(lin, "model_hash"),
        _attr_string_or_none(lin, "results_hash"),
    )


def _attr_string_or_none(
    grp: "h5py.Group", key: str,
) -> "str | None":
    """Return ``grp.attrs[key]`` as a string, or ``None`` when absent."""
    if key not in grp.attrs:
        return None
    raw = grp.attrs[key]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)
