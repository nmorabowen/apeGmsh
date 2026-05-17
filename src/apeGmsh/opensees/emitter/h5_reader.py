"""Reference reader for the bridge's ``model.h5`` archive.

This module ships:

* :class:`SchemaVersionError` / :class:`MalformedH5Error` — explicit
  exception types the viewer team can catch.
* :func:`open` — entry-point that opens a file, performs the strict
  ``major`` schema-version check, and returns an :class:`H5Model`.
* :class:`H5Model` — a thin read-only accessor over the file. The
  viewer team is expected to subclass or wrap this; the per-feature
  methods here cover the minimal contract.

The reader is intentionally not a transformation layer: it returns
``dict`` views of group attrs and numpy arrays of datasets directly,
so a viewer or test can keep working against the same h5py objects
without an indirection layer.

Schema-version compatibility:

* The reader is written for **schema major 2** (current: 2.0.0; the
  Phase 8.4 zone reshuffle moved bridge-written groups under
  ``/opensees/`` and bumped the major).
* It REFUSES files whose ``/meta/schema_version`` starts with anything
  other than ``"2."`` and raises :class:`SchemaVersionError`.
* Minor / patch differences are tolerated — unknown groups are
  simply not surfaced by the typed accessors here but remain
  reachable through :attr:`H5Model.handle`.
"""
from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import h5py


__all__ = [
    "EXPECTED_SCHEMA_MAJOR",
    "H5Model",
    "MalformedH5Error",
    "SchemaVersionError",
    "open",
]


#: Schema major version this reader supports. Bumped when a breaking
#: schema change requires viewer / reader coordination.  Phase 8.4
#: bumped this to 2 alongside the ``/opensees/`` namespace move.
EXPECTED_SCHEMA_MAJOR: int = 2


class SchemaVersionError(ValueError):
    """Raised when ``/meta/schema_version`` doesn't match the reader's
    expected major version. Catch this to surface "open this file with
    a different / newer apeGmsh release" guidance."""


class MalformedH5Error(ValueError):
    """Raised when a file is missing the mandatory ``/meta`` group or
    when a required group is structurally invalid (e.g. a compound
    dataset is missing one of its declared fields)."""


def open(path: str) -> H5Model:
    """Open ``path`` and version-check it.

    Returns
    -------
    H5Model
        Owns the underlying h5py handle; closes on
        :meth:`H5Model.close` or context-manager exit.

    Raises
    ------
    SchemaVersionError
        If ``/meta/schema_version`` major != :data:`EXPECTED_SCHEMA_MAJOR`.
    MalformedH5Error
        If ``/meta`` is missing entirely.
    """
    import h5py
    f = h5py.File(path, "r")
    try:
        if "meta" not in f:
            raise MalformedH5Error(
                f"{path}: missing /meta group; not a bridge model.h5"
            )
        version = str(f["meta"].attrs.get("schema_version", ""))
        if not version:
            raise MalformedH5Error(
                f"{path}: /meta/schema_version attribute is empty"
            )
        major_str = version.split(".", 1)[0]
        try:
            major = int(major_str)
        except ValueError as exc:
            raise MalformedH5Error(
                f"{path}: /meta/schema_version {version!r} is not "
                "semver-shaped"
            ) from exc
        if major != EXPECTED_SCHEMA_MAJOR:
            raise SchemaVersionError(
                f"{path}: schema_version={version} (major {major}) is "
                f"not supported by this reader "
                f"(expected major {EXPECTED_SCHEMA_MAJOR})"
            )
        return H5Model(f, path=path)
    except Exception:
        f.close()
        raise


class H5Model:
    """Read-only accessor over a bridge model.h5 file.

    Construct via :func:`open`. Iterate or use the typed accessors:

    >>> import apeGmsh.opensees.emitter.h5_reader as reader
    >>> m = reader.open("model.h5")
    >>> m.schema_version
    '2.0.0'
    >>> m.meta()["ndm"]
    3
    >>> m.materials()                          # doctest: +SKIP
    {'uniaxial': {...}, 'nd': {...}}
    """

    def __init__(self, f: h5py.File, *, path: str) -> None:
        self._f: h5py.File = f
        self._path: str = path

    # -- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HDF5 handle."""
        self._f.close()

    def __enter__(self) -> H5Model:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def handle(self) -> h5py.File:
        """Raw h5py file handle — for advanced use beyond this API."""
        return self._f

    # -- Required reads --------------------------------------------------

    @property
    def schema_version(self) -> str:
        version: str = str(self._f["meta"].attrs["schema_version"])
        return version

    def meta(self) -> dict[str, Any]:
        """Return the ``/meta`` group's attributes as a plain dict."""
        return _attrs_as_dict(self._f["meta"])

    # -- Optional reads --------------------------------------------------

    def materials(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return ``{family: {name: attrs}}`` for ``/opensees/materials/{family}``.

        Returns an empty dict if ``/opensees/materials`` is missing.
        """
        out: dict[str, dict[str, dict[str, Any]]] = {}
        materials = self._f.get("opensees/materials")
        if materials is None:
            return out
        for family in materials:
            fam_group = materials[family]
            out[family] = {
                name: _attrs_as_dict(fam_group[name])
                for name in fam_group
            }
        return out

    def material(self, family: str, name: str) -> dict[str, Any]:
        """Return one material's attrs.

        Raises :class:`KeyError` if missing.
        """
        return _attrs_as_dict(self._f[f"opensees/materials/{family}/{name}"])

    def sections(self) -> dict[str, dict[str, Any]]:
        """Return ``{name: attrs}`` for ``/opensees/sections``."""
        return self._group_attrs_map("opensees/sections")

    def transforms(self) -> dict[str, dict[str, Any]]:
        return self._group_attrs_map("opensees/transforms")

    def transform_arrays(self, name: str) -> dict[str, Any]:
        """Return ``{per_element_vecxz?, per_element_emitted_tag?}`` for
        one ``/opensees/transforms/{name}`` group.

        Companion to :meth:`transforms` (which exposes only attrs).
        Raises ``KeyError`` if the transform group is missing.  The H5
        emitter writes one group per ``geomTransf`` call (documented
        schema deviation), so both datasets are single-row — one for an
        explicit-``vecxz`` transform, one per distinct vecxz for an
        orientation-driven fan-out.
        """
        sub = self._f[f"opensees/transforms/{name}"]
        out: dict[str, Any] = {}
        if "per_element_vecxz" in sub:
            out["per_element_vecxz"] = sub["per_element_vecxz"][:]
        if "per_element_emitted_tag" in sub:
            out["per_element_emitted_tag"] = sub["per_element_emitted_tag"][:]
        return out

    def beam_integration(self) -> dict[str, dict[str, Any]]:
        return self._group_attrs_map("opensees/beam_integration")

    def elements(self) -> dict[str, dict[str, Any]]:
        return self._group_attrs_map("elements")

    def time_series(self) -> dict[str, dict[str, Any]]:
        return self._group_attrs_map("opensees/time_series")

    def patterns(self) -> dict[str, dict[str, Any]]:
        return self._group_attrs_map("opensees/patterns")

    def recorders(self) -> dict[str, dict[str, Any]]:
        """Return ``{name: attrs}`` for ``/opensees/recorders``.

        Schema 2.3.0 unifies typed and declared recorders into one
        group. Every entry carries a ``kind`` attr — ``"typed"`` for
        ``Node`` / ``Element`` / ``MPCO`` primitives (1:1 with an
        OpenSees recorder command), ``"declared"`` for fan-out calls
        produced by ``ops.recorder.declare(...)``. Declared entries
        also expose the original declaration metadata as attrs
        (``declaration_name``, ``record_name``, ``category``,
        ``components``, ``raw``, ``pg``, ``label``, ``selection``,
        ``ids``, ``dt``, ``n_steps``, ``file_root``).

        For legacy 2.0.0 – 2.2.0 archives (no ``kind`` attr at write
        time) this accessor synthesizes ``kind="typed"`` so callers
        can branch on the field uniformly without a version probe.
        """
        out = self._group_attrs_map("opensees/recorders")
        for entry in out.values():
            entry.setdefault("kind", "typed")
        return out

    def analysis(self) -> dict[str, Any] | None:
        a = self._f.get("opensees/analysis")
        if a is None:
            return None
        return _attrs_as_dict(a)

    def element_meta(self) -> dict[str, dict[str, Any]]:
        """Return ``{type_token: attrs}`` for ``/opensees/element_meta``.

        Phase 8.5 added this OpenSees-specific element-metadata zone
        (positional args, cross-references) alongside the broker's
        neutral ``/elements/{gmsh_alias}`` group.  Companion to
        :meth:`element_meta_arrays` for the actual datasets.
        """
        return self._group_attrs_map("opensees/element_meta")

    def element_meta_arrays(self, type_token: str) -> dict[str, Any]:
        """Return ``{ids, fem_eids?, args?, args_str?}`` for one
        element-meta group.

        Raises ``KeyError`` if the type group is missing.

        ``fem_eids`` is the Phase 8.6 mapping that pairs each
        OpenSees tag with the FEM element id it was fanned out from.
        Sentinel value ``-1`` (matches
        :data:`apeGmsh.opensees._internal.tag_resolution.MISSING_FEM_ELEMENT_ID`)
        marks records emitted outside a bridge fan-out.
        """
        sub = self._f[f"opensees/element_meta/{type_token}"]
        out: dict[str, Any] = {"ids": sub["ids"][:]}
        if "fem_eids" in sub:
            out["fem_eids"] = sub["fem_eids"][:]
        if "args" in sub:
            out["args"] = sub["args"][:]
        if "args_str" in sub:
            out["args_str"] = sub["args_str"][:]
        return out

    def element_local_axes_vecxz(self) -> dict[int, Any]:
        """Return ``{fem_element_id: vecxz}`` for every beam-column element.

        Joins three zones the archive keeps separate:

        * ``/opensees/transforms/*`` — each group carries the geomTransf
          ``tag`` (``per_element_emitted_tag``) and its ``vecxz``
          reference vector (``per_element_vecxz``).
        * ``/opensees/element_meta/{type_token}`` — each element's
          positional ``args`` tail (one slot is the geomTransf tag) and
          ``fem_eids`` mapping the row back to the broker FEM element id.
        * The OpenSees element vocabulary
          (:mod:`apeGmsh.opensees._element_capabilities`) — tells us
          *which* ``args`` slot holds the transf tag, per element type.

        Keyed by FEM element id so a viewer (which speaks the broker's
        ``/elements/{gmsh_alias}`` ids) can look up the orientation of
        any line element without knowing OpenSees tags.

        Returns an empty dict when the file carries no transforms or no
        element-meta (mesh-only / pre-bridge archives) — never raises
        for a well-formed but enrichment-light file.
        """
        import numpy as np

        transforms_grp = self._f.get("opensees/transforms")
        meta_grp = self._f.get("opensees/element_meta")
        if transforms_grp is None or meta_grp is None:
            return {}

        # emitted geomTransf tag -> vecxz (3,)
        tag_to_vecxz: dict[int, Any] = {}
        for tname in transforms_grp:
            arr = self.transform_arrays(tname)
            vec = arr.get("per_element_vecxz")
            tags = arr.get("per_element_emitted_tag")
            if vec is None or tags is None:
                continue
            vec = np.asarray(vec, dtype=np.float64).reshape(-1, 3)
            tags = np.asarray(tags).reshape(-1)
            for row in range(min(len(tags), len(vec))):
                tag_to_vecxz[int(tags[row])] = vec[row].copy()
        if not tag_to_vecxz:
            return {}

        ndm = int(self.meta().get("ndm", 3) or 3)

        # h5_reader lives in opensees.emitter, so it may consult the
        # element vocabulary directly (the viewer cannot — that's why
        # this join lives here; see ADR 0014 / viewer-integration.md).
        from apeGmsh.opensees._element_capabilities import _ELEM_REGISTRY

        out: dict[int, Any] = {}
        for type_token in meta_grp:
            idx = _transf_arg_tail_index(type_token, ndm, _ELEM_REGISTRY)
            if idx is None:
                continue
            try:
                arrays = self.element_meta_arrays(type_token)
            except KeyError:
                continue
            args = arrays.get("args")
            fem_eids = arrays.get("fem_eids")
            if args is None or fem_eids is None:
                continue
            args = np.asarray(args, dtype=np.float64)
            fem_eids = np.asarray(fem_eids).reshape(-1)
            if args.ndim != 2 or idx >= args.shape[1]:
                continue
            for row in range(min(args.shape[0], len(fem_eids))):
                fem_eid = int(fem_eids[row])
                if fem_eid < 0:        # -1: emitted outside a bridge fan-out
                    continue
                raw = args[row, idx]
                if not np.isfinite(raw):
                    continue
                vec = tag_to_vecxz.get(int(round(float(raw))))
                if vec is not None:
                    out[fem_eid] = vec
        return out

    # -- Neutral-zone reads (Phase 8.5) ---------------------------------

    def nodes(self) -> dict[str, Any]:
        """Return ``{ids, coords}`` for ``/nodes`` (empty dict if absent).

        ``ids`` is a 1-D int64 array; ``coords`` is ``(N, 3)`` float64.
        """
        g = self._f.get("nodes")
        if g is None:
            return {}
        return {
            "ids": g["ids"][:],
            "coords": g["coords"][:],
        }

    def element_arrays(self, type_name: str) -> dict[str, Any]:
        """Return ``{ids, connectivity}`` for one ``/elements/{type}`` group.

        Companion to :meth:`elements` (which exposes only attrs).
        Raises ``KeyError`` if the type group is missing.
        """
        sub = self._f[f"elements/{type_name}"]
        out: dict[str, Any] = {"ids": sub["ids"][:]}
        if "connectivity" in sub:
            out["connectivity"] = sub["connectivity"][:]
        return out

    def physical_groups(self) -> dict[str, dict[str, Any]]:
        """Return root-level ``/physical_groups`` shape.

        Each entry is ``{dim, tag, name, node_ids, node_coords,
        element_ids?}`` — ``element_ids`` is present only for dim>=1
        groups.  Empty dict if ``/physical_groups`` is absent.
        """
        return self._read_named_index("physical_groups")

    def labels(self) -> dict[str, dict[str, Any]]:
        """Return root-level ``/labels`` shape (same fields as :meth:`physical_groups`)."""
        return self._read_named_index("labels")

    def mesh_selections(self) -> dict[str, dict[str, Any]]:
        """Return root-level ``/mesh_selections`` shape.

        Same fields as :meth:`physical_groups` / :meth:`labels` —
        post-mesh selection sets (``fem.mesh_selection``) the broker
        captures at ``get_fem_data()`` time.  Schema 2.4.0 addition
        (Phase 8.7 commit 2); empty dict for pre-2.4.0 files.
        """
        return self._read_named_index("mesh_selections")

    def constraints(self) -> dict[str, Any]:
        """Return ``{kind: compound_array}`` for ``/constraints/{kind}``.

        Each value is the raw compound h5py dataset content (the
        symmetric ``target_kind`` / ``target`` / ``payload_kind`` /
        ``payload`` rows from :mod:`apeGmsh.mesh._record_h5`).  Empty
        dict if no ``/constraints`` group is present.
        """
        g = self._f.get("constraints")
        if g is None:
            return {}
        return {kind: g[kind][:] for kind in g}

    def loads(self) -> dict[str, dict[str, Any]]:
        """Return ``{kind: {pattern: compound_array}}`` for ``/loads``.

        Kinds are ``"nodal"``, ``"element"``, ``"sp"`` — each maps to
        a per-pattern dict of compound arrays.  Empty dict if no
        ``/loads`` group is present.
        """
        g = self._f.get("loads")
        if g is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for kind in g:
            sub = g[kind]
            out[kind] = {pat: sub[pat][:] for pat in sub}
        return out

    def masses(self) -> Any:
        """Return the ``/masses`` compound array, or ``None`` if absent."""
        ds = self._f.get("masses")
        if ds is None:
            return None
        return ds[:]

    def _read_named_index(
        self, group_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Walk a root-level named index (``physical_groups`` or ``labels``).

        Each child group becomes one entry in the returned dict.
        """
        parent = self._f.get(group_name)
        if parent is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for sub_name in parent:
            sub = parent[sub_name]
            entry: dict[str, Any] = _attrs_as_dict(sub)
            if "node_ids" in sub:
                entry["node_ids"] = sub["node_ids"][:]
            if "node_coords" in sub:
                entry["node_coords"] = sub["node_coords"][:]
            if "element_ids" in sub:
                entry["element_ids"] = sub["element_ids"][:]
            out[sub_name] = entry
        return out

    def _group_attrs_map(
        self, group: str,
    ) -> dict[str, dict[str, Any]]:
        g = self._f.get(group)
        if g is None:
            return {}
        return {name: _attrs_as_dict(g[name]) for name in g}

    # -- Validation ------------------------------------------------------

    def validate(self) -> list[str]:
        """Walk the file checking every schema invariant we can mechanically
        verify. Returns a list of violation messages — empty means valid.

        Checked invariants:

        * ``/meta`` has all the required attrs.
        * ``schema_version`` parses as semver.
        * Material / section names match their internal ``tag`` attr
          (i.e. ``Steel02_3``'s ``tag`` attr is 3) — this is the H5
          emitter's convention and a fast inconsistency check.
        * Patches' ``material_ref`` paths resolve to existing groups.
        * Fibers' ``material_ref`` paths resolve.
        * Pattern ``series_ref`` paths resolve.
        """
        errors: list[str] = []
        errors.extend(self._validate_meta())
        errors.extend(self._validate_materials_naming())
        errors.extend(self._validate_section_refs())
        errors.extend(self._validate_pattern_refs())
        return errors

    def _validate_meta(self) -> list[str]:
        out: list[str] = []
        meta = self._f.get("meta")
        if meta is None:
            out.append("missing /meta group")
            return out
        required = (
            "schema_version", "apeGmsh_version", "created_iso",
            "ndm", "ndf", "snapshot_id", "model_name",
        )
        for attr in required:
            if attr not in meta.attrs:
                out.append(f"/meta missing required attr {attr!r}")
        return out

    def _validate_materials_naming(self) -> list[str]:
        out: list[str] = []
        materials = self._f.get("opensees/materials")
        if materials is None:
            return out
        for family in materials:
            fam_group = materials[family]
            for name in fam_group:
                g = fam_group[name]
                attr_tag = int(g.attrs.get("tag", -1))
                expected_suffix = f"_{attr_tag}"
                if not name.endswith(expected_suffix):
                    out.append(
                        f"/opensees/materials/{family}/{name}: name does not "
                        f"end with _{attr_tag} (tag attr says {attr_tag})"
                    )
        return out

    def _validate_section_refs(self) -> list[str]:
        out: list[str] = []
        sections = self._f.get("opensees/sections")
        if sections is None:
            return out
        for name in sections:
            sec = sections[name]
            patches = sec.get("patches")
            if patches is not None:
                for row in patches[:]:
                    ref = _decode_bytes(row["material_ref"])
                    if ref and ref not in self._f:
                        out.append(
                            f"/opensees/sections/{name}/patches: material_ref "
                            f"{ref!r} not found in file"
                        )
            fibers = sec.get("fibers")
            if fibers is not None:
                for row in fibers[:]:
                    ref = _decode_bytes(row["material_ref"])
                    if ref and ref not in self._f:
                        out.append(
                            f"/opensees/sections/{name}/fibers: material_ref "
                            f"{ref!r} not found in file"
                        )
        return out

    def _validate_pattern_refs(self) -> list[str]:
        out: list[str] = []
        patterns = self._f.get("opensees/patterns")
        if patterns is None:
            return out
        for name in patterns:
            p = patterns[name]
            ref = p.attrs.get("series_ref")
            if ref:
                ref_str = ref if isinstance(ref, str) else str(ref)
                if ref_str not in self._f:
                    out.append(
                        f"/opensees/patterns/{name}: series_ref {ref_str!r} "
                        "not found in file"
                    )
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attrs_as_dict(group: Any) -> dict[str, Any]:
    """Convert an h5py attrs view to a plain dict, decoding bytes."""
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        out[key] = _decode_bytes(value)
    return out


def _decode_bytes(v: Any) -> Any:
    """Decode bytes / arrays-of-bytes to UTF-8 strings. Leave others alone."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    # h5py sometimes returns 0-D arrays for scalar string attrs.
    try:
        import numpy as np
        if isinstance(v, np.ndarray) and v.dtype.kind in ("O", "S"):
            if v.shape == ():
                item = v.item()
                if isinstance(item, bytes):
                    return item.decode("utf-8", errors="replace")
                return item
    except ImportError:  # pragma: no cover  - numpy is a hard dep
        pass
    return v


#: Beam-column element types whose geomTransf tag is the first
#: positional arg after connectivity (``args`` tail index 0).  These
#: are absent from
#: :data:`apeGmsh.opensees._element_capabilities._ELEM_REGISTRY`
#: (which only carries the scalar-property beam forms); the position is
#: a stable OpenSees convention:
#: ``element forceBeamColumn $ele $iN $jN $transfTag $integrationTag``.
_FORCE_DISP_BEAMS: frozenset = frozenset({"forceBeamColumn", "dispBeamColumn"})


def _transf_arg_tail_index(
    type_token: str, ndm: int, registry: dict,
) -> "int | None":
    """Return the ``args``-tail index of the geomTransf tag, or ``None``.

    ``args`` is the element's positional list *after* the connectivity
    prefix is dropped (h5-schema.md ``/opensees/element_meta``).  In the
    vocabulary the connectivity prefix is the leading ``"nodes"`` slot,
    so the tail index is ``slots.index("transfTag") - 1``.  ``None``
    means "this element type carries no geomTransf" (solids, trusses,
    shells) — the caller skips it.
    """
    if type_token in _FORCE_DISP_BEAMS:
        return 0
    spec = registry.get(type_token)
    if spec is None:
        return None
    if ndm == 2:
        slots = getattr(spec, "slots_2d", None)
    elif ndm == 3:
        slots = getattr(spec, "slots_3d", None)
    else:
        slots = None
    if slots is None:
        slots = (
            getattr(spec, "slots_3d", None)
            or getattr(spec, "slots_2d", None)
            or getattr(spec, "slots", None)
        )
    if not slots or "transfTag" not in slots:
        return None
    return slots.index("transfTag") - 1


# Re-export under the builtin name shadow so users can write
# ``h5_reader.open(...)``. Keep ``builtins.open`` accessible via a
# different alias inside this module if ever needed.
_ = builtins.open  # silence linters about the shadow
