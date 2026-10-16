"""Stamp ``INFO/PROVENANCE`` into a closed ``.ladruno`` (ADR 0122 D3).

A ``.ladruno`` pairs with the ``model.h5`` that produced it through six
attributes apeGmsh writes into the recorder's empty ``INFO/PROVENANCE``
group after the recorder has closed the file. The fork blesses those
consumer writes in its schema, ``ladruno_schema_v1.md`` §7b.1 (fork doc
PR nmorabowen/OpenSees #992); ADR 0122 D4 says the fork owns the format
and apeGmsh owns only these attributes.

==========================  ===============================================
Attribute                   Value (scalar, variable-length UTF-8 string)
==========================  ===============================================
``model_hash``              ``<model_h5>/meta/lineage@model_hash``, byte-equal
``fem_hash``                ``<model_h5>/meta/lineage@fem_hash``, byte-equal
``stage_names``             JSON array: the program's stage names, written
                            only when the file's ``MODEL_STAGE`` count and
                            ``KIND`` s match the program's stages
``element_index_space``     ``"fem"``, or absent (unverified)
``unsupported``             JSON array of the lossy categories; ``"[]"``
``writer``                  the apeGmsh component and version
==========================  ===============================================

Only lowercase names are written; UPPERCASE names in ``PROVENANCE`` are
fork-reserved and are never touched. Nothing outside ``INFO/PROVENANCE``
is written.

It lives beside the ``.ladruno`` reader because it encodes the fork's
format (the fork-token ratchet exempts ``readers/_ladruno*``);
:mod:`apeGmsh.results.writers` re-exports the public names.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import h5py

    from ...opensees.opensees_model import OpenSeesModel

__all__ = [
    "PROVENANCE_ATTRS",
    "ProvenanceStampRefused",
    "StageNamesOmittedWarning",
    "stamp_provenance",
]

#: The attributes this module owns, in ``INFO/PROVENANCE`` (§7b.1).
PROVENANCE_ATTRS: tuple[str, ...] = (
    "model_hash", "fem_hash", "stage_names", "element_index_space",
    "unsupported", "writer",
)

_PROVENANCE = "INFO/PROVENANCE"
_STAGE_PREFIX = "MODEL_STAGE["


class ProvenanceStampRefused(RuntimeError):
    """:func:`stamp_provenance` refused, and wrote nothing to any part.

    Raised before the first write when a part file is open or locked
    (another process, a viewer, an ``h5py`` session, or a file that was
    never closed cleanly), is not a ``.ladruno`` this reader supports,
    lacks ``INFO/PROVENANCE``; when the model archive has no complete
    ``/meta/lineage``; when the parts of a partitioned set disagree on
    their MODEL; or when a different ``model_hash`` is already stamped
    and ``overwrite`` is False.

    §7b.1 rule 3 has a consumer that cannot open the file for writing
    *skip and warn*. That policy belongs to the caller: the resolver
    (ADR 0122 slice 5) catches this class, warns, and leaves the file
    unstamped. Nothing else is caught there.
    """


class StageNamesOmittedWarning(UserWarning):
    """``stage_names`` was left out of a part's stamp.

    §7b.1 writes ``stage_names`` only when the file's ``MODEL_STAGE``
    groups match the program's stages in count and in kind; otherwise the
    pairing is positional guesswork. The message names the part and the
    mismatch. The rest of the stamp is written.
    """


def stamp_provenance(
    path: "str | Path",
    model_h5: "str | Path",
    *,
    stage_names: "Optional[Sequence[str]]" = None,
    unsupported: "Sequence[str]" = (),
    writer: "Optional[str]" = None,
    overwrite: bool = False,
) -> list[Path]:
    """Write the ADR 0122 D3 pairing attributes into a closed ``.ladruno``.

    ``path`` is one ``.ladruno``, or any ``<stem>.part-<N>.ladruno`` of a
    partitioned run: every part of the set is stamped (§7b.1, partitioned
    sets). ``model_h5`` is the bridge-written model archive the run was
    emitted from; ``model_hash`` and ``fem_hash`` are copied from its
    ``/meta/lineage`` byte for byte.

    ``stage_names`` defaults to the archive's ``ops.stage(name=...)``
    names; passing it replaces the names, never the kinds, which always
    come from each program stage's ``analysis`` record. A part whose
    ``MODEL_STAGE`` count or ``KIND`` s differ from the program's gets no
    ``stage_names`` and :class:`StageNamesOmittedWarning`; a program with
    no stages has nothing to name and stays silent.

    ``element_index_space`` is ``"fem"`` only when the part records at
    least one element and every element id it records (``MODEL/ELEMENTS``
    connectivity and ``ON_ELEMENTS`` bucket ids) is an element the archive
    pairs with a FEM id equal to that id. An element the bridge minted
    without a FEM id (``fem_eid`` -1), or one the archive does not know,
    leaves it absent, which a reader treats as unverified (#2079).

    ``unsupported`` lists the lossy categories the producer could not carry
    (``"[]"`` when none). ``writer`` defaults to ``"apeGmsh <version>"``.

    Re-stamping with identical values writes nothing. A part already
    stamped with another ``model_hash`` refuses unless ``overwrite=True``;
    a re-stamp removes an owned attribute the new stamp leaves out.

    Every check runs on every part before the first write; a refusal
    raises :class:`ProvenanceStampRefused` and leaves every file
    byte-identical. A failure *during* the writes (disk full) can leave a
    set stamped in part, which §7b.1 reads as unpaired.

    Returns the part paths, in partition order.
    """
    names_override = _string_list(stage_names, "stage_names")
    lossy = _string_list(unsupported, "unsupported") or []
    writer_text = _writer_text(writer)

    parts = _discover_parts(Path(path))
    _refuse_if_open_here(parts)
    model_hash, fem_hash = _lineage_hashes(Path(model_h5))
    program_names, program_kinds, kind_error, pairing = _program_facts(
        Path(model_h5),
    )
    if names_override is not None:
        program_names = names_override
    _check_part_set(parts)

    plans: list[tuple[Path, dict[str, str]]] = []
    for part in parts:
        facts = _read_part(part)
        wanted: dict[str, str] = {
            "model_hash": model_hash,
            "fem_hash": fem_hash,
            "unsupported": _json(lossy),
            "writer": writer_text,
        }
        names = _stage_names_for(
            part, facts.kinds, program_names, program_kinds, kind_error,
        )
        if names is not None:
            wanted["stage_names"] = _json(names)
        if _is_fem_index_space(facts.element_tags, pairing):
            wanted["element_index_space"] = "fem"
        stamped = facts.owned.get("model_hash")
        if stamped is not None and stamped != model_hash and not overwrite:
            raise ProvenanceStampRefused(
                f"{part}: INFO/PROVENANCE is already stamped with "
                f"model_hash={stamped!r}, not {model_hash!r} from "
                f"{model_h5}. Pass overwrite=True to re-pair the file."
            )
        if facts.owned != wanted or not facts.all_vlen_utf8:
            plans.append((part, wanted))

    if not plans:
        return parts
    handles = _open_for_writing([p for p, _ in plans])
    try:
        for (part, wanted), h5 in zip(plans, handles):
            _write(h5[_PROVENANCE], wanted)
    finally:
        for h5 in handles:
            h5.close()
    return parts


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _string_list(
    value: "Optional[Sequence[str]]", name: str,
) -> "Optional[list[str]]":
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"stamp_provenance: {name} must be a sequence of "
                        f"str, not one {type(value).__name__}.")
    out = list(value)
    bad = [v for v in out if not isinstance(v, str)]
    if bad:
        raise TypeError(f"stamp_provenance: {name} holds non-str items "
                        f"{bad[:5]!r}.")
    return out


def _writer_text(writer: "Optional[str]") -> str:
    if writer is None:
        import apeGmsh
        return f"apeGmsh {apeGmsh.__version__}"
    if not isinstance(writer, str) or not writer.strip():
        raise TypeError("stamp_provenance: writer must be a non-empty str "
                        f"naming the component and version, got {writer!r}.")
    return writer


def _json(items: "list[str]") -> str:
    return json.dumps(items, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _discover_parts(path: Path) -> list[Path]:
    from ._ladruno_multi import discover_partition_files

    try:
        parts = discover_partition_files(path)
    except ValueError as e:
        raise ProvenanceStampRefused(f"{path}: {e}") from e
    missing = [p for p in parts if not p.is_file()]
    if missing:
        raise ProvenanceStampRefused(f"no such results file: {missing[0]}")
    return parts


def _refuse_if_open_here(parts: list[Path]) -> None:
    """Refuse a part this process already holds open.

    HDF5's file lock stops another process; inside this one, a second
    ``h5py.File`` on an open file reuses the open handle, so a write would
    land under a live ``r+`` session (or a reader's view) unnoticed.
    """
    import os

    import h5py

    held: dict[str, int] = {}
    for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE):
        name = os.path.normcase(os.path.realpath(os.fsdecode(fid.name)))
        held[name] = fid.get_intent()
    for part in parts:
        key = os.path.normcase(os.path.realpath(part))
        if key in held:
            mode = "writing" if held[key] & h5py.h5f.ACC_RDWR else "reading"
            raise ProvenanceStampRefused(
                f"{part} is open for {mode} in this process (an h5py "
                "session or a Results reader); close it first. Nothing "
                "was stamped."
            )


def _lineage_hashes(model_h5: Path) -> tuple[str, str]:
    """``(model_hash, fem_hash)`` from ``/meta/lineage``, exactly as stored."""
    import h5py

    if not model_h5.is_file():
        raise ProvenanceStampRefused(f"no such model archive: {model_h5}")
    with h5py.File(model_h5, "r") as f:
        if "meta" not in f or "lineage" not in f["meta"]:
            raise ProvenanceStampRefused(
                f"{model_h5} has no /meta/lineage: it was not written by "
                "the bridge (ops.h5) or predates the lineage chain, so "
                "there is no model_hash to pair with."
            )
        lin = f["meta/lineage"].attrs
        out = []
        for key in ("model_hash", "fem_hash"):
            raw = lin[key] if key in lin else None
            text = _as_text(raw, f"{model_h5}:/meta/lineage@{key}")
            if not text:
                raise ProvenanceStampRefused(
                    f"{model_h5}: /meta/lineage carries no {key}; a pairing "
                    "needs both model_hash and fem_hash."
                )
            out.append(text)
    return out[0], out[1]


def _as_text(raw: Any, where: str) -> str:
    """An h5py string attribute as ``str``; ``""`` when it is absent."""
    if raw is None:
        return ""
    if isinstance(raw, np.ndarray) and raw.shape == (1,):
        raw = raw[0]
    if isinstance(raw, (bytes, np.bytes_)):
        try:
            return bytes(raw).decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProvenanceStampRefused(
                f"{where} is not UTF-8 text ({e}).") from e
    if isinstance(raw, str):
        return raw
    raise ProvenanceStampRefused(
        f"{where} is a {type(raw).__name__}, not a string attribute.")


def _program_facts(
    model_h5: Path,
) -> "tuple[list[str], list[str], Optional[str], dict[int, int]]":
    """The program's stage names and kinds, and its ops-tag → fem_eid map.

    ``kind_error`` is the reason the kinds are unknown (a stage with no
    ``analysis`` record); stage names are then never written.
    """
    from ...opensees.opensees_model import OpenSeesModel
    from .._bind import _stage_kind

    model: "OpenSeesModel" = OpenSeesModel.from_h5(model_h5)
    stages = list(model.stages())
    names = [str(s.name) for s in stages]
    kinds: list[str] = []
    kind_error: "Optional[str]" = None
    try:
        kinds = [_stage_kind(s, model_path=model_h5) for s in stages]
    except ValueError as e:
        kind_error = str(e)
    pairing = {int(rec.tag): int(rec.fem_eid) for rec in model.elements()}
    return names, kinds, kind_error, pairing


# ---------------------------------------------------------------------------
# The part files
# ---------------------------------------------------------------------------


class _PartFacts:
    __slots__ = ("kinds", "element_tags", "owned", "all_vlen_utf8")

    def __init__(
        self, kinds: "list[Optional[str]]", element_tags: "np.ndarray",
        owned: "dict[str, str]", all_vlen_utf8: bool,
    ) -> None:
        self.kinds = kinds
        self.element_tags = element_tags
        self.owned = owned
        self.all_vlen_utf8 = all_vlen_utf8


def _open_read(path: Path) -> "h5py.File":
    import h5py

    try:
        return h5py.File(path, "r")
    except OSError as e:
        raise ProvenanceStampRefused(
            f"{path} cannot be opened ({e}); it is still being written, "
            "is locked by another process, or was never closed cleanly."
        ) from e


def _stage_groups(h5: "h5py.File") -> list[str]:
    def key(name: str) -> "tuple[int, str]":
        inner = name[len(_STAGE_PREFIX):-1]
        return (int(inner), name) if inner.isdigit() else (1 << 62, name)

    return sorted((k for k in h5.keys() if k.startswith(_STAGE_PREFIX)),
                  key=key)


def _read_part(path: Path) -> _PartFacts:
    import h5py

    with _open_read(path) as h5:
        if "INFO" not in h5 or "PROVENANCE" not in h5["INFO"]:
            raise ProvenanceStampRefused(
                f"{path} has no INFO/PROVENANCE group. The fork's recorder "
                "creates it empty at open (fork schema v1 §7b); a file "
                "without it predates that, so there is nowhere to stamp."
            )
        prov = h5[_PROVENANCE]
        if not isinstance(prov, h5py.Group):
            raise ProvenanceStampRefused(
                f"{path}: INFO/PROVENANCE is not a group.")
        owned: dict[str, str] = {}
        all_vlen = True
        for name in PROVENANCE_ATTRS:
            if name not in prov.attrs:
                continue
            dtype = prov.attrs.get_id(name).dtype
            info = h5py.check_string_dtype(dtype)
            shape = prov.attrs.get_id(name).shape
            if info is None or info.length is not None \
                    or info.encoding != "utf-8" or shape != ():
                all_vlen = False
            owned[name] = _as_text(prov.attrs[name], f"{path}:{_PROVENANCE}@{name}")
        kinds: "list[Optional[str]]" = []
        tags = [np.empty(0, dtype=np.int64)]
        for gname in _stage_groups(h5):
            grp = h5[gname]
            kinds.append(
                _as_text(grp.attrs["KIND"], f"{path}:{gname}@KIND")
                if "KIND" in grp.attrs else None
            )
            tags.extend(_stage_element_tags(grp))
    return _PartFacts(kinds, np.unique(np.concatenate(tags)), owned, all_vlen)


def _stage_element_tags(stage: "h5py.Group") -> "list[np.ndarray]":
    """Element ids one stage records: ``MODEL/ELEMENTS`` and bucket ``ID`` s."""
    import h5py

    out: list[np.ndarray] = []
    if "MODEL" in stage and "ELEMENTS" in stage["MODEL"]:
        for grp in stage["MODEL/ELEMENTS"].values():
            if isinstance(grp, h5py.Group) and "CONNECTIVITY" in grp:
                conn = grp["CONNECTIVITY"]
                if conn.ndim == 2 and conn.shape[0]:
                    out.append(np.asarray(conn[:, 0], dtype=np.int64))
    if "RESULTS" in stage and "ON_ELEMENTS" in stage["RESULTS"]:
        for token in stage["RESULTS/ON_ELEMENTS"].values():
            if not isinstance(token, h5py.Group):
                continue
            for bucket in token.values():
                if isinstance(bucket, h5py.Group) and "ID" in bucket:
                    out.append(np.asarray(bucket["ID"][...],
                                          dtype=np.int64).ravel())
    return out


def _check_part_set(parts: list[Path]) -> None:
    """Refuse a file the reader rejects, or parts that disagree on MODEL.

    The readers' own checks run first: ``GENERATOR`` and ``FORMAT_VERSION``
    on each file; on a set, the partition manifest, the run identity, and
    stage counts, kinds and time vectors. A node two parts share must sit
    at the same coordinates, and an element two parts both list must have
    the same connectivity.
    """
    from ._ladruno import LadrunoReader
    from ._ladruno_multi import LadrunoMultiPartitionReader

    # Each file on its own first: the set reader is built only from files
    # that open, so a refused part never leaves a half-built set reader.
    for part in parts:
        _with_reader(LadrunoReader, part, where=str(part))
    if len(parts) > 1:
        _with_reader(LadrunoMultiPartitionReader, parts, where=str(parts[0]))
        _check_shared_model(parts)


def _with_reader(cls: "Any", arg: "Any", *, where: str) -> None:
    """Construct a reader (its checks run in ``__init__``), then close it."""
    try:
        reader = cls(arg)
    except OSError as e:
        raise ProvenanceStampRefused(
            f"{where} cannot be opened ({e}); it is still being written, is "
            "locked by another process, or was never closed cleanly."
        ) from e
    except ValueError as e:
        raise ProvenanceStampRefused(str(e)) from e
    reader.close()


def _check_shared_model(parts: list[Path]) -> None:
    per_part: list[list[tuple[dict[int, np.ndarray], dict[str, dict[int, tuple[int, ...]]]]]] = []
    for part in parts:
        with _open_read(part) as h5:
            stages = []
            for gname in _stage_groups(h5):
                grp = h5[gname]
                nodes: dict[int, np.ndarray] = {}
                elements: dict[str, dict[int, tuple[int, ...]]] = {}
                if "MODEL" in grp:
                    model = grp["MODEL"]
                    if "NODES" in model and "ID" in model["NODES"]:
                        ids = np.asarray(model["NODES/ID"][...]).ravel()
                        xyz = np.asarray(model["NODES/COORDINATES"][...],
                                         dtype=np.float64)
                        nodes = {int(n): xyz[i] for i, n in enumerate(ids)}
                    if "ELEMENTS" in model:
                        for bname, b in model["ELEMENTS"].items():
                            if "CONNECTIVITY" not in b:
                                continue
                            table = np.asarray(b["CONNECTIVITY"][...])
                            elements[bname] = {
                                int(r[0]): tuple(int(v) for v in r[1:])
                                for r in table.reshape(table.shape[0], -1)
                            }
                stages.append((nodes, elements))
            per_part.append(stages)
    for k in range(len(per_part[0])):
        seen_nodes: dict[int, tuple[int, np.ndarray]] = {}
        seen_elems: dict[tuple[str, int], tuple[int, tuple[int, ...]]] = {}
        for p, stages in enumerate(per_part):
            nodes, elements = stages[k]
            for nid, xyz in nodes.items():
                if nid in seen_nodes:
                    q, other = seen_nodes[nid]
                    scale = max(1.0, float(np.max(np.abs(other))))
                    if other.shape != xyz.shape or not np.allclose(
                            other, xyz, rtol=0.0, atol=1e-12 * scale):
                        raise ProvenanceStampRefused(
                            f"{parts[q].name} and {parts[p].name} disagree "
                            f"on their MODEL: stage {k} places node {nid} at "
                            f"{other.tolist()} and {xyz.tolist()}. The parts "
                            "are not from one run."
                        )
                else:
                    seen_nodes[nid] = (p, xyz)
            for bname, rows in elements.items():
                for eid, conn in rows.items():
                    key = (bname, eid)
                    if key in seen_elems and seen_elems[key][1] != conn:
                        q = seen_elems[key][0]
                        raise ProvenanceStampRefused(
                            f"{parts[q].name} and {parts[p].name} disagree "
                            f"on their MODEL: stage {k} element {eid} "
                            f"({bname}) connects {seen_elems[key][1]} and "
                            f"{conn}. The parts are not from one run."
                        )
                    seen_elems.setdefault(key, (p, conn))


# ---------------------------------------------------------------------------
# The decisions
# ---------------------------------------------------------------------------


def _stage_names_for(
    part: Path, file_kinds: "list[Optional[str]]",
    names: "list[str]", kinds: "list[str]", kind_error: "Optional[str]",
) -> "Optional[list[str]]":
    if not names:
        return None

    def omit(reason: str) -> None:
        warnings.warn(StageNamesOmittedWarning(
            f"{part.name}: stage_names not stamped: {reason}"), stacklevel=4)

    if kind_error is not None:
        omit(f"the program's stage kinds are unknown ({kind_error})")
        return None
    if len(names) != len(kinds):
        omit(f"{len(names)} stage names for {len(kinds)} program stages")
        return None
    if len(file_kinds) != len(names):
        omit(f"the file holds {len(file_kinds)} MODEL_STAGE groups and the "
             f"program {len(names)} stages ({names})")
        return None
    if list(file_kinds) != list(kinds):
        omit(f"the file's stage KINDs {file_kinds} differ from the "
             f"program's {kinds}")
        return None
    return list(names)


def _is_fem_index_space(tags: "np.ndarray", pairing: "dict[int, int]") -> bool:
    if tags.size == 0:
        return False
    return all(int(t) in pairing and pairing[int(t)] == int(t) for t in tags)


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------


def _open_for_writing(paths: list[Path]) -> "list[h5py.File]":
    """Open every part ``r+`` or none: a lock on one refuses them all."""
    import h5py

    handles: list[h5py.File] = []
    try:
        for p in paths:
            handles.append(h5py.File(p, "r+"))
    except OSError as e:
        for h in handles:
            h.close()
        raise ProvenanceStampRefused(
            f"{paths[len(handles)]} will not open for writing ({e}): "
            "another process (OpenSees, a viewer, an h5py session) holds "
            "it open. Nothing was stamped."
        ) from e
    return handles


def _write(prov: "h5py.Group", wanted: "dict[str, str]") -> None:
    import h5py

    vlen = h5py.string_dtype("utf-8")
    for name in PROVENANCE_ATTRS:
        if name in wanted:
            prov.attrs.create(name, data=wanted[name], dtype=vlen)
        elif name in prov.attrs:
            del prov.attrs[name]
