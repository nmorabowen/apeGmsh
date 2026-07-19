"""``/opensees/computed_sections`` sidecar — ComputedSection provenance
(ADR 0078 Amendment A1).

The resolved numbers of a ``ComputedSection`` already persist through
the ordinary section capture (``H5Emitter.section()`` →
``/opensees/sections/*`` — the emitted deck line is indistinguishable
from a hand-typed section, by design).  This sidecar adds the missing
**provenance**: which sections were analyzer-derived, from which
analyzer, under which materials/policy/reference moduli.  It answers
"where did these numbers come from", never "re-run the solve" — the
analyzer mesh is deliberately NOT persisted (the authoring script is
the reproducible source).

Layout (only written when at least one ``ComputedSection`` emitted)::

    /opensees/computed_sections
        tag            (int64)       — joins to /opensees/sections/*
        analyzer_name  (vlen utf-8)  — the analyzer handle ("" if unnamed)
        payload        (vlen utf-8)  — JSON provenance blob (kind,
                                       reference moduli / GJ, policy,
                                       part/element counts, materials)

``computed_sections`` is in
:data:`apeGmsh.opensees._internal.lineage.MODEL_HASH_EXCLUDED_CHILDREN`:
provenance metadata, not authored model state — same carve-out as
``names``.  Empty ⇒ no group, preserving byte-equivalence of files
without ``ComputedSection``s (mirrors ``write_names_into``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import h5py


__all__ = [
    "COMPUTED_SECTIONS_GROUP",
    "computed_section_payload",
    "read_computed_sections",
    "write_computed_sections_into",
]


#: Group name under ``/opensees`` holding the provenance table.
COMPUTED_SECTIONS_GROUP: str = "computed_sections"


def computed_section_payload(prim: object) -> str:
    """Build the deterministic JSON provenance blob for one
    ``ComputedSection`` (sorted keys; resolved values).

    Called after a successful emission, so the analyzer's memoized
    analyses are already cached — the reference-moduli / ``GJ``
    resolution below re-reads cached values, it never re-solves.
    """
    import json

    from apeGmsh.opensees.section.computed import ComputedSection

    assert isinstance(prim, ComputedSection)
    analysis = prim.analysis
    materials = {
        pg: {
            "E": mat.E,
            "nu": mat.nu,
            "G": mat.G,
            "fy": mat.fy,
        }
        for pg, mat in analysis.materials.items()
    }
    payload: dict[str, object] = {
        "kind": prim.kind,
        "disconnected": analysis.disconnected,
        "geometric_only": analysis.geometric_only,
        "n_parts": analysis.n_parts,
        "n_elements": analysis._snapshot.n_elements,
        "materials": materials,
    }
    if prim.kind == "elastic":
        from apeGmsh.sections._lowering import lower_to_elastic

        params = lower_to_elastic(analysis, E=prim.E, G=prim.G)
        payload["ndm"] = prim.ndm
        payload["E_ref"] = params.E
        payload["G_ref"] = params.G
    else:
        gj = prim.GJ if prim.GJ is not None else analysis.warping().GJ
        payload["GJ"] = gj
        assert prim.fibers is not None
        payload["fiber_pgs"] = sorted(prim.fibers)
    return json.dumps(payload, sort_keys=True)


def write_computed_sections_into(
    f: "h5py.File",
    records: "Sequence[tuple[int, str, str]]",
    *,
    opensees_root: str = "opensees",
) -> None:
    """Write ``(tag, analyzer_name, payload)`` rows under
    ``/opensees/computed_sections``.

    No-op when ``records`` is empty — the group is not created, so a
    model with no ``ComputedSection``s is byte-identical to the
    pre-sidecar layout.  Records are written in caller order (the
    bridge sorts by tag for determinism).
    """
    if not records:
        return

    import h5py

    grp = f.require_group(opensees_root)
    if COMPUTED_SECTIONS_GROUP in grp:
        del grp[COMPUTED_SECTIONS_GROUP]
    g = grp.create_group(COMPUTED_SECTIONS_GROUP)

    str_dt = h5py.string_dtype(encoding="utf-8")
    g.create_dataset(
        "tag", data=[int(t) for t, _, _ in records], dtype="int64"
    )
    g.create_dataset(
        "analyzer_name", data=[n for _, n, _ in records], dtype=str_dt
    )
    g.create_dataset(
        "payload", data=[p for _, _, p in records], dtype=str_dt
    )


def read_computed_sections(
    path: str,
    *,
    opensees_root: str = "/opensees",
) -> tuple[tuple[int, str, str], ...]:
    """Read the ``/opensees/computed_sections`` provenance table.

    Returns an empty tuple when the file has no bridge zone or no
    sidecar group (the common case, and every pre-2.20.0 file).  Uses
    the ``name in group`` probe per the repo's h5py optional-child
    convention — never ``Group.get``.
    """
    import h5py

    root = opensees_root.strip("/")
    with h5py.File(path, "r") as f:
        if root not in f:
            return ()
        grp = f[root]
        if COMPUTED_SECTIONS_GROUP not in grp:
            return ()
        g = grp[COMPUTED_SECTIONS_GROUP]
        raw_tags = g["tag"][()]
        raw_names = g["analyzer_name"][()]
        raw_payloads = g["payload"][()]

    out: list[tuple[int, str, str]] = []
    for tg, nm, pl in zip(raw_tags, raw_names, raw_payloads):
        out.append((int(tg), _as_str(nm), _as_str(pl)))
    return tuple(out)


def _as_str(value: object) -> str:
    """Decode an h5py vlen-string element to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
