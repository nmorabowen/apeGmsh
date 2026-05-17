"""Shared symbolic-target resolver for Loads + Masses (selection-unification S1).

``LoadsComposite._resolve_target`` and ``MassesComposite._resolve_target``
were a byte-for-byte clone of one another (only the human-facing error
wording differed).  This module holds the **one** shared engine; both
composites are thin facades that delegate here.

This is a pure de-duplication: :func:`resolve_target` reproduces the
prior behaviour byte-identically for every input — including the
mesh-selection sentinel ``[("__ms__", dim, tag)]`` and the
``expected_dim`` scoping (15+ internal call sites depend on the exact
return shape).  The only per-composite difference — two distinct error
strings (Loads says ``"Target ..."`` / ``"... this load requires
..."``; Masses says ``"Mass target ..."`` / ``"... this mass requires
..."``) — is threaded through explicit parameters so each composite's
message is emitted verbatim.

Leaf invariant (selection-unification §3 keystone / FP-1): this module
imports only ``gmsh`` + stdlib at module level.  It must NOT import
``apeGmsh.mesh`` / ``apeGmsh.viz`` / ``apeGmsh.results`` — doing so
would add a new eager cross-package edge and trip
``tests/test_import_dag_polarity.py``.  The session/composite ``parent``
(``self._parent`` in the callers) is passed in explicitly so this stays
package-clean; the only intra-``core`` symbol it needs
(``apeGmsh.core.Labels.add_prefix``) is imported deferred inside the
function exactly as the original methods did.
"""
from __future__ import annotations

import gmsh


def resolve_target(parent, target, source: str = "auto", *,
                   expected_dim: int | None = None,
                   not_found_prefix: str,
                   noun: str) -> list:
    """Resolve a target identifier to a list of ``(dim, tag)`` pairs.

    Lookup order (for ``source="auto"``):
        1. ``list[tuple[int, int]]``  -> as-is
        2. mesh selection name        -> entities from g.mesh_selection
        3. label name (Tier 1)        -> ``_label:``-prefixed PG
        4. physical group name (Tier 2)-> user PG
        5. part label                 -> entities from g.parts.instances

    When ``source="pg"`` only step 4 is tried.
    When ``source="label"`` only step 3 is tried.

    A label may span several dimensions and a part owns entities
    across dims; both are returned as the **union** of every
    matching ``(dim, tag)`` — never the first dim only (silent
    truncation otherwise).  ``expected_dim`` — the dimension the
    calling load/mass semantically needs (1 line, 2 surface, 3
    volume) — scopes a name-resolved target to that dimension and
    **fails loud** if the name resolved to entities but none at
    ``expected_dim`` (a wrong-dimension reference, or a multi-dim
    label that doesn't cover it).  Raw ``(dim, tag)`` lists and
    mesh selections bypass this scoping.

    ``parent`` is the session/composite that owns ``.mesh_selection``
    and ``.parts`` (the original methods reached ``self._parent``).
    ``not_found_prefix`` is the exact text preceding
    ``" {target!r} not found ..."`` in the not-found ``KeyError``
    (``"Target"`` for loads, ``"Mass target"`` for masses).  ``noun``
    is the lowercase noun used in the wrong-dimension ``ValueError``
    (``"load"`` / ``"mass"``).
    """
    # 1. Raw DimTag list — explicit user intent, returned verbatim.
    if isinstance(target, (list, tuple)) and len(target) > 0 \
            and isinstance(target[0], (list, tuple)):
        return [(int(d), int(t)) for d, t in target]

    if not isinstance(target, str):
        raise TypeError(
            f"target must be a string label or list of (dim, tag), "
            f"got {type(target).__name__}"
        )

    # 2. Mesh selection name (only in auto mode) — sentinel,
    #    bypasses expected_dim scoping (consumers special-case it).
    if source == "auto":
        ms = getattr(parent, "mesh_selection", None)
        if ms is not None and hasattr(ms, "_sets"):
            for (dim, tag), info in ms._sets.items():
                if info.get("name") == target:
                    return [("__ms__", dim, tag)]

    out: list[tuple[int, int]] = []

    # 3. Label name (Tier 1 — _label: prefixed PG).  A label may
    #    span dims — collect the union of every matching dim.
    if source in ("auto", "label"):
        try:
            from apeGmsh.core.Labels import add_prefix
            prefixed = add_prefix(target)
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                try:
                    if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                        ents = gmsh.model.getEntitiesForPhysicalGroup(
                            pg_dim, pg_tag)
                        out.extend((pg_dim, int(t)) for t in ents)
                except Exception:
                    pass
        except Exception:
            pass

    # 4. Physical group name (Tier 2 — user PGs).  A PG name maps
    #    to a single dimension; fail loud if a legacy model
    #    carries the name at several dims rather than silently
    #    binding the load/mass to whichever dim is found first.
    if not out and source in ("auto", "pg"):
        pg_matches: list[tuple[int, int]] = []
        try:
            for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                try:
                    if gmsh.model.getPhysicalName(pg_dim, pg_tag) == target:
                        pg_matches.append((pg_dim, pg_tag))
                except Exception:
                    pass
        except Exception:
            pass
        if pg_matches:
            pg_dims = {d for d, _ in pg_matches}
            if len(pg_dims) > 1:
                raise ValueError(
                    f"Physical group {target!r} exists at multiple "
                    f"dimensions {sorted(pg_dims)}. Multi-dimensional "
                    f"physical groups are not supported; assign one "
                    f"dimension per group name."
                )
            pg_dim, pg_tag = pg_matches[0]
            out.extend(
                (pg_dim, int(t))
                for t in gmsh.model.getEntitiesForPhysicalGroup(
                    pg_dim, pg_tag)
            )

    # 5. Part label — a part owns entities across dims; union them.
    if not out and source == "auto":
        parts = getattr(parent, "parts", None)
        if parts is not None and hasattr(parts, "_instances"):
            inst = parts._instances.get(target)
            if inst is not None:
                for d, ts in inst.entities.items():
                    out.extend((int(d), int(t)) for t in ts)

    if not out:
        raise KeyError(
            f"{not_found_prefix} {target!r} not found as label, "
            f"physical group, part label, or mesh selection."
        )

    if expected_dim is not None:
        scoped = [(d, t) for d, t in out if d == expected_dim]
        if not scoped:
            found = sorted({d for d, _ in out})
            raise ValueError(
                f"Target {target!r} resolved to dimension(s) "
                f"{found}, but this {noun} requires dim={expected_dim}. "
                f"Give it a target of the right dimension (a label "
                f"must cover dim={expected_dim}; multi-dimensional "
                f"physical groups are not supported)."
            )
        return scoped

    return out
