"""
_opensees_constraints — constraint emission sub-module for the OpenSees bridge.

Accessed indirectly via ``emit_tie_elements(ops)`` from
``_opensees_build.run_build`` (and, later, ``emit_node_pairs`` /
``emit_diaphragms`` once those paths are implemented).

Phase 11a scope
---------------
This module currently only emits **tie** constraints as
``element ASDEmbeddedNodeElement`` commands.  A tie is a zero-thickness
penalty element that constrains one "constrained" node (the slave)
to ride on the linear shape-function interpolation of 3 retained
nodes forming a triangle (or 4 retained nodes forming a tet — the
embedded-rebar case, deferred to a later phase).

See the OpenSees ``ASDEmbeddedNodeElement`` source for the kinematics:

    element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 $Rnode2 $Rnode3
                                   [-rot] [-K $K]

``-rot`` ties the rotational DOFs of the constrained node in addition
to the translations (required for beam-tip-in-shell ties).  The
element self-calibrates the initial geometric offset via its internal
``m_U0`` snapshot, so our resolver's ``projected_point`` only needs
to be within the master-face tolerance — it does not need to coincide
with the master surface to machine precision.

Deferred emission paths
-----------------------
``emit_node_pairs``      — ``equalDOF`` / ``rigidLink``
``emit_diaphragms``      — ``rigidDiaphragm``
``emit_embedded_tets``   — ``ASDEmbeddedNodeElement`` with 4 retained
                           (tet embedding, requires
                           ``resolve_embedded`` to be implemented)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .OpenSees import OpenSees


# =====================================================================
# Tag allocation
# =====================================================================

def _make_tie_tag_allocator(ops: "OpenSees"):
    """Return a closure that hands out monotonically increasing element
    tags for tied-interface elements.

    The first tie element starts one past the largest existing
    ``_elements_df`` index.  When ``_elements_df`` is empty (no user
    elements declared), we start at ``1_000_000`` to leave room for
    later user-declared elements without collisions.
    """
    if len(ops._elements_df) > 0:
        start = int(ops._elements_df.index.max()) + 1
    else:
        start = 1_000_000
    counter = [start]

    def next_tag() -> int:
        t = counter[0]
        counter[0] += 1
        return t

    return next_tag


# =====================================================================
# Face-order handling
# =====================================================================

def _quad4_split_pick(rec) -> list[int] | None:
    """Choose 3 retained nodes from a Quad4 master face.

    ASDEmbeddedNodeElement only accepts 3 or 4 retained nodes, and
    4 means **tetrahedron** (volumetric embedding), NOT a Quad4 face.
    So we must split the quad at its (0,2) diagonal and pick the
    triangle that contains the slave's parametric coordinates.

    Quad4 isoparametric layout in (ξ, η) ∈ [-1, 1]²::

        3 (-1, 1) ─────── 2 (1, 1)
           │                 │
           │     Tri B       │
           │    /            │
           │  /  diagonal    │
           │/    Tri A       │
        0 (-1,-1) ─────── 1 (1,-1)

    Triangle A (below the diagonal from 0 to 2) = nodes [0, 1, 2]
    Triangle B (above the diagonal from 0 to 2) = nodes [0, 2, 3]

    The split test is ``ξ >= η`` (Triangle A) vs ``ξ < η``
    (Triangle B).  Both triangles meet at the diagonal itself.

    Returns the 3 chosen master node tags, or None if the record
    has no parametric coordinates (shouldn't happen for a valid
    tie record but we guard anyway).
    """
    if rec.parametric_coords is None:
        return None
    xi, eta = float(rec.parametric_coords[0]), float(rec.parametric_coords[1])
    m = rec.master_nodes
    if xi >= eta:
        return [int(m[0]), int(m[1]), int(m[2])]
    return [int(m[0]), int(m[2]), int(m[3])]


def _pick_retained_nodes(
    rec,
    logger,
    slave_node: int,
) -> list[int] | None:
    """Choose the 3 retained master nodes for one tie record.

    Handles Tri3, Quad4 (split), Tri6 (corner downgrade), and Quad8
    (corner downgrade + split).  Returns None when the record's
    master face is unsupported or has missing parametric coords.
    """
    n = len(rec.master_nodes)
    if n == 3:
        # Tri3 — straight through
        return [int(m) for m in rec.master_nodes]

    if n == 4:
        # Quad4 — split at the diagonal
        chosen = _quad4_split_pick(rec)
        if chosen is None:
            if logger is not None:
                logger(
                    f"tie: Quad4 master face for slave {slave_node} has no "
                    f"parametric coordinates — skipped"
                )
            return None
        return chosen

    if n == 6:
        # Tri6 — use the first 3 corners (ordered corners come first
        # in gmsh's node convention).  Log once per record.
        if logger is not None:
            logger(
                f"tie: Tri6 master face downgraded to Tri3 corners for "
                f"slave {slave_node} — ASDEmbeddedNodeElement only "
                f"supports linear retained faces"
            )
        return [int(m) for m in rec.master_nodes[:3]]

    if n == 8:
        # Quad8 — use the first 4 corners, then split like a Quad4.
        # Temporarily rebuild a proxy with just the corner nodes and
        # re-run the quad-split logic against its parametric coords
        # (which are the same Quad4 isoparametric frame).
        if logger is not None:
            logger(
                f"tie: Quad8 master face downgraded to Quad4 corners for "
                f"slave {slave_node}"
            )
        if rec.parametric_coords is None:
            return None
        xi, eta = float(rec.parametric_coords[0]), float(rec.parametric_coords[1])
        m = rec.master_nodes
        if xi >= eta:
            return [int(m[0]), int(m[1]), int(m[2])]
        return [int(m[0]), int(m[2]), int(m[3])]

    # Unsupported face cardinality (shouldn't happen from the
    # resolver but we guard anyway).
    if logger is not None:
        logger(
            f"tie: UNSUPPORTED master face with {n} nodes for slave "
            f"{slave_node} — skipped"
        )
    return None


# =====================================================================
# Tie emission
# =====================================================================

def emit_tie_elements(ops: "OpenSees") -> dict[str, int]:
    """Emit all tie interpolation records as ASDEmbeddedNodeElement.

    Walks ``ops._constraint_records.interpolations()``, filters to
    ``kind == "tie"``, and for each accepted record appends an
    entry to ``ops._tie_elements`` that the exporters will render
    as an ``element ASDEmbeddedNodeElement`` command.

    Called from ``_opensees_build.run_build`` at the end of the
    build pipeline, after ``_elements_df`` has been populated with
    user-declared elements, so the tie element tags start one past
    the highest user-element tag.

    Parameters
    ----------
    ops : OpenSees
        The broker instance.  Reads ``ops._constraint_records`` and
        ``ops._tie_penalty``, writes to ``ops._tie_elements``.

    Returns
    -------
    dict[str, int]
        Per-kind emission counts for logging:

        ``{"tie": N, "tie_skipped": M}``
    """
    cs = ops._constraint_records
    counts = {"tie": 0, "tie_skipped": 0}
    if cs is None or not cs:
        return counts

    tag_alloc = _make_tie_tag_allocator(ops)

    def _log(msg: str) -> None:
        ops._log(msg)

    for rec in cs.interpolations():
        if rec.kind != "tie":
            continue

        retained = _pick_retained_nodes(rec, _log, int(rec.slave_node))
        if retained is None:
            counts["tie_skipped"] += 1
            continue

        # -rot flag: required when the slave node has rotational DOFs
        # and the user declared them in the tie definition's DOF list.
        # ASDEmbeddedNodeElement rejects empty/malformed rotation
        # combinations, so we emit -rot only when the record's dofs
        # actually contain 4/5/6.  A user passing dofs=[4,5,6] alone
        # (rotations only, no translations) gets the warning noted
        # below — the element always ties all 3 translations.
        dofs = rec.dofs or []
        has_rot = any(int(d) >= 4 for d in dofs)
        has_trans = any(1 <= int(d) <= 3 for d in dofs)
        if has_rot and not has_trans:
            _log(
                f"tie: slave {rec.slave_node} has rotation-only dofs "
                f"{list(dofs)} — element will still tie translations "
                f"(ASDEmbeddedNodeElement does not support rotation-only)"
            )

        ops._tie_elements.append({
            "ele_tag":     tag_alloc(),
            "cNode":       int(rec.slave_node),
            "rNodes":      retained,
            "use_rot":     has_rot,
            "penalty":     ops._tie_penalty,
            "source_kind": rec.kind,
        })
        counts["tie"] += 1

    if counts["tie"] or counts["tie_skipped"]:
        ops._log(
            f"emit_tie_elements(): {counts['tie']} tie(s) emitted, "
            f"{counts['tie_skipped']} skipped"
        )
    return counts


# =====================================================================
# Rendering helpers — used by _opensees_export
# =====================================================================

def render_tie_tcl(entry: dict) -> str:
    """Render one tie entry as an OpenSees Tcl ``element ...`` line."""
    tag    = entry["ele_tag"]
    cnode  = entry["cNode"]
    rnodes = entry["rNodes"]
    rot    = "  -rot" if entry.get("use_rot") else ""
    kflag  = ""
    if entry.get("penalty") is not None:
        kflag = f"  -K {entry['penalty']:.10g}"
    r_str = "  ".join(str(n) for n in rnodes)
    return (
        f"element ASDEmbeddedNodeElement  {tag}  {cnode}  {r_str}"
        f"{rot}{kflag}"
    )


def render_tie_py(entry: dict) -> str:
    """Render one tie entry as an openseespy ``ops.element(...)`` call."""
    tag    = entry["ele_tag"]
    cnode  = entry["cNode"]
    rnodes = entry["rNodes"]
    args: list[str] = [
        "'ASDEmbeddedNodeElement'",
        str(tag),
        str(cnode),
        *(str(n) for n in rnodes),
    ]
    if entry.get("use_rot"):
        args.append("'-rot'")
    if entry.get("penalty") is not None:
        args.extend(["'-K'", f"{entry['penalty']:.10g}"])
    return f"ops.element({', '.join(args)})"
