"""up: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ...types import Element
from ....pattern.pattern import _SPRecord
from ..errors import BridgeError
from ..records import FixRecord, SupportRecord
from ..pg_expand import expand_pg_to_nodes
from ..ndf import (
    _ADAPTIVE_NDF_OK,
)


if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


# ---------------------------------------------------------------------------
# LadrunoUP build gates (ADR 0074 D3/D4)
# ---------------------------------------------------------------------------
# The shape tables (etypes / node counts / TH subset / mid-edge slots) are
# owned by _element_capabilities — the SINGLE source of truth (imported at
# use to keep this module cycle-light).

#: The fork's straight-side guard tolerance (LadrunoUP.cpp setDomain):
#: a mid-edge node more than ``1e-6 * edge_length`` off the true midpoint
#: deactivates the element (zero stiffness -> the solve singularizes).
_UP_STRAIGHT_SIDE_RTOL = 1e-6


def _format_bridge_examples(rows: "list[str]", *, cap: int = 3) -> str:
    """Join the first ``cap`` example strings with a ``(+N more)`` tail —
    the shared truncation formatter for the geometry-validator BridgeErrors
    (LadrunoUP shape / curved / ASDAbsorbingBoundary skew)."""
    head = "; ".join(rows[:cap])
    more = f" (+{len(rows) - cap} more)" if len(rows) > cap else ""
    return head + more


def validate_ladruno_up_specs(
    fem: "FEMData", elements: "Iterable[Element]", ndm: int,
) -> None:
    """ADR 0074 — fail loud on LadrunoUP mesh/kwarg combinations the fork
    would reject at parse (or worse, deactivate silently at setDomain).

    Per spec, over the pg fan-out (evaluated per element-type GROUP so the
    columnar fan-out never boxes a connectivity row):

    * **Shape legality** — every cell's Gmsh ETYPE must have a shape
      provider at this ``ndm`` ((2,3) T3 · (2,4) Q4 · (2,6) T6 · (3,8) H8 ·
      (3,10) Tet10); a tet4, prism, or a *count-aliasing* cell (an 8-node
      serendipity quad8 surface in 3D, a 3-node line3 curve in 2D) has none.
      The etype is authoritative — node count alone would wave quad8 through
      as an "H8".
    * **Dimension coherence** — ``perm``/``permH`` (and ``thick``) must
      match the model ``ndm`` (the parser reads exactly ``ndm`` values).
    * **``-stab`` on Taylor–Hood** — parser-fatal (TH is inf-sup stable;
      ``-stab`` is refused on quadratic shapes).  Caught here with the pg
      name instead of a per-element parse error.
    * **Straight sides (D3)** — the Bézier providers assume an affine map;
      the fork checks every mid-edge node against its edge midpoint with a
      ``1e-6 * edge_length`` tolerance at ``setDomain`` and DEACTIVATES the
      element on violation — from openseespy that surfaces only as a
      cryptic ``analyze()`` failure.  Checked here (vectorized) with mesh
      context.
    """
    from ...._element_capabilities import (
        LADRUNO_UP_ETYPES_BY_NDM,
        LADRUNO_UP_MIDEDGE_SLOTS,
        LADRUNO_UP_TH_ETYPES,
    )
    from ....element.solid import LadrunoUP

    legal_etypes = LADRUNO_UP_ETYPES_BY_NDM.get(int(ndm))

    # Sorted node ids + argsort once per fem for the vectorized coord gather
    # (the same searchsorted idiom infer_node_ndf uses).  Lazy — only built
    # when a TH group needs coordinates.
    _id_sort: "tuple[np.ndarray, np.ndarray] | None" = None

    def _coords_for(conn_block: np.ndarray) -> np.ndarray:
        """Gather (rows, k, 3) coordinates for an int64[rows, k] conn block."""
        nonlocal _id_sort
        if _id_sort is None:
            ids = np.asarray(fem.nodes.ids, dtype=np.int64)
            order = np.argsort(ids, kind="stable")
            _id_sort = (ids[order], order)
        sorted_ids, order = _id_sort
        pos = order[np.searchsorted(sorted_ids, conn_block.ravel())]
        coords = np.asarray(fem.nodes.coords, dtype=float)[pos]
        return coords.reshape(conn_block.shape[0], conn_block.shape[1], 3)

    for spec in elements:
        if not isinstance(spec, LadrunoUP):
            continue
        if legal_etypes is None:
            raise BridgeError(
                f"LadrunoUP over pg {spec.pg!r}: model ndm must be 2 or 3, "
                f"got {ndm}."
            )
        if spec.perm_dim != int(ndm):
            raise BridgeError(
                f"LadrunoUP over pg {spec.pg!r}: permeability carries "
                f"{spec.perm_dim} components but the model is ndm={ndm} — "
                f"the fork parser reads exactly ndm values for -perm/-permH "
                f"(and -body/-fluidBody). Match perm= to the model dimension."
            )
        bad_shape: list[str] = []
        curved: list[str] = []
        has_th = False

        # Iterate per element-type GROUP (etype known, conn columnar) — the
        # shape/legality decision is one check per group, not per element.
        result = fem.elements.select(pg=spec.pg).groups()
        for group in _iter_element_groups(result):
            etype = group.element_type
            code = int(getattr(etype, "code", -1))
            gids = np.asarray(group.ids, dtype=np.int64)
            gconn = np.asarray(group.connectivity, dtype=np.int64)
            if gconn.ndim != 2:
                gconn = gconn.reshape(gids.shape[0], -1)
            if code not in legal_etypes:
                nm = getattr(etype, "name", None) or f"etype {code}"
                bad_shape.append(
                    f"{int(gids.shape[0])} {nm} cell(s) (e.g. element "
                    f"{int(gids[0])})" if gids.size else f"{nm} cells"
                )
                continue
            if code not in LADRUNO_UP_TH_ETYPES:
                continue
            has_th = True
            k = gconn.shape[1]
            X = _coords_for(gconn)                       # (rows, k, 3)
            for m_slot, a_slot, b_slot in LADRUNO_UP_MIDEDGE_SLOTS[k]:
                xa = X[:, a_slot, :]
                xb = X[:, b_slot, :]
                xm = X[:, m_slot, :]
                edge = np.linalg.norm(xb - xa, axis=1)
                off = np.linalg.norm(xm - 0.5 * (xa + xb), axis=1)
                viol = off > _UP_STRAIGHT_SIDE_RTOL * np.maximum(edge, 1e-300)
                for r in np.nonzero(viol)[0]:
                    curved.append(
                        f"element {int(gids[r])}: mid-edge node "
                        f"{int(gconn[r, m_slot])} is {float(off[r]):.3g} off "
                        f"the midpoint (edge length {float(edge[r]):.3g})"
                    )
        if bad_shape:
            raise BridgeError(
                f"LadrunoUP over pg {spec.pg!r}: cell(s) with no shape "
                f"provider at ndm={ndm} — {_format_bridge_examples(bad_shape)}. "
                f"Supported: tri3/quad4/tri6 (2D), hexa8/tet10 (3D). A pg's "
                f"cell TYPE (not just node count) must match — an 8-node "
                f"quad8 surface is not an H8, a 3-node line3 is not a T3. "
                f"Point the element at the volume/surface pg meshed with a "
                f"supported cell type (tet4 has no u-p provider; use tet10 "
                f"or hexa8)."
            )
        if has_th and spec.stab is not None:
            raise BridgeError(
                f"LadrunoUP over pg {spec.pg!r}: stab= is set but the pg "
                f"contains Taylor–Hood (tri6/tet10) cells — TH is inf-sup "
                f"stable and the fork parser is FATAL on -stab there. Drop "
                f"stab= for this pg (equal-order pgs keep it)."
            )
        if curved:
            raise BridgeError(
                f"LadrunoUP over pg {spec.pg!r}: {len(curved)} Bézier "
                f"cell(s) violate the straight-side requirement "
                f"(tolerance {_UP_STRAIGHT_SIDE_RTOL:g}·edge) — "
                f"{_format_bridge_examples(curved)}. The fork deactivates "
                f"such elements at setDomain (zero stiffness -> the solve "
                f"singularizes with a cryptic analyze() failure). Generate "
                f"order-2 meshes on straight geometry, or turn Gmsh "
                f"high-order optimization off so mid-edge nodes stay at "
                f"edge midpoints."
            )


def _iter_element_groups(result: "Any") -> "Iterable[Any]":
    """Yield per-type element groups (``element_type`` / ``ids`` /
    ``connectivity``) from whatever ``fem.elements.select(pg=).groups()``
    returned, normalizing three shapes:

    * a real ``GroupResult`` — iterable of ``ElementGroup`` (each with true
      Gmsh ``element_type``);
    * the live-mesh flat ``dict`` (``{'element_ids', 'connectivity'}``) —
      one homogeneous block, etype derived from the connectivity width;
    * a lightweight test stub's ``list`` of ``[(eid, conn), ...]`` blocks —
      etype derived from node count per block.

    Only the ``GroupResult`` path carries an authoritative etype (so it
    catches count-aliasing shapes like quad8-as-H8); the count-derived
    fallbacks pick the canonical interpretation of each node count.
    """
    if isinstance(result, dict):
        yield _StubElementGroup(
            list(zip(result["element_ids"], result["connectivity"])),
            conn_is_rows=True,
        )
        return
    for group in result:
        if hasattr(group, "element_type") and hasattr(group, "connectivity"):
            yield group
        else:
            yield _StubElementGroup(group)


class _StubElementGroup:
    """Adapter over an etype-less block of ``(eid, conn)`` rows (live-mesh
    flat dict or test stub) — derives a canonical etype from the node count.
    """

    __slots__ = ("element_type", "ids", "connectivity")

    def __init__(self, block: "Any", *, conn_is_rows: bool = False) -> None:
        rows = list(block)
        width = len(rows[0][1]) if rows else 0
        self.element_type = _StubTypeInfo(width)
        self.ids = np.asarray([int(eid) for eid, _c in rows], dtype=np.int64)
        self.connectivity = (
            np.asarray([[int(n) for n in c] for _e, c in rows], dtype=np.int64)
            if rows else np.empty((0, 0), dtype=np.int64)
        )
        _ = conn_is_rows  # both branches materialize rows identically


class _StubTypeInfo:
    """Etype stand-in for a metadata-less test block: derives a code from the
    node count via the standard Gmsh alias table so straight-topology unit
    tests (which build conforming tri6/tet10 blocks) still exercise the
    etype path, while genuinely unknown counts fall to the -1 sentinel."""

    __slots__ = ("code", "name")

    #: node count -> canonical Gmsh code, for the shapes the u-p gate cares
    #: about (mirrors _element_types._KNOWN_ALIASES for these counts).
    _COUNT_TO_CODE: "dict[int, int]" = {
        3: 2, 4: 3, 6: 9, 8: 5, 10: 11,
    }

    def __init__(self, npe: int) -> None:
        self.code = self._COUNT_TO_CODE.get(int(npe), -1)
        self.name = f"{npe}-node cell"


def _ladruno_up_carrier_blocks(
    fem: "FEMData", up_specs: "Sequence[Element]",
) -> "Iterator[np.ndarray]":
    """Yield ``(n_elem, n_carrier)`` pressure-CARRIER connectivity blocks.

    One block per mesh element group of each LadrunoUP spec, sliced down to
    the slots that actually carry ``p``: equal-order shapes carry it on every
    node, Taylor–Hood shapes only on the leading vertex slots (mid-edge nodes
    are pure displacement).  SINGLE SOURCE of the carrier-slot rule — both
    :func:`_ladruno_up_carrier_nodes` (which wants the flat set) and
    :func:`validate_up_pressure_datum` (which wants per-element rows, to walk
    connectivity) read it from here.
    """
    from ...._element_capabilities import LADRUNO_UP_TH_ETYPES

    th_vertex_count = {6: 3, 10: 4}
    for spec in up_specs:
        result = fem.elements.select(pg=spec.pg).groups()  # type: ignore[attr-defined]
        for group in _iter_element_groups(result):
            gids = np.asarray(group.ids, dtype=np.int64)
            gconn = np.asarray(group.connectivity, dtype=np.int64)
            if gconn.ndim != 2:
                gconn = gconn.reshape(gids.shape[0], -1)
            code = int(getattr(group.element_type, "code", -1))
            k = gconn.shape[1]
            if code in LADRUNO_UP_TH_ETYPES or k in th_vertex_count:
                yield gconn[:, :th_vertex_count.get(k, k)]
            else:
                yield gconn


def _ladruno_up_carrier_nodes(
    fem: "FEMData", up_specs: "Sequence[Element]",
) -> "set[int]":
    """The pressure-CARRIER node tags of the given LadrunoUP specs.

    Equal-order shapes carry ``p`` on every node; Taylor–Hood shapes carry
    it only on the vertex slots (mid-edge nodes are pure displacement).  The
    pressure DOF lives at slot ``ndm+1`` on exactly these nodes.
    """
    carrier: set[int] = set()
    for block in _ladruno_up_carrier_blocks(fem, up_specs):
        carrier.update(int(t) for t in block.ravel())
    return carrier


def validate_ladruno_up_pressure_dof(
    fem: "FEMData", elements: "Iterable[Element]", ndm: int,
) -> None:
    """ADR 0074 — fail loud on the rotation-vs-pressure DOF aliasing trap.

    On a LadrunoUP pressure-carrier node the DOF at slot ``ndm+1`` is the
    pore pressure ``p``.  A 2D frame element (elasticBeamColumn &c.) puts a
    ROTATION at that same slot (``ux, uy, rz`` — floor ``ndm+1``), so a
    beam sharing a saturated equal-order soil node silently assembles its
    bending stiffness into the pressure row: both require ``ndf=ndm+1``, so
    the count-based ndf gate (:func:`infer_node_ndf`) and the disjoint-set
    guard (:func:`validate_node_ndf_element_compat`) both PASS, and the fork
    setDomain checks only the DOF count — nothing catches it, and the run
    returns garbage pore pressures and moments with ``rc=0``.

    An element that requires exactly ``ndm+1`` DOFs and is neither a u-p
    element nor an adaptive spring uses that extra slot structurally (pure
    translation is only ``ndm`` DOFs; higher-DOF neighbours like 6-DOF
    shells/3D-beams are already caught by the strict ndf-set mismatch).  So
    a carrier node shared with any such element is the aliasing trap — raise
    with the ADR-0069 separate-node fix.  (TH mid-edge nodes are ``ndf=ndm``
    and a floor-``ndm+1`` neighbour there resolves to a mismatch the strict
    gate already rejects, so only carrier nodes need this pass.)
    """
    from ...._element_capabilities import (
        element_class_ndf_ok,
        element_required_floor,
    )
    from ....element.solid import LadrunoUP

    up_specs = [s for s in elements if isinstance(s, LadrunoUP)]
    if not up_specs:
        return
    carrier_floor = int(ndm) + 1
    carrier = _ladruno_up_carrier_nodes(fem, up_specs)
    if not carrier:
        return

    for spec in elements:
        if isinstance(spec, LadrunoUP):
            continue
        cls = type(spec).__name__
        try:
            floor = element_required_floor(cls, ndm)
        except (ValueError, KeyError):
            continue
        ndf_ok = element_class_ndf_ok(cls)
        if floor is None or ndf_ok == _ADAPTIVE_NDF_OK:
            continue
        if floor != carrier_floor:
            continue
        pg = getattr(spec, "pg", None)
        if pg is None:
            continue
        hit: int | None = None
        for group in _iter_element_groups(
            fem.elements.select(pg=pg).groups()
        ):
            gconn = np.asarray(group.connectivity, dtype=np.int64).ravel()
            for t in gconn:
                if int(t) in carrier:
                    hit = int(t)
                    break
            if hit is not None:
                break
        if hit is not None:
            raise BridgeError(
                f"node {hit}: a LadrunoUP pressure-carrier node (DOF "
                f"{carrier_floor} is the pore pressure p) is shared with "
                f"{cls!r} (pg {pg!r}), which places a structural DOF "
                f"(rotation) at that same slot — both require ndf="
                f"{carrier_floor}, so the count-based ndf gate passes, but "
                f"OpenSees would silently assemble {cls}'s stiffness into the "
                f"pressure row and return garbage pore pressures. Give the "
                f"interface SEPARATE coincident nodes and tie only the shared "
                f"displacement DOFs with g.constraints.equal_dof (the "
                f"mixed-ndf / structure-on-soil idiom, ADR 0069 / fork guide "
                f"§6.3)."
            )


def up_pressure_components(
    element_carriers: "Iterable[Sequence[int]]",
) -> "dict[int, list[int]]":
    """Connected components of the pressure graph, keyed by root node tag.

    *element_carriers* is one sequence of pressure-CARRIER node tags per
    element (mid-edge / non-pressure slots already dropped — dropping them is
    what keeps a Taylor–Hood mesh ONE region instead of shattering it).  Two
    carrier nodes are connected when some element carries both, which is
    exactly the sparsity of the ``H`` seepage / ``S`` storage blocks: a
    pressure DOF is coupled only to the pressure DOFs it shares an element
    with.  Union-find, so the walk is ``O(N α(N))`` in the carrier count.

    Returns ``{min node tag of the component: sorted node tags}``.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for conn in element_carriers:
        head: int | None = None
        for raw in conn:
            tag = int(raw)
            parent.setdefault(tag, tag)
            if head is None:
                head = tag
                continue
            ra, rb = find(head), find(tag)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for tag in parent:
        groups.setdefault(find(tag), []).append(tag)
    return {min(members): sorted(members) for members in groups.values()}


def validate_up_pressure_datum(
    fem: "FEMData",
    elements: "Iterable[Element]",
    ndm: int,
    *,
    enforce: bool,
    fix_records: "Iterable[FixRecord]" = (),
    sp_records: "Iterable[_SPRecord]" = (),
    support_records: "Iterable[SupportRecord]" = (),
) -> None:
    """Refuse a STATIC u-p deck in which some pressure region has no datum.

    A saturated (u-p) region whose pressure DOFs are ALL free is impervious:
    its static tangent is singular in ``p`` (the ``H`` seepage block has the
    constant-pressure vector in its null space, exactly like an all-Neumann
    Laplacian, and static has no ``S/Δt`` storage term to regularise it).
    Nothing downstream reports it: the fork MEASURED (2026-07-11) that every
    serial general solver — UmfPack / FullGeneral / BandGeneral /
    SparseGeneral — factorises the singular sealed system through round-off
    and returns ``rc = 0`` with an arbitrary, solver-dependent pressure level,
    because the p-RHS is consistent so no solver sees the rank deficiency
    (``tests/test_ladruno_up_element_analytic.py:533-548``, a ``strict``
    xfail pinning the refuted "it fails loudly" claim of fork ADR-71
    §3.2/§7).  A silent wrong answer, so this is a build-time gate.

    Contract:

    * **Pressure node** — a LadrunoUP pressure CARRIER node
      (:func:`_ladruno_up_carrier_blocks`): every node of an equal-order
      shape, the vertex slots only of a Taylor–Hood shape.  LadrunoUP is the
      only u-p element class in the capability registry, so it is the only
      one walked; a stock ``quadUP`` / ``brickUP`` cannot reach emit at all
      (:func:`infer_node_ndf` refuses an unregistered class).
    * **Datum** — any single-point constraint on DOF ``ndm+1`` of a carrier
      node: a broker or stage-claimed ``fix`` / ``s.support`` whose 0/1 mask
      flags that slot, or a pattern ``sp`` whose 1-based DOF index is that
      slot.  All three make the DOF constrained, which is what removes the
      null space; ``fix`` alone would false-refuse the ``sp``-datum idiom.
    * **Exempt** — everything the caller does not mark ``enforce``: H5
      archival, and any deck that does not declare ``ops.analysis.Static()``.
      A sealed region is PHYSICALLY CORRECT under Transient (undrained
      loading: the storage term ``1/Q̄`` puts a nonzero diagonal on the p
      rows, so the sealed tangent is regular — that is what the fork's own
      Terzaghi lane runs), so refusing it there would be a false positive.

    Known limitation (not enforced): a datum contributed only by a
    stage-scoped ``s.support`` is counted for the whole model, though it holds
    the DOF only inside its own stage; and a pressure DOF grounded indirectly
    through ``equal_dof`` to a fixed one is not traced.
    """
    from ....element.solid import LadrunoUP

    if not enforce:
        return
    up_specs = [s for s in elements if isinstance(s, LadrunoUP)]
    if not up_specs:
        return

    components = up_pressure_components(
        conn
        for block in _ladruno_up_carrier_blocks(fem, up_specs)
        for conn in block
    )
    if not components:
        return

    p_dof = int(ndm) + 1
    carriers = {n for members in components.values() for n in members}

    def _pg_or_nodes(rec: "FixRecord | SupportRecord") -> "list[int]":
        if rec.nodes is not None:
            return [int(n) for n in rec.nodes]
        if rec.pg is not None:
            return list(expand_pg_to_nodes(fem, rec.pg))
        return []

    datum: set[int] = set()
    fix_like: "list[FixRecord | SupportRecord]" = [
        *fix_records, *support_records,
    ]
    for frec in fix_like:
        flags = frec.dofs
        if len(flags) < p_dof or not flags[p_dof - 1]:
            continue
        datum.update(_pg_or_nodes(frec))
    for srec in sp_records:
        if int(srec.dof) != p_dof:
            continue
        if srec.target_kind == "pg":
            datum.update(int(n) for n in expand_pg_to_nodes(fem, srec.target))
        else:
            datum.add(int(srec.target))
    # A fix on a TH mid-edge node addresses a slot that node does not carry —
    # it is not a pressure datum (G3 already refuses the over-long mask).
    datum &= carriers

    sealed = sorted(
        root for root, members in components.items()
        if not datum.intersection(members)
    )
    if not sealed:
        return

    mask = "(" + "0, " * int(ndm) + "1)"
    raise BridgeError(
        f"LadrunoUP node {sealed[0]}: its pressure region has NO fixed "
        f"pressure DOF ({len(sealed)} of {len(components)} u-p region(s) in "
        f"this model are sealed). A STATIC solve on an all-impervious "
        f"region is SINGULAR in p, and it does not fail loudly: every serial "
        f"general solver (UmfPack / FullGeneral / BandGeneral / "
        f"SparseGeneral) factorises it through round-off and returns rc = 0 "
        f"with an arbitrary, solver-dependent pressure level (fork "
        f"tests/test_ladruno_up_element_analytic.py:533-548, MEASURED "
        f"2026-07-11) — a silent wrong answer. Fix the pressure DOF (slot "
        f"{p_dof}) of at least one carrier node of that region, typically a "
        f"drained surface: ops.fix(pg=\"Top\", dofs={mask}). On a "
        f"Taylor-Hood mesh (tri6 / tet10) that pg mask over-runs the "
        f"mid-edge nodes' ndf={ndm} and G3 refuses it, so pass the VERTEX "
        f"nodes of the drained surface as ops.fix(nodes=[...], dofs={mask}) "
        f"instead. Each element-disconnected u-p region needs its own datum. "
        f"(A sealed "
        f"region is legitimate under ops.analysis.Transient() — the storage "
        f"term regularises the p rows — and this gate only runs for a deck "
        f"that declares ops.analysis.Static().)"
    )
