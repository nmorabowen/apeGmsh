"""
EmbedmentsComposite -- declare general node-to-host embedment, resolve to ties.

``g.embed(host=..., nodes=..., ...)`` is the apeGmsh-side generator for the
Ladruno fork's ``LadrunoEmbeddedNode`` coupling element (ELE 33006) — the
isotropic sibling of ``g.reinforce`` / ``LadrunoEmbeddedRebar``. It ties a
**constrained node set** into a **non-matching** solid host: at FEM-build
time each node is inverse-mapped into the host element it falls inside,
producing one :class:`~apeGmsh._kernel.records._constraints.EmbedTieRecord`
per node. The bridge build step (``opensees._internal.build.emit_embed_ties``)
turns each record into ``element LadrunoEmbeddedNode`` via the
``embedded_node_args`` builder.

This is the conditioned fork upgrade over ``g.constraints.embedded`` (which
emits the upstream ``ASDEmbeddedNodeElement`` with a raw 1e18 penalty): it
ships ``-k``-numeric / augmented-Lagrangian enforcement, explicit-safe
bipenalty, and g0 stress-free birth. It is a distinct, opt-in generator;
``g.constraints.embedded`` is left untouched.

Two-stage pipeline, mirroring :class:`ReinforcementsComposite`:

1. **Declare** (pre-mesh): :meth:`embed` (also ``g.embed(...)``) stores an
   :class:`~apeGmsh._kernel.defs.constraints.EmbedDef`.
2. **Resolve** (post-mesh): :meth:`resolve` — called by
   ``Mesh.queries.get_fem_data`` — pulls the host elements + the constrained
   node set from the live Gmsh session and delegates to
   :func:`~apeGmsh._kernel.resolvers._embed.resolve_embed`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh._kernel.defs.constraints import EmbedDef
from apeGmsh._kernel.records._constraints import EmbedTieRecord
from apeGmsh._kernel.resolvers._embed import resolve_embed

# Reuse the reinforce composite's gmsh host-kind maps (single source of
# truth for the straight-sided host taxonomy).
from apeGmsh.core.ReinforcementsComposite import (
    _GMSH_HOST_FULL_NPE,
    _GMSH_HOST_KIND,
)


def _host_has_curved_edge(code, full_npe, row, coord_of) -> bool:
    """True iff a higher-order host element of gmsh type ``code`` has a curved
    edge — i.e. an edge mid-side node deviates from the midpoint of its two
    corner endpoints (in any direction).

    The edge→corner-pair association is discovered from gmsh's reference
    parametric coordinates (``getElementProperties``): an edge mid-side node
    sits at the parametric midpoint of exactly two corner (primary) nodes.
    This is table-free and works for every supported higher-order host
    (tri6/quad8/quad9/tet10/hex20). Face / volume / centre nodes (no matching
    corner pair) are skipped — only edge straightness is tested, which is what
    the corner-sub-element linearisation depends on.
    """
    import gmsh

    props = gmsh.model.mesh.getElementProperties(int(code))
    local = np.asarray(props[4], dtype=float)   # flat param coords, npe*dref
    n_prim = int(props[5])                       # number of corner nodes
    if local.size == 0 or n_prim <= 0:
        return False
    dref = local.size // full_npe
    param = np.asarray(local, dtype=float).reshape(full_npe, dref)
    corner_xyz = [coord_of[int(row[c])] for c in range(n_prim)]
    cc = np.vstack(corner_xyz)
    diag = float(np.linalg.norm(cc.max(axis=0) - cc.min(axis=0))) or 1.0
    tol = 1.0e-6 * diag
    for i in range(n_prim, full_npe):
        pi = param[i]
        # Collect ALL corner pairs whose parametric midpoint is this node. An
        # EDGE mid-node matches exactly ONE pair (its two endpoints); a FACE /
        # CENTRE node matches >1 (e.g. the quad9 centre at (0,0) is the midpoint
        # of BOTH diagonals (0,2) and (1,3)) or 0. Only run the straightness
        # test for genuine edge nodes — a face/centre node sits at the bilinear
        # image of its corners, not a corner-pair midpoint, so testing it would
        # spuriously flag a straight non-parallelogram element as curved.
        pairs = [
            (a, b)
            for a in range(n_prim)
            for b in range(a + 1, n_prim)
            if np.allclose((param[a] + param[b]) * 0.5, pi, atol=1e-9)
        ]
        if len(pairs) != 1:
            continue
        a, b = pairs[0]
        straight = 0.5 * (corner_xyz[a] + corner_xyz[b])
        actual = coord_of[int(row[i])]
        if float(np.linalg.norm(actual - straight)) > tol:
            return True
    return False


class EmbedmentsComposite:
    """General node-to-host embedment generator — declare on geometry,
    resolve to ``LadrunoEmbeddedNode`` ties after meshing.

    Examples
    --------
    Embed a refined sub-block's node set into a coarse non-matching host
    volume, stress-free birth (default), augmented-Lagrangian::

        g.embed(host="coarse", nodes="fine_iface", enforce="al")

    A single control point tied into a solid host::

        g.embed(host="block", nodes="probe_pt")
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.embed_defs: list[EmbedDef] = []
        self.embed_records: list[EmbedTieRecord] = []

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs) -> EmbedDef:
        """``g.embed(...)`` — alias for :meth:`embed`."""
        return self.embed(*args, **kwargs)

    def embed(
        self, host, nodes, *,
        k=None, k_alpha=None,
        enforce="penalty", explicit=False, dtcr=None,
        staged=True,
        tolerance=1.0e-6, snap=False,
        host_entities=None, nodes_entities=None,
        name=None,
    ) -> EmbedDef:
        """Declare a node-to-host embedment tie.

        Parameters
        ----------
        host : str
            The solid host physical group / part label (3-D hex/tet, or
            2-D quad/tri for a 2-D model).
        nodes : str
            The constrained node set physical group / part label. Every
            mesh node of its entities is inverse-mapped into ``host`` and
            tied with one ``LadrunoEmbeddedNode``.
        k : float, optional
            Isotropic penalty ``Ku`` (``-k``). ``None`` → fork default.
            ``"auto"`` is deferred (needs the ``-xi`` host-query path);
            pass a numeric value or leave ``None``.
        k_alpha : float, optional
            Auto-scale multiplier (``-kAlpha``) — only with ``k="auto"``
            (deferred); accepted for forward compatibility.
        enforce : {"penalty", "al"}
            Constraint enforcement (default ``"penalty"``). ``"al"``
            (augmented Lagrangian) drives the gap → 0 at a moderate penalty.
        explicit : bool
            Enable explicit bipenalty critical-time-step control
            (``-bipenalty``). Penalty-enforcement only; needs ``dtcr``.
        dtcr : float, optional
            The explicit critical-time-step budget for ``explicit``
            (``-dtcr``). (The ``-wcap`` host-frequency form is deferred with
            the ``-xi`` path.)
        staged : bool
            ``True`` (default) → g0 stress-free birth (a node added onto an
            already-deformed host activates at zero force). ``False`` →
            emit ``-absolute`` (legacy absolute tie).
        tolerance : float
            Inverse-map acceptance threshold on the parametric excess.
        snap : bool
            ``False`` (default) → a node outside every host raises; ``True``
            → project it onto the nearest host + warn.
        host_entities, nodes_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities; when omitted the
            whole label is used.
        name : str, optional
            Friendly name (round-trips into the emitted deck comment).

        Returns
        -------
        EmbedDef
        """
        defn = EmbedDef(
            master_label=host, slave_label=nodes,
            host_entities=host_entities, nodes_entities=nodes_entities,
            k=k, k_alpha=k_alpha,
            enforce=enforce, explicit=explicit, dtcr=dtcr,
            staged=staged,
            tolerance=tolerance, snap=snap,
            name=name,
        )
        self.embed_defs.append(defn)
        return defn

    def validate_pre_mesh(self) -> None:
        """No-op — embedment resolves at ``get_fem_data`` time."""
        return None

    # ------------------------------------------------------------------
    # Resolution (post-mesh)
    # ------------------------------------------------------------------
    def resolve(self, node_tags, node_coords) -> list[EmbedTieRecord]:
        """Resolve every :meth:`embed` def to ``EmbedTieRecord``\\ s.

        Pulls the host elements (full node lists + kind) and the constrained
        node set from the live Gmsh session, builds a tag → coord map from
        ``node_tags`` / ``node_coords``, and delegates each def to
        :func:`resolve_embed`. Fail-loud throughout.
        """
        records: list[EmbedTieRecord] = []
        if not self.embed_defs:
            self.embed_records = records
            return records

        coord_of = {
            int(t): np.asarray(node_coords[i], dtype=float)
            for i, t in enumerate(node_tags)
        }

        for defn in self.embed_defs:
            host_entities = (
                defn.host_entities if defn.host_entities
                else self._entities_for_label(defn.master_label)
            )
            nodes_entities = (
                defn.nodes_entities if defn.nodes_entities
                else self._entities_for_label(defn.slave_label)
            )

            host_node_ids, host_node_coords, host_kinds = \
                self._collect_hosts(host_entities, coord_of, defn.master_label)
            node_ids = self._collect_nodes(nodes_entities, defn.slave_label)
            node_coords_arr = np.vstack([coord_of[n] for n in node_ids])

            records.extend(resolve_embed(
                node_ids=node_ids,
                node_coords=node_coords_arr,
                host_node_ids=host_node_ids,
                host_node_coords=host_node_coords,
                host_kinds=host_kinds,
                k=defn.k,
                k_alpha=defn.k_alpha,
                enforce=defn.enforce,
                explicit=defn.explicit,
                dtcr=defn.dtcr,
                staged=defn.staged,
                tolerance=defn.tolerance,
                snap=defn.snap,
                name=defn.name,
            ))

        self.embed_records = records
        return records

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------
    def _collect_hosts(self, entities, coord_of, label):
        """Per host element: (node-id list, (n,3) coord array, host kind).

        Higher-order hosts are reduced to their corner subset + corner kind,
        which linearises the element — VALID ONLY for STRAIGHT-SIDED higher-
        order hosts (mid-side nodes on the straight edges). On a genuinely
        CURVED host (bulging mid-side nodes) the corner linearisation +
        nearest-centroid prefilter can mislocate an embedded node (wrong host
        element / barycentric coords); a one-time warning fires when curvature
        is detected (mid-side node outside the corner bounding box). Unsupported
        host kinds (prism / pyramid) fail loud. Mirrors
        ``ReinforcementsComposite._collect_hosts``.
        """
        import gmsh
        import warnings as _warnings

        host_node_ids: list[list[int]] = []
        host_node_coords: list[np.ndarray] = []
        host_kinds: list[str] = []
        curved_warned = False

        for dim, tag in entities:
            try:
                etypes, _, enodes = gmsh.model.mesh.getElements(
                    dim=int(dim), tag=int(tag))
            except Exception as exc:
                raise ValueError(
                    f"embed: cannot get mesh elements for host entity "
                    f"(dim={dim}, tag={tag}) of label {label!r}: {exc}"
                ) from exc
            for etype, nodes in zip(etypes, enodes):
                if len(nodes) == 0:
                    continue
                code = int(etype)
                if code not in _GMSH_HOST_KIND:
                    raise ValueError(
                        f"embed: host label {label!r} carries gmsh element "
                        f"type {code}, which is not a supported straight-sided "
                        f"host. v1 supports tri3/quad4 (2-D) and tet4/hex8 "
                        f"(3-D), plus their straight-sided higher-order forms "
                        f"(tri6/quad8/quad9/tet10/hex20). Prism / pyramid "
                        f"hosts are deferred."
                    )
                kind, n_corner = _GMSH_HOST_KIND[code]
                full_npe = _GMSH_HOST_FULL_NPE[code]
                conn = np.asarray(nodes, dtype=int).reshape(-1, full_npe)
                for row in conn:
                    corners = [int(n) for n in row[:n_corner]]
                    cc = np.vstack([coord_of[n] for n in corners])
                    host_node_ids.append(corners)
                    host_node_coords.append(cc)
                    host_kinds.append(kind)
                    # Straight-sided check (warn once): compare each EDGE
                    # mid-side node to the midpoint of its two corner endpoints
                    # — a deviation in ANY direction means a curved edge (the
                    # corner linearisation then mislocates embedded nodes). The
                    # edge→corner-pair association is discovered generically
                    # from gmsh's reference parametric coords (an edge node sits
                    # at the parametric midpoint of its two corners), so this
                    # works for tri6/quad8/quad9/tet10/hex20 with no hardcoded
                    # edge tables. Face/centre nodes (no corner pair) are
                    # skipped — edge curvature is the signal.
                    if full_npe > n_corner and not curved_warned:
                        if _host_has_curved_edge(code, full_npe, row, coord_of):
                            _warnings.warn(
                                f"embed: host label {label!r} has CURVED "
                                f"higher-order elements (an edge mid-side node "
                                f"is off its corner-pair midpoint). g.embed "
                                f"linearises hosts to corner sub-elements, so "
                                f"embedded-node location may be inaccurate on a "
                                f"curved host. Use a straight-sided mesh for the "
                                f"host, or verify the resolved ties.",
                                stacklevel=2,
                            )
                            curved_warned = True

        if not host_node_ids:
            raise ValueError(
                f"embed: host label {label!r} resolved to entities but none "
                f"carry supported host elements (is the host meshed?)."
            )
        return host_node_ids, host_node_coords, host_kinds

    def _collect_nodes(self, entities, label):
        """The constrained node-id list (unique, first-seen order).

        Reads the mesh nodes of the ``nodes`` label's entities (any
        dimension — a 0-D point PG, a face PG, a refined-block boundary).
        """
        import gmsh

        seen: dict[int, None] = {}
        for dim, tag in entities:
            try:
                ntags, _, _ = gmsh.model.mesh.getNodes(
                    int(dim), int(tag), includeBoundary=True)
            except Exception as exc:
                raise ValueError(
                    f"embed: cannot get mesh nodes for node entity "
                    f"(dim={dim}, tag={tag}) of label {label!r}: {exc}"
                ) from exc
            for n in ntags:
                seen.setdefault(int(n), None)

        if not seen:
            raise ValueError(
                f"embed: nodes label {label!r} resolved to entities but "
                f"carries no mesh nodes (is it meshed?)."
            )
        return list(seen.keys())

    def _entities_for_label(self, label: str) -> list[tuple[int, int]]:
        """Geometric entities for *label* — part instance, then PG. Fail
        loud (never returns ``[]``). Mirrors
        ``ReinforcementsComposite._entities_for_label``."""
        import gmsh
        parts = getattr(self._parent, "parts", None)
        if parts is not None and label in getattr(parts, "_instances", {}):
            inst = parts._instances[label]
            return [
                (int(dim), int(tag))
                for dim, tags in inst.entities.items()
                for tag in tags
            ]
        ents: list[tuple[int, int]] = []
        pg_dims: set[int] = set()
        for d, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(int(d), int(pg_tag))
            except Exception:
                continue
            if name != label:
                continue
            pg_dims.add(int(d))
            for ent in gmsh.model.getEntitiesForPhysicalGroup(
                    int(d), int(pg_tag)):
                ents.append((int(d), int(ent)))
        if len(pg_dims) > 1:
            raise ValueError(
                f"embed: physical group {label!r} exists at multiple "
                f"dimensions {sorted(pg_dims)}. Assign one dimension per "
                f"group name."
            )
        if not ents:
            raise KeyError(
                f"embed: label {label!r} resolved to neither a g.parts "
                f"instance nor a physical group. Register the part or create "
                f"the physical group before resolving."
            )
        return ents

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_defs(self) -> list[dict]:
        return [
            {"kind": d.kind, "host": d.master_label, "nodes": d.slave_label,
             "enforce": d.enforce, "name": d.name}
            for d in self.embed_defs]

    def clear(self) -> None:
        self.embed_defs.clear()
        self.embed_records.clear()

    def __repr__(self) -> str:
        return (
            f"<EmbedmentsComposite {len(self.embed_defs)} defs, "
            f"{len(self.embed_records)} ties>")
