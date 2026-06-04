"""
ReinforcementsComposite -- declare embedded reinforcement, resolve to ties.

``g.reinforce(host=..., bars=..., ...)`` is the apeGmsh-side generator for
the Ladruno fork's ``LadrunoEmbeddedRebar`` coupling element (ADR 20). It
embeds a **pre-meshed rebar line physical group** into a **non-matching**
solid host: at FEM-build time each rebar node is inverse-mapped into the
host element it falls inside, producing one
:class:`~apeGmsh._kernel.records._constraints.ReinforceTieRecord` per rebar
node. The bridge build step (``opensees._internal.build.emit_reinforce_ties``)
turns each record into ``element LadrunoEmbeddedRebar`` via the R0
``embedded_rebar_args`` builder.

Two-stage pipeline, mirroring :class:`ConstraintsComposite`:

1. **Declare** (pre-mesh): :meth:`reinforce` (also reachable by calling the
   composite directly — ``g.reinforce(...)``) stores a
   :class:`~apeGmsh._kernel.defs.constraints.ReinforceDef`.
2. **Resolve** (post-mesh): :meth:`resolve` — called by
   ``Mesh.queries.get_fem_data`` — pulls the host elements + rebar segments
   from the live Gmsh session and delegates to
   :func:`~apeGmsh._kernel.resolvers._reinforce.resolve_reinforce`.

Option B layering (locked with the user): this composite is **pure
geometry**, exactly like ``g.constraints.embedded()``. The rebar
``corotTruss`` and the steel / ``LadrunoBondSlip`` materials are declared
**separately on the bridge**; the def references the bond material **by
name** only. The composite never touches OpenSees tags — name → tag
resolution happens at bridge emit time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh._kernel.defs.constraints import ReinforceDef
from apeGmsh._kernel.records._constraints import ReinforceTieRecord
from apeGmsh._kernel.resolvers._reinforce import resolve_reinforce


# gmsh element-type code → (inverse-map host kind, corner-node count).
#
# v1 = straight-sided hosts (``_inverse_map.HOST_KINDS``): the four linear
# kinds map directly; a straight-sided higher-order host maps with its
# **corner** kind + corner subset (gmsh orders corner nodes first), since
# the host's geometry / ξ domain is defined by the corner sub-element.
_GMSH_HOST_KIND: dict[int, tuple[str, int]] = {
    2:  ("tri3", 3),    # tri3
    3:  ("quad4", 4),   # quad4
    4:  ("tet4", 4),    # tet4
    5:  ("hex8", 8),    # hex8
    9:  ("tri3", 3),    # tri6  → corner tri3
    10: ("quad4", 4),   # quad9 → corner quad4
    11: ("tet4", 4),    # tet10 → corner tet4
    16: ("quad4", 4),   # quad8 → corner quad4
    17: ("hex8", 8),    # hex20 → corner hex8
}

# gmsh host element-type code → full per-element node count (corner +
# midside), used to reshape the flat connectivity gmsh returns.
_GMSH_HOST_FULL_NPE: dict[int, int] = {
    2: 3, 3: 4, 4: 4, 5: 8,
    9: 6, 10: 9, 11: 10, 16: 8, 17: 20,
}

# gmsh line-element codes → corner-node count (Line2 / Line3).
_GMSH_LINE_NPE: dict[int, int] = {1: 2, 8: 2}


class ReinforcementsComposite:
    """Embedded-reinforcement generator — declare on geometry, resolve to
    ``LadrunoEmbeddedRebar`` ties after meshing.

    Examples
    --------
    Embed a pre-meshed rebar line PG into a non-matching concrete hex
    block, perfect bond::

        g.reinforce(host="concrete", bars="rebar", perfect=1.0e12,
                    bar_diameter=0.025)

    Bond-slip law (declared separately on the bridge, referenced by name)::

        ops.uniaxialMaterial.LadrunoBondSlip(name="bond1", ...)
        ...
        g.reinforce(host="concrete", bars="rebar",
                    bond="bond1", bar_diameter=0.025)
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.reinforce_defs: list[ReinforceDef] = []
        self.reinforce_records: list[ReinforceTieRecord] = []

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs) -> ReinforceDef:
        """``g.reinforce(...)`` — alias for :meth:`reinforce`."""
        return self.reinforce(*args, **kwargs)

    def reinforce(
        self, host, bars, *,
        bond=None, perfect=None,
        bar_diameter=None, bar_area=None,
        kt=None, kt_alpha=None,
        enforce="penalty", bipenalty=False, dtcr=None,
        tolerance=1.0e-6, snap=False,
        host_entities=None, bars_entities=None,
        name=None,
    ) -> ReinforceDef:
        """Declare an embedded-reinforcement tie.

        Parameters
        ----------
        host : str
            The solid host physical group / part label (3-D hex/tet, or
            2-D quad/tri for a 2-D model).
        bars : str
            The pre-meshed rebar **line** physical group / part label
            (Line2 segments). The generator inverse-maps each of its
            nodes into ``host``.
        bond : str, optional
            Name of a ``LadrunoBondSlip`` material declared on the bridge,
            for the axial τ–s law. Mutually exclusive with ``perfect``.
            Needs ``bar_diameter`` / ``bar_area`` for
            ``bondScale = π·d_b·L_trib``.
        perfect : float, optional
            Perfect-bond axial penalty ``kAxial``. Mutually exclusive with
            ``bond``.
        bar_diameter, bar_area : float, optional
            Bar geometry for ``bondScale`` (``bar_area`` derives
            ``d_b = 2·√(A/π)``).
        kt, kt_alpha : float, optional
            Transverse penalty (``-kt`` / ``-ktAlpha``). ``"auto"`` is
            deferred (needs the ``-xi`` host-query path); pass a numeric
            value or leave ``None`` (fork default).
        enforce : {"penalty", "al"}
            Constraint enforcement (default ``"penalty"``). ``"al"``
            (augmented Lagrangian) gives near-exact bond at a moderate
            transverse penalty.
        bipenalty : bool
            Enable explicit bipenalty critical-time-step control
            (``-bipenalty``). Penalty-enforcement only; needs ``dtcr``.
        dtcr : float, optional
            The explicit critical-time-step budget for ``bipenalty``
            (``-dtcr``). The fork keeps the coupling from shrinking the
            explicit step below this. (The ``-wcap`` host-frequency form is
            deferred with the ``-xi`` path.)
        tolerance : float
            Inverse-map acceptance threshold on the parametric excess
            (ADR 20 D3).
        snap : bool
            ``False`` (default) → a rebar node outside every host raises;
            ``True`` → project it onto the nearest host + warn.
        host_entities, bars_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities; when omitted the
            whole label is used.
        name : str, optional
            Friendly name (round-trips into the emitted deck comment).

        Returns
        -------
        ReinforceDef
        """
        defn = ReinforceDef(
            master_label=host, slave_label=bars,
            host_entities=host_entities, bars_entities=bars_entities,
            bond=bond, perfect=perfect,
            bar_diameter=bar_diameter, bar_area=bar_area,
            kt=kt, kt_alpha=kt_alpha,
            enforce=enforce, bipenalty=bipenalty, dtcr=dtcr,
            tolerance=tolerance, snap=snap,
            name=name,
        )
        self.reinforce_defs.append(defn)
        return defn

    def validate_pre_mesh(self) -> None:
        """No-op — reinforcement resolves at ``get_fem_data`` time."""
        return None

    # ------------------------------------------------------------------
    # Resolution (post-mesh)
    # ------------------------------------------------------------------
    def resolve(self, node_tags, node_coords) -> list[ReinforceTieRecord]:
        """Resolve every :meth:`reinforce` def to ``ReinforceTieRecord``\\ s.

        Pulls the host elements (full node lists + kind) and the rebar
        line segments from the live Gmsh session, builds a tag → coord
        map from ``node_tags`` / ``node_coords``, and delegates each def
        to :func:`resolve_reinforce`. Fail-loud throughout (a stray rebar
        node, an empty host, an unsupported host kind all raise).
        """
        records: list[ReinforceTieRecord] = []
        if not self.reinforce_defs:
            self.reinforce_records = records
            return records

        coord_of = {
            int(t): np.asarray(node_coords[i], dtype=float)
            for i, t in enumerate(node_tags)
        }

        for defn in self.reinforce_defs:
            host_entities = (
                defn.host_entities if defn.host_entities
                else self._entities_for_label(defn.master_label)
            )
            bars_entities = (
                defn.bars_entities if defn.bars_entities
                else self._entities_for_label(defn.slave_label)
            )

            host_node_ids, host_node_coords, host_kinds = \
                self._collect_hosts(host_entities, coord_of, defn.master_label)
            bar_node_ids, bar_segments = \
                self._collect_bars(bars_entities, defn.slave_label)
            bar_node_coords = np.vstack([coord_of[n] for n in bar_node_ids])

            records.extend(resolve_reinforce(
                bar_node_ids=bar_node_ids,
                bar_node_coords=bar_node_coords,
                bar_segments=bar_segments,
                host_node_ids=host_node_ids,
                host_node_coords=host_node_coords,
                host_kinds=host_kinds,
                bond=defn.bond,
                perfect=defn.perfect,
                diameter=defn.diameter,
                kt=defn.kt,
                kt_alpha=defn.kt_alpha,
                enforce=defn.enforce,
                bipenalty=defn.bipenalty,
                dtcr=defn.dtcr,
                tolerance=defn.tolerance,
                snap=defn.snap,
                name=defn.name,
            ))

        self.reinforce_records = records
        return records

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------
    def _collect_hosts(self, entities, coord_of, label):
        """Per host element: (node-id list, (n,3) coord array, host kind).

        Higher-order straight-sided hosts are reduced to their corner
        subset + corner kind (the ``-shape`` weights then couple to the
        corner nodes). Unsupported host kinds (prism / pyramid in v1)
        fail loud.
        """
        import gmsh

        host_node_ids: list[list[int]] = []
        host_node_coords: list[np.ndarray] = []
        host_kinds: list[str] = []

        for dim, tag in entities:
            try:
                etypes, _, enodes = gmsh.model.mesh.getElements(
                    dim=int(dim), tag=int(tag))
            except Exception as exc:
                raise ValueError(
                    f"reinforce: cannot get mesh elements for host entity "
                    f"(dim={dim}, tag={tag}) of label {label!r}: {exc}"
                ) from exc
            for etype, nodes in zip(etypes, enodes):
                if len(nodes) == 0:
                    continue
                code = int(etype)
                if code not in _GMSH_HOST_KIND:
                    raise ValueError(
                        f"reinforce: host label {label!r} carries gmsh "
                        f"element type {code}, which is not a supported "
                        f"straight-sided host. v1 supports tri3/quad4 (2-D) "
                        f"and tet4/hex8 (3-D), plus their straight-sided "
                        f"higher-order forms (tri6/quad8/quad9/tet10/hex20). "
                        f"Prism / pyramid hosts are deferred."
                    )
                kind, n_corner = _GMSH_HOST_KIND[code]
                full_npe = _GMSH_HOST_FULL_NPE[code]
                conn = np.asarray(nodes, dtype=int).reshape(-1, full_npe)
                for row in conn:
                    corners = [int(n) for n in row[:n_corner]]
                    host_node_ids.append(corners)
                    host_node_coords.append(
                        np.vstack([coord_of[n] for n in corners]))
                    host_kinds.append(kind)

        if not host_node_ids:
            raise ValueError(
                f"reinforce: host label {label!r} resolved to entities but "
                f"none carry supported host elements (is the host meshed?)."
            )
        return host_node_ids, host_node_coords, host_kinds

    def _collect_bars(self, entities, label):
        """(rebar node-id list, list of (a, b) segment node pairs).

        Reads the 1-D Line2 / Line3 elements of the rebar PG. Line3 uses
        its two corner endpoints (the midside node is dropped — the rebar
        ``corotTruss`` and the coupling are straight-segment in v1).
        Node ids are returned in first-seen order for deterministic
        record ordering.
        """
        import gmsh

        seen: dict[int, None] = {}
        segments: list[tuple[int, int]] = []

        for dim, tag in entities:
            if int(dim) != 1:
                continue
            try:
                etypes, _, enodes = gmsh.model.mesh.getElements(
                    dim=1, tag=int(tag))
            except Exception as exc:
                raise ValueError(
                    f"reinforce: cannot get mesh elements for rebar entity "
                    f"(dim=1, tag={tag}) of label {label!r}: {exc}"
                ) from exc
            for etype, nodes in zip(etypes, enodes):
                code = int(etype)
                if code not in _GMSH_LINE_NPE or len(nodes) == 0:
                    continue
                full_npe = 2 if code == 1 else 3
                conn = np.asarray(nodes, dtype=int).reshape(-1, full_npe)
                for row in conn:
                    a, b = int(row[0]), int(row[1])
                    segments.append((a, b))
                    seen.setdefault(a, None)
                    seen.setdefault(b, None)

        if not segments:
            raise ValueError(
                f"reinforce: rebar label {label!r} resolved to entities but "
                f"carries no 1-D line elements — the bars must be a "
                f"pre-meshed Line physical group."
            )
        return list(seen.keys()), segments

    def _entities_for_label(self, label: str) -> list[tuple[int, int]]:
        """Geometric entities for *label* — part instance, then PG. Fail
        loud (never returns ``[]``). Mirrors
        ``ConstraintsComposite._entities_for_label``."""
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
                f"reinforce: physical group {label!r} exists at multiple "
                f"dimensions {sorted(pg_dims)}. Assign one dimension per "
                f"group name."
            )
        if not ents:
            raise KeyError(
                f"reinforce: label {label!r} resolved to neither a g.parts "
                f"instance nor a physical group. Register the part or "
                f"create the physical group before resolving."
            )
        return ents

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_defs(self) -> list[dict]:
        return [
            {"kind": d.kind, "host": d.master_label, "bars": d.slave_label,
             "bond": d.bond, "perfect": d.perfect, "name": d.name}
            for d in self.reinforce_defs]

    def clear(self) -> None:
        self.reinforce_defs.clear()
        self.reinforce_records.clear()

    def __repr__(self) -> str:
        return (
            f"<ReinforcementsComposite {len(self.reinforce_defs)} defs, "
            f"{len(self.reinforce_records)} ties>")
