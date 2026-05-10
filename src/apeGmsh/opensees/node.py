"""
Node aggregator — typed Python convenience over ``fem.nodes``.

A :class:`Node` instance is a lightweight wrapper around one node from
the FEM snapshot: it carries the tag, coords, and a back-reference to
the bridge so that ``n.fix(dofs=...)`` and ``n.mass(values=...)``
delegate to the bridge's model-level records.

A :class:`NodeSet` is the multi-node analogue — apply ``.fix`` /
``.mass`` to a whole group resolved from a physical group, or iterate
to get individual :class:`Node` instances.

:class:`_NodeAccessor` is what ``ops.nodes`` exposes. It mirrors
:class:`apeGmsh.mesh.FEMData.NodeComposite` shape (``get(tag=...)``,
``get(pg=...)``, container protocol) and adds the bridge-side verbs.

Per P1 (relaxed): the bridge **still** exposes the flat
``ops.fix(pg=..., dofs=...)`` shape. ``Node.fix`` and ``NodeSet.fix``
are convenience layers on top — both routes go through the same
:class:`apeGmsh.opensees._internal.build.FixRecord` and the same
fan-out at emit time.

Examples
--------

.. code-block:: python

    # Single-node query + verbs
    n = ops.nodes.get(tag=2)
    n.fix(dofs=(1, 1, 1, 1, 1, 1))
    n.mass(values=(50, 50, 50, 0, 0, 0))
    n.coords          # (0.0, 0.0, 1.0)
    n.tag             # 2

    # Multi-node query via physical group
    base = ops.nodes.get(pg="Base")
    base.fix(dofs=(1, 1, 1, 1, 1, 1))    # applies to every node in the PG
    base.summary()                        # DataFrame of tag, x, y, z

    # Iteration
    for n in base:
        print(n.tag, n.coords)

    # Inside a pattern — load() accepts a Node directly
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=ops.nodes.get(tag=2), forces=(100e3, 0, 0))
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    import pandas as pd

    from .apesees import apeSees


__all__ = ["Node", "NodeSet", "_NodeAccessor"]


class Node:
    """One node of the FEM snapshot, surfaced as a typed aggregate.

    Construct via :meth:`_NodeAccessor.get` — never directly. The
    instance is a thin wrapper over the bridge's FEM snapshot: the
    tag and coords are read once at lookup time and cached on the
    instance; subsequent verbs delegate back to the bridge.

    Equality is structural on ``(tag, bridge)`` — two nodes from the
    same bridge with the same tag compare equal; from different
    bridges they do not.
    """

    __slots__ = ("tag", "coords", "_bridge")

    def __init__(
        self,
        *,
        tag: int,
        coords: tuple[float, float, float],
        bridge: "apeSees",
    ) -> None:
        self.tag = int(tag)
        self.coords = (float(coords[0]), float(coords[1]), float(coords[2]))
        self._bridge = bridge

    # -- Bridge-side verbs ---------------------------------------------------

    def fix(self, *, dofs: tuple[int, ...]) -> None:
        """Apply a homogeneous SP constraint at this node.

        Equivalent to ``ops.fix(nodes=(self.tag,), dofs=dofs)`` — both
        paths produce the same :class:`FixRecord` consumed at emit time.
        """
        self._bridge.fix(nodes=(self.tag,), dofs=dofs)

    def mass(self, *, values: tuple[float, ...]) -> None:
        """Attach lumped nodal mass at this node."""
        self._bridge.mass(nodes=(self.tag,), values=values)

    # -- Container conveniences ---------------------------------------------

    def __int__(self) -> int:
        """Make Node usable wherever an integer tag is expected.

        Callers passing a Node to a Phase-3/4 API that takes ``int``
        (e.g. internal Tcl/Py rendering) get the tag transparently.
        Prefer the explicit ``.tag`` attribute in new code; this exists
        for forward-compatibility with primitive APIs that haven't
        been updated to accept ``Node``.
        """
        return self.tag

    def __hash__(self) -> int:
        return hash((self.tag, id(self._bridge)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.tag == other.tag and self._bridge is other._bridge

    def __repr__(self) -> str:
        return f"Node(tag={self.tag}, coords={self.coords})"


class NodeSet:
    """A collection of :class:`Node` instances drawn from one bridge.

    Returned by :meth:`_NodeAccessor.get` when a physical group (or no
    selector) is supplied. Verbs (``fix``, ``mass``) apply to every
    node in the set at once; iteration yields individual :class:`Node`
    instances; container protocol mirrors :class:`apeGmsh.mesh.FEMData`
    sets.
    """

    __slots__ = ("_nodes", "_bridge")

    def __init__(
        self,
        *,
        nodes: tuple[Node, ...],
        bridge: "apeSees",
    ) -> None:
        self._nodes = tuple(nodes)
        self._bridge = bridge

    # -- Bridge-side verbs (whole-set) --------------------------------------

    def fix(self, *, dofs: tuple[int, ...]) -> None:
        """Apply ``fix`` to every node in the set.

        Equivalent to a single ``ops.fix(nodes=tuple-of-tags, dofs=dofs)``
        — one record holding the explicit tag list (no PG fan-out at
        emit time, because the resolution already happened here).
        """
        if not self._nodes:
            return
        self._bridge.fix(
            nodes=tuple(n.tag for n in self._nodes),
            dofs=dofs,
        )

    def mass(self, *, values: tuple[float, ...]) -> None:
        """Attach mass to every node in the set."""
        if not self._nodes:
            return
        self._bridge.mass(
            nodes=tuple(n.tag for n in self._nodes),
            values=values,
        )

    # -- Container protocol --------------------------------------------------

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __bool__(self) -> bool:
        return bool(self._nodes)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Node):
            return item in self._nodes
        if isinstance(item, int):
            return any(n.tag == item for n in self._nodes)
        return False

    def __getitem__(self, idx: int) -> Node:
        return self._nodes[idx]

    # -- Introspection -------------------------------------------------------

    def summary(self) -> "pd.DataFrame":
        """Return a DataFrame with columns ``tag``, ``x``, ``y``, ``z``."""
        import pandas as pd
        return pd.DataFrame(
            [
                {"tag": n.tag, "x": n.coords[0], "y": n.coords[1], "z": n.coords[2]}
                for n in self._nodes
            ]
        )

    @property
    def tags(self) -> tuple[int, ...]:
        """Tuple of node tags in the set's iteration order."""
        return tuple(n.tag for n in self._nodes)

    def __repr__(self) -> str:
        return f"NodeSet(n={len(self)})"


class _NodeAccessor:
    """Backs ``ops.nodes`` — accessor over the FEM snapshot's nodes.

    The accessor never holds nodes itself; it builds :class:`Node` /
    :class:`NodeSet` instances on demand from ``self._bridge.fem.nodes``.
    Apart from rare hot loops, this is fine — the FEM snapshot is the
    source of truth and ``_NodeAccessor`` is a thin facade.
    """

    __slots__ = ("_bridge",)

    def __init__(self, bridge: "apeSees") -> None:
        self._bridge = bridge

    # -- Selection -----------------------------------------------------------

    def get(
        self,
        *,
        tag: int | None = None,
        pg: str | None = None,
    ) -> "Node | NodeSet":
        """Return a :class:`Node` (single tag) or :class:`NodeSet` (PG / all).

        Exactly-one-of-or-neither:

        - ``tag=N`` → :class:`Node` with that tag (raises ``KeyError``
          if absent from the FEM).
        - ``pg=name`` → :class:`NodeSet` of every node in the PG
          (raises ``KeyError`` if the PG is unknown).
        - no args → :class:`NodeSet` of every FEM node.
        """
        if tag is not None and pg is not None:
            raise ValueError(
                "ops.nodes.get: supply at most one of tag= or pg= "
                f"(got tag={tag!r}, pg={pg!r})."
            )
        if tag is not None:
            return self._node_by_tag(tag)
        if pg is not None:
            return self._set_by_pg(pg)
        return self._all()

    # -- Container protocol --------------------------------------------------

    def __iter__(self) -> Iterator[Node]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._bridge.fem.nodes.ids)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Node):
            return item._bridge is self._bridge and self._has_tag(item.tag)
        if isinstance(item, int):
            return self._has_tag(item)
        return False

    def __getitem__(self, tag: int) -> Node:
        """``ops.nodes[N]`` shorthand for ``ops.nodes.get(tag=N)``."""
        return self._node_by_tag(int(tag))

    # -- Introspection -------------------------------------------------------

    def summary(self) -> "pd.DataFrame":
        """DataFrame of every FEM node — columns ``tag``, ``x``, ``y``, ``z``."""
        return self._all().summary()

    def __repr__(self) -> str:
        return f"_NodeAccessor(n={len(self)})"

    # -- Internal helpers ----------------------------------------------------

    def _has_tag(self, tag: int) -> bool:
        try:
            self._bridge.fem.nodes.index(tag)
        except KeyError:
            return False
        return True

    def _node_by_tag(self, tag: int) -> Node:
        nodes = self._bridge.fem.nodes
        try:
            idx = nodes.index(tag)
        except KeyError as e:
            raise KeyError(
                f"ops.nodes.get(tag={tag}): no such node in the FEM snapshot."
            ) from e
        xyz = nodes.coords[idx]
        return Node(
            tag=int(tag),
            coords=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            bridge=self._bridge,
        )

    def _set_by_pg(self, pg: str) -> NodeSet:
        try:
            result = self._bridge.fem.nodes.get(pg=pg)
        except (KeyError, ValueError) as e:
            raise KeyError(
                f"ops.nodes.get(pg={pg!r}): no such node PG in the FEM "
                "snapshot."
            ) from e
        ids = result.ids
        coords = result.coords
        members: list[Node] = []
        for i, nid in enumerate(ids):
            xyz = coords[i]
            members.append(
                Node(
                    tag=int(nid),
                    coords=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    bridge=self._bridge,
                )
            )
        return NodeSet(nodes=tuple(members), bridge=self._bridge)

    def _all(self) -> NodeSet:
        nodes_comp = self._bridge.fem.nodes
        members: list[Node] = []
        for i, nid in enumerate(nodes_comp.ids):
            xyz = nodes_comp.coords[i]
            members.append(
                Node(
                    tag=int(nid),
                    coords=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    bridge=self._bridge,
                )
            )
        return NodeSet(nodes=tuple(members), bridge=self._bridge)


# Helper for primitives that want to accept "node or node tag" — used by
# Plain.load and the bridge's fix() / mass() to normalize their `node=` /
# `nodes=` args.
def _to_tag(node_or_tag: "int | Node") -> int:
    """Coerce a :class:`Node` or plain int to its tag."""
    if isinstance(node_or_tag, Node):
        return node_or_tag.tag
    return int(node_or_tag)


def _iter_tags(nodes: "Iterable[int | Node]") -> tuple[int, ...]:
    """Coerce an iterable of (Node | int) to a tuple of tags."""
    return tuple(_to_tag(n) for n in nodes)
