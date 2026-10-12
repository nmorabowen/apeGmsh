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
:class:`apeGmsh.opensees._internal.build.records.FixRecord` and the same
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

from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from apeGmsh.mesh.FEMData import FEMData

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

    def region(self, name: str) -> None:
        """Assign this node to the named OpenSees Region.

        Multiple ``.region(name)`` calls with the same ``name`` (from
        any node or set) accumulate into one ``region $tag -node ...``
        line at build time, with a freshly allocated tag.
        ``Emitter.region()`` (ADR 0024) is the underlying Protocol
        method.
        """
        self._bridge.region(name=name, nodes=(self.tag,))

    # -- Live read-back (ADR 0115 D2) ---------------------------------------

    def read(self, component: str) -> float:
        """``component`` at this node, in model units, at the last
        committed state of the bridge's live :meth:`~apeSees.analyze`.

        ``component`` is a canonical results name (``displacement_x``,
        ``rotation_z``, ``velocity_y``, ``acceleration_x``,
        ``reaction_force_y``, ``reaction_moment_z``, ...), mapped to this
        node's DOF through the bridge's DOF layout for its ``ndf``; a
        reaction read calls ``reactions()`` first.

        Raises :class:`BridgeError` for an unknown component, a
        component the node lacks (naming the node and its ``ndf``), when
        no live analyze has run on the bridge (a deck run is read with
        ``ops.results(path)``), and when the session is stale: an
        ``eigen``, another ``analyze``/``run``, a deck run or another
        bridge rebuilt the domain since. A rotational component
        (``rotation_*``, ``reaction_moment_*``, ``angular_*``) is read
        only on a node a beam-column or shell touches, and no element
        other than a truss: on a u-p node (``ndm=2, ndf=3``) that DOF
        slot is pore pressure, and on any other family it is undefined,
        so the read raises.

        The staleness guard sees the bridge's own live work only: a raw
        ``get_ops().wipe()`` followed by a raw rebuild of the domain is
        outside it, and a read may then answer from that other domain.
        """
        from ._readback import read_nodes

        return float(read_nodes(
            self._bridge, (self.tag,), component,
            where=f"Node {self.tag}.read")[0])

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

    def region(self, name: str) -> None:
        """Assign every node in the set to the named OpenSees Region.

        Equivalent to one ``apeSees.region(name=..., nodes=tags)`` call
        with the set's tags; see :meth:`Node.region`.
        """
        if not self._nodes:
            return
        self._bridge.region(
            name=name,
            nodes=tuple(n.tag for n in self._nodes),
        )

    # -- Live read-back (ADR 0115 D2) ---------------------------------------

    def read(self, component: str) -> "np.ndarray":
        """``component`` at every node, ``ndarray (N,)`` in :attr:`tags`
        order; see :meth:`Node.read`. A sum over a group is plain numpy:
        ``ops.nodes.get(pg="base").read("reaction_force_y").sum()``.
        """
        from ._readback import read_nodes

        return read_nodes(
            self._bridge, self.tags, component, where="NodeSet.read")

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
        label: str | Sequence[str] | None = None,
    ) -> "Node | NodeSet":
        """Return a :class:`Node` (tag) or :class:`NodeSet` (PG, label, all).

        At most one selector:

        - ``tag=N`` → :class:`Node` with that tag (raises ``KeyError``
          if absent from the FEM).
        - ``pg=name`` → :class:`NodeSet` of every node in the PG
          (raises ``KeyError`` if the PG is unknown).
        - ``label=name`` (or a list of names) → :class:`NodeSet` of
          every node under the geometry label(s), in snapshot order
          (ADR 0115 D4). An unknown label raises :class:`BridgeError`
          listing the labels that exist; it never falls back to a
          physical group of the same name.
        - no args → :class:`NodeSet` of every FEM node.

        A ``pg=`` / ``label=`` selection is always a set, whatever its
        size (ADR 0115 D5); :meth:`one` is the single-node form.
        """
        given = [k for k, v in (("tag", tag), ("pg", pg), ("label", label))
                 if v is not None]
        if len(given) > 1:
            raise ValueError(
                "ops.nodes.get: supply at most one of tag=, pg= or label= "
                f"(got {', '.join(given)})."
            )
        if tag is not None:
            return self._node_by_tag(tag)
        if pg is not None:
            return self._set_by_pg(pg)
        if label is not None:
            return self._set_by_ids(_resolve_label(
                self._bridge.fem, label, where="ops.nodes.get"))
        return self._all()

    def one(
        self,
        *,
        pg: str | None = None,
        label: str | Sequence[str] | None = None,
    ) -> Node:
        """Return the single :class:`Node` a selector picks (ADR 0115 D2/D5).

        Exactly one of ``pg=`` / ``label=``. Raises :class:`BridgeError`,
        naming the count, unless exactly one node is selected; use
        :meth:`get` for a set.
        """
        from ._internal.build.errors import BridgeError

        if pg is not None and label is None:
            found = self._set_by_pg(pg)
            what = f"pg={pg!r}"
        elif label is not None and pg is None:
            found = self._set_by_ids(_resolve_label(
                self._bridge.fem, label, where="ops.nodes.one"))
            what = f"label={label!r}"
        else:
            raise ValueError(
                "ops.nodes.one: supply exactly one of pg= or label= "
                f"(got pg={pg!r}, label={label!r})."
            )
        if len(found) != 1:
            raise BridgeError(
                f"ops.nodes.one({what}) selects {len(found)} nodes, not "
                f"exactly one (tags {list(found.tags)[:10]}"
                f"{' ...' if len(found) > 10 else ''}). Use "
                "ops.nodes.get(...) for a set."
            )
        return found[0]

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
            # selection-unification v2 P3-R / §6.3 §2 #8 (P-NODE+
            # P-COORD; m3 — resolution raises at .select()).
            result = self._bridge.fem.nodes.select(pg=pg)
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

    def _set_by_ids(self, ids: tuple[int, ...]) -> NodeSet:
        return NodeSet(
            nodes=tuple(self._node_by_tag(int(n)) for n in ids),
            bridge=self._bridge,
        )

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


def _resolve_label(
    fem: "FEMData", label: "str | Sequence[str]", *, where: str,
) -> tuple[int, ...]:
    """Node ids under the geometry label(s) ``label``, in snapshot order
    (ADR 0115 D4).

    ``label`` is one name or a non-empty list of names; several names
    union, first-seen order kept.  The lookup is the snapshot's own
    ``fem.nodes.select(label=...)``: a label never falls back to a
    physical group (or a part) of the same name.

    Raises
    ------
    ValueError
        ``label`` is empty, or a name is not a non-blank string.
    BridgeError
        A name is not a label of the snapshot (the message lists the
        labels that exist), or the labels select no node.
    """
    from ._internal.build.errors import BridgeError

    if isinstance(label, str):
        names: tuple[str, ...] = (label,)
    elif isinstance(label, (list, tuple)):
        names = tuple(label)
    else:
        raise ValueError(
            f"{where}: label= must be a string or a list of strings, got "
            f"{type(label).__name__} ({label!r})."
        )
    if not names:
        raise ValueError(f"{where}: label= is empty; name at least one label.")
    bad = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad:
        raise ValueError(
            f"{where}: every label must be a non-blank string, got {bad!r}."
        )
    labels = fem.nodes.labels
    missing = [n for n in names if n not in labels]
    if missing:
        raise BridgeError(
            f"{where}: no label {', '.join(repr(n) for n in missing)} on the "
            f"FEM snapshot. Labels that exist: {sorted(set(labels.names()))}. "
            "A label never falls back to a physical group of the same name."
        )
    selector: "str | list[str]" = names[0] if len(names) == 1 else list(names)
    out = tuple(int(n) for n in fem.nodes.select(label=selector).ids)
    if not out:
        raise BridgeError(
            f"{where}: label={label!r} selects 0 nodes on the FEM snapshot."
        )
    return out


def _select_nodes(
    fem: "FEMData",
    *,
    pg: str | None,
    nodes: "Iterable[int | Node] | None",
    label: "str | Sequence[str] | None",
    where: str,
) -> tuple[str | None, tuple[int, ...] | None]:
    """The ``(pg, nodes)`` pair a ``fix`` / ``mass`` record stores, from
    exactly one of ``pg=`` / ``nodes=`` / ``label=`` (ADR 0115 D4).

    A label resolves here, at the call, to its node ids, so every
    refusal (no selector, two selectors, an empty or unknown label, a
    label with no nodes) raises before the caller registers anything.
    The record then carries those ids as ``nodes``: it emits exactly
    the lines ``pg=`` over the same nodes emits (Invariant 6), and the
    record classes, the deck and the archive of every existing model
    are unchanged (Invariants 7 and 9).
    """
    given = sum(x is not None for x in (pg, nodes, label))
    if given != 1:
        raise ValueError(
            f"{where}: supply exactly one of pg= or nodes= (or label=) "
            f"(got pg={pg!r}, nodes={nodes!r}, label={label!r})."
        )
    if label is not None:
        return None, _resolve_label(fem, label, where=where)
    return pg, (_iter_tags(nodes) if nodes is not None else None)
