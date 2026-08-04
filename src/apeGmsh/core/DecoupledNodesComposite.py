"""DecoupledNodesComposite — declare auxiliary (decoupled) nodes.

A *decoupled node* is a node that is **not** a Gmsh mesh vertex —
a spring/dashpot ground for SSI, a ``rigidDiaphragm`` master, a control
node, or a load/mass anchor.  The user declares one on the session and
the FEM factory appends it to the broker's node arrays at extraction
time, assigning a deterministic tag above every mesh node so it is
dedup-immune by construction (it never reaches Gmsh's
``removeDuplicateNodes`` pass — see ADR 0049).

Two-stage pipeline, mirroring :class:`MassesComposite`:

1. **Declare** (pre-mesh): :meth:`add` (reached via the session verb
   ``g.decouple_node(...)``) stores a :class:`DecoupledNodeDef`.
2. **Resolve** (post-mesh): the FEM factory
   (``apeGmsh.mesh._fem_factory``) walks the stored defs, snapshots any
   ``point=`` coordinate, assigns the deterministic tag, appends the row
   to ``fem.nodes``, and writes the tag back onto the def.

PR-4 scope is **identity only** — coordinates + an optional label.  A
decoupled node carries **no** ``ndf``/DOF count: per ADR 0049 the DOF
count lives on the bridge (``ops.ndf``), never on the session or the
neutral broker.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from apeGmsh._kernel.defs.decoupled import DecoupledNodeDef

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession


class DecoupledNodesComposite:
    """Solver-agnostic decoupled-node collector — declare on the
    session, inject into the broker at mesh-extraction time.

    Declared defs land on ``g.decoupled_nodes.node_defs``; the resolved
    rows land on ``fem.nodes`` with ``provenance == "decoupled"`` and
    tags above every mesh node.
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.node_defs: list[DecoupledNodeDef] = []

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        coords: tuple[float, float, float] | None = None,
        point: str | None = None,
        label: str | None = None,
    ) -> DecoupledNodeDef:
        """Declare a decoupled node; return its :class:`DecoupledNodeDef`.

        Exactly one of ``coords`` / ``point`` locates the node:

        * ``coords=(x, y, z)`` — an explicit coordinate triple.
        * ``point="label"`` — a geometry point label, resolved to its
          coordinates at mesh-extraction time (a snapshot, not tracked
          through later transforms).

        ``label`` is an optional friendly name for the node.

        The returned def carries ``tag=None`` until the FEM factory
        resolves the model; after ``g.mesh.queries.get_fem_data(...)`` it
        holds the deterministic tag assigned to the node.
        """
        from ._compose_errors import raise_if_from_h5_session
        raise_if_from_h5_session(self._parent, "g.decouple_node()")
        coords_norm = _validate_location(coords=coords, point=point)
        defn = DecoupledNodeDef(coords=coords_norm, point=point, label=label)
        self.node_defs.append(defn)
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()
        return defn

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.node_defs)

    def __repr__(self) -> str:
        if not self.node_defs:
            return "DecoupledNodesComposite(empty)"
        return f"DecoupledNodesComposite({len(self.node_defs)} defs)"


def _validate_location(
    *,
    coords: tuple[float, float, float] | None,
    point: str | None,
) -> tuple[float, float, float] | None:
    """Validate that exactly one of ``coords`` / ``point`` is given.

    Returns the normalised ``coords`` triple (floats) when ``coords`` was
    passed, else ``None`` (the ``point=`` path is snapshotted later by
    the factory).
    """
    specified = [n for n, v in (("coords", coords), ("point", point))
                 if v is not None]
    if len(specified) != 1:
        raise ValueError(
            "g.decouple_node requires exactly one of coords=(x, y, z) or "
            f"point='label'; got {specified or 'neither'}."
        )
    if coords is not None:
        triple = tuple(coords)
        if len(triple) != 3:
            raise ValueError(
                f"coords= must be a length-3 (x, y, z) tuple; got "
                f"length {len(triple)}."
            )
        return tuple(float(v) for v in triple)
    if not isinstance(point, str) or not point:
        raise ValueError(
            f"point= must be a non-empty geometry-label string; got "
            f"{point!r}."
        )
    return None
