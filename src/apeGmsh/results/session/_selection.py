"""Session selection — nodes XOR Gauss over the ONE store (ADR 0098 §8).

``SessionSelection`` is the owner-facing surface of ADR 0045's
``SelectionState`` / ``SelectionLog`` — never a second store (INV-5).
The session creates the store; the Qt client (S2+) binds to the same
instance via :attr:`state`.

Layering note (S0 decision 1, recorded in the package docstring): this
module imports ``apeGmsh.viewers.core.selection`` directly.

Target encoding (S0 decision 2 — a one-way door into the S5 snapshot,
matching ADR 0045's RESULTS_TOPO table):

* results node  → ``SelectionTarget(RESULTS_TOPO, dim=0, key=node_id)``
* Gauss point   → ``SelectionTarget(RESULTS_TOPO, dim=0,
  key=element_id, sub=gp_index)``

Both are point-like under the 0/1/2/3 dimensional filter, so both take
``dim=0``; the kind discriminator is ``sub`` (``None`` = node, set =
Gauss). ``parent`` (the owning element target, which would carry the
cell dim) stays ``None`` at this surface — S4's post-hit pick resolver
may attach it, an additive widening per 0045 INV-8.

The XOR law: the set is homogeneous and its kind follows the last
writer — ANY write of the other kind replaces the whole set (recorded
as one ``OpKind.SET`` gesture in the real log). Elements are never
targets here (§8: an element is a membership query, not a hit).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from apeGmsh.viewers.core.selection import SelectionState
from apeGmsh.viewers.scene_ir import SelectionTarget, Substrate


def node_target(node_id: int) -> SelectionTarget:
    """A results-node selection target (the S0-decided encoding)."""
    return SelectionTarget(Substrate.RESULTS_TOPO, dim=0, key=int(node_id))


def gauss_target(element_id: int, gp_index: int) -> SelectionTarget:
    """A Gauss-point selection target (the S0-decided encoding)."""
    return SelectionTarget(
        Substrate.RESULTS_TOPO, dim=0, key=int(element_id),
        sub=int(gp_index),
    )


class SessionSelection:
    """Nodes-XOR-Gauss surface over one real ``SelectionState``.

    Four writers (click, window, outline select-all, code) all land
    here; S0 ships the code writer. The per-view Nodes|Gauss radio
    (``MeshView.pick_target``) only AIMS clicks — it neither owns nor
    clears this set.
    """

    def __init__(
        self, on_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        self._state = SelectionState()
        if on_changed is not None:
            self._state.on_changed.append(on_changed)

    @property
    def state(self) -> SelectionState:
        """The one underlying store (ADR 0045 / INV-5). The Qt client
        binds here; nothing may construct a second store."""
        return self._state

    # -- readers -------------------------------------------------------

    @property
    def targets(self) -> list[SelectionTarget]:
        return self._state.targets

    @property
    def kind(self) -> Optional[str]:
        """``"nodes"`` | ``"gauss"`` | ``None`` (empty set). Derived
        from the set — the writers keep it homogeneous by law.

        A heterogeneous or foreign-substrate set means some writer
        bypassed this surface (the raw store enforces no law); that is
        a programming error and reads FAIL LOUD rather than return a
        subset that silently misrepresents the store (probe H26)."""
        targets = self._state.targets
        if not targets:
            return None
        bad = [t for t in targets if t.substrate is not Substrate.RESULTS_TOPO]
        if bad:
            raise ValueError(
                f"Selection store holds {len(bad)} non-results "
                f"target(s) (e.g. {bad[0]!r}) — a writer bypassed the "
                f"SessionSelection surface (ADR 0098 §8)."
            )
        kinds = {t.sub is not None for t in targets}
        if len(kinds) > 1:
            raise ValueError(
                "Selection store holds nodes AND gauss targets — the "
                "set must be homogeneous (ADR 0098 §8 XOR law); a "
                "writer bypassed the SessionSelection surface."
            )
        return "gauss" if kinds.pop() else "nodes"

    @property
    def nodes(self) -> tuple[int, ...]:
        """The selected node ids (empty unless ``kind == "nodes"``)."""
        if self.kind != "nodes":
            return ()
        return tuple(t.key for t in self._state.targets)

    @property
    def gauss(self) -> tuple[tuple[int, int], ...]:
        """The selected ``(element_id, gp_index)`` pairs (empty unless
        ``kind == "gauss"``)."""
        if self.kind != "gauss":
            return ()
        return tuple((t.key, t.sub) for t in self._state.targets)

    # -- writers -------------------------------------------------------

    def set_nodes(self, node_ids: Iterable[int]) -> None:
        """Replace the set with these nodes (one ``SET`` gesture)."""
        self._state.select_batch(
            [node_target(i) for i in node_ids], replace=True,
        )

    def set_gauss(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Replace the set with these Gauss points (one ``SET``)."""
        self._state.select_batch(
            [gauss_target(e, gp) for e, gp in pairs], replace=True,
        )

    def add_nodes(self, node_ids: Iterable[int]) -> None:
        """Extend a node set — or, on a Gauss set, REPLACE it (the §8
        last-writer law: a write of the other kind replaces)."""
        targets = [node_target(i) for i in node_ids]
        if not targets:
            return
        self._state.select_batch(targets, replace=self.kind == "gauss")

    def add_gauss(self, pairs: Iterable[tuple[int, int]]) -> None:
        """Extend a Gauss set — or, on a node set, REPLACE it."""
        targets = [gauss_target(e, gp) for e, gp in pairs]
        if not targets:
            return
        self._state.select_batch(targets, replace=self.kind == "nodes")

    def toggle_node(self, node_id: int) -> None:
        """Ctrl+click on a node: add it, or drop it if already in.

        On a Gauss set this is a write of the other kind, so the §8
        last-writer law applies unchanged — the set is REPLACED by this
        one node rather than becoming heterogeneous.
        """
        if self.kind == "gauss":
            self.set_nodes([node_id])
            return
        self._state.toggle(node_target(node_id))

    def toggle_gauss(self, element_id: int, gp_index: int) -> None:
        """Ctrl+click on a Gauss point — the mirror of
        :meth:`toggle_node`, replacing a node set."""
        if self.kind == "nodes":
            self.set_gauss([(element_id, gp_index)])
            return
        self._state.toggle(gauss_target(element_id, gp_index))

    def clear(self) -> None:
        self._state.clear()

    def __len__(self) -> int:
        return len(self._state)

    def __repr__(self) -> str:
        return f"<SessionSelection {self.kind or 'empty'} ({len(self)})>"


__all__ = ["SessionSelection", "node_target", "gauss_target"]
