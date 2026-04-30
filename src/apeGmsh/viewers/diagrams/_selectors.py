"""SlabSelector — frozen record of how a Diagram picks its data subset.

Mirrors the selection vocabulary of ``Results.nodes.get(...)`` /
``Results.elements.gauss.get(...)``: a selector carries one of
``pg=``, ``label=``, ``selection=``, or ``ids=`` plus the canonical
component name. A Diagram resolves its selector to concrete node /
element IDs **once at attach time** (never re-resolved on step change)
so per-step reads stay scoped.

The component is the canonical name (e.g. ``"displacement_x"``), never
a shorthand. Shorthand expansion happens upstream — the user picks a
component in the Add Diagram dialog from a list of canonical names
that the bound stage / level exposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


@dataclass(frozen=True)
class SlabSelector:
    """Where data comes from, scoped to one canonical component.

    At most one of ``pg``, ``label``, ``selection``, ``ids`` may be
    set. If none are set, the selector resolves to "all nodes" /
    "all elements" depending on the topology level the Diagram
    consumes.

    Attributes
    ----------
    component
        Canonical component name. Never a shorthand.
    pg, label, selection
        Names of physical groups, labels, or post-mesh selection sets.
        A tuple resolves to the union.
    ids
        Raw node / element IDs. Bypasses ``pg/label/selection``
        resolution — useful for pick-driven diagram creation.
    """

    component: str
    pg: tuple[str, ...] | None = None
    label: tuple[str, ...] | None = None
    selection: tuple[str, ...] | None = None
    ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        named = [x for x in (self.pg, self.label, self.selection, self.ids)
                 if x is not None]
        if len(named) > 1:
            raise ValueError(
                "SlabSelector accepts at most one of "
                "pg=, label=, selection=, ids= (got "
                f"{len([x for x in (self.pg, self.label, self.selection) if x])} named "
                f"+ {'ids' if self.ids is not None else 'no ids'})."
            )
        if not isinstance(self.component, str) or not self.component:
            raise ValueError(
                f"SlabSelector.component must be a non-empty string, "
                f"got {self.component!r}."
            )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_node_ids(self, fem: "FEMData") -> Optional[ndarray]:
        """Return a node ID array, or None for 'all nodes'."""
        if self.ids is not None:
            return np.asarray(self.ids, dtype=np.int64)
        if self._is_unrestricted():
            return None

        out: list[ndarray] = []
        if self.pg is not None:
            for name in self.pg:
                out.append(np.asarray(
                    fem.nodes.physical.node_ids(name), dtype=np.int64,
                ))
        elif self.label is not None:
            for name in self.label:
                out.append(np.asarray(
                    fem.nodes.labels.node_ids(name), dtype=np.int64,
                ))
        elif self.selection is not None:
            store = getattr(fem, "mesh_selection", None)
            if store is None:
                raise RuntimeError(
                    "selection= requires fem.mesh_selection — none captured "
                    "in this FEMData snapshot."
                )
            for name in self.selection:
                out.append(np.asarray(
                    store.node_ids(name), dtype=np.int64,
                ))
        if not out:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(out))

    def resolve_element_ids(self, fem: "FEMData") -> Optional[ndarray]:
        """Return an element ID array, or None for 'all elements'."""
        if self.ids is not None:
            return np.asarray(self.ids, dtype=np.int64)
        if self._is_unrestricted():
            return None

        out: list[ndarray] = []
        if self.pg is not None:
            for name in self.pg:
                out.append(np.asarray(
                    fem.elements.physical.element_ids(name), dtype=np.int64,
                ))
        elif self.label is not None:
            for name in self.label:
                out.append(np.asarray(
                    fem.elements.labels.element_ids(name), dtype=np.int64,
                ))
        elif self.selection is not None:
            store = getattr(fem, "mesh_selection", None)
            if store is None:
                raise RuntimeError(
                    "selection= requires fem.mesh_selection — none captured "
                    "in this FEMData snapshot."
                )
            for name in self.selection:
                out.append(np.asarray(
                    store.element_ids(name), dtype=np.int64,
                ))
        if not out:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(out))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_unrestricted(self) -> bool:
        return (
            self.pg is None and self.label is None
            and self.selection is None and self.ids is None
        )

    def short_label(self) -> str:
        """Compact one-line description for the Diagrams tab list."""
        if self.ids is not None:
            return f"{self.component} [{len(self.ids)} ids]"
        if self.pg is not None:
            return f"{self.component} @ pg={'+'.join(self.pg)}"
        if self.label is not None:
            return f"{self.component} @ label={'+'.join(self.label)}"
        if self.selection is not None:
            return f"{self.component} @ sel={'+'.join(self.selection)}"
        return f"{self.component} (all)"


def normalize(
    *,
    component: str,
    pg: str | tuple[str, ...] | None = None,
    label: str | tuple[str, ...] | None = None,
    selection: str | tuple[str, ...] | None = None,
    ids: tuple[int, ...] | list[int] | ndarray | None = None,
) -> SlabSelector:
    """Build a SlabSelector from loose user inputs.

    Strings auto-tuple; lists / arrays of IDs convert to int tuples.
    """
    def _as_tuple(x):
        if x is None:
            return None
        if isinstance(x, str):
            return (x,)
        return tuple(x)

    if ids is not None:
        ids_t = tuple(int(i) for i in ids)
    else:
        ids_t = None

    return SlabSelector(
        component=component,
        pg=_as_tuple(pg),
        label=_as_tuple(label),
        selection=_as_tuple(selection),
        ids=ids_t,
    )
