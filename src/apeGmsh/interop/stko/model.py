"""ScdModel — the contents of an STKO ``.scd`` document, as plain data.

STKO (ASDEA's OpenSees pre/post-processor) saves its CAE document as
HDF5. :func:`~apeGmsh.interop.stko.read_scd` reads it into these frozen
dataclasses; nothing here touches gmsh, OCC or OpenSees.

Addressing follows STKO: a *geometry* (``GEOMETRIES/GEOM_<id>``) is one
child of the document's OCC compound, and its sub-shapes are 0-based
indices per kind (vertices, edges, faces, solids). Selection sets,
property assignments and conditions all address sub-shapes that way.
The mesh links each sub-shape to the elements generated on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Sub-shape kinds, in STKO's order.
KINDS: tuple[str, ...] = ("vertices", "edges", "faces", "solids")

#: STKO mesh element type codes seen in ``MESH/ELEMENTS``.
ELEMENT_TYPES: dict[int, str] = {
    102: "line2",
    303: "quad4",
    401: "hex8",
    500: "hex8",
    600: "link",        # interaction-generated (e.g. embedded node)
}


@dataclass(frozen=True, slots=True)
class SubShapes:
    """0-based sub-shape indices of one geometry, per kind."""

    vertices: tuple[int, ...] = ()
    edges: tuple[int, ...] = ()
    faces: tuple[int, ...] = ()
    solids: tuple[int, ...] = ()

    def of(self, kind: str) -> tuple[int, ...]:
        return getattr(self, kind)

    def __bool__(self) -> bool:
        return any(self.of(k) for k in KINDS)


@dataclass(frozen=True, slots=True)
class XObject:
    """One STKO object: a physical / element property, condition,
    definition or analysis step.

    ``type`` is STKO's ``XOBJ_META`` (``"shell.ASDShellQ4"``,
    ``"Mass.FaceMass"``, …). ``attributes`` maps each parameter name as
    STKO shows it (``"E"``, ``"h"``, ``"Fx"``) to its value: ``bool``,
    ``int``, ``float``, ``str``, a tuple of numbers, ``None`` for an
    empty slot, or a dict of arrays for a nested custom object (section
    outlines, fiber sections). ``INDEX`` / ``INDEX_VEC`` parameters are
    IDs of other objects; their names are listed in ``references``.
    """

    id: int
    name: str
    type: str
    attributes: dict[str, Any]
    references: frozenset[str] = frozenset()

    def __getitem__(self, key: str) -> Any:
        return self.attributes[key]


@dataclass(frozen=True, slots=True)
class Geometry:
    """One geometry: a child of the document's OCC compound.

    ``shape_index`` is its 1-based position among the compound's
    children. ``element_property`` / ``physical_property`` /
    ``local_axes`` give, per kind, one ID per sub-shape (0 = none).
    """

    id: int
    name: str
    shape_index: int
    counts: dict[str, int]
    element_property: dict[str, np.ndarray]
    physical_property: dict[str, np.ndarray]
    local_axes: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SelectionSet:
    id: int
    name: str
    items: dict[int, SubShapes]
    whole: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class Condition:
    """A condition (fixity, mass, load, MP constraint, DRM input) and
    what it is assigned to: sub-shapes and / or interaction IDs."""

    xobject: XObject
    geometry: dict[int, SubShapes] = field(default_factory=dict)
    interactions: tuple[int, ...] = ()

    @property
    def id(self) -> int:
        return self.xobject.id

    @property
    def name(self) -> str:
        return self.xobject.name

    @property
    def type(self) -> str:
        return self.xobject.type


@dataclass(frozen=True, slots=True)
class SubShapeRef:
    """``(geometry, kind code, index)`` as STKO stores interaction
    sides. Kind code 3 is a face; other codes are kept as read."""

    geometry: int
    kind: int
    index: int


@dataclass(frozen=True, slots=True)
class Interaction:
    id: int
    name: str
    type: str                               # "NE" / "NN"
    masters: tuple[SubShapeRef, ...]
    slaves: tuple[SubShapeRef, ...]
    elements: tuple[int, ...] = ()           # generated in the mesh
    element_property: int = 0
    physical_property: int = 0


@dataclass(frozen=True, slots=True)
class LocalAxes:
    id: int
    name: str
    type: str
    origin: tuple[float, float, float]
    x: tuple[float, float, float]
    y: tuple[float, float, float]
    z: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MeshElement:
    type: int
    rule: int
    nodes: tuple[int, ...]

    @property
    def type_name(self) -> str:
        return ELEMENT_TYPES.get(self.type, f"type{self.type}")


@dataclass(frozen=True, slots=True)
class Mesh:
    """STKO's mesh: every meshed entity, analysis elements or not.

    ``domains[(geometry, kind)][index]`` are the element IDs generated
    on that sub-shape; ``vertex_nodes[geometry][i]`` is the node on
    vertex ``i``. ``orientation`` maps element ID to its local-frame
    quaternion ``(x, y, z, w)``.
    """

    node_ids: np.ndarray
    coordinates: np.ndarray
    node_flags: np.ndarray
    elements: dict[int, MeshElement]
    domains: dict[tuple[int, str], dict[int, np.ndarray]]
    vertex_nodes: dict[int, np.ndarray]
    orientation: dict[int, tuple[float, float, float, float]]

    def node_xyz(self, node_id: int) -> np.ndarray:
        index = np.searchsorted(self.node_ids, node_id)
        if index >= len(self.node_ids) or self.node_ids[index] != node_id:
            raise KeyError(f"node {node_id} is not in the mesh")
        return self.coordinates[index]


@dataclass(frozen=True, slots=True)
class AnalysisElement:
    """A mesh element OpenSees receives: one on a sub-shape with an
    element property, or one an interaction generated."""

    id: int
    element: MeshElement
    geometry: int | None            # None for interaction elements
    kind: str | None
    index: int | None
    element_property: int
    physical_property: int
    interaction: int | None = None


@dataclass(frozen=True, slots=True)
class ScdModel:
    path: Path
    version: tuple[int, ...]
    geometries: dict[int, Geometry]
    mesh: Mesh
    selection_sets: dict[str, SelectionSet]
    physical_properties: dict[int, XObject]
    element_properties: dict[int, XObject]
    conditions: dict[int, Condition]
    interactions: dict[int, Interaction]
    local_axes: dict[int, LocalAxes]
    definitions: dict[int, XObject]
    analysis_steps: dict[int, XObject]      # in ID (execution) order

    # ── selection sets → mesh IDs ─────────────────────────────────

    def set_elements(self, name: str) -> set[int]:
        """Element IDs generated on the set's edges, faces and solids
        (the same expansion STKO writes to its ``.mpco.cdata``)."""
        out: set[int] = set()
        for geom, sub in self._set_items(name):
            for kind in ("edges", "faces", "solids"):
                domain = self.mesh.domains.get((geom, kind), {})
                for index in sub.of(kind):
                    out.update(domain.get(index, np.empty(0, int)).tolist())
        return out

    def set_nodes(self, name: str) -> set[int]:
        """Nodes of the set's elements plus the nodes on its vertices."""
        out: set[int] = set()
        for e in self.set_elements(name):
            out.update(self.mesh.elements[e].nodes)
        for geom, sub in self._set_items(name):
            if sub.vertices:
                out.update(self.mesh.vertex_nodes[geom][list(sub.vertices)].tolist())
        return out

    def _set_items(self, name: str) -> list[tuple[int, SubShapes]]:
        try:
            sset = self.selection_sets[name]
        except KeyError:
            raise KeyError(
                f"no selection set {name!r}; the document has "
                f"{sorted(self.selection_sets)}"
            ) from None
        items = dict(sset.items)
        for geom in sset.whole:
            g = self.geometries[geom]
            items[geom] = SubShapes(**{k: tuple(range(g.counts.get(k, 0))) for k in KINDS})
        return list(items.items())

    # ── what OpenSees receives ────────────────────────────────────

    def analysis_elements(self) -> dict[int, AnalysisElement]:
        """The mesh elements that become OpenSees elements: those on a
        sub-shape with an element property, plus interaction-generated
        ones. The rest of the mesh (face meshes of solids, edge meshes
        of shells) only carries geometry."""
        out: dict[int, AnalysisElement] = {}
        for gid, g in self.geometries.items():
            for kind in ("edges", "faces", "solids"):
                eprop = g.element_property.get(kind)
                if eprop is None:
                    continue
                pprop = g.physical_property[kind]
                domain = self.mesh.domains.get((gid, kind), {})
                for index in np.flatnonzero(eprop):
                    for e in domain.get(int(index), ()):
                        out[int(e)] = AnalysisElement(
                            id=int(e), element=self.mesh.elements[int(e)],
                            geometry=gid, kind=kind, index=int(index),
                            element_property=int(eprop[index]),
                            physical_property=int(pprop[index]),
                        )
        for inter in self.interactions.values():
            for e in inter.elements:
                out[e] = AnalysisElement(
                    id=e, element=self.mesh.elements[e],
                    geometry=None, kind=None, index=None,
                    element_property=inter.element_property,
                    physical_property=inter.physical_property,
                    interaction=inter.id,
                )
        return out

    # ── lookups by name ───────────────────────────────────────────

    def physical_property(self, name: str) -> XObject:
        return _by_name(self.physical_properties, name, "physical property")

    def element_property(self, name: str) -> XObject:
        return _by_name(self.element_properties, name, "element property")

    def condition(self, name: str) -> Condition:
        return _by_name(self.conditions, name, "condition")


def _by_name(objects: dict[int, Any], name: str, what: str) -> Any:
    matches = [o for o in objects.values() if o.name == name]
    if len(matches) != 1:
        names = sorted(o.name for o in objects.values())
        raise KeyError(
            f"{len(matches)} {what}(s) named {name!r}; the document has {names}"
        )
    return matches[0]
