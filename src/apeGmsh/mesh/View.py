from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tag = int


class View:
    """
    Solver-agnostic post-processing view composite attached to a ``apeGmsh``
    instance as ``g.view``.

    Wraps ``gmsh.view`` to inject scalar and vector fields onto the active
    mesh.  **No solver dependency** — you compute the result arrays yourself
    and pass them in.

    Usage::

        # Element-wise scalar (constant per element)
        g.view.add_element_scalar("VonMises", elem_tags, values)

        # Nodal scalar (smooth contour via Gmsh interpolation)
        g.view.add_node_scalar("sigma_xx avg", node_tags, values)

        # Nodal vector (displacement arrows / deformed shape)
        g.view.add_node_vector("Displacement", node_tags, vectors)

    All ``add_*`` methods return the Gmsh view tag (``int``).

    Parameters
    ----------
    parent : _SessionBase
        Owning instance — used for ``name`` and ``_verbose``.
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self._views: dict[Tag, str] = {}          # view_tag -> name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[View] {msg}")

    # ------------------------------------------------------------------
    # ElementData
    # ------------------------------------------------------------------

    def add_element_scalar(
        self,
        name       : str,
        elem_tags  : list[int] | ndarray,
        values     : list[float] | ndarray,
        *,
        step       : int   = 0,
        time       : float = 0.0,
    ) -> Tag:
        """
        Add a scalar field with one value per element.

        Parameters
        ----------
        name      : view name shown in the Gmsh GUI sidebar
        elem_tags : Gmsh element tags (from ``g.mesh.get_elements``)
        values    : one scalar per element, same order as *elem_tags*

        Returns
        -------
        int  Gmsh view tag
        """
        tags = [int(t) for t in elem_tags]
        data = [[float(v)] for v in values]

        v = gmsh.view.add(name)
        gmsh.view.addModelData(
            v, step, self._parent.name, "ElementData",
            tags, data, time, 1,
        )
        gmsh.view.option.setNumber(v, "IntervalsType", 3)  # continuous map

        self._views[v] = name
        self._log(f"add_element_scalar({name!r}) -> view {v}  "
                  f"({len(tags)} elements)")
        return v

    def add_element_vector(
        self,
        name       : str,
        elem_tags  : list[int] | ndarray,
        vectors    : ndarray,
        *,
        step       : int   = 0,
        time       : float = 0.0,
    ) -> Tag:
        """
        Add a vector field with one 3-component vector per element.

        Parameters
        ----------
        vectors : shape ``(nElem, 3)`` — ``[vx, vy, vz]`` per element
        """
        tags = [int(t) for t in elem_tags]
        data = [[float(vectors[i, 0]), float(vectors[i, 1]), float(vectors[i, 2])]
                for i in range(len(tags))]

        v = gmsh.view.add(name)
        gmsh.view.addModelData(
            v, step, self._parent.name, "ElementData",
            tags, data, time, 3,
        )
        self._views[v] = name
        self._log(f"add_element_vector({name!r}) -> view {v}")
        return v

    # ------------------------------------------------------------------
    # NodeData
    # ------------------------------------------------------------------

    def add_node_scalar(
        self,
        name       : str,
        node_tags  : list[int] | ndarray,
        values     : list[float] | ndarray,
        *,
        step       : int   = 0,
        time       : float = 0.0,
    ) -> Tag:
        """
        Add a scalar field with one value per node.

        Parameters
        ----------
        node_tags : Gmsh node tags
        values    : one scalar per node, same order as *node_tags*
        """
        tags = [int(t) for t in node_tags]
        data = [[float(v)] for v in values]

        v = gmsh.view.add(name)
        gmsh.view.addModelData(
            v, step, self._parent.name, "NodeData",
            tags, data, time, 1,
        )
        gmsh.view.option.setNumber(v, "IntervalsType", 3)

        self._views[v] = name
        self._log(f"add_node_scalar({name!r}) -> view {v}  ({len(tags)} nodes)")
        return v

    def add_node_vector(
        self,
        name       : str,
        node_tags  : list[int] | ndarray,
        vectors    : ndarray,
        *,
        step       : int   = 0,
        time       : float = 0.0,
        vector_type: int   = 5,
    ) -> Tag:
        """
        Add a vector field with one 3-component vector per node.

        Parameters
        ----------
        vectors     : shape ``(nNode, 2)`` or ``(nNode, 3)`` — missing components are zero-padded
        vector_type : Gmsh display style (1=arrows, 2=cones, 5=displacement)
        """
        vecs = np.asarray(vectors)
        if vecs.ndim == 1:
            vecs = vecs.reshape(-1, 1)
        ncols = vecs.shape[1]
        if ncols < 3:
            vecs = np.pad(vecs, ((0, 0), (0, 3 - ncols)))
        tags = [int(t) for t in node_tags]
        data = [[float(vecs[i, 0]), float(vecs[i, 1]), float(vecs[i, 2])]
                for i in range(len(tags))]

        v = gmsh.view.add(name)
        gmsh.view.addModelData(
            v, step, self._parent.name, "NodeData",
            tags, data, time, 3,
        )
        gmsh.view.option.setNumber(v, "VectorType", vector_type)

        self._views[v] = name
        self._log(f"add_node_vector({name!r}) -> view {v}  ({len(tags)} nodes)")
        return v

    # ------------------------------------------------------------------
    # View management
    # ------------------------------------------------------------------

    def list_views(self) -> dict[Tag, str]:
        """Return ``{tag: name}`` for all views created through this class."""
        return dict(self._views)

    def count(self) -> int:
        """Number of views created."""
        return len(self._views)

    def __repr__(self) -> str:
        return f"View(model={self._parent.name!r}, n_views={len(self._views)})"
