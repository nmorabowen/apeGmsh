"""
_Queries — read-only extraction from the live Gmsh mesh.

Accessed via ``g.mesh.queries``.  Returns data rather than chaining:
get_nodes, get_elements, get_element_properties, get_fem_data,
get_element_qualities, quality_report.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import numpy as np
from numpy import ndarray

from .FEMData import FEMData

if TYPE_CHECKING:
    import pandas as pd
    from .Mesh import Mesh


class _Queries:
    """Read-only mesh data extraction and quality reporting."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Nodes / elements
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        *,
        dim              : int  = -1,
        tag              : int  = -1,
        include_boundary : bool = False,
        return_parametric: bool = False,
    ) -> dict:
        """
        Query mesh nodes.

        Returns
        -------
        dict
            ``'tags'``              : ndarray(N,)   — node tags
            ``'coords'``            : ndarray(N, 3) — XYZ coordinates
            ``'parametric_coords'`` : ndarray       — only if requested
        """
        node_tags, coords, param = gmsh.model.mesh.getNodes(
            dim=dim, tag=tag,
            includeBoundary=include_boundary,
            returnParametricCoord=return_parametric,
        )
        result: dict = {
            'tags'  : np.array(node_tags, dtype=np.int64),
            'coords': np.array(coords).reshape(-1, 3),
        }
        if return_parametric and len(param):
            result['parametric_coords'] = np.array(param)
        self._mesh._log(f"get_nodes -> {len(node_tags)} nodes")
        return result

    def get_elements(
        self,
        *,
        dim: int = -1,
        tag: int = -1,
    ) -> dict:
        """
        Query mesh elements.

        Returns
        -------
        dict
            ``'types'``     : list[int]         — gmsh element type codes
            ``'tags'``      : list[ndarray]     — element tags per type
            ``'node_tags'`` : list[ndarray]     — connectivity per type
        """
        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(
            dim=dim, tag=tag
        )
        result = {
            'types'    : list(elem_types),
            'tags'     : [np.array(t, dtype=np.int64) for t in elem_tags],
            'node_tags': [np.array(n, dtype=np.int64) for n in node_tags],
        }
        total = sum(len(t) for t in result['tags'])
        self._mesh._log(
            f"get_elements -> {total} elements "
            f"({len(elem_types)} types)"
        )
        return result

    def get_element_properties(self, element_type: int) -> dict:
        """
        Return metadata for a given gmsh element type code.

        Returns
        -------
        dict
            ``'name'``, ``'dim'``, ``'order'``, ``'n_nodes'``,
            ``'n_primary_nodes'``, ``'local_coords'``.
        """
        name, dim, order, n_nodes, local_coords, n_primary = \
            gmsh.model.mesh.getElementProperties(element_type)
        d = max(dim, 1)
        return {
            'name'           : name,
            'dim'            : dim,
            'order'          : order,
            'n_nodes'        : n_nodes,
            'n_primary_nodes': n_primary,
            'local_coords'   : np.array(local_coords).reshape(-1, d),
        }

    # ------------------------------------------------------------------
    # FEM data
    # ------------------------------------------------------------------

    def get_fem_data(self, dim: int = 2) -> FEMData:
        """
        Extract solver-ready FEM data as a :class:`FEMData` object.

        Returns node IDs, coordinates, element IDs, connectivity,
        mesh statistics (``.info``), and physical group data
        (``.physical``).

        Must be called **after** ``generate()``.

        Parameters
        ----------
        dim : int
            Element dimension to extract (2 = triangles/quads,
            3 = tets/hexes).

        Example
        -------
        ::

            g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
            fem = g.mesh.queries.get_fem_data(dim=2)
        """
        parent = self._mesh._parent
        result = FEMData.from_gmsh(dim=dim, session=parent)

        self._mesh._log(
            f"get_fem_data(dim={dim}) -> "
            f"{result.info.n_nodes} nodes, "
            f"{result.info.n_elems} elements, "
            f"bw={result.info.bandwidth}"
        )

        return result

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def get_element_qualities(
        self,
        element_tags: list[int] | ndarray,
        quality_name: str = "minSICN",
    ) -> ndarray:
        """
        Compute quality metrics for the given elements.

        Parameters
        ----------
        element_tags : element tags to evaluate
        quality_name : ``"minSICN"``, ``"minSIGE"``, ``"gamma"``, or
                       ``"minSJ"``
        """
        tags = list(element_tags) if not isinstance(element_tags, list) else element_tags
        q = gmsh.model.mesh.getElementQualities(tags, qualityName=quality_name)
        return np.asarray(q)

    def quality_report(
        self,
        *,
        dim: int = -1,
        metrics: list[str] | None = None,
    ) -> "pd.DataFrame":
        """
        Compute a summary quality report for all mesh elements.

        For each element type and quality metric, reports count, min,
        max, mean, std, and the percentage of elements below common
        thresholds.

        Must be called **after** ``generate()``.

        Example
        -------
        ::

            g.mesh.generation.generate(2)
            print(g.mesh.queries.quality_report().to_string())
        """
        import pandas as pd

        if metrics is None:
            metrics = ["minSICN", "minSIGE", "gamma", "minSJ"]

        elems = self.get_elements(dim=dim)

        rows: list[dict] = []
        for etype, etags in zip(elems['types'], elems['tags']):
            if len(etags) == 0:
                continue
            props = self.get_element_properties(etype)
            etype_name = props.get('name', str(etype))

            for metric in metrics:
                try:
                    q = gmsh.model.mesh.getElementQualities(
                        list(etags.astype(int)), qualityName=metric,
                    )
                    q = np.asarray(q)
                except Exception:
                    continue  # metric not supported for this element type

                if len(q) == 0:
                    continue

                row: dict = {
                    'element_type' : etype_name,
                    'gmsh_code'    : int(etype),
                    'metric'       : metric,
                    'count'        : len(q),
                    'min'          : float(q.min()),
                    'max'          : float(q.max()),
                    'mean'         : float(q.mean()),
                    'std'          : float(q.std()),
                    'pct_below_0.1': float((q < 0.1).sum() / len(q) * 100),
                    'pct_below_0.3': float((q < 0.3).sum() / len(q) * 100),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index(['element_type', 'metric']).sort_index()

        self._mesh._log(
            f"quality_report(dim={dim}) -> "
            f"{len(rows)} metric rows across "
            f"{df.index.get_level_values('element_type').nunique() if not df.empty else 0} "
            f"element types"
        )

        if self._mesh._parent._verbose and not df.empty:
            print("\n--- Mesh Quality Report ---")
            print(df.to_string())

        return df
