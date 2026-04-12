"""
_Generation — mesh generation & algorithm sub-composite.

Accessed via ``g.mesh.generation``.  Owns the "run the mesher" surface:
generate, set_order, refine, optimize, set_algorithm (and the
``_by_physical`` variant).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from ._mesh_algorithms import _normalize_algorithm

if TYPE_CHECKING:
    from .Mesh import Mesh


class _Generation:
    """Mesh generation, high-order elevation, refinement, optimisation,
    and algorithm selection."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, dim: int = 3) -> "_Generation":
        """
        Generate a mesh up to the given dimension.

        Parameters
        ----------
        dim : 1 = edges only, 2 = surface mesh, 3 = volume mesh (default)
        """
        gmsh.model.mesh.generate(dim)
        self._mesh._log(f"generate(dim={dim})")
        return self

    def set_order(self, order: int) -> "_Generation":
        """
        Elevate elements to high order.

        Parameters
        ----------
        order : 1 = linear, 2 = quadratic, 3 = cubic, …
        """
        gmsh.model.mesh.setOrder(order)
        self._mesh._log(f"set_order({order})")
        return self

    def refine(self) -> "_Generation":
        """Uniformly refine by splitting every element once."""
        gmsh.model.mesh.refine()
        self._mesh._log("refine()")
        return self

    def optimize(
        self,
        method  : str  = "",
        *,
        force   : bool = False,
        niter   : int  = 1,
        dim_tags: list[tuple[int, int]] | None = None,
    ) -> "_Generation":
        """
        Optimise mesh quality.

        Parameters
        ----------
        method   : one of ``OptimizeMethod.*`` constants
        force    : apply optimisation even to already-valid elements
        niter    : number of passes
        dim_tags : limit to specific entities (``None`` = all)

        Example
        -------
        ::

            g.mesh.generation.generate(3).optimize(OptimizeMethod.NETGEN, niter=5)
        """
        gmsh.model.mesh.optimize(method, force=force, niter=niter,
                                  dimTags=dim_tags or [])
        self._mesh._log(f"optimize(method={method!r}, niter={niter})")
        return self

    # ------------------------------------------------------------------
    # Algorithm selection
    # ------------------------------------------------------------------

    def set_algorithm(
        self,
        tag,
        algorithm,
        *,
        dim      : int = 2,
    ) -> "_Generation":
        """
        Choose the meshing algorithm for a surface (dim=2) or globally
        for all volumes (dim=3).

        ``tag`` accepts int, label, or PG name.  If it resolves to
        multiple surfaces, the algorithm is applied to each.

        Example
        -------
        ::

            g.mesh.generation.set_algorithm("col.web", "frontal_delaunay_quads")
            g.mesh.generation.set_algorithm(0, "hxt", dim=3)
        """
        from apeGmsh.core._helpers import resolve_to_tags

        if dim not in (2, 3):
            raise ValueError(f"set_algorithm: dim must be 2 or 3, got {dim!r}")

        alg_int = _normalize_algorithm(algorithm, dim)

        if dim == 2:
            tags_resolved = resolve_to_tags(tag, dim=2, session=self._mesh._parent)
            for t in tags_resolved:
                gmsh.model.mesh.setAlgorithm(2, t, alg_int)
                self._mesh._directives.append({
                    'kind': 'algorithm', 'dim': 2, 'tag': t,
                    'algorithm': alg_int, 'requested': algorithm,
                })
                self._mesh._log(
                    f"set_algorithm(dim=2, tag={t}, alg={algorithm!r} -> {alg_int})"
                )
        else:  # dim == 3
            gmsh.option.setNumber("Mesh.Algorithm3D", alg_int)
            self._mesh._directives.append({
                'kind': 'algorithm', 'dim': 3, 'tag': 0,
                'algorithm': alg_int, 'requested': algorithm,
            })
            self._mesh._log(
                f"set_algorithm(dim=3, alg={algorithm!r} -> {alg_int})  [global option]"
            )
        return self

    def set_algorithm_by_physical(
        self,
        name     : str,
        algorithm,
        *,
        dim      : int = 2,
    ) -> "_Generation":
        """
        Apply :meth:`set_algorithm` to every entity in a physical group.

        Only meaningful for ``dim=2`` (per-surface algorithm selection).
        With ``dim=3`` this still calls ``set_algorithm`` once because
        ``Mesh.Algorithm3D`` is a global option.
        """
        if dim not in (2, 3):
            raise ValueError(
                f"set_algorithm_by_physical: dim must be 2 or 3, got {dim!r}"
            )
        tags = self._mesh._resolve_physical(name, dim)
        self._mesh._log(
            f"set_algorithm_by_physical(name={name!r}, dim={dim}, "
            f"tags={tags}, alg={algorithm!r})"
        )
        if dim == 3:
            return self.set_algorithm(0, algorithm, dim=3)
        for t in tags:
            self.set_algorithm(t, algorithm, dim=2)
        return self
