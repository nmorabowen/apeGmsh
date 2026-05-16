"""
_Options — global Gmsh mesher options.

Accessed via ``g.mesh.options``.  Wraps the most commonly-tuned
``Mesh.*`` options with apeGmsh-style methods that accept either a
string enum (typo-safe, autocomplete-friendly) or the raw int code
(for power users who already know the Gmsh option grammar).

The wrapper is intentionally narrow — Gmsh exposes 50+ ``Mesh.*``
options and most users only ever touch a handful.  If you need
something not wrapped here, fall through to::

    import gmsh
    gmsh.option.setNumber("Mesh.MyOption", value)

The wrapped options are:

* ``set_subdivision_algorithm(...)`` — post-process tets/prisms into
  hexes (Gmsh's ``Mesh.SubdivisionAlgorithm``).
* ``set_smoothing(iterations=)`` — global Laplacian smoothing passes
  applied after generation (``Mesh.Smoothing``).
* ``set_element_order(order)`` — element interpolation order
  (``Mesh.ElementOrder``); 1 for linear, 2 for quadratic, …
* ``set_algorithm_2d(...)`` — 2-D meshing algorithm
  (``Mesh.Algorithm``).
* ``set_algorithm_3d(...)`` — 3-D meshing algorithm
  (``Mesh.Algorithm3D``).
* ``set_recombination_algorithm(...)`` — recombination strategy
  used when triangles are merged into quads
  (``Mesh.RecombinationAlgorithm``).

Each setter has a matching ``get_*`` method that returns the string
form when the current value matches a known enum, otherwise the raw
int.

Note
----
``g.mesh.structured.set_smoothing(tag, val, dim=)`` is a *different*
method — it applies a smoothing constraint to a single entity by
calling ``gmsh.model.mesh.setSmoothing``.  The method on this
composite is the global ``Mesh.Smoothing`` option that runs after
generation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh


# ---------------------------------------------------------------------------
# Enum tables
# ---------------------------------------------------------------------------

_SUBDIVISION_ALGORITHM = {
    "none":     0,
    "all_quad": 1,
    "all_hex":  2,
}

_ALGORITHM_2D = {
    "meshadapt":              1,
    "automatic":              2,
    "delaunay":               5,
    "frontal":                6,
    "bamg":                   7,
    "frontal_quads":          8,
    "packing":                9,
    "quasi_structured_quad": 11,
}

_ALGORITHM_3D = {
    "delaunay":          1,
    "frontal":           4,
    "frontal_delaunay":  5,
    "mmg3d":             7,
    "rtree":             9,
    "hxt":              10,
}

_RECOMBINATION_ALGORITHM = {
    "simple":       0,
    "blossom":      1,
    "simple_full":  2,
    "blossom_full": 3,
}


def _resolve_enum(value, table: dict[str, int], option_name: str) -> int:
    """Resolve a string-or-int enum value to its int code.

    Raises ``ValueError`` for unknown strings with the valid options
    listed.
    """
    if isinstance(value, str):
        key = value.lower()
        if key not in table:
            raise ValueError(
                f"{option_name}: unknown value {value!r}. "
                f"Valid options: {sorted(table)}"
            )
        return table[key]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(
        f"{option_name}: expected str or int, got {type(value).__name__}"
    )


def _invert(table: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in table.items()}


def _enum_string_or_int(code: int, table: dict[str, int]) -> "str | int":
    """Return the string form for *code* if it matches a known enum, else int."""
    return _invert(table).get(code, code)


# ---------------------------------------------------------------------------
# _Options composite
# ---------------------------------------------------------------------------

class _Options:
    """Global Gmsh mesher options.

    Each ``set_*`` method maps to one ``gmsh.option.setNumber("Mesh.X", v)``
    call but adds string-enum input, validation, and logging.  Methods
    return ``self`` for chaining.
    """

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Subdivision algorithm
    # ------------------------------------------------------------------

    def set_subdivision_algorithm(self, algorithm) -> "_Options":
        """Post-process tets/prisms into hexes after generation.

        Maps to ``Mesh.SubdivisionAlgorithm``.

        Parameters
        ----------
        algorithm : str or int
            One of:

            - ``"none"`` (0)     no subdivision (default)
            - ``"all_quad"`` (1) split tris into 3 quads each
            - ``"all_hex"`` (2)  split tets/prisms into hexes

        Notes
        -----
        Redundant for transfinite + face-recombined volumes — those
        produce hexes natively via the structured mesher.  This option
        is for the unstructured-tet → hex conversion path.

        Example
        -------
        ::

            g.mesh.options.set_subdivision_algorithm("all_hex")
        """
        code = _resolve_enum(algorithm, _SUBDIVISION_ALGORITHM,
                             "set_subdivision_algorithm")
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", code)
        self._mesh._log(f"options.set_subdivision_algorithm({algorithm!r})")
        return self

    def get_subdivision_algorithm(self):
        """Return the current ``Mesh.SubdivisionAlgorithm`` as a string or int."""
        return _enum_string_or_int(
            int(gmsh.option.getNumber("Mesh.SubdivisionAlgorithm")),
            _SUBDIVISION_ALGORITHM,
        )

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def set_smoothing(self, iterations: int) -> "_Options":
        """Number of global Laplacian smoothing passes applied after generation.

        Maps to ``Mesh.Smoothing``.

        Parameters
        ----------
        iterations : int
            Number of smoothing passes.  ``0`` disables.  Higher values
            improve element quality on unstructured meshes; on
            transfinite meshes the option is effectively a no-op since
            interior nodes are already on the structured grid.

        See Also
        --------
        g.mesh.structured.set_smoothing :
            Per-entity smoothing constraint (different — sets
            ``gmsh.model.mesh.setSmoothing`` on a specific tag).
        """
        gmsh.option.setNumber("Mesh.Smoothing", int(iterations))
        self._mesh._log(f"options.set_smoothing(iterations={iterations})")
        return self

    def get_smoothing(self) -> int:
        """Return the current ``Mesh.Smoothing`` value."""
        return int(gmsh.option.getNumber("Mesh.Smoothing"))

    # ------------------------------------------------------------------
    # Element order
    # ------------------------------------------------------------------

    def set_element_order(self, order: int) -> "_Options":
        """Element interpolation order.

        Maps to ``Mesh.ElementOrder``.

        Parameters
        ----------
        order : int
            ``1`` for linear elements (default), ``2`` for quadratic,
            higher orders supported but rarely used.  Higher-order
            meshes have additional mid-edge / mid-face nodes; downstream
            FEM code must support the element type.

        Example
        -------
        ::

            g.mesh.options.set_element_order(2)  # quadratic hexes/tets
        """
        gmsh.option.setNumber("Mesh.ElementOrder", int(order))
        self._mesh._log(f"options.set_element_order({order})")
        return self

    def get_element_order(self) -> int:
        """Return the current ``Mesh.ElementOrder`` value."""
        return int(gmsh.option.getNumber("Mesh.ElementOrder"))

    # ------------------------------------------------------------------
    # 2-D meshing algorithm
    # ------------------------------------------------------------------

    def set_algorithm_2d(self, algorithm) -> "_Options":
        """2-D meshing algorithm.

        Maps to ``Mesh.Algorithm``.

        Parameters
        ----------
        algorithm : str or int
            One of:

            - ``"meshadapt"`` (1)             adaptive Delaunay
            - ``"automatic"`` (2)             default — let Gmsh choose
            - ``"delaunay"`` (5)              pure Delaunay
            - ``"frontal"`` (6)               frontal-Delaunay
            - ``"bamg"`` (7)                  anisotropic remeshing
            - ``"frontal_quads"`` (8)         frontal-Delaunay for quads
            - ``"packing"`` (9)               packing of parallelograms
            - ``"quasi_structured_quad"`` (11) quasi-structured quads
              (Gmsh ≥ 4.10)
        """
        code = _resolve_enum(algorithm, _ALGORITHM_2D, "set_algorithm_2d")
        gmsh.option.setNumber("Mesh.Algorithm", code)
        self._mesh._log(f"options.set_algorithm_2d({algorithm!r})")
        return self

    def get_algorithm_2d(self):
        """Return the current ``Mesh.Algorithm`` as a string or int."""
        return _enum_string_or_int(
            int(gmsh.option.getNumber("Mesh.Algorithm")),
            _ALGORITHM_2D,
        )

    # ------------------------------------------------------------------
    # 3-D meshing algorithm
    # ------------------------------------------------------------------

    def set_algorithm_3d(self, algorithm) -> "_Options":
        """3-D meshing algorithm.

        Maps to ``Mesh.Algorithm3D``.

        Parameters
        ----------
        algorithm : str or int
            One of:

            - ``"delaunay"`` (1)         pure Delaunay (default)
            - ``"frontal"`` (4)          frontal
            - ``"frontal_delaunay"`` (5) frontal-Delaunay
            - ``"mmg3d"`` (7)            anisotropic remeshing
            - ``"rtree"`` (9)            R-tree based
            - ``"hxt"`` (10)             HXT — much faster than Delaunay
              on large unstructured tet meshes
        """
        code = _resolve_enum(algorithm, _ALGORITHM_3D, "set_algorithm_3d")
        gmsh.option.setNumber("Mesh.Algorithm3D", code)
        self._mesh._log(f"options.set_algorithm_3d({algorithm!r})")
        return self

    def get_algorithm_3d(self):
        """Return the current ``Mesh.Algorithm3D`` as a string or int."""
        return _enum_string_or_int(
            int(gmsh.option.getNumber("Mesh.Algorithm3D")),
            _ALGORITHM_3D,
        )

    # ------------------------------------------------------------------
    # Recombination algorithm
    # ------------------------------------------------------------------

    def set_recombination_algorithm(self, algorithm) -> "_Options":
        """Strategy used when recombining triangles into quads.

        Maps to ``Mesh.RecombinationAlgorithm``.

        Parameters
        ----------
        algorithm : str or int
            One of:

            - ``"simple"`` (0)       simple
            - ``"blossom"`` (1)      Blossom — better quality, default
            - ``"simple_full"`` (2)  simple, full recombination
            - ``"blossom_full"`` (3) Blossom, full recombination —
              highest quality for all-quad meshes
        """
        code = _resolve_enum(algorithm, _RECOMBINATION_ALGORITHM,
                             "set_recombination_algorithm")
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", code)
        self._mesh._log(f"options.set_recombination_algorithm({algorithm!r})")
        return self

    def get_recombination_algorithm(self):
        """Return the current ``Mesh.RecombinationAlgorithm`` as a string or int."""
        return _enum_string_or_int(
            int(gmsh.option.getNumber("Mesh.RecombinationAlgorithm")),
            _RECOMBINATION_ALGORITHM,
        )
