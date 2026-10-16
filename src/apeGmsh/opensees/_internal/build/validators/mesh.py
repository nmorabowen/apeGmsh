"""mesh: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Iterable

from ...types import Element
from ..errors import BridgeError
from ..pg_expand import PGElementFanout, expand_spec_to_elements
from ..transforms import _node_coord


if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


#: Gmsh quadrilateral codes (quad4, quad8) and the node counts of the
#: triangle cells (tri3, tri6) a surface meshed without recombination gives.
_QUAD_ETYPES: frozenset[int] = frozenset({3, 16})


_TRI_NODE_COUNTS: frozenset[int] = frozenset({3, 6})


def check_quad_element_on_triangles(
    spec: Element,
    elements: "PGElementFanout | list[tuple[int, tuple[int, ...]]]",
) -> None:
    """Raise, naming the PG, when a quad-only element meets triangle cells.

    A surface meshed without recombination is all-triangle (most often
    ``g.mesh.structured.recombine()`` called before ``generate()``, which
    is a no-op). The element's own ``_emit`` would then fail with a bare
    "expected 4 node tags, got 3" that says nothing about recombination
    (#1327), so :func:`allocate_element_tags`, which plans every emit
    route (flat, staged, partitioned), runs this first and names the cause.

    Only classes whose registry entry accepts quadrilaterals alone are
    checked. A class with no registry entry, or one that accepts other
    shapes, keeps its own node-count check in ``_emit``, which still
    fails loud on a mismatch.
    """
    from ...._element_capabilities import _ETYPE_INFO, Unknown, element_capability

    cap = element_capability(type(spec).__name__)
    if cap is Unknown or not cap.gmsh_etypes or not cap.gmsh_etypes <= _QUAD_ETYPES:
        return
    if isinstance(elements, PGElementFanout) and elements.conn.ndim == 2:
        counts = {int(elements.conn.shape[1])}
    else:
        counts = {len(nodes) for _, nodes in elements}
    tri = sorted(counts & _TRI_NODE_COUNTS)
    if not tri:
        return
    cls = type(spec).__name__
    pg = getattr(spec, "pg", None)
    want = sorted({_ETYPE_INFO[e][0] for e in cap.gmsh_etypes})
    raise BridgeError(
        f"{cls}(pg={pg!r}): the physical group holds "
        f"{'/'.join(map(str, tri))}-node triangles, but {cls} needs "
        f"{'/'.join(map(str, want))}-node quadrilaterals. Request quads "
        f"before meshing with g.mesh.structured.set_recombine({pg!r}) or "
        "g.mesh.recipe.structured(...), then generate(); "
        "g.mesh.structured.recombine() only acts on an already generated "
        "mesh. Otherwise declare a triangle element on this PG (e.g. "
        "ASDShellT3 for shells, Tri31 for plane solids)."
    )


def validate_absorbing_quad_geometry(
    fem: "FEMData", elements: "Iterable[Element]",
) -> None:
    """ADR 0054 (AB-5) — fail loud on a distorted 2D absorbing quad.

    ``ASDAbsorbingBoundary2D`` has **no source-side distortion handling**: it
    sizes its dashpots / free-field column from the sorted nodal x/y
    coordinates assuming an axis-aligned rectangle
    (``getElementSizes``, ASDAbsorbingBoundary2D.cpp:986-1003 — no Jacobian
    check, no normal check), so a skewed / rotated / degenerate quad runs
    with silently wrong terms.  The 3D element guards itself
    (``handleDistortion`` + singular-Jacobian exit), so only 2D is checked.

    Every fan-out quad of every ``ASDAbsorbingBoundary2D`` spec must have its
    4 nodes on exactly 2 distinct x and 2 distinct y stations (the 4 corners
    of an axis-aligned rectangle), coplanar in z, with non-degenerate spans.
    """
    # Deferred import: avoids an _internal -> element import cycle at load.
    from ....element.absorbing import ASDAbsorbingBoundary2D

    for spec in elements:
        if not isinstance(spec, ASDAbsorbingBoundary2D):
            continue
        bad: list[tuple[int, str]] = []
        for eid, conn in expand_spec_to_elements(fem, spec):
            coords = [_node_coord(fem, int(t)) for t in conn]
            if len(coords) != 4:
                bad.append((int(eid), f"{len(coords)} nodes (expected 4)"))
                continue
            xs = sorted(float(c[0]) for c in coords)
            ys = sorted(float(c[1]) for c in coords)
            zs = [float(c[2]) for c in coords]
            dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
            scale = max(dx, dy, 1e-300)
            tol = 1e-6 * scale
            if dx <= tol or dy <= tol:
                bad.append((int(eid), f"degenerate spans dx={dx:.3g} dy={dy:.3g}"))
                continue
            if max(zs) - min(zs) > tol:
                bad.append((int(eid), "nodes not coplanar in z"))
                continue
            on_corners = all(
                min(abs(float(c[0]) - xs[0]), abs(float(c[0]) - xs[-1])) <= tol
                and min(abs(float(c[1]) - ys[0]), abs(float(c[1]) - ys[-1])) <= tol
                for c in coords
            )
            corners = {
                (round(float(c[0]) / tol), round(float(c[1]) / tol))
                for c in coords
            }
            if not on_corners or len(corners) != 4:
                bad.append((int(eid), "skewed / non-axis-aligned quad"))
        if bad:
            examples = "; ".join(
                f"element {eid}: {why}" for eid, why in bad[:3]
            )
            more = f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""
            raise BridgeError(
                f"ASDAbsorbingBoundary2D over pg {spec.pg!r}: {len(bad)} "
                f"quad(s) are not axis-aligned rectangles — {examples}{more}. "
                "The 2D element has NO distortion handling in OpenSees "
                "(it sizes itself from sorted nodal x/y coordinates), so a "
                "skewed or rotated skin runs with silently wrong "
                "dashpot/stiffness terms.  Build the absorbing skin "
                "axis-aligned (see g.parts.add_plane_wave_box_2d / "
                "add_absorbing_shell_2d)."
            )
