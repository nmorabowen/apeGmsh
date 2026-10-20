"""The refusal every Gauss reader raises for a join of mixed widths (#2221).

A :class:`~apeGmsh.results._slabs.GaussSlab` carries one
``natural_coords`` array, ``(sum_GP, dim)``. A read that joins a 2-D
parent class (a quad or shell, ``(xi, eta)``) with a 3-D parent class (a
tet or brick, ``(xi, eta, zeta)``) has no faithful slab: padding the 2-D
rows, or truncating the 3-D ones, places Gauss points wrongly. Each
reader (MPCO single-file and partitioned, ``.ladruno``, native) refuses
that read by name, and a read narrowed to one class answers.

A leaf module: it imports only the standard library, so every reader can
import it without a cycle.
"""
from __future__ import annotations

from collections.abc import Iterable


class GaussDimensionMixed(ValueError):
    """One Gauss read joins classes whose natural coordinates differ in
    width (a 2-D ``FourNodeQuad`` beside a 3-D ``FourNodeTetrahedron``).

    The read refuses, naming one source of each width and the file; a
    read narrowed to one class's elements (``element_ids``, or a
    physical group holding one class) answers.
    """


def refuse_mixed_gauss_dims(
    parts: Iterable[tuple[str, int]],
    *,
    component: str,
    reader: str,
    source: str,
    error: type[ValueError] = GaussDimensionMixed,
) -> None:
    """Raise ``error`` when ``parts`` hold more than one coordinate width.

    ``parts`` are ``(label, width)`` pairs, one per bucket, group or
    partition the read joins; the label names the class and where it
    sits. A width of 0 is a source with no natural coordinates at all
    (the reader warns about that on its own) and has no width to clash.
    The error names the first source of each width, the ``reader`` kind
    and the file ``source``.
    """
    first_of_width: dict[int, str] = {}
    for label, width in parts:
        if int(width) > 0:
            first_of_width.setdefault(int(width), label)
    if len(first_of_width) < 2:
        return
    listing = "; ".join(
        f"{dim}-D natural coordinates in {label}"
        for dim, label in sorted(first_of_width.items())
    )
    raise error(
        f"{reader} gauss read of {component!r} in {source} joins classes "
        f"with different natural-coordinate widths ({listing}). A "
        f"GaussSlab holds one (sum_GP, dim) natural_coords array, and "
        f"padding or truncating the coordinates would misplace Gauss "
        f"points. Read one class's elements at a time (element_ids, or a "
        f"physical group holding one class)."
    )


__all__ = ["GaussDimensionMixed", "refuse_mixed_gauss_dims"]
