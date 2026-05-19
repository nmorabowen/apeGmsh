"""``SectionCutDef`` — apeGmsh-side section-cut definition.

A :class:`SectionCutDef` is the apeGmsh-side carrier for a section cut.
Pure data, picklable, no STKO_to_python dependency at construction.
Convert to STKO's ``SectionCutSpec`` via :meth:`SectionCutDef.to_spec`
when ready to consume MPCO output.

Storage rationale
-----------------
The plane is stored as ``(point_tuple, normal_tuple)`` rather than as
a STKO ``Plane`` instance, so a Def can be constructed and pickled
without STKO installed. STKO's ``Plane`` is reconstructed inside
:meth:`to_spec`. Validation matches STKO's contract — nonzero normal,
nonempty filter, valid side, convex on-plane bounding polygon — so
``to_spec()`` doesn't blow up on bad data; bad data is rejected at
construction.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData

    from ._preflight import PreflightReport
    from ._tag_map import FemToOpsTagMap


Vec3 = tuple[float, float, float]
Side = Literal["positive", "negative"]
PolyVerts = Iterable[Iterable[float]] | np.ndarray


def _coerce_vec3(v: Iterable[float] | np.ndarray, *, label: str) -> Vec3:
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(f"{label} must be length-3, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite, got {arr.tolist()}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _coerce_int_tuple(values: Iterable[int] | np.ndarray, *, label: str) -> tuple[int, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{label} must be 1-D, got shape {values.shape}.")
        return tuple(int(x) for x in values)
    try:
        return tuple(int(x) for x in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain integers: {exc}") from None


def _coerce_polygon(
    values: Iterable[Iterable[float]] | np.ndarray,
) -> tuple[Vec3, ...]:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"bounding_polygon must be a sequence of (x, y, z) triples or "
            f"shape (M, 3); got shape {arr.shape}."
        )
    if arr.shape[0] < 3:
        raise ValueError(
            f"bounding_polygon must have at least 3 vertices; got {arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounding_polygon must contain only finite coordinates.")
    return tuple((float(v[0]), float(v[1]), float(v[2])) for v in arr)


@dataclass(frozen=True)
class SectionCutDef:
    """apeGmsh-side definition of a section cut.

    Parameters
    ----------
    plane_point:
        Any point on the cut plane, length-3.
    plane_normal:
        Outward normal of the cut plane, length-3. Need not be unit
        — auto-normalized on construction.
    element_ids:
        OpenSees element tags the cut filters to. Required and
        non-empty; ``SectionCutDef`` does not resolve from physical
        groups in v1 (callers must do that themselves until the
        Phase 3 ``FemToOpsTagMap`` lands).
    side:
        ``"positive"`` (default) treats the side along the plane normal
        as the kept side; ``"negative"`` flips that. The resultant
        downstream is the force the discarded side exerts on the kept
        side — classic internal-force sign convention.
    label:
        Optional display label.
    bounding_polygon:
        Optional convex polygon on the cut plane restricting the cut
        to elements whose intersection falls inside it. Vertices must
        lie on the plane (checked on conversion to STKO spec).
    """

    plane_point: Vec3
    plane_normal: Vec3
    element_ids: tuple[int, ...]
    side: Side = "positive"
    label: str | None = None
    bounding_polygon: tuple[Vec3, ...] | None = None

    def __post_init__(self) -> None:
        p = _coerce_vec3(self.plane_point, label="plane_point")
        n_arr = np.asarray(_coerce_vec3(self.plane_normal, label="plane_normal"), dtype=float)
        n_norm = float(np.linalg.norm(n_arr))
        if n_norm < 1e-300:
            raise ValueError(
                f"plane_normal must be nonzero, got {n_arr.tolist()}."
            )
        n_unit: Vec3 = tuple((n_arr / n_norm).tolist())  # type: ignore[assignment]
        object.__setattr__(self, "plane_point", p)
        object.__setattr__(self, "plane_normal", n_unit)

        if self.side not in ("positive", "negative"):
            raise ValueError(
                f"side must be 'positive' or 'negative', got {self.side!r}."
            )

        eids = _coerce_int_tuple(self.element_ids, label="element_ids")
        if not eids:
            raise ValueError("element_ids must be non-empty.")
        object.__setattr__(self, "element_ids", eids)

        if self.bounding_polygon is not None:
            poly = _coerce_polygon(self.bounding_polygon)
            object.__setattr__(self, "bounding_polygon", poly)

    @property
    def plane_normal_arr(self) -> np.ndarray:
        return np.asarray(self.plane_normal, dtype=float)

    @property
    def plane_point_arr(self) -> np.ndarray:
        return np.asarray(self.plane_point, dtype=float)

    # ------------------------------------------------------------------ #
    # Builders — synthesize Phases 1+2+3
    # ------------------------------------------------------------------ #
    @classmethod
    def from_plane_and_pg(
        cls,
        *,
        plane: tuple[Vec3, Vec3],
        elements_pg: str,
        fem: "FEMData",
        model_h5: str | Path,
        side: Side = "positive",
        label: str | None = None,
        bounding_polygon: PolyVerts | None = None,
    ) -> "SectionCutDef":
        """Build a ``SectionCutDef`` from a plane and a physical group.

        Resolves ``elements_pg`` → FEM eids via :class:`FEMData`, then
        maps those FEM eids to OpenSees tags via
        :class:`FemToOpsTagMap` reading ``model_h5``.

        Parameters
        ----------
        plane:
            ``(point, normal)`` tuple — easiest produced by one of the
            helpers in :mod:`apeGmsh.cuts._planes` (``plane_horizontal``,
            ``plane_vertical``, ``plane_from_three_points``,
            ``plane_from_physical_surface``).
        elements_pg:
            Physical-group name selecting which elements the cut
            filters to. May span multiple element types.
        fem:
            Solver-ready :class:`apeGmsh.mesh.FEMData.FEMData` produced
            from the same mesh that wrote ``model_h5``.
        model_h5:
            Path to a Phase-8.6+ ``model.h5`` containing the
            ``/opensees/element_meta/{type}/fem_eids`` side-channel.
        side, label, bounding_polygon:
            Pass-through to :class:`SectionCutDef`.

        Raises
        ------
        ValueError
            Empty PG, missing ``fem_eids`` dataset, cross-type id
            collision, etc. (propagated from the tag map / FEMData).
        KeyError
            Some FEM eid in the PG has no OpenSees tag in ``model_h5``
            — indicates the PG was added after the bridge emit, or the
            wrong ``model_h5`` was supplied.
        """
        from ._tag_map import FemToOpsTagMap

        point, normal = plane
        # selection-unification v2 P3-R / §6.3 M-MINOR-a: ``.ids`` is a
        # list (MeshSelection) — wrap in ``np.asarray`` so the ``.size``
        # guard below and the tag-map lookup keep working.
        fem_eids = np.asarray(fem.elements.select(pg=elements_pg).ids)
        if fem_eids.size == 0:
            raise ValueError(
                f"Physical group {elements_pg!r} resolves to zero "
                "elements; cannot build a cut filter."
            )

        tag_map = FemToOpsTagMap.from_h5(model_h5)
        ops_tags = tag_map.ops_tags_for_fem_eids(fem_eids)

        # bounding_polygon is intentionally typed wider than the field;
        # __post_init__ coerces. mypy doesn't see the coercion through
        # the dataclass field annotation, so a narrow cast keeps the
        # downstream `.bounding_polygon` typed as the tuple form.
        return cls(
            plane_point=point,
            plane_normal=normal,
            element_ids=ops_tags,
            side=side,
            label=label,
            bounding_polygon=cast(
                "tuple[Vec3, ...] | None", bounding_polygon,
            ),
        )

    @classmethod
    def from_planar_pg(
        cls,
        *,
        plane_pg: str,
        elements_pg: str,
        fem: "FEMData",
        model_h5: str | Path,
        side: Side = "positive",
        label: str | None = None,
        normal_hint: Iterable[float] | np.ndarray | None = None,
        tol: float = 1e-6,
        bounding_polygon: PolyVerts | None = None,
        with_bounding: bool = False,
    ) -> "SectionCutDef":
        """Build a cut where the plane is derived from one PG and the
        element filter from another.

        The ergonomic entry point. ``plane_pg`` is typically a 2-D
        physical group representing the cut plane geometry (e.g. a
        diaphragm surface); ``elements_pg`` is typically a 3-D / 2-D
        PG representing the elements to integrate over (e.g. a story
        of columns + walls).

        Parameters
        ----------
        plane_pg:
            Physical group whose nodes define the cut plane. Must be
            planar within ``tol``.
        elements_pg:
            Physical group selecting the element filter.
        fem, model_h5:
            Same as :meth:`from_plane_and_pg`.
        normal_hint:
            Optional length-3 outward direction; the fit normal is
            flipped to agree.
        tol:
            Coplanarity tolerance for ``plane_pg``.
        side, label, bounding_polygon:
            Pass-through. If ``label`` is omitted, an auto-label
            ``"plane=<plane_pg>, elements=<elements_pg>"`` is set for
            traceability in plots / logs.
        with_bounding:
            When ``True``, auto-derive a convex bounding polygon from
            ``plane_pg``'s node convex hull (see
            :func:`bounding_polygon_from_physical_surface`). Mutually
            exclusive with passing ``bounding_polygon`` explicitly —
            raises ``ValueError`` if both are set.
        """
        from ._planes import plane_from_physical_surface

        plane = plane_from_physical_surface(
            fem, plane_pg, tol=tol, normal_hint=normal_hint,
        )

        if with_bounding:
            if bounding_polygon is not None:
                raise ValueError(
                    "Pass either bounding_polygon=... OR "
                    "with_bounding=True, not both."
                )
            from ._polygons import bounding_polygon_from_physical_surface
            bounding_polygon = bounding_polygon_from_physical_surface(
                fem, plane_pg, tol=tol, normal_hint=normal_hint,
            )

        return cls.from_plane_and_pg(
            plane=plane,
            elements_pg=elements_pg,
            fem=fem,
            model_h5=model_h5,
            side=side,
            label=(
                label
                if label is not None
                else f"plane={plane_pg}, elements={elements_pg}"
            ),
            bounding_polygon=bounding_polygon,
        )

    # ------------------------------------------------------------------ #
    # Preflight (v2.3)
    # ------------------------------------------------------------------ #
    def preflight(
        self,
        fem: "FEMData",
        *,
        model_h5: str | Path | None = None,
        tag_map: "FemToOpsTagMap | None" = None,
        tol: float = 1e-6,
    ) -> "PreflightReport":
        """Validate this cut against a live ``FEMData`` snapshot.

        Returns a :class:`PreflightReport` with structured errors and
        warnings; does not mutate the cut. See
        :mod:`apeGmsh.cuts._preflight` for the issue catalog
        (E1–E4, W1) and design rationale.

        Parameters
        ----------
        fem:
            Solver-ready :class:`apeGmsh.mesh.FEMData.FEMData` to check
            against.
        model_h5, tag_map:
            Provide one (not both) to enable the OpenSees-tag checks
            (E1, E2, E4) and the W1 AABB scan. Without either, only
            the polygon-on-plane check (E3) runs.
        tol:
            Tolerance for the polygon-on-plane check (E3) and the
            AABB-straddle check (W1). Default ``1e-6``.
        """
        from ._preflight import run_cut_checks
        return run_cut_checks(
            self, fem, model_h5=model_h5, tag_map=tag_map, tol=tol,
        )

    # ------------------------------------------------------------------ #
    # STKO_to_python interop
    # ------------------------------------------------------------------ #
    def to_spec(self):
        """Convert to a ``STKO_to_python.cuts.SectionCutSpec``.

        Lazy-imports STKO_to_python. Raises :class:`ImportError` with a
        ``pip install`` hint if the package is not installed.

        Returns
        -------
        STKO_to_python.cuts.SectionCutSpec
            A picklable, hashable, validated spec ready to feed to
            ``MPCODataSet.section_cut(...)``.
        """
        from ._optional_stko import load_stko_cuts
        stko = load_stko_cuts()
        plane = stko.Plane(point=self.plane_point, normal=self.plane_normal)
        kwargs = {
            "plane": plane,
            "element_ids": self.element_ids,
            "side": self.side,
        }
        if self.label is not None:
            kwargs["label"] = self.label
        if self.bounding_polygon is not None:
            kwargs["bounding_polygon"] = self.bounding_polygon
        return stko.SectionCutSpec(**kwargs)

    # ------------------------------------------------------------------ #
    # Pickle
    # ------------------------------------------------------------------ #
    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """Pickle to ``path``. ``.gz`` suffix → gzip unless overridden."""
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"
        opener = gzip.open if compress else open
        with opener(p, "wb") as f:
            pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(cls, path: str | Path) -> "SectionCutDef":
        p = Path(path)
        opener = gzip.open if p.suffix.lower() == ".gz" else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj
