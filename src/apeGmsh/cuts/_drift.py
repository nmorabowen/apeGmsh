"""``DriftDef`` / ``DriftSweepDef`` — node-pair drift carriers (v5).

Sibling types to :class:`SectionCutDef` / :class:`SectionSweepDef`,
but operating on node pairs instead of element filters. A drift is
``Δu = u(top) - u(bottom)``, optionally projected onto a unit axis,
optionally normalized by a story height. The natural carrier for
inter-story drift in building models.

v5 scope is **carrier only** — no ``.to_spec()`` (STKO has no
``DriftSpec`` counterpart yet), no ``.extract()`` (no demanded
consumer). Mirrors :class:`SectionCutDef` v1 philosophy: build the
spec; defer consumption.

Why this lives in ``apeGmsh.cuts``
----------------------------------
The subpackage name is a slight misnomer — drift isn't a "cut". But
the plumbing is identical (frozen dataclass + builders + pickle +
preflight) and the import line reads naturally
(``from apeGmsh.cuts import DriftDef``). Reorganizing to
``apeGmsh.outputs`` becomes worth the churn only when a third
post-process type lands.

Preflight catalog
-----------------
``D-E1``  top_node not in ``fem.nodes``.
``D-E2``  bottom_node not in ``fem.nodes``.
``D-W1``  top and bottom node coordinates coincident within ``tol``.

Issue codes share the :class:`PreflightReport` shape with cuts —
prefixed ``D-`` to keep them distinct from cut codes ``E1``–``E4`` /
``W1`` (Option α from the v5 design pass).
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np

from ._defs import Vec3, _coerce_vec3
from ._preflight import PreflightIssue, PreflightReport

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


def _coerce_node_id(value: object, *, label: str) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer: {exc}") from None


@dataclass(frozen=True)
class DriftDef:
    """apeGmsh-side definition of a node-pair drift.

    Parameters
    ----------
    top_node:
        FEM node id at the "high" end of the drift (e.g. upper floor).
    bottom_node:
        FEM node id at the "low" end. Must differ from ``top_node``.
    direction:
        Optional unit axis to project the drift onto. ``None`` (default)
        means consumers receive the raw ``Δu`` vector. A nonzero
        length-3 sequence is auto-normalized to unit length.
    story_height:
        Optional positive distance between the two reference points.
        Pure metadata in v5 — no in-package consumer reads it yet.
    label:
        Optional display label.
    """

    top_node: int
    bottom_node: int
    direction: Vec3 | None = None
    story_height: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        top = _coerce_node_id(self.top_node, label="top_node")
        bottom = _coerce_node_id(self.bottom_node, label="bottom_node")
        if top == bottom:
            raise ValueError(
                f"top_node and bottom_node must differ; both are {top}."
            )
        object.__setattr__(self, "top_node", top)
        object.__setattr__(self, "bottom_node", bottom)

        if self.direction is not None:
            d_arr = np.asarray(
                _coerce_vec3(self.direction, label="direction"),
                dtype=float,
            )
            d_norm = float(np.linalg.norm(d_arr))
            if d_norm < 1e-300:
                raise ValueError(
                    f"direction must be nonzero, got {d_arr.tolist()}."
                )
            d_unit: Vec3 = tuple((d_arr / d_norm).tolist())  # type: ignore[assignment]
            object.__setattr__(self, "direction", d_unit)

        if self.story_height is not None:
            sh = float(self.story_height)
            if not np.isfinite(sh) or sh <= 0.0:
                raise ValueError(
                    f"story_height must be a positive finite number; got {sh}."
                )
            object.__setattr__(self, "story_height", sh)

    @property
    def direction_arr(self) -> np.ndarray | None:
        if self.direction is None:
            return None
        return np.asarray(self.direction, dtype=float)

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    @classmethod
    def from_node_pair(
        cls,
        *,
        top_node: int,
        bottom_node: int,
        direction: Iterable[float] | np.ndarray | None = None,
        story_height: float | None = None,
        label: str | None = None,
    ) -> "DriftDef":
        """Named-classmethod twin of ``DriftDef(...)`` for API symmetry with cuts."""
        return cls(
            top_node=top_node,
            bottom_node=bottom_node,
            direction=tuple(direction) if direction is not None else None,  # type: ignore[arg-type]
            story_height=story_height,
            label=label,
        )

    @classmethod
    def from_pgs(
        cls,
        *,
        top_pg: str,
        bottom_pg: str,
        fem: "FEMData",
        direction: Iterable[float] | np.ndarray | None = None,
        story_height: float | None = None,
        label: str | None = None,
    ) -> "DriftDef":
        """Build a drift from two single-node physical groups.

        Each PG must resolve to exactly one FEM node — typically a
        floor-CM tag, a diaphragm centroid, or a single reference
        node deliberately placed in the model.

        Parameters
        ----------
        top_pg, bottom_pg:
            Physical group names. Each must contain exactly one node;
            multi-node PGs raise (silent averaging would hide tagging
            bugs).
        fem:
            Solver-ready :class:`apeGmsh.mesh.FEMData.FEMData`.
        direction, story_height, label:
            Pass-through to the constructor. If ``label`` is omitted,
            an auto-label ``"drift top=<top_pg>, bottom=<bottom_pg>"``
            is set.

        Raises
        ------
        ValueError
            Either PG is empty or contains more than one node.
        """
        top_id = _single_node_from_pg(fem, top_pg, kind="top_pg")
        bottom_id = _single_node_from_pg(fem, bottom_pg, kind="bottom_pg")
        return cls(
            top_node=top_id,
            bottom_node=bottom_id,
            direction=tuple(direction) if direction is not None else None,  # type: ignore[arg-type]
            story_height=story_height,
            label=(
                label
                if label is not None
                else f"drift top={top_pg}, bottom={bottom_pg}"
            ),
        )

    # ------------------------------------------------------------------ #
    # Preflight
    # ------------------------------------------------------------------ #
    def preflight(
        self,
        fem: "FEMData",
        *,
        tol: float = 1e-6,
    ) -> PreflightReport:
        """Check this drift against a live ``FEMData`` snapshot.

        Returns a :class:`PreflightReport` with structured errors
        (D-E1, D-E2) and warnings (D-W1). The shared
        :class:`PreflightReport` shape with cuts keeps both
        post-process types under one inspection surface; codes are
        prefixed ``D-`` to keep them distinct.

        Parameters
        ----------
        fem:
            Solver-ready :class:`apeGmsh.mesh.FEMData.FEMData` to check
            against.
        tol:
            Coordinate-coincidence tolerance for D-W1. Default ``1e-6``.
        """
        issues: list[PreflightIssue] = []
        top_coord = _try_node_coord(fem, self.top_node)
        bottom_coord = _try_node_coord(fem, self.bottom_node)
        if top_coord is None:
            issues.append(PreflightIssue(
                code="D-E1",
                severity="error",
                message=(
                    f"top_node {self.top_node} is not in the current FEM."
                ),
            ))
        if bottom_coord is None:
            issues.append(PreflightIssue(
                code="D-E2",
                severity="error",
                message=(
                    f"bottom_node {self.bottom_node} is not in the "
                    "current FEM."
                ),
            ))
        if top_coord is not None and bottom_coord is not None:
            dist = float(np.linalg.norm(top_coord - bottom_coord))
            if dist < tol:
                issues.append(PreflightIssue(
                    code="D-W1",
                    severity="warning",
                    message=(
                        f"top_node {self.top_node} and bottom_node "
                        f"{self.bottom_node} are coincident "
                        f"(|Δx| = {dist:g} < tol={tol:g})."
                    ),
                    detail={"distance": dist},
                ))
        return PreflightReport(cut_label=self.label, issues=tuple(issues))

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
    def load_pickle(cls, path: str | Path) -> "DriftDef":
        p = Path(path)
        opener = gzip.open if p.suffix.lower() == ".gz" else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj


@dataclass(frozen=True)
class DriftSweepDef:
    """Frozen sequence of :class:`DriftDef` instances.

    Construct via :meth:`from_pg_pairs` (one drift per ``(top_pg,
    bottom_pg)`` tuple, all sharing one direction). Iterates in
    drift order; index accessors mirror :class:`SectionSweepDef`.
    """

    drifts: tuple[DriftDef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "drifts", tuple(self.drifts))

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pg_pairs(
        cls,
        *,
        pg_pairs: Sequence[tuple[str, str]],
        fem: "FEMData",
        direction: Iterable[float] | np.ndarray | None = None,
        story_height: float | None = None,
    ) -> "DriftSweepDef":
        """Sweep across an ordered sequence of ``(top_pg, bottom_pg)`` pairs.

        Each pair is resolved through :meth:`DriftDef.from_pgs` (each
        PG must contain exactly one node). All drifts share the same
        ``direction`` and ``story_height`` metadata.
        """
        d_tuple = (
            tuple(direction) if direction is not None else None
        )
        drifts = tuple(
            DriftDef.from_pgs(
                top_pg=top,
                bottom_pg=bot,
                fem=fem,
                direction=d_tuple,  # type: ignore[arg-type]
                story_height=story_height,
            )
            for top, bot in pg_pairs
        )
        return cls(drifts=drifts)

    # ------------------------------------------------------------------ #
    # Preflight
    # ------------------------------------------------------------------ #
    def preflight(
        self,
        fem: "FEMData",
        *,
        tol: float = 1e-6,
    ) -> tuple[PreflightReport, ...]:
        """Run :meth:`DriftDef.preflight` on every drift in the sweep."""
        return tuple(d.preflight(fem, tol=tol) for d in self.drifts)

    # ------------------------------------------------------------------ #
    # Aggregations
    # ------------------------------------------------------------------ #
    def elevations(
        self,
        axis: str = "z",
        *,
        fem: "FEMData",
    ) -> np.ndarray:
        """Top-node coordinate along ``axis`` for every drift in the sweep.

        Handy for plotting drift profiles vs elevation::

            ax.plot(values, sweep.elevations(fem=fem), "o-")

        Parameters
        ----------
        axis:
            ``"x"``, ``"y"``, or ``"z"`` (default).
        fem:
            FEM to look up node coordinates against. Required —
            ``DriftDef`` stores node IDs, not coords.
        """
        key = axis.strip().lower()
        if key not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}.")
        col = {"x": 0, "y": 1, "z": 2}[key]
        coords: list[float] = []
        for d in self.drifts:
            top_coord = _try_node_coord(fem, d.top_node)
            if top_coord is None:
                raise KeyError(
                    f"top_node {d.top_node} not in fem.nodes; "
                    "run .preflight(fem) to surface the problem."
                )
            coords.append(float(top_coord[col]))
        return np.asarray(coords, dtype=float)

    # ------------------------------------------------------------------ #
    # Container protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.drifts)

    def __getitem__(self, i: int) -> DriftDef:
        return self.drifts[i]

    def __iter__(self) -> Iterator[DriftDef]:
        return iter(self.drifts)

    def __repr__(self) -> str:
        return f"DriftSweepDef(n_drifts={len(self.drifts)})"

    @property
    def n_drifts(self) -> int:
        return len(self.drifts)

    @property
    def is_empty(self) -> bool:
        return not self.drifts

    # ------------------------------------------------------------------ #
    # Pickle — matches DriftDef
    # ------------------------------------------------------------------ #
    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"
        opener = gzip.open if compress else open
        with opener(p, "wb") as f:
            pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(cls, path: str | Path) -> "DriftSweepDef":
        p = Path(path)
        opener = gzip.open if p.suffix.lower() == ".gz" else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj


# --------------------------------------------------------------------- #
# Helpers (module-private)
# --------------------------------------------------------------------- #
def _single_node_from_pg(
    fem: "FEMData", pg_name: str, *, kind: str,
) -> int:
    """Resolve a PG name to exactly one FEM node id.

    Raises ``ValueError`` with a clear message if the PG is empty or
    contains more than one node.
    """
    ids = fem.nodes.select(pg=pg_name).ids
    arr = np.asarray(ids).ravel()
    if arr.size == 0:
        raise ValueError(
            f"{kind}={pg_name!r} resolves to zero nodes — drift cannot be built."
        )
    if arr.size > 1:
        sample = arr[:5].tolist()
        tail = "…" if arr.size > 5 else ""
        raise ValueError(
            f"{kind}={pg_name!r} resolves to {arr.size} nodes; expected exactly 1. "
            f"Sample ids: {sample}{tail}. Tag a single representative node "
            "(e.g. a floor-CM node) and try again."
        )
    return int(arr[0])


def _try_node_coord(fem: "FEMData", node_id: int) -> np.ndarray | None:
    """Return the coordinates of ``node_id`` or ``None`` if absent."""
    try:
        idx = fem.nodes.index(int(node_id))
    except KeyError:
        return None
    return np.asarray(fem.nodes.coords[idx], dtype=float)
