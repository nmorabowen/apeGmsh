"""
``FootfallResult`` — the return type of
:meth:`apeGmsh.opensees.apeSees.footfall_walking` (ADR 0109 D5).

One row per response node: the dominant frequency the FRF peaked at,
the peak FRF ordinate itself, the two Design Guide branches (Eq 7-1
low-frequency, Eq 7-6 high-frequency ESPA), the governing acceleration
and which branch produced it, the excitation node that governed under
``excitation="full"``, and the Fig 2-1 / Table 4-1 limit with the
demand/capacity ratio.

**Accelerations are fractions of g.** The kernel in
:mod:`.footfall` works in the model's own acceleration unit; the
driver divides by the ``g`` it was given, once, so everything on this
object is dimensionless (``0.005`` is 0.5 %g). ``frf_max`` is the one
exception — it is acceleration per unit force in *model* units, which
is what the FRF is.

The sweep table and the mode rows are kept for inspection:
:meth:`frf` returns ``(freq, |A_ij|)`` for any pair the matrix was
built over, and :meth:`mode_table` returns the Eq 7-5 per-mode rows —
Example 7.1's Table 7-2 is exactly that table.

Nothing here reads a live domain: the FRF matrix it holds carries its
own mode-shape columns, so a later ``eigen`` or sweep call does not
invalidate the result (unlike
:class:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .footfall import effective_impulse, harmonic_for_dominant
from .footfall_frf import FRFMatrix
from .modal import _node_tag

if TYPE_CHECKING:
    from pathlib import Path

    from ..node import Node


__all__ = ["FootfallModes", "FootfallResult"]


@dataclass(frozen=True, slots=True)
class FootfallModes:
    """The mode basis behind a :class:`FootfallResult`."""

    f_n: np.ndarray
    """Natural frequencies of every extracted mode, Hz."""

    beta: np.ndarray
    """Damping ratio of every extracted mode, as resolved from the
    ADR 0075 channel (ADR 0109 D6)."""


@dataclass(frozen=True, slots=True)
class FootfallResult:
    """Walking-excitation footfall evaluation, one row per response node.

    Attributes
    ----------
    nodes
        Response node tags, in the order they were requested.
    f_dom
        Dominant frequency of the governing branch, Hz.
    frf_max
        ``|A_ij|`` at :attr:`f_dom` for the governing pair —
        acceleration per unit force in **model** units.
    a_p_lf
        Eq 7-1 peak acceleration, fraction of g; ``NaN`` when no sweep
        point lies below 9 Hz.
    a_espa_hf
        Eq 7-6 equivalent sinusoidal peak acceleration, fraction of g;
        ``NaN`` when the 9-``f_max`` Hz slice is empty or no mode lies
        at or below ``f_max``.
    a_p
        The governing acceleration — the larger of the two branches,
        fraction of g.
    regime
        ``"low"`` / ``"high"`` / ``"both"`` by which branches evaluated
        (``"none"`` if neither could).
    exc_node
        Excitation node tag that produced :attr:`a_p` (Robot's
        ``ExcitationNode``); under ``excitation="self"`` this is the
        response node itself.
    limit
        Acceptance limit at :attr:`f_dom`, fraction of g.
    ratio
        ``a_p / limit`` — acceptable below 1.
    """

    nodes: tuple[int, ...]
    f_dom: np.ndarray
    frf_max: np.ndarray
    a_p_lf: np.ndarray
    a_espa_hf: np.ndarray
    a_p: np.ndarray
    regime: np.ndarray
    exc_node: np.ndarray
    limit: np.ndarray
    ratio: np.ndarray

    occupancy: str
    """Occupancy the limit was read for."""

    limit_kind: str
    """``"curve"`` (Fig 2-1, frequency-scaled) or ``"table"`` (flat)."""

    g: float
    """Gravitational acceleration in model units — the divisor."""

    body_weight: float
    """Walker weight ``Q`` in model force units."""

    normalization: str
    """``"asserted(k of p)"`` — the ADR 0109 D2 provenance of
    ``m̃ = 1``."""

    freq: np.ndarray
    """The sweep grid, Hz (ADR 0109 D4)."""

    modes: FootfallModes
    """Per-mode ``(f_n, β)`` rows of the whole basis."""

    dof: int
    """1-based DOF the FRF and the mode shapes were read at."""

    dt: float
    """Sampling step of the Eq 7-5 history, s."""

    f_max: float
    """Upper band edge, Hz — also the Eq 7-5 basis cut."""

    _matrix: FRFMatrix
    _phi: Mapping[int, np.ndarray]
    _f_dom_hf: np.ndarray

    # -- inspection -------------------------------------------------------

    def frf(
        self, j: "int | Node", i: "int | Node | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(freq, |A_ij|)`` for one excitation/response pair.

        ``i`` defaults to :attr:`exc_node` for ``j`` — the excitation
        node that governed there.
        """
        row = self._row(j)
        exc = self.exc_node[row] if i is None else _node_tag(i)
        return self.freq, self._matrix.magnitude(int(exc), _node_tag(j))

    def mode_table(
        self, j: "int | Node", i: "int | Node | None" = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Eq 7-5 per-mode rows ``(f_n, φ_i, φ_j, a_p,m)`` for the
        high-frequency branch.

        One row per mode with ``f_n <= f_max`` (the Eq 7-5 basis), with
        ``a_p,m = 2π f_n φ_i φ_j I_eff`` as a **fraction of g** — the
        Design Guide's Table 7-2 layout.  Raises if the high-frequency
        branch did not evaluate at ``j`` (no dominant frequency in
        9-``f_max`` Hz).
        """
        row = self._row(j)
        f_dom_hf = float(self._f_dom_hf[row])
        if not np.isfinite(f_dom_hf):
            raise ValueError(
                f"FootfallResult.mode_table: the high-frequency branch "
                f"did not evaluate at node {_node_tag(j)} — no sweep "
                f"point lies in 9-{self.f_max} Hz, so Eq 7-5 has no "
                "resonant harmonic (regime "
                f"{str(self.regime[row])!r})."
            )
        exc = int(self.exc_node[row]) if i is None else _node_tag(i)
        mask = self.modes.f_n <= self.f_max
        f_n = self.modes.f_n[mask]
        phi_i = self._phi_column(exc)[mask]
        phi_j = self._phi_column(_node_tag(j))[mask]
        f_step = f_dom_hf / harmonic_for_dominant(f_dom_hf)
        i_eff = np.asarray([
            effective_impulse(f_step, float(f), self.body_weight)
            for f in f_n
        ])
        a_p_m = 2.0 * np.pi * f_n * phi_i * phi_j * i_eff / self.g
        return f_n, phi_i, phi_j, np.asarray(a_p_m)

    def to_results(self, fem: Any, path: "str | Path") -> "Path":
        """Write this evaluation as a one-frame native results file.

        ADR 0109 D5: one stage (``kind="static"``), one time station
        (``t = 0``), one partition, three nodal components —
        ``footfall_ap``, ``footfall_ratio``, ``footfall_fdom`` — each
        shape ``(1, N)`` over **every** node in ``fem``. Nodes outside
        :attr:`nodes` (the response set) are ``NaN``. ``fem`` needs only
        ``fem.nodes.ids`` — the model itself is not embedded here; feed
        the returned path and the same ``fem`` to
        ``Results.from_fem(fem, path, kind="native").viewer()`` (that
        call materialises the model side) and the ratio map renders
        with the ordinary nodal-scalar machinery — not a recorder, not
        a ``RESPONSE_CATALOG`` entry.

        Returns the path written, as a :class:`pathlib.Path`.
        """
        from pathlib import Path as _Path

        from ...results.writers._native import NativeWriter

        out_path = _Path(path)
        all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        row_of = {int(tag): row for row, tag in enumerate(all_ids)}

        def _scatter(values: np.ndarray) -> np.ndarray:
            out = np.full((1, all_ids.size), np.nan, dtype=np.float64)
            for j, tag in enumerate(self.nodes):
                out[0, row_of[int(tag)]] = values[j]
            return out

        writer = NativeWriter(out_path)
        writer.open(source_type="footfall_walking", source_path="")
        try:
            sid = writer.begin_stage(
                name="footfall", kind="static", time=np.array([0.0]),
            )
            writer.write_nodes(
                sid, "partition_0",
                node_ids=all_ids,
                components={
                    "footfall_ap": _scatter(self.a_p),
                    "footfall_ratio": _scatter(self.ratio),
                    "footfall_fdom": _scatter(self.f_dom),
                },
            )
            writer.end_stage()
        finally:
            writer.close()
        return out_path

    def to_dataframe(self) -> Any:
        """Per-response-node table as a ``pandas.DataFrame``."""
        import pandas as pd

        return pd.DataFrame(
            {
                "f_dom": self.f_dom,
                "frf_max": self.frf_max,
                "a_p_lf": self.a_p_lf,
                "a_espa_hf": self.a_espa_hf,
                "a_p": self.a_p,
                "regime": self.regime,
                "exc_node": self.exc_node,
                "limit": self.limit,
                "ratio": self.ratio,
            },
            index=pd.Index(self.nodes, name="node"),
        )

    # -- internals --------------------------------------------------------

    def _row(self, node: "int | Node") -> int:
        tag = _node_tag(node)
        try:
            return self.nodes.index(tag)
        except ValueError:
            raise KeyError(
                f"FootfallResult: node {tag} is not a response node — "
                f"the evaluation covered {list(self.nodes)}."
            ) from None

    def _phi_column(self, tag: int) -> np.ndarray:
        try:
            return self._phi[tag]
        except KeyError:
            raise KeyError(
                f"FootfallResult: no mode-shape column for node {tag} — "
                f"columns were read for {sorted(self._phi)}."
            ) from None
