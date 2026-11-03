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


__all__ = ["USER_COMPONENTS", "USER_RESULT", "FootfallModes", "FootfallResult"]

#: The ``.user.h5`` entry :meth:`FootfallResult.to_results` writes (ADR 0131 D6).
USER_RESULT = "footfall"
#: Its components, in ``DATA`` column order: fraction of g, ratio, Hz.
USER_COMPONENTS = ("footfall_ap", "footfall_ratio", "footfall_fdom")


def _writer_text() -> str:
    """``/meta@writer``: this driver, apeGmsh's version, and the F2-b backend.

    footfall_walking ran on the live modal basis, in this process, on the
    module the live resolver bound, so that module's verdict is the one to
    stamp: ``opensees fork <sha>`` or ``opensees stock``.
    """
    import apeGmsh

    from ..emitter.live import get_backend_info

    backend = get_backend_info()
    engine = f"opensees {backend.kind}"
    if backend.build:
        engine += f" {backend.build}"
    return f"apeGmsh footfall_walking {apeGmsh.__version__}; {engine}"


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

    def to_results(
        self, fem: Any, path: "str | Path", *, model_h5: "str | Path",
    ) -> "Path":
        """Write this evaluation into ``<stem>.user.h5`` as a user result.

        ADR 0131 D6: footfall is a **state-free** user result (ADR 0130
        D7), written by :func:`~apeGmsh.results.user.write_user_file`.
        The entry is named :data:`USER_RESULT` (``"footfall"``), on the
        nodes, ``@basis="model"``, with three components —
        ``footfall_ap`` (fraction of g), ``footfall_ratio`` and
        ``footfall_fdom`` (Hz) — over **every** node in ``fem``, so
        ``DATA`` has shape ``(1, N, 3)``. Nodes outside :attr:`nodes`
        (the response set) are ``NaN``.

        ``path`` must be named ``<stem>.user.h5``; an existing file
        gains (or replaces) the ``"footfall"`` entry and keeps the rest,
        under that writer's rules. ``model_h5`` is the model archive
        of ``fem`` (``ops.h5(...)`` or ``fem.to_h5(...)``): it supplies
        the pairing hashes, and its ``fem_hash`` must equal
        ``fem.snapshot_id``. There is no default — a guessed archive
        would pair the map with another model.

        ``/meta@writer`` carries the provenance: this driver, the
        apeGmsh version, and the OpenSees backend (and fork build) the
        live modal basis ran on.

        Read it back with
        ``UserFile(path).read_result("footfall")`` from
        :mod:`apeGmsh.results.user`.

        Returns the path written, as a :class:`pathlib.Path`.

        Raises :class:`~apeGmsh.results.user.UserWriteRefused` when
        ``model_h5`` is missing or belongs to another fem, when ``fem``
        has no nodes, or when a response node is not in ``fem``; nothing
        is written then.
        """
        from pathlib import Path as _Path

        from ...results.user import (
            UserResult,
            UserWriteRefused,
            model_hashes,
            write_user_file,
        )

        if model_h5 is None:
            raise UserWriteRefused(
                "FootfallResult.to_results: model_h5= is required — the model "
                "archive of this fem, whose /meta/lineage pairs the .user.h5."
            )
        all_ids = np.asarray(fem.nodes.ids, dtype=np.int64).ravel()
        if all_ids.size == 0:
            raise UserWriteRefused(
                "FootfallResult.to_results: fem has no nodes — there is "
                "nothing to place the footfall map on."
            )
        row_of = {int(tag): row for row, tag in enumerate(all_ids)}
        missing = [int(t) for t in self.nodes if int(t) not in row_of]
        if missing:
            raise UserWriteRefused(
                f"FootfallResult.to_results: response nodes {missing} are not "
                "in fem.nodes.ids — the evaluation was run on another model."
            )
        fem_hash, _ = model_hashes(model_h5)
        snapshot = str(fem.snapshot_id)
        if fem_hash != snapshot:
            raise UserWriteRefused(
                f"FootfallResult.to_results: {model_h5} is not this fem's "
                f"archive (its fem_hash {fem_hash} is not fem.snapshot_id "
                f"{snapshot})."
            )

        data = np.full((1, all_ids.size, 3), np.nan, dtype=np.float64)
        for j, tag in enumerate(self.nodes):
            row = row_of[int(tag)]
            data[0, row, 0] = self.a_p[j]
            data[0, row, 1] = self.ratio[j]
            data[0, row, 2] = self.f_dom[j]

        result = UserResult(
            name=USER_RESULT, target="nodes", basis="model",
            components=USER_COMPONENTS, units=("g", "1", "Hz"),
            data=data, ids=all_ids,
        )
        return write_user_file(
            _Path(path), model_h5, results=(result,), writer=_writer_text(),
        )

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
