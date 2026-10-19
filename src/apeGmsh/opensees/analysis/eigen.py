"""
``EigenResult`` — the return type of :meth:`apeGmsh.opensees.apeSees.eigen`.

Eigen is the one OpenSees analysis call that is **not** an Analysis
primitive: it does not emit ``analysis <Type>``, has no analysis chain,
takes no stepping, and returns values directly. Modelled as a bridge
method (``apeSees.eigen``) that drives a :class:`LiveOpsEmitter` end-
to-end and wraps the returned eigenvalues in this dataclass.

The dataclass carries the eigenvalue array plus a back-reference to the
live emitter so :meth:`mode_shape` can query ``ops.nodeEigenvector``
without re-running the eigen solve. :meth:`to_native` persists the modes
as ``kind="mode"`` stages of a native results file (the
modes route on pip ``openseespy``), and :meth:`results` opens that file.
:meth:`to_ladruno` writes the same modes as a stamped ``.ladruno`` (ADR
0122), the format the native file retires into. No modal-mass /
participation calculations live here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._atomic_io import replace_with_retry
from ...results.readers._ladruno_writer import write_modes

if TYPE_CHECKING:
    from ...results.Results import Results
    from ..apesees import apeSees
    from ..emitter.live import LiveOpsEmitter
    from ..node import Node


__all__ = ["EigenResult"]


@dataclass(frozen=True, slots=True)
class EigenResult:
    """Eigenvalues + lazy mode-shape access from one ``eigen`` call.

    Attributes
    ----------
    eigenvalues
        1-D ``np.ndarray`` of ``λ_i = ω_i²`` in the order returned by
        OpenSees (ascending magnitude for the default solver).

    Notes
    -----
    The eigenvectors are NOT eagerly fetched — they live in openseespy's
    domain state until :meth:`mode_shape` queries them via
    ``ops.nodeEigenvector``. Calling :meth:`mode_shape` after another
    ``apeSees.eigen(...)`` or ``ops.wipe()`` will either return the new
    solve's vectors or raise from openseespy. The :class:`LiveOpsEmitter`
    back-reference is kept so the call can still be made; no attempt is
    made to detect staleness there. :meth:`to_native` does detect it
    (the domain epoch) and refuses a rebuilt domain.
    """

    eigenvalues: np.ndarray

    # Implementation handles for lazy mode-shape access. Underscore-
    # prefixed; not part of the user-facing surface.
    _live: "LiveOpsEmitter"

    # Filled by ``apeSees.eigen`` / ``apeSees.eigen_feast``
    # for :meth:`to_native`: the bridge that solved (its ``fem``
    # and ``h5`` describe the file), its ``ndm`` / envelope ``ndf``, and
    # ``live.domain_epoch()`` read right after the solve. ``None`` only
    # for a result built by hand, which ``to_native`` refuses.
    _bridge: "apeSees | None" = None
    _ndm: int | None = None
    _ndf: int | None = None
    _epoch: int | None = None

    @property
    def omega(self) -> np.ndarray:
        """Natural circular frequencies ``ω_i = √λ_i`` (rad/s)."""
        return np.asarray(np.sqrt(self.eigenvalues))

    @property
    def freq(self) -> np.ndarray:
        """Natural frequencies ``f_i = ω_i / (2π)`` (Hz)."""
        return self.omega / (2.0 * np.pi)

    @property
    def periods(self) -> np.ndarray:
        """Natural periods ``T_i = 2π / ω_i = 1 / f_i``, one per mode.

        A 1-D ``np.ndarray`` in the order of :attr:`eigenvalues` (mode 1
        first, so the longest period first for the default solver), in
        seconds when the model's time unit is the second: ``λ_i`` comes
        from the model's own mass and stiffness, with no unit conversion.
        A zero eigenvalue (a mechanism, or a DOF with no mass) gives
        ``inf``; a negative one gives ``nan``.

        This is the eigen read-back ADR 0115 D1 documents: ``eigen``
        keeps returning :class:`EigenResult`, and the periods are read
        here rather than through ``Node.read`` (which reads the static
        or transient state an ``analyze`` left).
        """
        return 1.0 / self.freq

    def mode_shape(self, node: "int | Node", mode: int) -> np.ndarray:
        """Return the mode shape for ``node`` in ``mode`` (1-indexed).

        Parameters
        ----------
        node
            Either a plain integer node tag or a :class:`Node` instance
            from ``ops.nodes.get(...)``.
        mode
            1-indexed mode number, matching OpenSees' ``$modeNumber``
            convention.

        Returns
        -------
        np.ndarray
            Length-``ndf`` vector of the eigenvector's DOF entries at
            ``node``, fetched live via ``ops.nodeEigenvector``.
        """
        from ..node import Node as _Node  # local import — avoid cycle

        if isinstance(node, _Node):
            tag = int(node.tag)
        else:
            tag = int(node)
        values: Any = self._live.ops.nodeEigenvector(tag, int(mode))
        return np.asarray(values, dtype=np.float64)

    def to_native(self, path: "str | os.PathLike[str] | None" = None) -> Path:
        """Write the modes as ``kind="mode"`` stages of a native results
        file and return its path.

        One stage per eigenvalue (``mode_<k>``, ``mode_index`` 1..n, at
        ``time = [0.0]``) with ``eigenvalue``, ``frequency_hz = ω/2π`` and
        ``period_s = 2π/ω`` (``ω = √λ``), the layout
        ``DomainCapture.capture_modes`` writes. A non-positive eigenvalue
        writes 0 for both and raises one ``UserWarning`` naming the mode.

        Each fem node's vector is read once per mode with
        ``ops.nodeEigenvector``: ``displacement_*`` holds its first
        ``min(3, ndm)`` entries and, when the bridge's envelope ``ndf``
        is at least 6, ``rotation_*`` holds entries 4-6. A node whose own
        vector is shorter (a solid node in a solid-and-frame model)
        writes 0.0 for the entries it lacks.

        The file embeds ``/model`` (the bridge's ``fem``) and
        ``/opensees`` (``bridge.h5``), so it opens alone::

            p = ops.eigen(3).to_native()
            r = Results.from_native(p, model=OpenSeesModel.from_h5(p))

        ``source_type`` is ``"eigen"``. The root attrs name the module
        that solved: ``opensees_backend="stock"``, or ``"fork"`` plus its
        ``ladrunoBuild()`` sha. On the fork this stays an explicit export
        (it also covers :meth:`apeSees.eigen_feast`); the results-format
        resolver never offers it there.

        Parameters
        ----------
        path
            The file to write. ``None`` writes ``modes.h5`` in a fresh
            directory under ``tempfile.gettempdir()``.

        Raises
        ------
        ValueError
            The solve returned no modes (an ``eigen_feast`` band with no
            eigenvalue in it). Nothing is written.
        RuntimeError
            The result was not returned by ``apeSees.eigen`` /
            ``eigen_feast``, or the live domain was rebuilt since the
            solve: any later ``LiveOpsEmitter`` (``modal_properties``,
            another ``eigen``, ``analyze``, a second bridge's solve, ...)
            moves ``live.domain_epoch()``, and the vectors in the domain
            are no longer this solve's. A deck-only ``ops.tcl(path)`` /
            ``ops.py(path)`` does not rebuild it. One safe false positive:
            a moment-curvature call (``sections/_mc``, ``wipe=False``)
            also moves the epoch, so ``to_native`` raises after it even
            though the eigenvectors may survive. Re-run ``eigen`` then.
            Mutating the bridge between ``eigen`` and ``to_native``
            without a re-emit is unsupported.
        """
        return self._write_native(path)

    def _checked_solve(self, method: str) -> "tuple[apeSees, int, int]":
        """``(bridge, ndm, ndf)`` of the solve, or raise what
        :meth:`to_native` documents; ``method`` names the caller."""
        from ..emitter.live import domain_epoch

        if len(self.eigenvalues) == 0:
            raise ValueError(
                f"EigenResult.{method}: the solve returned no modes (an "
                "eigen_feast band with no eigenvalue in it), so there is "
                "nothing to write; no file was created."
            )
        bridge = self._bridge
        if (
            bridge is None or self._ndm is None or self._ndf is None
            or self._epoch is None
        ):
            raise RuntimeError(
                f"EigenResult.{method}: this result was not returned by "
                "apeSees.eigen / apeSees.eigen_feast, so it names no "
                "bridge or solve to write."
            )
        now = domain_epoch()
        if now != self._epoch:
            raise RuntimeError(
                f"EigenResult.{method}: the live OpenSees domain was "
                f"rebuilt since this eigen solve (domain epoch {self._epoch} "
                f"-> {now}): a later LiveOpsEmitter (another eigen, "
                "modal_properties, analyze, a second bridge, ...) replaced "
                "the eigenvectors. Re-run ops.eigen(...) and export that "
                "result."
            )
        return bridge, self._ndm, self._ndf

    def to_ladruno(self, path: "str | os.PathLike[str] | None" = None) -> Path:
        """Write the modes as a stamped ``.ladruno`` and return its path.

        One ``KIND="eigen"`` ``MODEL_STAGE`` holds every mode, in the
        layout the fork recorder writes for ``modesOfVibration`` +
        ``modesOfVibrationRotational`` (the table is in
        :mod:`apeGmsh.results.readers._ladruno_writer`): each eigenvalue
        as ``LAMBDA``, ``Ux..`` from the first ``ndm`` entries of each
        fem node's ``ops.nodeEigenvector`` and, where the model has
        them, the rotations (entry 3 in 2-D, entries 4-6 in 3-D). A node
        whose own vector is shorter writes 0.0 for the entries it lacks.
        A non-positive eigenvalue writes 0 Hz / 0 s and raises one
        ``UserWarning`` naming the mode, as :meth:`to_native` does.

        ``MODEL/NODES`` is the bridge's fem. ``MODEL/ELEMENTS`` is empty
        (apeGmsh holds no C++ class tag to name a bucket with), and
        ``INFO/PROVENANCE@unsupported`` says so. The bridge's model
        archive (``ops.h5``) is written beside the file as ``<stem>.h5``,
        the name schema §7b.1 pairs a ``.ladruno`` with, and the file is
        stamped against it::

            p = ops.eigen(3).to_ladruno()
            r = Results.from_ladruno(p, model_h5=p.with_suffix(".h5"))

        ``r.modes`` lists one ``mode`` stage per eigenvalue.

        Parameters
        ----------
        path
            The ``.ladruno`` to write (replaced when it exists). ``None``
            writes ``modes.ladruno`` in a fresh directory under
            ``tempfile.gettempdir()``.

        Raises
        ------
        FileExistsError
            ``<stem>.h5`` exists and is not this bridge's archive: its
            ``/meta/lineage`` hashes differ, or it has none. Nothing is
            written.
        ValueError, RuntimeError
            As :meth:`to_native`.
        """
        import tempfile

        bridge, ndm, ndf = self._checked_solve("to_ladruno")
        ops = self._live.ops
        node_ids = np.asarray(bridge.fem.nodes.ids, dtype=np.int64)
        coords = np.asarray(bridge.fem.nodes.coords, dtype=np.float64)

        # The recorder's rule: translations are a node's first ndm entries,
        # rotations its entry 3 (2-D) or 4-6 (3-D) only when its own
        # vector reaches them, so a u-p node's pressure is no rotation.
        rot_cols = [c for c in ((2,) if ndm == 2 else (3, 4, 5)) if c < ndf] \
            if ndm in (2, 3) else []

        def field(mode: int) -> np.ndarray:
            # One nodeEigenvector call per node; a node's vector is as
            # long as ITS ndf, so a shorter one fills only what it has.
            out = np.zeros((node_ids.size, ndf), dtype=np.float64)
            for row, nid in enumerate(node_ids):
                vec = np.asarray(
                    ops.nodeEigenvector(int(nid), mode), dtype=np.float64,
                ).ravel()
                n_t = min(ndm, ndf, vec.size)
                out[row, :n_t] = vec[:n_t]
                if rot_cols and vec.size > rot_cols[-1]:
                    out[row, rot_cols] = vec[rot_cols]
            return out

        if path is None:
            out_path = Path(
                tempfile.mkdtemp(prefix="apegmsh_modes_")
            ) / "modes.ladruno"
        else:
            out_path = Path(os.fspath(path))
        archive = out_path.with_suffix(".h5")
        with tempfile.TemporaryDirectory(
            prefix="apegmsh_modes_", dir=out_path.parent,
        ) as tmp:
            staged = Path(tmp) / archive.name
            bridge.h5(str(staged))
            _place_archive(staged, archive)
        return write_modes(
            out_path, model_h5=archive, node_ids=node_ids, coords=coords,
            ndm=ndm, ndf=ndf, eigenvalues=self.eigenvalues, fields=field,
            source="eigen", stacklevel=3,
        )

    def _write_native(self, path: "str | os.PathLike[str] | None") -> Path:
        """:meth:`to_native`'s body, called straight from each public
        method so a mode warning's ``stacklevel=4`` names their caller."""
        import tempfile

        from ...results.writers._native import NativeWriter
        from ..emitter.live import _backend_of
        from .modal import _write_mode_stages

        bridge, ndm, ndf = self._checked_solve("to_native")
        ops = self._live.ops
        backend = _backend_of(ops)
        n_trans = min(3, ndm)
        rotations = ndf >= 6
        node_ids = np.asarray(bridge.fem.nodes.ids, dtype=np.int64)

        def field(mode: int) -> np.ndarray:
            # One nodeEigenvector call per node, sliced per node: a
            # node's vector is as long as ITS ndf, so a solid node in a
            # frame model (3 < envelope 6) fills only what it has.
            out = np.zeros((node_ids.size, 6), dtype=np.float64)
            for row, nid in enumerate(node_ids):
                vec = ops.nodeEigenvector(int(nid), mode)
                for axis in range(min(n_trans, len(vec))):
                    out[row, axis] = vec[axis]
                if rotations and len(vec) >= 6:
                    for axis in range(3):
                        out[row, 3 + axis] = vec[3 + axis]
            return out

        if path is None:
            out_path = Path(
                tempfile.mkdtemp(prefix="apegmsh_modes_")
            ) / "modes.h5"
        else:
            out_path = Path(os.fspath(path))

        with tempfile.TemporaryDirectory(prefix="apegmsh_modes_") as tmp:
            # Named as DomainCapture names its sidecar, so /model/meta
            # carries the same model_name ("<stem>.model").
            sidecar = Path(tmp) / out_path.with_suffix(".model.h5").name
            bridge.h5(str(sidecar))
            writer = NativeWriter(out_path)
            writer.open(
                fem=bridge.fem,
                source_type="eigen",
                source_path="<openseespy>",
                model_h5_src=sidecar,
                opensees_backend=backend.kind,
                opensees_build=backend.build,
            )
            try:
                _write_mode_stages(
                    writer, self.eigenvalues, node_ids, field,
                    n_trans=n_trans, rotations=rotations, stacklevel=4,
                )
            finally:
                writer.close()
        return out_path

    def results(
        self, model_h5: "str | os.PathLike[str] | None" = None,
    ) -> "Results":
        """Write the modes with :meth:`to_native` (to a temp file) and
        open them as a :class:`~apeGmsh.results.Results`; ``.modes``
        lists one mode-scoped ``Results`` per eigenvalue.

        ``model_h5`` names the ``model.h5`` archive to bind; ``None``
        binds the ``/opensees`` zone the file itself embeds. Raises what
        :meth:`to_native` raises.
        """
        from ...results.Results import Results as _Results
        from ..opensees_model import OpenSeesModel

        p = self._write_native(None)
        src = p if model_h5 is None else Path(os.fspath(model_h5))
        return _Results.from_native(
            p,
            model=OpenSeesModel.from_h5(str(src)),
            model_path=None if model_h5 is None else src,
        )


def _lineage(path: Path) -> "tuple[str, str] | None":
    """``(model_hash, fem_hash)`` of an archive's ``/meta/lineage``."""
    import h5py

    with h5py.File(path, "r") as f:
        if "meta" not in f or "lineage" not in f["meta"]:
            return None
        attrs = f["meta"]["lineage"].attrs
        if "model_hash" not in attrs or "fem_hash" not in attrs:
            return None
        return str(attrs["model_hash"]), str(attrs["fem_hash"])


def _place_archive(staged: Path, archive: Path) -> None:
    """Move the freshly written ``staged`` archive to ``archive``.

    An existing ``archive`` with the same lineage hashes is this
    bridge's and stays as it is; any other existing file refuses, so a
    session's own ``<model_name>.h5`` is never replaced.
    """
    if archive.exists():
        have = _lineage(archive) if archive.is_file() else None
        if have is None or have != _lineage(staged):
            raise FileExistsError(
                f"EigenResult.to_ladruno: {archive} exists and is not this "
                "bridge's model archive (its /meta/lineage differs or is "
                "missing), so the .ladruno beside it cannot pair with it. "
                "Pass a path whose <stem>.h5 is free, or remove that file. "
                "Nothing was written."
            )
        return
    replace_with_retry(staged, archive)
