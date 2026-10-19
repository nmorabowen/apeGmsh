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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._atomic_io import replace_with_retry
from ...results.readers._ladruno_writer import check_target, write_modes

if TYPE_CHECKING:
    from ...results.Results import Results
    from ..apesees import apeSees
    from ..emitter.live import LiveOpsEmitter
    from ..node import Node


__all__ = ["EigenResult", "RotationSlotUnresolvedWarning"]


class RotationSlotUnresolvedWarning(UserWarning):
    """A ``.ladruno`` export wrote a node's rotation slots as 0.0 because
    the model's DOF roles cannot say whether they hold a rotation or a
    pore pressure (an element-less node beside a u-p element, a node no
    beam or shell touches). The message names the nodes and the reason."""


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
        as ``LAMBDA``, and each fem node's ``ops.nodeEigenvector`` laid
        onto ``Ux..Rz`` by its own DOF layout (its vector's length is its
        ndf). A rotation slot holds the rotation only where the model's
        DOF roles say so: a beam or shell touches the node, or it is
        element-less in a model without a u-p element. A u-p node's pore
        pressure is never written as a rotation. A slot the roles cannot
        settle is written as 0.0, with one
        :class:`RotationSlotUnresolvedWarning` naming the nodes. A
        non-positive eigenvalue writes 0 Hz / 0 s and raises one
        ``UserWarning`` naming the mode, as :meth:`to_native` does.

        ``MODEL/NODES`` is the bridge's fem. ``MODEL/ELEMENTS`` is empty
        (apeGmsh holds no C++ class tag to name a bucket with), and
        ``INFO/PROVENANCE@unsupported`` says so. The bridge's model
        archive (``ops.h5``) is written beside the file as ``<stem>.h5``,
        the name schema §7b.1 pairs a ``.ladruno`` with, and the file is
        stamped against it::

            p = ops.eigen(3).to_ladruno()
            r = Results.from_ladruno(p, model_h5=p.with_suffix(".h5"))

        ``r.modes`` lists one ``mode`` stage per eigenvalue. The modes
        are written and stamped first and the archive follows, so a
        refusal or a failed write leaves no new file behind. A
        ``.ladruno`` already at ``path`` is replaced (the caller chose
        that path) and is not restored if a later step fails, e.g. the
        archive move. An existing ``<stem>.h5`` of this bridge (the same
        lineage) is kept as it is.

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
        ValueError
            ``path`` has an ``.h5`` suffix, so it would be its own
            archive (``to_ladruno("model.h5")`` would replace the
            archive); and as :meth:`to_native`. Nothing is written.
        RuntimeError
            As :meth:`to_native`.
        """
        import tempfile

        bridge, ndm, _ = self._checked_solve("to_ladruno")
        if path is None:
            out_path = Path(
                tempfile.mkdtemp(prefix="apegmsh_modes_")
            ) / "modes.ladruno"
        else:
            out_path = Path(os.fspath(path))
        archive = out_path.with_suffix(".h5")
        check_target(out_path, archive)

        ops = self._live.ops
        node_ids = np.asarray(bridge.fem.nodes.ids, dtype=np.int64)
        coords = np.asarray(bridge.fem.nodes.coords, dtype=np.float64)
        n_modes = len(self.eigenvalues)
        # One nodeEigenvector call per node and mode; a node's vector is
        # as long as ITS ndf, so its length is the node's own DOF count.
        vectors = [
            [np.asarray(ops.nodeEigenvector(int(nid), k + 1),
                        dtype=np.float64).ravel() for nid in node_ids]
            for k in range(n_modes)
        ]
        ndf_of = {int(n): int(v.size) for n, v in zip(node_ids, vectors[0])}

        with tempfile.TemporaryDirectory(
            prefix="apegmsh_modes_", dir=out_path.parent,
        ) as tmp:
            staged = Path(tmp) / archive.name
            bridge.h5(str(staged))
            place = _archive_is_free(staged, archive)
            layouts, rotations, unresolved = _ladruno_slots(
                staged, node_ids, ndf_of, ndm, where="EigenResult.to_ladruno")
            shapes = np.stack([_spatial(rows, layouts) for rows in vectors])
            _warn_unresolved("EigenResult.to_ladruno", unresolved, 2)
            # The modes go first, stamped against the staged copy (the
            # same lineage), so a refused or failed write leaves no
            # archive behind; the archive follows them into place.
            out = write_modes(
                out_path, model_h5=staged, node_ids=node_ids,
                coords=coords, ndm=ndm,
                eigenvalues=self.eigenvalues,
                fields=lambda k: shapes[k - 1], rotations=rotations,
                source="eigen", stacklevel=3,
            )
            if place:
                try:
                    replace_with_retry(staged, archive)
                except BaseException:
                    out.unlink(missing_ok=True)
                    raise
        return out

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


def _ladruno_slots(
    model_h5: Path, node_ids: "Any", ndf_of: "Mapping[int, int]", ndm: int,
    *, where: str,
) -> "tuple[list[tuple[int, ...]], np.ndarray, list[tuple[int, str]]]":
    """How each node's DOFs land on ``ux..rz`` for a ``.ladruno`` export.

    Returns, per node of ``node_ids``: its DOF layout (DOF index -> spatial
    slot, the emit's ``_node_layout(ndf, ndm)``), whether its rotation
    slots hold rotations, and the ``(node, reason)`` of each node whose
    layout keeps a rotation slot that the model's DOF roles
    (:func:`~apeGmsh.opensees._recorder_translate.model_h5_rotation_roles`)
    refuse for any reason but a u-p element touching it. The layout alone
    cannot tell a 2-D frame node (ux, uy, rz) from a 2-D u-p node (ux, uy,
    p); the roles do, so a pore pressure is never written as a rotation.
    """
    from .._internal.build.recorders import _node_layout
    from .._readback import _UP_FAMILIES
    from .._recorder_translate import model_h5_rotation_roles

    ids = [int(n) for n in node_ids]
    layouts = [_node_layout(int(ndf_of[n]), ndm) for n in ids]
    roles = model_h5_rotation_roles(model_h5, ids, where=where)
    up = {cls.__name__ for cls in _UP_FAMILIES}
    rotations = np.zeros(len(ids), dtype=bool)
    unresolved: list[tuple[int, str]] = []
    for i, (n, lay) in enumerate(zip(ids, layouts)):
        if not any(slot >= 3 for slot in lay):
            continue                    # no rotation slot at all
        why = roles.refusal(n)
        if why is None:
            rotations[i] = True
        elif not (n in roles.blocked and roles.blocked[n] in up):
            unresolved.append((n, why))
    return layouts, rotations, unresolved


def _spatial(rows: "Any", layouts: "list[tuple[int, ...]]") -> np.ndarray:
    """``[n x 6]`` by spatial slot from per-node DOF rows (each row the
    node's own DOFs, padded or not); a DOF past a row's end reads 0.0."""
    out = np.zeros((len(layouts), 6), dtype=np.float64)
    for i, (row, lay) in enumerate(zip(rows, layouts)):
        vals = np.asarray(row, dtype=np.float64).ravel()
        for dof, slot in enumerate(lay):
            if dof < vals.size:
                out[i, slot] = vals[dof]
    return out


def _warn_unresolved(
    where: str, unresolved: "list[tuple[int, str]]", stacklevel: int,
) -> None:
    """One :class:`RotationSlotUnresolvedWarning` for every unsettled node."""
    import warnings

    if not unresolved:
        return
    node, why = unresolved[0]
    warnings.warn(
        f"{where}: the rotation slots of {len(unresolved)} node(s) "
        f"({[n for n, _ in unresolved[:10]]}) are written as 0.0, neither "
        f"a rotation nor a pore pressure: for node {node}, {why}.",
        RotationSlotUnresolvedWarning,
        stacklevel=stacklevel + 1,
    )


def _lineage(path: Path) -> "tuple[str, str] | None":
    """``(model_hash, fem_hash)`` of an archive's ``/meta/lineage``;
    ``None`` when the file has none or is not HDF5."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f or "lineage" not in f["meta"]:
                return None
            attrs = f["meta"]["lineage"].attrs
            if "model_hash" not in attrs or "fem_hash" not in attrs:
                return None
            return str(attrs["model_hash"]), str(attrs["fem_hash"])
    except OSError:
        return None


def _archive_is_free(staged: Path, archive: Path) -> bool:
    """``True`` when ``archive`` is absent and ``staged`` must move there.

    An existing ``archive`` with the lineage hashes of ``staged`` is
    this bridge's and stays as it is (``False``). Any other existing
    file refuses, so a session's own ``<model_name>.h5`` is never
    replaced.
    """
    if not archive.exists():
        return True
    have = _lineage(archive) if archive.is_file() else None
    if have is None or have != _lineage(staged):
        raise FileExistsError(
            f"EigenResult.to_ladruno: {archive} exists and is not this "
            "bridge's model archive (its /meta/lineage differs or is "
            "missing), so the .ladruno beside it cannot pair with it. "
            "Pass a path whose <stem>.h5 is free, or remove that file. "
            "Nothing was written."
        )
    return False
