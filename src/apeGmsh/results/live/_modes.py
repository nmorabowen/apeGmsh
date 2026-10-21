"""capture_live_modes — the modes of a plain openseespy deck, on any build (#2342).

A deck the bridge did not build (hand-written ``ops.node`` /
``ops.element`` calls over a FEMData) has no :class:`EigenResult`
(``apeSees.eigen`` makes one) and no default recorder route for modes
(``emit_default`` refuses modal records). :func:`capture_live_modes`
solves ``ops.eigen(n)`` on the live domain and writes one native
``mode_<k>`` stage per mode, in the layout ``EigenResult.to_native``
writes, through the same writer helpers.

Each fem node is paired with the live node of its tag by fact
(:class:`~apeGmsh.opensees._recorder_translate.LiveDomainFacts`): the
live domain must hold it, and its ``nodeCoord`` must be the FEM node's
first ``ndm`` coordinates. Its ndf is the live ``getNDF``, and the rotation
roles come from the live elements' ``eleType``. The vectors are the raw
``nodeEigenvector`` values, unscaled.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ...opensees._recorder_translate import LiveDomainFacts
from ...opensees._target import backend_info_of
from ...opensees.analysis.eigen import _live_slots, _warn_unresolved
from ...opensees.analysis.modal import _native_fields, _write_mode_stages
from ...opensees.emitter.live import get_ops
from ...opensees.opensees_model import OpenSeesModel
from ..writers._native import NativeWriter

if TYPE_CHECKING:
    from ...mesh.FEMData import FEMData

__all__ = ["capture_live_modes"]

_WHERE = "capture_live_modes"

#: The ``ops.eigen`` solver flags ``solver=`` forwards: the ones stock
#: openseespy accepts that solve the generalised problem ``K x = λ M x``.
#: openseespy ignores a flag it does not know and solves with its default,
#: so any other value refuses by name instead of being passed on.
_SOLVERS = ("-genBandArpack", "-fullGenLapack")


def capture_live_modes(
    model_h5_or_fem: "str | os.PathLike[str] | FEMData",
    n_modes: int,
    path: "str | os.PathLike[str]",
    *,
    ops: Optional[Any] = None,
    solver: Optional[str] = None,
) -> Path:
    """Solve ``ops.eigen(n_modes)`` on the live domain and write the modes
    as ``kind="mode"`` stages of a native results file; return its path.

    One stage per mode (``mode_<k>``, ``mode_index`` 1..n, at ``time =
    [0.0]``) carries ``eigenvalue``, ``frequency_hz = ω/2π`` and
    ``period_s = 2π/ω`` (``ω = √λ``), the layout ``EigenResult.to_native``
    writes. Each fem node's ``nodeEigenvector`` is written raw (unscaled),
    laid onto ``ux..rz`` by its own live ndf: ``displacement_*`` holds the
    first ``min(3, ndm)`` slots. A rotation slot holds the rotation only
    where the live elements' DOF roles say so (a beam, shell or
    zero-length element touches the node); a slot they cannot settle is
    written as 0.0 with one ``RotationSlotUnresolvedWarning``, except at a
    node an ``ASDEmbeddedNodeElement`` touches, where it refuses. That
    element counts as translational-only unless every node it names has
    ndf ``ndm + 1`` (a possible ``-p`` pressure), so a beam node embedded
    in a solid keeps the beam's rotation.

    The file embeds the FEMData as ``/model`` and no ``/opensees`` zone
    (the deck is not the bridge's). Read it with the FEMData archive as
    the model, or with ``Results.from_fem(fem, p, kind="native")``::

        p = capture_live_modes("model.h5", 3, "modes.h5", ops=ops)
        r = Results.from_native(p, model=OpenSeesModel.from_h5("model.h5"))

    The capture trusts the eigenvalues its own ``ops.eigen`` call returns.
    On an indefinite stiffness (a negative spring, a softened member) the
    stock default eigen solver can return plausible positive eigenvalues
    without raising: a -10 spring under ``ops.eigen(3)`` gave 0.02 three
    times, where ``-fullGenLapack`` or a ``system BandGeneral`` deck gives
    the true -0.2. No cheap check tells them apart, and a solve made
    outside the capture does not guard the capture's own solve. When the
    stiffness may be indefinite, declare ``ops.system("BandGeneral")``
    before capturing, or pass ``solver="-fullGenLapack"`` (a dense, O(n³)
    solve, for small models only). ``ops.wipeAnalysis()`` drops the
    system too, so declare it again after one. ``DomainCapture.capture_modes``
    had the same exposure.

    Parameters
    ----------
    model_h5_or_fem
        The FEMData the deck was built from, or a ``model.h5`` holding it
        (``fem.to_h5``, or a bridge archive, whose FEMData alone is read).
        Every fem node is captured.
    n_modes
        How many modes to solve for, at least 1.
    path
        The native file to write (replaced when it exists).
    ops
        The OpenSees module holding the deck. ``None`` takes the one the
        bridge resolves (:func:`~apeGmsh.opensees.emitter.live.get_ops`).
    solver
        The eigen solver flag the capture's solve passes,
        ``ops.eigen(solver, n_modes)``: ``"-genBandArpack"`` or
        ``"-fullGenLapack"``. ``None`` solves with the module's default.
        ``"-genBandArpack"`` is the stock default, so it does not guard on
        its own: on an indefinite stiffness, declare
        ``ops.system("BandGeneral")`` or pass ``"-fullGenLapack"``.

    Raises
    ------
    ValueError
        ``n_modes`` is not a positive integer; ``solver`` is not one of
        the flags above; the live domain declares no
        ndm; a fem node is not in the live domain, or its live coordinates
        are not the FEM node's; a node's eigenvector length is not its
        live ndf; ``ops.eigen`` raised (on stock openseespy, a second
        eigen solve in a domain with no analysis, even with a ``system``
        declared: call ``ops.wipeAnalysis()``, then
        ``ops.system("BandGeneral")`` again, before ``capture_live_modes``,
        since ``wipeAnalysis()`` also drops the system that guards against
        the wrong positive eigenvalues above; or, to keep your own system,
        define ``ops.analysis(...)`` with it before the first eigen); or
        the solve returned
        another number of modes, or a
        non-positive or non-finite eigenvalue (a mechanism or an
        unconverged solve); or a node an ``ASDEmbeddedNodeElement``
        touches keeps a rotation slot its elements do not settle. Nothing
        is written.
    FileNotFoundError
        ``model_h5_or_fem`` names no file.
    """
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 1:
        raise ValueError(
            f"{_WHERE}: n_modes must be a positive integer, got {n_modes!r}.")
    if solver is not None and solver not in _SOLVERS:
        raise ValueError(
            f"{_WHERE}: solver={solver!r} is not an eigen solver flag this "
            f"capture forwards; pass one of {list(_SOLVERS)}, or None for the "
            "default. openseespy ignores a flag it does not know, so it is "
            "refused here rather than passed on.")
    out_path = Path(os.fspath(path))
    if ops is None:
        ops = get_ops()
    with tempfile.TemporaryDirectory(prefix="apegmsh_live_modes_") as tmp:
        archive = _archive(model_h5_or_fem, out_path, Path(tmp))
        return _capture(archive, n_modes, out_path, ops, solver)


def _archive(
    model_h5_or_fem: "str | os.PathLike[str] | FEMData", out_path: Path,
    tmp: Path,
) -> OpenSeesModel:
    """The archive broker of a ``model.h5`` path, or of a FEMData written
    to ``tmp``; refuses an ``out_path`` that is the archive itself."""
    if isinstance(model_h5_or_fem, (str, os.PathLike)):
        src = Path(os.fspath(model_h5_or_fem))
        if not src.is_file():
            raise FileNotFoundError(
                f"{_WHERE}: no model archive at {src}; pass the FEMData the "
                "deck was built from, or its fem.to_h5(...) file.")
        if out_path.exists() and out_path.resolve() == src.resolve():
            raise ValueError(
                f"{_WHERE}: path {out_path} is the model archive itself; "
                "writing the modes there would replace it. Pass another "
                "path.")
    else:
        src = tmp / "model.h5"
        model_h5_or_fem.to_h5(str(src))
    return OpenSeesModel.from_h5(src)


def _capture(
    archive: OpenSeesModel, n_modes: int, out_path: Path, ops: Any,
    solver: Optional[str],
) -> Path:
    fem = archive.fem
    facts = LiveDomainFacts(ops, archive)
    ndm = facts.ndm
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    # Every fem node is in the live domain, at the FEM coordinates.
    ndf_of = facts.nodes_ndf(node_ids, where=_WHERE)

    call = ("" if solver is None else f"{solver!r}, ") + str(n_modes)
    try:
        solved = (ops.eigen(n_modes) if solver is None
                  else ops.eigen(solver, n_modes))
    except Exception as exc:  # the module's own error class (OpenSeesError)
        raise ValueError(
            f"{_WHERE}: ops.eigen({call}) failed ({exc!r}; the solver's "
            "own message is on stderr). On stock openseespy a second eigen "
            "solve in a domain with no analysis fails with 'no EigenSOE has "
            "been set', even with a system declared: call ops.wipeAnalysis(), "
            "then ops.system(\"BandGeneral\") again, before "
            "capture_live_modes; or, to keep your own system, define "
            "ops.analysis(...) with it before the first eigen. "
            "wipeAnalysis() also drops the system, and "
            "without BandGeneral the stock default solver can return wrong "
            "positive eigenvalues on an indefinite stiffness. Nothing was "
            "written.") from exc
    eigenvalues = np.asarray(solved, dtype=np.float64).ravel()
    if eigenvalues.size != n_modes:
        raise ValueError(
            f"{_WHERE}: ops.eigen({call}) returned {eigenvalues.size} "
            "eigenvalue(s); nothing was written.")
    bad = [(k, float(lam)) for k, lam in enumerate(eigenvalues, start=1)
           if not (np.isfinite(lam) and lam > 0.0)]
    if bad:
        k, lam = bad[0]
        raise ValueError(
            f"{_WHERE}: mode {k} has a non-positive or non-finite eigenvalue "
            f"({lam:.6g}) "
            f"(modes {[m for m, _ in bad][:10]}): a rigid-body mechanism or "
            "an unconverged eigensolve, with no frequency to write. Restrain "
            "the mechanism or check the solver; nothing was written.")

    # One nodeEigenvector call per node and mode; the vector of a node is
    # as long as its live ndf.
    vectors = [
        [np.asarray(ops.nodeEigenvector(int(nid), k), dtype=np.float64).ravel()
         for nid in node_ids]
        for k in range(1, n_modes + 1)
    ]
    short = [(int(n), int(v.size), ndf_of[int(n)])
             for n, v in zip(node_ids, vectors[0]) if v.size != ndf_of[int(n)]]
    if short:
        n, size, ndf = short[0]
        raise ValueError(
            f"{_WHERE}: node {n}'s eigenvector has {size} entries but its "
            f"live ndf is {ndf} ({len(short)} node(s)); the DOFs cannot be "
            "laid out. Nothing was written.")

    layouts, rotations, unresolved = _live_slots(facts, node_ids, ndf_of, ndm)
    if unresolved:
        _refuse_embedded(facts, unresolved)
    spatial, rotation_slots = _native_fields(vectors, layouts, rotations, ndm)
    _warn_unresolved(_WHERE, unresolved, 3)
    info = backend_info_of(ops)
    writer = NativeWriter(out_path)
    writer.open(
        fem=fem,
        source_type="eigen",
        source_path="<openseespy>",
        opensees_backend=info.kind,
        opensees_build=info.build,
    )
    try:
        _write_mode_stages(
            writer, eigenvalues, node_ids, lambda k: spatial[k - 1],
            n_trans=min(3, ndm), rotation_slots=rotation_slots,
        )
    finally:
        writer.close()
    return out_path


def _refuse_embedded(
    facts: LiveDomainFacts, unresolved: "list[tuple[int, str]]",
) -> None:
    """Raise when an unsettled rotation slot is at a node an
    ``ASDEmbeddedNodeElement`` touches (#2407): there it may hold a
    rotation the element ties to the host (``-rot``), which a 0.0 would
    replace, or a pore pressure (``-p``), and the live domain does not say
    which flags the element was given."""
    embedded = facts.embedded_nodes()
    hit = [(n, why) for n, why in unresolved if n in embedded]
    if not hit:
        return
    node, why = hit[0]
    raise ValueError(
        f"{_WHERE}: the rotation slots of {len(hit)} node(s) an "
        f"ASDEmbeddedNodeElement touches ({[n for n, _ in hit[:10]]}) are "
        f"not settled by their elements: for node {node}, {why}. That "
        "element may tie the slot to the host's rotation (-rot) or to a "
        "pore pressure (-p), and the live domain does not say which, so "
        "the capture refuses rather than write 0.0 there. Nothing was "
        "written.")
