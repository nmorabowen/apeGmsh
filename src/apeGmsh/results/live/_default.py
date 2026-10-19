"""LiveDefault — the default results route for an in-process run (#1950).

:meth:`ResolvedRecorderSpec.emit_default` returns one. At ``__enter__`` it
asks the bound OpenSees module what it is
(:func:`~apeGmsh.opensees._target.backend_info_of`) and picks a route:

==============  ==========================  =================================
Build           Records                     :meth:`LiveDefault.results`
==============  ==========================  =================================
fork            the fork's own results      stamps ``INFO/PROVENANCE`` with
                file, ``<stem>.ladruno``    ``model_h5``'s pairing (a refusal
                                            warns and is skipped), then opens
                                            it with ``Results.from_ladruno``
stock with      ``<stem>.mpco``             ``Results.from_mpco``, after one
MPCO                                        :class:`DefaultRouteWarning`
stock without   classic recorder files in   ``Results.from_recorders``, after
MPCO (pip)      ``<stem>_out/``             one :class:`DefaultRouteWarning`
==============  ==========================  =================================

Why the stock MPCO run is not converted: the conversion maps a program with
no ``ops.stage`` block to one implicit stage only when its kind is a fact,
and in process it never is. The loop inside the ``with`` block drives the
module directly (``opspy.loadConst``, ``opspy.analysis("Transient")``),
those calls never reach the model archive, and MPCO opens a new stage on a
domain change only. A deck the bridge emitted is the program that ran, so
:func:`~apeGmsh.results.convert_mpco` converts its ``.mpco`` instead.

The pip route probes the MPCO recorder first: the probe call is the real
recorder when it works, and it works when it returns a recorder tag. When
it raises or returns no tag, no recorder was made, a file the probe created
is deleted, and the classic recorders open one stage over the whole block.

Modal records have no default route and refuse at ``__enter__``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ...mesh.FEMData import FEMData
from ...opensees._target import backend_info_of
from ...opensees.emitter.live import get_ops
from ...opensees.opensees_model import OpenSeesModel
from ..readers._ladruno_live import (
    FORK_SUFFIX,
    existing_fork_files,
    fork_recorder_args,
    open_fork_results,
    stamp_fork_file,
)
from ..Results import Results
from ..spec._emit import mpco_ops_args
from ._recorders import LiveRecorders

if TYPE_CHECKING:
    from ..spec._resolved import ResolvedRecorderSpec

__all__ = ["DefaultRouteWarning", "LiveDefault"]

_MPCO_SUFFIX = ".mpco"
#: The one stage the classic-recorder route opens over the whole block.
_STAGE = "analysis"
#: Route names, as :attr:`LiveDefault.route` reports them.
ROUTES = ("fork", "mpco", "recorders")


class DefaultRouteWarning(UserWarning):
    """The default route recorded less than the fork's file, or skipped a step.

    Raised once per run, with the reason: the build has no MPCO recorder
    (the classic-recorder fallback), the run is a stock in-process MPCO run
    (read with ``from_mpco``, not converted), or the fork's file refused its
    provenance stamp (read unstamped).
    """


class LiveDefault:
    """Context manager that records one in-process run on the default route.

    Parameters
    ----------
    spec
        The :class:`ResolvedRecorderSpec` whose records to emit.
    path
        Names the run: the route replaces its suffix (``""``, ``.mpco`` or
        the fork's suffix), so ``"run"`` records ``run.mpco`` on a stock
        build.
    model_h5
        The bridge-written model archive (``ops.h5``) the live domain was
        built from. Required: every route's read pairs with it.
    ops
        The OpenSees module, or a stand-in for testing. Defaults to the
        module the bridge drives.
    overwrite
        Replace the fork's file of an earlier run. Without it a rerun onto
        an existing file refuses at ``__enter__``.
    """

    def __init__(
        self,
        spec: "ResolvedRecorderSpec",
        path: "str | Path",
        *,
        model_h5: "str | Path",
        ops: Any = None,
        overwrite: bool = False,
    ) -> None:
        self._spec = spec
        self._path = Path(path)
        self._model_h5 = Path(model_h5)
        self._ops = ops
        self._overwrite = overwrite
        self._opened = False
        self._exited = False
        self._route: Optional[str] = None
        self._target: Optional[Path] = None
        self._tag: Optional[int] = None
        self._live: Optional[LiveRecorders] = None
        self._model: Optional[OpenSeesModel] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "LiveDefault":
        if self._opened:
            raise RuntimeError("LiveDefault is single-use; create a new instance.")
        self._opened = True
        modal = [r.name for r in self._spec.records if r.category == "modal"]
        if modal:
            raise ValueError(
                f"emit_default has no route for modal records {modal}; "
                "capture modes with spec.emit_mpco(...) or "
                "ops.domain_capture(...) instead."
            )
        if self._path.suffix not in ("", _MPCO_SUFFIX, FORK_SUFFIX):
            raise ValueError(
                f"emit_default path {str(self._path)!r} has suffix "
                f"{self._path.suffix!r}; the route sets the suffix, so pass "
                f"a name with none, {_MPCO_SUFFIX!r} or {FORK_SUFFIX!r}."
            )
        if not self._model_h5.is_file():
            raise FileNotFoundError(
                f"emit_default needs the model archive the run was built "
                f"from (ops.h5); no such file: {self._model_h5}"
            )
        if self._ops is None:
            self._ops = get_ops()
        if self._path.parent and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        if backend_info_of(self._ops).kind == "fork":
            self._enter_fork()
        else:
            self._enter_stock()
        return self

    def _enter_fork(self) -> None:
        target = self._path.with_suffix(FORK_SUFFIX)
        existing = existing_fork_files(target)
        if existing and not self._overwrite:
            raise FileExistsError(
                f"emit_default would overwrite {[str(p) for p in existing]} "
                "from an earlier run; pass overwrite=True to replace them."
            )
        for old in existing:
            old.unlink()
        args = fork_recorder_args(self._spec.records, target)
        try:
            tag = self._ops.recorder(*args)
        except Exception as exc:
            raise RuntimeError(
                f"the fork build refused its own results recorder "
                f"(ops.recorder{args!r}): {type(exc).__name__}: {exc}"
            ) from exc
        if not _is_tag(tag):
            raise RuntimeError(
                f"the fork build's results recorder returned {tag!r}, not a "
                "recorder tag; it cannot be removed, so the run would "
                "never flush."
            )
        self._route, self._target, self._tag = "fork", target, int(tag)

    def _enter_stock(self) -> None:
        target = self._path.with_suffix(_MPCO_SUFFIX)
        existed = target.exists()
        out_dir = "" if str(target.parent) == "." else str(target.parent)
        args = mpco_ops_args(
            self._spec.records, output_dir=out_dir, filename=target.name,
        )
        reason = ""
        try:
            tag = self._ops.recorder(*args)
        except Exception as exc:
            tag, reason = None, f"{type(exc).__name__}: {exc}"
        if tag is not None and not _is_tag(tag):
            # No recorder tag, so no recorder to remove: the call failed.
            reason = f"ops.recorder('mpco', ...) returned {tag!r}, not a tag"
            tag = None
        if tag is not None:
            self._route, self._target, self._tag = "mpco", target, int(tag)
            try:
                warnings.warn(DefaultRouteWarning(
                    f"emit_default: a stock in-process run is recorded to "
                    f"{target} and read with Results.from_mpco, not "
                    "converted: its analysis kind is not a fact (the loop's "
                    "own analysis calls never reach the model archive). "
                    "convert_mpco converts the .mpco of a deck the bridge "
                    "emitted."), stacklevel=3)
            except BaseException:
                self._remove(int(tag))
                self._tag = None
                raise
            return
        if target.exists() and not existed:
            target.unlink()
        out = target.parent / f"{target.stem}_out"
        warnings.warn(DefaultRouteWarning(
            f"emit_default: this OpenSees build has no MPCO recorder "
            f"({reason}); recording classic recorder files in {out} instead "
            "(one stage over the whole block; fiber and layer records are "
            "not recorded on this route)."), stacklevel=3)
        self._model = OpenSeesModel.from_h5(self._model_h5)
        live = LiveRecorders(self._spec, out, ops=self._ops, model=self._model)
        live.__enter__()
        self._live = live
        try:
            # The classic-recorder route's own default kind, as
            # emit_recorders and from_recorders already use it; the
            # program does not state the kind of an in-process block.
            live.begin_stage(_STAGE, kind="transient")
        except BaseException:
            live.__exit__(None, None, None)
            self._live = None
            raise
        self._route, self._target = "recorders", out

    def __exit__(self, exc_type, exc, tb) -> None:
        # Removing the recorder flushes the file. Cleanup never raises, so
        # an exception in flight is never masked.
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
        elif self._tag is not None:
            self._remove(self._tag)
        self._exited = True

    def _remove(self, tag: int) -> None:
        try:
            self._ops.remove("recorder", tag)
        except Exception:  # noqa: BLE001 - best-effort flush
            warnings.warn(
                f"emit_default: failed to remove recorder tag {tag}; the "
                "output may not be flushed.", stacklevel=3)

    # ------------------------------------------------------------------
    # Read back
    # ------------------------------------------------------------------

    def results(self) -> Results:
        """Open what the run recorded, on the route ``__enter__`` picked."""
        if not self._exited:
            raise RuntimeError(
                "LiveDefault.results() reads the run after the with block "
                "has closed; call it outside the block."
            )
        target = self._target
        assert target is not None
        if self._route == "fork":
            refused = stamp_fork_file(target, self._model_h5)
            if refused is not None:
                warnings.warn(DefaultRouteWarning(
                    f"emit_default: {target} was left unstamped and is read "
                    f"without its provenance: {refused}"), stacklevel=2)
            return open_fork_results(target, self._model_h5)
        if self._route == "mpco":
            return Results.from_mpco(str(target), model_h5=str(self._model_h5))
        if self._route == "recorders":
            assert self._live is not None and self._model is not None
            return Results.from_recorders(
                self._spec, target, fem=FEMData.from_h5(self._model_h5),
                model=self._model, stages=self._live.stages,
            )
        raise RuntimeError(f"LiveDefault has no route {self._route!r}; "
                           f"expected one of {ROUTES}.")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def route(self) -> Optional[str]:
        """``"fork"``, ``"mpco"`` or ``"recorders"``; ``None`` before entry."""
        return self._route

    @property
    def path(self) -> Optional[Path]:
        """The file (or, on the recorder route, the directory) written."""
        return self._target

    @property
    def tag(self) -> Optional[int]:
        """The recorder tag of the fork or MPCO route."""
        return self._tag


def _is_tag(tag: object) -> bool:
    return isinstance(tag, int) and not isinstance(tag, bool) and tag >= 0
