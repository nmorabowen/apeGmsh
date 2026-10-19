"""LiveDefault — the default results route for an in-process run (#1950).

:meth:`ResolvedRecorderSpec.emit_default` returns one. At ``__enter__`` it
asks the bound OpenSees module what it is
(:func:`~apeGmsh.opensees._target.backend_info_of`) and picks a route:

==============  ==========================  =================================
Build           Records                     :meth:`LiveDefault.results`
==============  ==========================  =================================
fork            the fork's own results      stamps ``INFO/PROVENANCE`` with
                file, ``<stem>.ladruno``    ``model_h5``'s pairing and the
                                            build that solved it (a refusal
                                            warns and is skipped), then opens
                                            it with ``Results.from_ladruno``
stock with      ``<stem>.mpco``             ``Results.from_mpco``, after one
MPCO                                        :class:`DefaultRouteWarning`
stock without   classic recorder files in   ``Results.from_recorders``, after
MPCO (pip)      ``<stem>_out/``             one :class:`DefaultRouteWarning`
==============  ==========================  =================================

Why the stock MPCO run is not converted: a program with no ``ops.stage``
block would have to map to one stage whose kind is a fact, and in process
it never is. The loop inside the ``with`` block drives the module directly
(``opspy.loadConst``, ``opspy.analysis("Transient")``), those calls never
reach the model archive, and MPCO opens a new stage on a domain change
only. :func:`read_deck_mpco` converts the ``.mpco`` of the bridge's own
exported deck when the program is staged (each stage records its kind);
an unstaged deck's kind is not archived either, so it reads the ``.mpco``
after one warning (coordinator ruling (iii) on #1950).

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

import h5py

from ...opensees._recorder_translate import LiveDomainFacts, live_ndm
from ...opensees._target import backend_info_of
from ...opensees.emitter.live import get_ops
from ...opensees.opensees_model import OpenSeesModel
from ..readers._ladruno_from_mpco import StageKindRefused
from ..readers._ladruno_live import (
    FORK_SUFFIX,
    convert_and_open,
    existing_fork_files,
    fork_recorder_args,
    fork_target_of,
    open_fork_results,
    stamp_fork_file,
)
from ..Results import Results
from ..spec._emit import mpco_ops_args
from ._recorders import LiveRecorders

if TYPE_CHECKING:
    from ..spec._resolved import ResolvedRecorderSpec

__all__ = ["DefaultRouteWarning", "LiveDefault", "read_deck_mpco"]

_MPCO_SUFFIX = ".mpco"
#: The one stage the classic-recorder route opens over the whole block.
_STAGE = "analysis"
#: Categories the classic-recorder route does not record (``LiveRecorders``
#: refuses layers by name; fibres are recorded), and the prefix of a
#: per-record skip warning.
_UNRECORDED = ("layers",)
_SKIP_PREFIX = "LiveRecorders: skipping record"
#: Route names, as :attr:`LiveDefault.route` reports them.
ROUTES = ("fork", "mpco", "recorders")


class DefaultRouteWarning(UserWarning):
    """The default route recorded less than the fork's file, or skipped a step.

    Raised once per run, with the reason: the build has no MPCO recorder
    (the classic-recorder fallback), the run is a stock in-process MPCO run
    (read with ``from_mpco``, not converted), a deck's ``.mpco`` refused
    conversion (read with ``from_mpco``), or the fork's file refused its
    provenance stamp (read unstamped).
    """


def read_deck_mpco(
    path: "str | Path",
    *,
    model_h5: "str | Path",
    overwrite: bool = False,
) -> Results:
    """Read the ``.mpco`` of a deck the bridge emitted, on the default route.

    ``path`` is the ``.mpco`` written by running the bridge's own exported
    deck (``ops.py`` / ``ops.tcl``) out of process, as it was emitted, and
    ``model_h5`` the archive of that bridge. An in-process run (``ops.run()``
    then your own ``opspy.analyze`` loop) is not a deck: read its ``.mpco``
    with :meth:`Results.from_mpco`, or record it with
    :meth:`ResolvedRecorderSpec.emit_default`.

    The ``.mpco`` (or any part of a partitioned set) converts to the fork's
    format beside it, which is then opened; the ``.mpco`` is always kept.
    A staged program converts: each ``ops.stage`` records its kind. An
    unstaged program does not (its kind is not archived as fact, and the
    archive does not record the deck's ``analyze`` call; #2210 will), so
    it refuses through ``convert_mpco``'s private
    ``_implicit_stage`` path, which only this function passes. A refusal
    about stage kinds
    (:class:`~apeGmsh.results.readers._ladruno_from_mpco.StageKindRefused`)
    reads the ``.mpco`` with :meth:`Results.from_mpco` after one
    :class:`DefaultRouteWarning` that quotes it. Every other refusal (the
    archive is another model's, an incomplete partition set, a staging
    directory in the way) raises. A converted file from an earlier run
    refuses unless ``overwrite=True``.
    """
    mpco, model = Path(path), Path(model_h5)
    existing = existing_fork_files(fork_target_of(mpco))
    if existing and not overwrite:
        raise FileExistsError(
            f"read_deck_mpco would overwrite {[str(p) for p in existing]} "
            "from an earlier run; pass overwrite=True to replace them."
        )
    try:
        return convert_and_open(mpco, model, overwrite=overwrite)
    except StageKindRefused as e:
        warnings.warn(DefaultRouteWarning(
            f"read_deck_mpco: {mpco} was not converted and is read with "
            f"Results.from_mpco: {e}"), stacklevel=2)
    return Results.from_mpco(str(mpco), model_h5=str(model))


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
        The model archive the live domain was built from: the bridge's
        (``ops.h5``) or a FEMData archive (``fem.to_h5``). Required: every
        route's read pairs with it. A FEMData archive declares no ndm and
        archives no per-node ndf or element classes, so the
        classic-recorder route reads them from the live domain
        (``getNDM``, ``getNDF``, ``eleType``; #2251) and records node
        records only: it pairs no FEM element id with an OpenSees tag.
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
        #: The classic-recorder route over a FEMData archive (no
        #: ``/opensees`` zone): read with no bridge zone embedded.
        self._neutral = False
        self._build = "unknown"

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
        info = backend_info_of(self._ops)
        if info.kind == "fork":
            # The build of the module this run solves on: what
            # get_backend_build() answers for the bridge's own module, and
            # still right when the caller passes another one.
            self._build = info.build or "unknown"
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
                    "read_deck_mpco converts the .mpco of the bridge's own "
                    "staged deck run out of process."), stacklevel=3)
            except BaseException:
                self._remove(int(tag))
                self._tag = None
                raise
            return
        if target.exists() and not existed:
            target.unlink()
        out = target.parent / f"{target.stem}_out"
        skipped = [r.name for r in self._spec.records
                   if r.category in _UNRECORDED]
        dropped = (f" Not recorded on this route: {skipped}."
                   if skipped else "")
        warnings.warn(DefaultRouteWarning(
            f"emit_default: this OpenSees build has no MPCO recorder "
            f"({reason}); recording classic recorder files in {out} instead, "
            f"one stage over the whole block.{dropped}"), stacklevel=3)
        self._model = OpenSeesModel.from_h5(self._model_h5)
        deck_model: Any = self._model
        if not _has_bridge_zone(self._model_h5):
            # A FEMData archive (#2251): ndm, per-node ndf and element
            # classes are read from the live domain. With no ops.model
            # declared there is nothing to read, and the archive's own
            # refusal ("declares no ndm") stands.
            self._neutral = True
            _refuse_element_records(self._spec, self._model_h5)
            if live_ndm(self._ops) >= 1:
                deck_model = LiveDomainFacts(self._ops, self._model)
        live = LiveRecorders(self._spec, out, ops=self._ops, model=deck_model)
        live.__enter__()
        self._live = live
        try:
            # The classic-recorder route's own default kind, as
            # emit_recorders and from_recorders already use it; the
            # program does not state the kind of an in-process block.
            _begin_quietly(live)
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

    def results(self, *, cache_root: "str | Path | None" = None) -> Results:
        """Open what the run recorded, on the route ``__enter__`` picked.

        ``cache_root`` is where the classic-recorder route caches its
        transcoded file (:meth:`Results.from_recorders`); the other routes
        read their file in place and ignore it.
        """
        if not self._exited:
            raise RuntimeError(
                "LiveDefault.results() reads the run after the with block "
                "has closed; call it outside the block."
            )
        target = self._target
        assert target is not None
        if self._route == "fork":
            refused = stamp_fork_file(target, self._model_h5, self._build)
            if refused is not None:
                warnings.warn(DefaultRouteWarning(
                    f"emit_default: {target} was left unstamped and is read "
                    f"without its provenance: {refused}"), stacklevel=2)
            return open_fork_results(target, self._model_h5)
        if self._route == "mpco":
            return Results.from_mpco(str(target), model_h5=str(self._model_h5))
        if self._route == "recorders":
            assert self._live is not None and self._model is not None
            if self._neutral:
                return _read_neutral_recorders(
                    self._spec, target, self._model, self._live.stages,
                    cache_root=cache_root,
                )
            return Results.from_recorders(
                self._spec, target, fem=self._model.fem,
                cache_root=cache_root, model=self._model,
                stages=self._live.stages,
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


def _begin_quietly(live: LiveRecorders) -> None:
    """``live.begin_stage`` without a per-record skip warning for an
    :data:`_UNRECORDED` category: the route's one :class:`DefaultRouteWarning` names those
    records already. Any other warning passes through unchanged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        live.begin_stage(_STAGE, kind="transient")
    for w in caught:
        text = str(w.message)
        if text.startswith(_SKIP_PREFIX) and any(
                f"category={c!r}" in text for c in _UNRECORDED):
            continue
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


#: Categories whose records name elements by FEM id: the emit translates
#: them through the model's ``(fem_eid, tag)`` pairing.
_ELEMENT_CATEGORIES = ("elements", "gauss", "line_stations", "fibers")


def _has_bridge_zone(model_h5: Path) -> bool:
    """``model_h5`` holds the bridge's ``/opensees`` zone (``ops.h5``), not
    only the neutral FEMData zones (``fem.to_h5``)."""
    with h5py.File(str(model_h5), "r") as f:
        return "opensees" in f


def _refuse_element_records(spec: "ResolvedRecorderSpec", model_h5: Path) -> None:
    """Refuse, by name, the records a FEMData archive cannot pair: it holds
    no ``(fem_eid, tag)`` pairing, and identity is never assumed."""
    named = [r.name for r in spec.records if r.category in _ELEMENT_CATEGORIES]
    if named:
        raise ValueError(
            f"emit_default: records {named} name elements by FEM id, and "
            f"{model_h5} is a FEMData archive (no /opensees zone), which "
            "pairs no FEM id with an OpenSees element tag. Record nodes "
            "only, or pass the bridge's model archive (ops.h5(...))."
        )


def _read_neutral_recorders(
    spec: "ResolvedRecorderSpec",
    out_dir: Path,
    model: OpenSeesModel,
    stages: "tuple[Any, ...]",
    *,
    cache_root: "str | Path | None",
) -> Results:
    """Read a FEMData-archive run's classic recorder files.

    The transcode embeds no ``/opensees`` zone (the archive has none, and
    :meth:`OpenSeesModel.to_h5` would refuse to write one), so the file
    holds ``/meta`` and ``/model`` only; it is cached like
    :meth:`Results.from_recorders` caches, keyed apart from every
    model-bearing transcode. The emit refused every rotation the read
    could not keep as written, so no read mask is owed.
    """
    from ..schema._versions import PARSER_VERSION
    from ..transcoders import RecorderTranscoder
    from ..writers import _cache

    fem = model.fem
    plan = RecorderTranscoder.stage_plan(stages)
    source_files = [
        f for *_, prefix in plan
        for f in _cache.list_source_files(spec, out_dir, stage_id=prefix)
    ]
    key = _cache.compute_cache_key(
        source_files, parser_version=PARSER_VERSION,
        fem_snapshot_id=fem.snapshot_id, model_token=None, stages=plan,
    )
    cached_h5, _ = _cache.cache_paths(_cache.resolve_cache_root(cache_root), key)
    if not cached_h5.exists():
        RecorderTranscoder(spec, out_dir, cached_h5, fem, stages=stages).run()
    return Results.from_native(cached_h5, fem=fem, model=model)


def _is_tag(tag: object) -> bool:
    return isinstance(tag, int) and not isinstance(tag, bool) and tag >= 0
