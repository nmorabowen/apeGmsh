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
A spec with a Truss ``axial_force`` gauss component is not probed: MPCO
writes that bucket with no ``GAUSS_IDS``, so it takes the classic recorders,
which record it (#2290). The record's class (or, when it names none, every
live ``eleType`` of its elements) must be one the catalog decodes
``axial_force`` for, the Truss family. Its other records must still pass
every MPCO refusal (:func:`~apeGmsh.results.spec._emit.collect_mpco_tokens`),
as on any stock run: only the Truss axial force is rerouted.

``model_h5`` is the bridge's archive or a FEMData archive (``fem.to_h5``,
no ``/opensees`` zone). On the classic-recorder route a FEMData archive's
missing facts (ndm, per-node ndf, element classes) are read from the live
domain (:class:`~apeGmsh.opensees._recorder_translate.LiveDomainFacts`,
#2251), and the read embeds no ``/opensees`` zone. Element records
(Gauss points, line stations, nodal forces) pair each FEM element with
the live element of its id, proved by ``eleNodes`` against the FEM
connectivity and typed by ``eleType`` (#2269); the read carries that
pairing, per-Gauss-point material records included (#2290). A FEMData
archive holds no ``beamIntegration``, so a fibres record's section
locations and fibre counts are read from the live domain at
``__enter__`` (``sectionLocation``, ``fiberData2``), and its read slices
the files by them (#2290).

Modal records have no default route and refuse at ``__enter__``.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import h5py

from ...opensees._recorder_translate import LiveDomainFacts, live_ndm
from ...opensees._response_catalog import (
    RESPONSE_CATALOG,
    catalog_token_for_keyword,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    is_nodal_force_catalogued,
)
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
from ..spec._emit import collect_mpco_tokens, element_tag_map, mpco_ops_args
from ._recorders import LiveRecorders

if TYPE_CHECKING:
    from ..spec._resolved import ResolvedRecorderRecord, ResolvedRecorderSpec

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
        (``getNDM``, ``getNDF``, ``eleType``; #2251). It pairs no FEM
        element id with an OpenSees tag either, so an element record pairs
        each FEM element with the live element of its own id only when
        that element's ``eleNodes`` is the FEM connectivity in order and
        its class is known; any other id refuses, naming it (#2269).
        A per-Gauss-point material record reads through that pairing
        (#2290), and so does a ``fibers`` record: its elements' section
        locations and fibre counts are read from the live domain at
        ``__enter__``, in place of a ``beamIntegration``.
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
        #: The live facts of that route, with the element pairing proved
        #: at ``__enter__``; the read carries the pairing.
        self._facts: Optional[LiveDomainFacts] = None
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

    def _live_classes(self, element_ids: Any) -> "set[str] | None":
        """The live ``eleType`` of the FEM elements ``element_ids``; their
        tags are the bridge archive's pairing, or the FEM ids over a
        FEMData archive (proved by ``eleNodes`` later). ``None`` when one
        has no live element."""
        tags = (element_tag_map(OpenSeesModel.from_h5(self._model_h5),
                                fem_snapshot_id=self._spec.fem_snapshot_id)
                if _has_bridge_zone(self._model_h5) else None)
        live = {int(t) for t in self._ops.getEleTags()}
        classes: "set[str]" = set()
        for e in (int(v) for v in element_ids):
            tag = e if tags is None else tags.get(e)
            if tag is None or tag not in live:
                return None
            classes.add(str(self._ops.eleType(tag)))
        return classes

    def _enter_stock(self) -> None:
        target = self._path.with_suffix(_MPCO_SUFFIX)
        existed = target.exists()
        out_dir = "" if str(target.parent) == "." else str(target.parent)
        lead, reason, tag = "this OpenSees build has no MPCO recorder", "", None
        kept, rerouted = _split_truss_axial(self._spec.records,
                                            self._live_classes)
        if rerouted:
            # Every other MPCO refusal stands; no probe, so no file, and
            # no MPCO line-stations warning for a recorder never made.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"MPCO line-stations record ")
                collect_mpco_tokens(kept)
            lead = "MPCO cannot write the Truss axial force"
            reason = (f"records {rerouted} record its axialForce bucket "
                      "with no GAUSS_IDS, which the MPCO reader refuses")
        else:
            args = mpco_ops_args(
                self._spec.records, output_dir=out_dir, filename=target.name,
            )
            try:
                tag = self._ops.recorder(*args)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
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
            f"emit_default: {lead} ({reason}); recording classic recorder "
            f"files in {out} instead, "
            f"one stage over the whole block.{dropped}"), stacklevel=3)
        self._model = OpenSeesModel.from_h5(self._model_h5)
        deck_model: Any = self._model
        if not _has_bridge_zone(self._model_h5):
            # A FEMData archive: ndm, per-node ndf and element
            # classes are read from the live domain. With no ops.model
            # declared there is nothing to read, and the archive's own
            # refusal ("declares no ndm") stands.
            self._neutral = True
            if live_ndm(self._ops) >= 1:
                self._facts = _pair_live_elements(
                    self._spec, LiveDomainFacts(self._ops, self._model),
                    self._model_h5)
                deck_model = self._facts
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
                    facts=self._facts, cache_root=cache_root,
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


#: The classes the catalog decodes the gauss ``axial_force`` for: the
#: Truss family.
_TRUSS_AXIAL_CLASSES = frozenset(
    c for (c, _rule, token) in RESPONSE_CATALOG
    if token == catalog_token_for_keyword("axialForce"))


def _split_truss_axial(
    records: "tuple[ResolvedRecorderRecord, ...]",
    live_classes: "Callable[[Any], set[str] | None]",
) -> "tuple[list[ResolvedRecorderRecord], dict[str, list[str]]]":
    """``(kept, rerouted)``: the records without their Truss axial-force
    gauss components (``axialForce``, which MPCO writes with no
    ``GAUSS_IDS``), a record left with none dropped; and ``{record name:
    those components}``. Only these move a stock run off MPCO.

    A record's ``axialForce`` components move only when its class is in
    the Truss family (:data:`_TRUSS_AXIAL_CLASSES`): the record's own, or,
    when it names none, every live class of its elements
    (``live_classes``). Any other stays, and MPCO refuses it by name."""
    kept: "list[ResolvedRecorderRecord]" = []
    rerouted: "dict[str, list[str]]" = {}
    for r in records:
        axial = [c for c in r.components if r.category == "gauss"
                 and gauss_keyword_for_canonical(c) == "axialForce"]
        if axial:
            classes = ({r.element_class_name} if r.element_class_name
                       is not None else live_classes(
                           r.element_ids if r.element_ids is not None else ()))
            if not classes or not classes <= _TRUSS_AXIAL_CLASSES:
                axial = []
        if axial:
            rerouted[r.name] = axial
        comps = tuple(c for c in r.components if c not in axial)
        if comps:
            kept.append(dataclasses.replace(r, components=comps)
                        if axial else r)
    return kept, rerouted


def _pair_live_elements(
    spec: "ResolvedRecorderSpec", facts: LiveDomainFacts, model_h5: Path,
) -> LiveDomainFacts:
    """Pair every FEM element an element record names with the live
    element of its own id, by fact (:meth:`LiveDomainFacts.pair_elements`),
    and check each record's class against the live classes.

    A FEMData archive pairs no FEM id with an OpenSees tag, and identity
    is never assumed: each element is proved to be the live one, or the
    run refuses, naming it. A record that names its element class (the
    response catalog's key on read) must name the live ``eleType`` of
    every element it records. A ``fibers`` record's elements then have
    their section locations and fibre counts read from the live domain
    (:meth:`LiveDomainFacts.record_fiber_sections`): a FEMData archive
    holds no ``beamIntegration`` to count them from.
    """
    records = [r for r in spec.records if r.category in _ELEMENT_CATEGORIES]
    if not records:
        return facts
    where = (f"emit_default: records {[r.name for r in records]} name "
             f"elements by FEM id, and {model_h5} is a FEMData archive (no "
             "/opensees zone), so each element is paired from the live "
             "domain")
    facts.pair_elements(
        (int(e) for r in records for e in (r.element_ids if r.element_ids
                                           is not None else ())),
        where=where)
    for r in records:
        cls = r.element_class_name
        if cls is None or r.element_ids is None:
            continue
        other = sorted({(facts.live_class(int(e)), int(e)) for e in r.element_ids
                        if facts.live_class(int(e)) != cls})
        if other:
            raise ValueError(
                f"emit_default: record {r.name!r} names element class "
                f"{cls!r}, but the live domain has FEM elements "
                + ", ".join(f"{e} as {c!r}" for c, e in other[:10])
                + ". Its results would be decoded as another class's; name "
                "the live class, or split the record by class.")
    for r in records:
        if r.category == "elements":
            _refuse_uncatalogued_nodal_forces(r, facts)
        elif r.category == "fibers":
            # No ids: the emit records nothing, as the pairing pairs none.
            facts.record_fiber_sections(
                (int(e) for e in (r.element_ids if r.element_ids is not None
                                  else ())),
                where=f"emit_default: fibres record {r.name!r}")
    return facts


def _refuse_uncatalogued_nodal_forces(
    rec: "ResolvedRecorderRecord", facts: LiveDomainFacts,
) -> None:
    """Refuse, before the run, an ``elements`` record the read could not
    decode: one with no ``nodal_resisting_*`` component (the read decodes
    nodal forces only), one that names no class (a ``.out`` file carries
    none), or whose elements' live class has no nodal-force layout for its
    frame."""
    tokens = sorted({routing[1] for c in rec.components if (
        routing := gauss_routing_for_canonical(c, topology="nodal_forces"))})
    if not tokens:
        raise ValueError(
            f"emit_default: elements record {rec.name!r} has no "
            f"nodal_resisting_* component ({list(rec.components)}), and an "
            "elements record is read as nodal forces only, so nothing it "
            "records could be decoded. Record nodal_resisting_* components, "
            "or the response as another category.")
    if rec.element_class_name is None:
        raise ValueError(
            f"emit_default: elements record {rec.name!r} names no element "
            "class, and a .out file carries none, so its nodal forces "
            "cannot be decoded. Name the live element class.")
    classes = sorted({facts.live_class(int(e)) for e in (
        rec.element_ids if rec.element_ids is not None else ())})
    missing = [(cls, t) for cls in classes for t in tokens
               if not is_nodal_force_catalogued(cls, t)]
    if missing:
        raise ValueError(
            f"emit_default: elements record {rec.name!r} records nodal "
            f"forces of live classes with no nodal-force layout {missing}, "
            "so the read could not decode them. Record a catalogued class, "
            "or its response as another category.")


def _read_neutral_recorders(
    spec: "ResolvedRecorderSpec",
    out_dir: Path,
    model: OpenSeesModel,
    stages: "tuple[Any, ...]",
    *,
    facts: Optional[LiveDomainFacts],
    cache_root: "str | Path | None",
) -> Results:
    """Read a FEMData-archive run's classic recorder files.

    The transcode embeds no ``/opensees`` zone (the archive has none, and
    :meth:`OpenSeesModel.to_h5` would refuse to write one), so the file
    holds ``/meta`` and ``/model`` only. The element pairing the run
    proved (``facts``) is carried into the transcode, so element rows are
    stamped as FEM ids and an elastic beam's ``localForce`` stations are
    checked by its live type token. The file is cached like
    :meth:`Results.from_recorders` caches, keyed by that pairing, apart
    from every model-bearing transcode. The emit refused every rotation
    the read could not keep as written, so no read mask is owed.
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
        fem_snapshot_id=fem.snapshot_id,
        model_token=None if facts is None else _cache.model_cache_token(facts),
        stages=plan,
    )
    cached_h5, _ = _cache.cache_paths(_cache.resolve_cache_root(cache_root), key)
    if not cached_h5.exists():
        RecorderTranscoder(spec, out_dir, cached_h5, fem, stages=stages,
                           pairing=facts).run()
    return Results.from_native(cached_h5, fem=fem, model=model)


def _is_tag(tag: object) -> bool:
    return isinstance(tag, int) and not isinstance(tag, bool) and tag >= 0
