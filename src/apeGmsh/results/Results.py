"""Results — backend-agnostic FEM post-processing container.

Mirrors the ``FEMData`` composite shape: ``results.nodes``,
``results.elements.gauss``, ``results.elements.fibers`` etc., with
the same ``pg=`` / ``label=`` / ``ids=`` selection vocabulary.

Construction
------------
::

    # Native HDF5 (apeGmsh-produced or domain capture).  Phase 8
    # (ADR 0020) — ``model=`` is required at every constructor.
    from apeGmsh.opensees import OpenSeesModel

    model = OpenSeesModel.from_h5("model.h5")  # or the same file as results
    results = Results.from_native("run.h5", model=model)

    # MPCO HDF5 — ``model_h5=`` is required (sibling model.h5 path).
    results = Results.from_mpco("run.mpco", model_h5="model.h5")

    # Recorder transcoder (Phase 6) — ``model=`` is required.
    results = Results.from_recorders(spec, output_dir="out/", fem=fem, model=model)

Stage scoping
-------------
Top-level ``Results`` carries all stages. Reads disambiguate
automatically when there is only one stage; with multiple stages,
pick one explicitly::

    gravity = results.stage("gravity")              # stage-scoped Results
    disp = gravity.nodes.get(component="displacement_z", pg="Top")

Modes
-----
Modes are stages with ``kind="mode"``. The ``.modes`` accessor is
sugar that filters and returns each as a stage-scoped ``Results``::

    for mode in results.modes:
        print(mode.mode_index, mode.frequency_hz, mode.period_s)
        shape = mode.nodes.get(component="displacement_z")

Bind contract
-------------
A results file embeds (native) or synthesizes (MPCO) a ``FEMData``
snapshot tagged with a ``snapshot_id``. Calling ``.bind(other_fem)``
swaps it in (useful when re-using session-side labels and Parts);
no hash validation is performed.

OpenSeesModel chain (ADR 0020 INV-1, Phase 8 — required)
--------------------------------------------------------
``Results`` carries an :class:`OpenSeesModel` broker handle exposed
via :attr:`Results.model`. The chain forward is

::

    results.model     -> OpenSeesModel
    results.model.fem -> FEMData         (the same neutral zone)
    results.lineage   -> Lineage         (fem_hash + model_hash chain)

Since the Phase 8 prune of the major architectural refactor, every
:class:`Results` constructor owns the chain — passing ``model=`` is
required, missing supply raises :class:`TypeError`. See ADR 0020
INV-1 for the structural-pairing contract this enforces.

Module-import discipline
------------------------
:class:`apeGmsh.opensees.opensees_model.OpenSeesModel` is referenced
under ``TYPE_CHECKING`` only and imported lazily inside the method
bodies that need it. This keeps the import-DAG polarity tested by
``tests/test_import_dag_polarity.py`` intact and avoids dragging
``apeGmsh.opensees`` into the ``apeGmsh.results`` eager graph.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from numpy import ndarray

from ._bind import (
    _bind_stage_names,
    _resolve_fem,
    _resolve_fem_via_model,
    resolve_bound_model,
)
from ._composites import (
    ElementResultsComposite,
    NodeResultsComposite,
)
from ._inspect import ResultsInspect
from .readers._protocol import EigenMode, ResultsReader, StageInfo

if TYPE_CHECKING:
    from apeGmsh.assess import AssessmentReport
    from ..mesh.FEMData import FEMData
    from ..opensees._internal.lineage import Lineage
    from ..opensees.opensees_model import OpenSeesModel
    from .plot import ResultsPlot


# Sentinel for ``Results._derive`` so we can distinguish
# "not passed" from "None" in the fem= / stage_id= overrides.
class _Sentinel:
    pass


_SENTINEL = _Sentinel()


_MODEL_REQUIRED_MESSAGE = (
    "model= is required. For a bridge-built model pass "
    "model=OpenSeesModel.from_h5('model.h5'). For a bare FEMData "
    "snapshot (e.g. get_fem_data()), the one-call route is "
    "Results.from_fem(fem, path)."
)


_MODEL_H5_REQUIRED_MESSAGE = (
    "model_h5= is required. For a model built through the apeSees "
    "bridge, write ops.h5('model.h5') and pass model_h5='model.h5': that "
    "archive carries the element tag map the bridge's dense renumbering "
    "needs. For a bare FEMData that drove a hand-written deck whose "
    "element tags are the fem element ids, use "
    "Results.from_fem(fem, path) or fem.to_h5('model.h5')."
)


#: The phrase a ``model hash mismatch`` lineage warning carries when one
#: side's ``model_hash`` hashes the declaration ids of
#: ``/opensees/program.decl`` and the other's does not (the
#: ``/opensees/decls@program_decl`` completeness marker). ``assess`` keys
#: its ``RES.LINEAGE`` message on it.
MODEL_HASH_ERA_CAUSE = "hash change (declaration ids entered model_hash)"


def _model_hash_era_cause(
    results_records: bool | None, model_records: bool,
) -> str | None:
    """Name the cause of a ``model_hash`` mismatch, or ``None``.

    ``results_records`` / ``model_records`` say whether each side's
    archive records its program declarations
    (:func:`h5_reader.records_program_decl`): only then does its
    ``model_hash`` include the declaration ids. The version stamp is not
    used, because a ``to_h5`` rewrite of an older archive is stamped
    current but keeps the old hash. Returns the cause text when the two
    sides differ; ``None`` when they agree, or when the results side is
    unknown (no ``/opensees`` zone): the cause is then never guessed.
    """
    from ..opensees.emitter.h5_reader import PROGRAM_DECL_FROM

    if results_records is None or results_records == model_records:
        return None
    boundary = ".".join(str(part) for part in PROGRAM_DECL_FROM)
    if model_records:
        sides = "the model's hash includes declaration ids, the results file's does not"
        remedy = (
            "re-run the analysis against the re-saved model, or open the "
            "results with the model.h5 they were run on"
        )
    else:
        sides = "the results file's hash includes declaration ids, the model's does not"
        remedy = (
            "open the results with the model.h5 they were run on, or "
            "re-run the analysis against this model"
        )
    return (
        f"likely cause: the opensees schema {boundary} {MODEL_HASH_ERA_CAUSE} "
        f"({sides}); {remedy}"
    )


#: The opensees schema step at which the side-channel calls' declaration
#: ids entered the hashed ``/opensees/program.decl`` column. The
#: ``program_decl`` marker reads ``"complete"`` on both sides of it, so
#: :func:`_model_hash_era_cause` cannot see it. This is the one step
#: named this way: a later ``model_hash`` change gets a new marker value,
#: never another version rule.
SIDE_CHANNEL_DECL_FROM: tuple[int, int, int] = (2, 32, 0)

#: The six verbs whose calls first carried a proved declaration at
#: :data:`SIDE_CHANNEL_DECL_FROM`: the content of that one step, so it
#: does not grow with the emitter.
SIDE_CHANNEL_DECL_VERBS: frozenset[str] = frozenset({
    "step_hook_ramp", "addToParameter", "flip_element_stage",
    "update_parameter", "set_node_vel", "set_node_accel"})

#: The phrase a ``model hash mismatch`` lineage warning carries when the
#: two sides differ across :data:`SIDE_CHANNEL_DECL_FROM`. ``assess``
#: keys its ``RES.LINEAGE`` message on it. It does not contain
#: :data:`MODEL_HASH_ERA_CAUSE`.
SIDE_CHANNEL_HASH_CAUSE = (
    "hash change (side-channel calls' declaration ids entered model_hash)"
)


def _side_channel_proof(runs: "Iterable[tuple[str, int]]") -> bool | None:
    """Whether an archive's side-channel calls carry a proved declaration.

    ``runs`` are the ``(method, decl)`` pairs of ``/opensees/program``.
    Returns ``None`` when no run is one of :data:`SIDE_CHANNEL_DECL_VERBS`,
    ``True`` when one of them names a declaration (``decl >= 0``), so the
    ids are in that archive's ``model_hash``, and ``False`` when every
    one is ``-1``: an emit below :data:`SIDE_CHANNEL_DECL_FROM`, or a
    ``to_h5`` rewrite of one, which echoes the old column.
    """
    proof: bool | None = None
    for method, decl in runs:
        if method in SIDE_CHANNEL_DECL_VERBS:
            if decl >= 0:
                return True
            proof = False
    return proof


def _side_channel_hash_cause(
    results_proof: bool | None, model_proof: bool | None,
) -> str | None:
    """Name the :data:`SIDE_CHANNEL_DECL_FROM` cause of a ``model_hash``
    mismatch, or ``None``.

    Called only when both sides record their program declarations (the
    ``program_decl`` markers agree at ``"complete"``). Each argument is
    :func:`_side_channel_proof` of one side's ``/opensees/program``. The
    cause is named when both sides carry side-channel calls and exactly
    one side's ``model_hash`` includes their declaration ids. It reads the
    hashed column, not the ``opensees_schema_version`` stamp: a ``to_h5``
    rewrite of an older archive is stamped current but keeps the old
    column and hash. Still a heuristic ("likely"): the two models may also
    differ in other ways.
    """
    if results_proof is None or model_proof is None:
        return None
    if results_proof == model_proof:
        return None
    boundary = ".".join(str(part) for part in SIDE_CHANNEL_DECL_FROM)
    if model_proof:
        remedy = (
            "re-run the analysis against the re-saved model, or open the "
            "results with the model.h5 they were run on"
        )
    else:
        remedy = (
            "open the results with the model.h5 they were run on, or "
            "re-run the analysis against this model"
        )
    return (
        f"likely cause: the opensees schema {boundary} "
        f"{SIDE_CHANNEL_HASH_CAUSE}; {remedy}"
    )


def _resolve_results_kind(kind: str, path: Any) -> str:
    """Resolve ``Results.from_fem``'s ``kind`` to a concrete reader.

    ``"auto"`` detects from the file suffix (``.mpco`` / ``.ladruno``);
    a native ``.h5`` is ambiguous and must be requested explicitly.  For
    a partition-list input the first path is peeked.
    """
    valid = {"auto", "mpco", "ladruno", "native"}
    if kind not in valid:
        raise ValueError(
            f"Results.from_fem: kind={kind!r} is not one of "
            f"{sorted(valid)}."
        )
    if kind != "auto":
        return kind
    first = path[0] if isinstance(path, (list, tuple)) else path
    suffix = Path(first).suffix.lower()
    if suffix == ".mpco":
        return "mpco"
    if suffix == ".ladruno":
        return "ladruno"
    raise ValueError(
        f"Results.from_fem: cannot auto-detect the result kind from "
        f"{str(first)!r} (suffix {suffix!r}). Pass kind='mpco' | "
        f"'ladruno' | 'native'."
    )


def _start_subprocess_monitor(handle: Any, args: list[str]) -> None:
    """Spawn a daemon thread that waits on ``handle`` and surfaces
    a non-zero exit code to stderr.

    Closes the silent-failure window the May 2026 audit identified:
    :meth:`Results._spawn_viewer_subprocess` returns a fire-and-forget
    :class:`subprocess.Popen`; without this monitor, a child failure
    (corrupt file, missing dependency, ``__main__.sys.exit(2)``) is
    invisible to the parent kernel.  PR1 fixed the specific
    MPCO-without-model-h5 path; this monitor catches the general case.

    The monitor uses :meth:`Popen.wait` (no busy-poll, no CPU cost) on
    a daemon thread (dies with the parent — no shutdown hook needed).
    Exit code 0 is silent; non-zero prints a single line with the
    code and the argv that was launched.  Crucially, the monitor
    never *raises* — surfacing-to-stderr is the right escalation
    level for a fire-and-forget IPC pattern.
    """
    import sys
    import threading

    def _wait_and_report() -> None:
        try:
            rc = handle.wait()
        except Exception as exc:
            print(
                f"[apeGmsh] viewer subprocess monitor failed: {exc!r}",
                file=sys.stderr, flush=True,
            )
            return
        if rc != 0:
            print(
                f"[apeGmsh] viewer subprocess exited with code {rc}; "
                f"argv={args!r}",
                file=sys.stderr, flush=True,
            )

    threading.Thread(
        target=_wait_and_report,
        daemon=True,
        name="apeGmsh-viewer-monitor",
    ).start()


def _in_notebook_kernel() -> bool:
    """True when running inside a Jupyter / IPython ZMQ kernel.

    The blocking Qt viewer freezes (and often kills) such a kernel, so
    :meth:`Results.viewer` defaults to the subprocess path there.
    The class-name probe is the standard idiom: ``ZMQInteractiveShell``
    is the notebook / lab / VS Code kernel, ``TerminalInteractiveShell``
    is plain ``ipython`` in a terminal — where blocking is fine and the
    subprocess hop would only cost ergonomics.
    """
    try:
        from IPython import get_ipython
    except Exception:
        return False
    shell = get_ipython()
    return type(shell).__name__ == "ZMQInteractiveShell"


class Results:
    """Top-level results object. Returned by ``Results.from_*`` constructors.

    Stage scoping
    -------------
    Instances may be unscoped (top-level — accesses any stage) or
    scoped to one stage (returned by ``.stage(name)``,
    ``.modes[i]``). Scoped instances expose stage metadata as
    properties (``.kind``, ``.time``, ``.n_steps``); mode-scoped
    instances additionally expose ``.eigenvalue``, ``.frequency_hz``,
    ``.period_s``, ``.mode_index``.
    """

    def __init__(
        self,
        reader: ResultsReader,
        *,
        fem: "Optional[FEMData]" = None,
        stage_id: Optional[str] = None,
        path: Optional[Path] = None,
        model: "OpenSeesModel",
        model_path: Optional[Path] = None,
    ) -> None:
        self._reader = reader
        self._fem = fem
        # ADR 0113 D9 — ``from_native`` sets this after construction
        # when no ``fem=`` was supplied and the file's embedded
        # ``/model`` is below its floor: the refusal text. ``/stages``
        # read; :attr:`fem` raises it until ``bind(fem)`` supplies a
        # snapshot. Not a constructor keyword: it stays off the public
        # ``Results`` signature.
        self._fem_unavailable: Optional[str] = None
        self._stage_id = stage_id
        self._path = path
        # ADR 0020 INV-1 (Phase 8 prune) — ``_model`` is required and
        # never None.  The three public constructors validate the
        # contract and raise :class:`TypeError` on missing supply;
        # internal callers (``_derive``) propagate the existing handle.
        self._model = model
        # Sibling-archive path for readers that carry no embedded
        # ``/opensees/`` zone (MPCO).  ``None`` when ``self._path``
        # already is the model archive (native Composed file).  The
        # subprocess viewer reads this to forward ``--model-h5`` to
        # the child process — without it, ``__main__.py`` exits(2)
        # on ``.mpco`` paths.
        self._model_path = model_path
        self._stages_cache: Optional[list[StageInfo]] = None
        self._units_override: Optional[str] = None
        self._source_units_override: Optional[str] = None
        self._base_fem: "Optional[FEMData]" = fem
        from ._units import resolve_results_units

        resolve_results_units(self)

        # Composites
        self.nodes = NodeResultsComposite(self)
        self.elements = ElementResultsComposite(self)
        self.inspect = ResultsInspect(self)
        self._plot: Optional["ResultsPlot"] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_native(
        cls,
        path: str | Path,
        *,
        fem: "Optional[FEMData]" = None,
        model: "Optional[OpenSeesModel]" = None,
        model_path: "Optional[str | Path]" = None,
    ) -> "Results":
        """Open an apeGmsh native HDF5 results file.

        Phase 8 (ADR 0020 INV-1) — ``model=`` is required. Missing
        supply raises :class:`TypeError`. Pass
        ``model=OpenSeesModel.from_h5(path_to_model_h5)`` (often the
        same path as ``path`` when the file is a Composed-file
        per ADR 0020).

        If ``fem`` is omitted, the embedded ``/model/`` snapshot is
        used as the bound FEMData.

        ADR 0113 D9 — a results file outlives its embedded model zones.
        When the embedded ``/model`` is below its floor the file still
        opens and ``/stages`` read: with ``fem=`` supplied the zone is
        never read; without it :attr:`fem` raises the reader's refusal
        (with a ``fem=`` hint) until :meth:`bind` supplies a snapshot.
        The embedded ``/opensees`` is never read here, since ``model=``
        is required; ``OpenSeesModel.from_h5`` on such a file refuses,
        so the model comes from a sidecar archive.

        ``model_path`` records the on-disk archive the ``model`` was read
        from, for when it is *not* ``path`` itself — e.g. results whose
        embedded ``/model`` zone is not independently readable. The
        non-blocking subprocess viewer forwards it as ``--model-h5`` so the
        child re-reads the model from there instead of from ``path``.
        """
        if model is None:
            raise TypeError(_MODEL_REQUIRED_MESSAGE)
        from ..opensees._internal.schema_version import NEUTRAL
        from .readers._native import NativeReader
        reader = NativeReader(path)
        fem_unavailable: Optional[str] = None
        if fem is None and NEUTRAL in reader.unavailable_zones:
            # D9: no snapshot to bind and the file's own is below its
            # floor. Open anyway; ``Results.fem`` refuses with the text.
            fem_unavailable = reader.unavailable_zones[NEUTRAL]
            bound_fem: "Optional[FEMData]" = None
        else:
            bound_fem = _resolve_fem(reader, fem)
        bound_model = resolve_bound_model(reader, model)
        # ``resolve_bound_model`` always returns ``model`` here since
        # we just asserted it is non-None, but route through the helper
        # to keep the resolution semantics in one place.
        assert bound_model is not None
        results = cls(
            reader, fem=bound_fem, path=Path(path), model=bound_model,
            model_path=Path(model_path) if model_path is not None else None,
        )
        results._fem_unavailable = fem_unavailable
        return results._with_autoloaded_definitions()

    @classmethod
    def from_recorders(
        cls,
        spec,
        output_dir: str | Path,
        *,
        fem: "FEMData",
        cache_root: str | Path | None = None,
        stage_name: str | None = None,
        stage_kind: str | None = None,
        file_format: str = "out",
        stage_id: str | None = None,
        model: "Optional[OpenSeesModel]" = None,
        stages: "Optional[Iterable[Any]]" = None,
    ) -> "Results":
        """Open the result of an OpenSees run driven by Tcl/Py recorders.

        ``model=`` is required (else :class:`TypeError`); its ``/opensees/``
        zone is embedded in the transcoded file, which
        :meth:`Results.from_native` then opens.

        Parses the ``.out`` / ``.xml`` files emitted at
        ``output_dir`` (matching what ``spec.emit_recorders(...)`` or
        the apeGmsh OpenSees bridge's Tcl/Py emit produced) into an
        apeGmsh native HDF5, caches the result at
        ``cache_root``, and opens it through ``NativeReader``.

        Caching: unchanged input files, model, stage name and stage kind
        reuse the cached HDF5 (``writers/_cache.py``). Recorders sample
        every step or every ``dt=``: sampling at arbitrary caller
        instants has no recorder equivalent.

        ``stage_id`` matches the per-stage filename prefix used by
        :meth:`ResolvedRecorderSpec.emit_recorders` together with
        ``begin_stage(stage_id, ...)``. When set, only files prefixed
        with ``<stage_id>__`` are read; ``stage_name`` defaults to
        ``stage_id``, else ``"analysis"``, and ``stage_kind`` to
        ``"transient"``. ``None`` (default) keeps the
        legacy flat-naming used by Tcl/Py exports. ``stages=live.stages``
        reads every stage of a :meth:`ResolvedRecorderSpec.emit_recorders`
        run into one file, one stage each with its name and kind; it
        excludes ``stage_id`` / ``stage_name`` / ``stage_kind``.

        Nodes, element forces, gauss points and line stations are
        transcoded; fiber and layer records are skipped (``.out`` cannot
        carry them; use MPCO) and modal records are not emitted. Rows
        are labelled with the spec's FEM ids, so the files must come from
        an emit that translated them to OpenSees tags
        (``spec.emit_recorders(..., model=)`` or
        ``spec.to_tcl_commands(model=)``).
        """
        if model is None:
            raise TypeError(_MODEL_REQUIRED_MESSAGE)
        from .schema._versions import PARSER_VERSION
        from .transcoders import RecorderTranscoder
        from .writers import _cache

        if fem is None:
            raise TypeError(
                "Results.from_recorders(...) requires fem= "
                "(the spec's snapshot_id must match)."
            )

        # When stage_id is provided and stage_name was left at its
        # default, mirror stage_id so the resulting Results stage is
        # named meaningfully (otherwise everything ends up as
        # "analysis" regardless of which stage the user loaded).
        if stage_id is not None and stage_name in (None, "analysis"):
            stage_name = stage_id
        stages = None if stages is None else tuple(stages)   # read a generator once
        plan = RecorderTranscoder.stage_plan(stages, stage_name, stage_kind, stage_id)

        out_dir = Path(output_dir)
        cache_dir = _cache.resolve_cache_root(cache_root)

        source_files = _cache.list_source_files(
            spec, out_dir, file_format=file_format, stage_id=stage_id,
        )
        if stages is not None:
            source_files = [f for *_, prefix in plan for f in _cache.list_source_files(
                spec, out_dir, file_format=file_format, stage_id=prefix)]
        key = _cache.compute_cache_key(
            source_files,
            parser_version=PARSER_VERSION, fem_snapshot_id=fem.snapshot_id,
            model_token=_cache.model_cache_token(model), stages=plan,
        )
        cached_h5, _ = _cache.cache_paths(cache_dir, key)

        if not cached_h5.exists():
            # The model's /opensees/ zone, written beside the target for
            # NativeWriter to copy in at open time (ADR 0020).
            model_h5_src: Path = cached_h5.with_suffix(".model.h5")
            model.to_h5(model_h5_src)
            RecorderTranscoder(
                spec, out_dir, cached_h5, fem, stage_name=stage_name,
                stage_kind=stage_kind, file_format=file_format, stage_id=stage_id,
                model_h5_src=model_h5_src, stages=stages,
            ).run()

        return cls.from_native(cached_h5, fem=fem, model=model)

    @classmethod
    def from_mpco(
        cls,
        path: "str | Path | list[str | Path]",
        *,
        fem: "Optional[FEMData]" = None,
        merge_partitions: bool = True,
        model_h5: "Optional[str | Path]" = None,
    ) -> "Results":
        """Open a STKO ``.mpco`` HDF5 results file.

        Phase 8 (ADR 0020 INV-1) — ``model_h5=`` is required. Missing
        supply raises :class:`TypeError`. The broker is loaded via
        :meth:`OpenSeesModel.from_h5` and attached to the resulting
        :class:`Results`; INV-3 — the broker is held *in memory only*
        (no derived ``results.h5`` is written copying the
        ``/opensees/`` zone in).

        What the archive carries, the results use (#1324, #1325):

        - **FEMData.** ``fem=`` wins when given. Otherwise the neutral
          FEMData stored in ``model_h5`` (physical groups, labels) is
          bound whenever its node ids cover the capture's and sit at
          the capture's coordinates (the model's ``ndm`` columns: a 2-D
          model on an offset plane compares in ``x, y``), so ``pg=``
          queries work from files alone. An archive written by
          ``fem.to_h5`` carries no element tag map, so it is bound
          only when the capture's element ids and nodes are its own
          (ops tags are read as fem element ids on that route); a ``model_h5`` from another model or another mesh of
          the same part warns (``ModelFemMismatchWarning``) and the
          partial FEMData synthesized from the MPCO ``MODEL/`` group is
          bound instead.
        - **Stage names.** ``ops.stage(name=...)`` names from
          ``/opensees/stages`` replace the file's ``MODEL_STAGE[<k>]``
          names in order (``results.stage("gravity")``);
          ``MODEL_STAGE[<k>]`` stays resolvable as an alias. A partial
          run (fewer capture stages than program stages) names the
          prefix and warns (``StageCountMismatchWarning``); more capture
          stages than program stages warns and keeps the file's names.
          Two program stages with one name warn
          (``DuplicateStageNameWarning``); ``stage(name)`` picks the
          first, and the ids ``stage_<k>`` stay unique.

        Single-file mode (default for non-partitioned analyses): pass
        the path of one ``.mpco`` file.

        Multi-partition mode (parallel OpenSees runs): pass either

        - a single ``<stem>.part-<N>.mpco`` path — siblings are
          discovered automatically by globbing ``<stem>.part-*.mpco``
          in the same directory and merged into one virtual reader;
        - an explicit list of partition paths.

        Boundary nodes deduplicate by ID (first-occurrence wins);
        elements concatenate (disjoint by partition); slabs stitch
        across partitions transparently. Stage and time vectors must
        match across partitions or construction raises.

        Pass ``merge_partitions=False`` to opt out of auto-discovery
        and read only the file at ``path`` even if it follows the
        ``.part-N`` naming convention.
        """
        if model_h5 is None:
            raise TypeError(_MODEL_H5_REQUIRED_MESSAGE)
        from .readers._mpco import MPCOReader
        from .readers._mpco_multi import (
            MPCOMultiPartitionReader, discover_partition_files,
        )

        if isinstance(path, (list, tuple)):
            paths = [Path(p) for p in path]
            reader = (
                MPCOMultiPartitionReader(paths)
                if len(paths) > 1
                else MPCOReader(paths[0])
            )
            anchor = paths[0]
        else:
            anchor = Path(path)
            if merge_partitions:
                discovered = discover_partition_files(anchor)
            else:
                discovered = [anchor]
            if len(discovered) > 1:
                reader = MPCOMultiPartitionReader(discovered)
            else:
                reader = MPCOReader(discovered[0])
        # A step after the open that raises closes the reader
        # first (the file is free again on Windows), then re-raises.
        try:
            # Per INV-3, this is an in-memory rehydrate from the sibling
            # file; we never copy the zone into a derived h5.
            from ..opensees.opensees_model import OpenSeesModel
            bound_model = OpenSeesModel.from_h5(model_h5)
            # The archive's neutral FEMData carries the physical groups
            # the MPCO MODEL/ group lacks; its /opensees/stages carry the
            # names the MODEL_STAGE[<k>] groups lack.
            bound_fem = _resolve_fem_via_model(
                reader, fem, bound_model, model_path=model_h5,
            )
            _bind_stage_names(reader, bound_model, model_path=model_h5)
            # ADR 0043 slice 1.3 — MPCO buckets key element results by
            # the OpenSees ops tag; the results API speaks fem_eid.
            # Whenever the bound model carries a real element_meta
            # pairing, the reader must relabel ops↔fem through it. This
            # is NOT compose-only: gmsh numbers lower-dimensional elements
            # first, so almost any solid model with surface physical
            # groups has fem_eid != ops_tag even uncomposed (the former
            # compose-provenance gate silently scrambled / dropped element
            # results for exactly that case — fixed 2026-08-04). A
            # deliberately-unrelated stub ``model_h5=`` (common in tests)
            # stays safe through the translator's all-or-nothing
            # `_relabel` contract: ids the model does not fully describe
            # pass through untranslated.
            from .readers._tag_translation import ElementTagTranslator
            _tag_map = ElementTagTranslator.from_model(bound_model)
            if not _tag_map.is_empty:
                reader.attach_tag_map(_tag_map)
            # The file holds no per-node ndf; the model's masks the
            # recorder's 0.0 rotations at nodes without rotational DOFs.
            from .readers._mpco import attach_model_nodes_ndf
            attach_model_nodes_ndf(reader, model_h5, bound_model.ndm)
            return cls(
                reader, fem=bound_fem, path=anchor, model=bound_model,
                model_path=Path(model_h5),
            )._with_autoloaded_definitions()
        except BaseException:
            reader.close()
            raise

    @classmethod
    def from_ladruno(
        cls,
        path: "str | Path | list[str | Path]",
        *,
        fem: "Optional[FEMData]" = None,
        merge_partitions: bool = True,
        model_h5: "Optional[str | Path]" = None,
    ) -> "Results":
        """Open a Ladruno ``.ladruno`` HDF5 results file.

        The Ladruno recorder is the fork's *canonical* recorder. Unlike
        ``.mpco`` (and unlike :meth:`from_mpco`, which **requires**
        ``model_h5=``), a ``.ladruno`` is **self-sufficient** — it carries
        its own geometry, regions and beam local axes (schema Principle 0:
        "this *is* the native path; no sibling file"). So ``model_h5=`` is
        **optional**:

        * omitted → the broker is built in-memory from the file's own
          ``MODEL`` group (geometry + inferred ``ndm``/``ndf``; bridge
          record zones empty). This is read-time interpretation, not a
          transcode. A file with no ``MODEL`` group raises
          :class:`ValueError` naming the file; ``model_h5=`` opens it.
        * supplied → the richer broker is loaded via
          :meth:`OpenSeesModel.from_h5` (full bridge records + lineage),
          and — whenever the model records an element_meta pairing —
          the fem_eid↔ops-tag translator is attached (ADR 0043; required
          for composed models AND for any sparsely-renumbered mesh, e.g.
          a gmsh solid whose 2-D boundary elements consumed the low ids).

        FEMData with ``model_h5=`` (#2329): as in :meth:`from_mpco`,
        ``fem=`` wins when given; otherwise the FEMData archived in
        ``model_h5`` is bound, with its physical groups and labels, so
        ``pg=`` and ``label=`` reads work. An archive of another model or
        another mesh warns ``ModelFemMismatchWarning`` and the file's own
        ``MODEL`` group (no physical groups) is bound instead. Without
        ``model_h5=`` the file's ``MODEL`` group is bound, as before.

        Stage names with ``model_h5=`` (ADR 0122 design point 9, #2101):
        as in :meth:`from_mpco`, the ``ops.stage(name=...)`` names and
        each stage's ``analysis`` kind from ``/opensees/stages`` replace
        the file's ``MODEL_STAGE[<k>]`` names in order, and
        ``MODEL_STAGE[<k>]`` stays resolvable as an alias. The count
        rules and warnings are :meth:`from_mpco`'s. A ``.ladruno`` stage
        also records the ``KIND`` it ran, so where that disagrees with
        the program kind the pairing is misaligned: nothing is renamed
        and ``StageKindMismatchWarning`` is the one warning (#2079 Q7).
        A stage recorded ``KIND=eigen`` (every stage of a recorder that
        asks for modes) or with no ``KIND`` is no evidence either way.
        Without ``model_h5=`` the file's names stand.

        The file records no per-node ndf, so the recorder writes 0.0 for
        a rotation a node does not have (a solid node beside a shell).
        With ``model_h5=`` a one-file read gives NaN there, from the
        model's ``/opensees/nodes_ndf``; without it a rotation read over
        continuum elements warns ``AbsentDofWarning`` once (#1971).

        Keys on ``INFO/GENERATOR="Ladruno"`` + a supported
        ``FORMAT_VERSION`` (the reader rejects a ``.mpco`` / foreign file
        or an out-of-window version loudly).

        Multi-partition merge: a parallel run writes one
        ``<stem>.part-<N>.ladruno`` per rank. Passing one partition path
        auto-discovers its siblings (``<stem>.part-*.ladruno``) and merges
        them into one virtual reader (node-union + element-concat);
        passing a list merges exactly those paths. ``merge_partitions=False``
        opts out of sibling auto-discovery.
        """
        from .readers._ladruno import LadrunoReader
        from .readers._ladruno_multi import (
            LadrunoMultiPartitionReader, discover_partition_files,
        )

        if isinstance(path, (list, tuple)):
            paths = [Path(p) for p in path]
            reader = (
                LadrunoMultiPartitionReader(paths)
                if len(paths) > 1
                else LadrunoReader(paths[0])
            )
            anchor = paths[0]
        else:
            anchor = Path(path)
            discovered = (
                discover_partition_files(anchor)
                if merge_partitions else [anchor]
            )
            reader = (
                LadrunoMultiPartitionReader(discovered)
                if len(discovered) > 1
                else LadrunoReader(discovered[0])
            )
        # A step after the open that raises closes the reader
        # first (the file is free again on Windows), then re-raises.
        try:
            bound_model: "Optional[OpenSeesModel]"
            if model_h5 is not None:
                from ..opensees.opensees_model import OpenSeesModel
                bound_model = OpenSeesModel.from_h5(model_h5)
                # As in from_mpco (#2329): the archive's FEMData carries
                # the physical groups and labels the file's MODEL group
                # lacks; a model of another mesh warns and binds the file's.
                bound_fem = _resolve_fem_via_model(
                    reader, fem, bound_model, model_path=model_h5,
                )
                # The program's stage names and kinds, as in from_mpco.
                _bind_stage_names(reader, bound_model, model_path=model_h5)
                # Attach the fem_eid↔ops-tag translator whenever the
                # model carries a real element_meta pairing — see the
                # twin comment in :meth:`from_mpco` (the pairing diverges
                # for ANY sparsely renumbered mesh, not just composed
                # models).
                from .readers._tag_translation import ElementTagTranslator
                _tag_map = ElementTagTranslator.from_model(bound_model)
                if not _tag_map.is_empty:
                    reader.attach_tag_map(_tag_map)
                # As in :meth:`from_mpco`: NaN, not the recorder's 0.0,
                # for rotations a node does not have.
                from .readers._ladruno import attach_model_nodes_ndf
                attach_model_nodes_ndf(reader, model_h5, bound_model.ndm)
                model_path: "Optional[Path]" = Path(model_h5)
            else:
                # Self-sufficient path — minimal broker from the file.
                bound_fem = _resolve_fem(reader, fem)
                bound_model = resolve_bound_model(reader, None)
                model_path = None
            if bound_model is None:
                # The reader builds its broker from the MODEL
                # group; a file without one needs model_h5=.
                raise ValueError(
                    f"{anchor}: this results file carries no MODEL group "
                    f"(node geometry), so the OpenSees model cannot be "
                    f"built from the file. Pass model_h5= to supply it."
                )
            return cls(
                reader, fem=bound_fem, path=anchor, model=bound_model,
                model_path=model_path,
            )._with_autoloaded_definitions()
        except BaseException:
            reader.close()
            raise

    @classmethod
    def from_fem(
        cls,
        fem: "FEMData",
        path: "str | Path | list[str | Path]",
        *,
        kind: str = "auto",
        merge_partitions: bool = True,
        cache_root: "str | Path | None" = None,
    ) -> "Results":
        """Open a results file against a bare :class:`FEMData` snapshot.

        The one-call route to :class:`Results` / :meth:`viewer` for a
        model that did **not** go through the ``apeSees`` bridge — e.g. a
        physical-group model where ``fem = g.mesh.queries.get_fem_data()``
        drove a hand-written OpenSees deck.  Without this the only routes
        were the bridge-bound constructors (which need a bridge-emitted
        ``model.h5`` / ``OpenSeesModel``) or the self-describing
        :meth:`from_ladruno`.

        ``from_fem`` materialises a neutral-only ``model.h5`` from ``fem``
        (cached, keyed by ``fem.snapshot_id``) and binds it to the reader
        for ``path``.  Materialising a file — rather than an in-memory
        model — is deliberate: it sets ``model_path`` so the non-blocking
        / web viewers work (they forward ``--model-h5``), not just data
        access.

        Parameters
        ----------
        fem
            The bound snapshot (from ``g.mesh.queries.get_fem_data()`` or
            ``FEMData.from_h5``).  A **composed** fem is refused — see
            below.
        path
            The results file (or a partition list).
        kind
            ``"mpco"`` / ``"ladruno"`` / ``"native"``, or ``"auto"``
            (default) to detect from the suffix (``.mpco`` / ``.ladruno``;
            a native ``.h5`` must pass ``kind="native"``).
        merge_partitions
            Forwarded to :meth:`from_mpco` / :meth:`from_ladruno` for
            ``.part-N`` auto-discovery.
        cache_root
            Where the materialised ``model.h5`` is written — under
            ``<cache_root>/from_fem/`` (default ``<cwd>/results/from_fem/``
            or ``$APEGMSH_RESULTS_DIR``).

        Raises
        ------
        ValueError
            When ``fem`` is **composed** (``g.compose`` / ``from_h5``
            assembly).  Element / Gauss results are relabelled through the
            ``fem_eid ↔ ops-tag`` map that only a real bridge run records;
            a neutral-only ``model.h5`` carries none, so a composed model
            would silently mislabel every element result.  Build the model
            through ``apeSees`` and pass its ``model.h5`` (e.g.
            ``apeSees(fem).h5(...)`` / ``g.save`` →
            ``from_mpco(path, model_h5=...)``).

        Notes
        -----
        A bare fem carries no envelope ``ndf`` (``MeshInfo`` has none), so
        the cached model's ``ndf`` is ``0``.  This is harmless for reading
        results and for the viewer; it only matters for deck **re-emit**
        (``model.build(...)``), which is not this path's purpose.
        """
        # ADR 0043 slice 1.3 — element results relabel through the
        # bridge-emitted element_meta.  A neutral-only model.h5 carries
        # none, but ``composed_from`` round-trips, so from_mpco /
        # from_ladruno would see a "composed" model, attach an
        # element-less (empty) tag translator, and silently mislabel
        # every element/gauss/fiber result.  Refuse loudly.
        if len(getattr(fem, "composed_from", ()) or ()) > 0:
            raise ValueError(
                "Results.from_fem: the FEMData is composed (g.compose / "
                "from_h5). Element/Gauss results need the bridge's "
                "fem_eid<->ops-tag map, which a neutral-only model.h5 "
                "cannot provide. Build the model through apeSees and pass "
                "its model.h5 — e.g. Results.from_mpco(path, "
                "model_h5='model.h5')."
            )

        resolved_kind = _resolve_results_kind(kind, path)

        # Materialise a neutral-only model.h5, cached by content hash.
        from .writers._cache import resolve_cache_root

        cache_dir = resolve_cache_root(cache_root) / "from_fem"
        cache_dir.mkdir(parents=True, exist_ok=True)
        snap = str(getattr(fem, "snapshot_id", "") or "model")
        units_tag = f".{fem.units}" if getattr(fem, "units", None) else ""
        cached = cache_dir / f"{snap}{units_tag}.model.h5"
        if not cached.exists():
            fem.to_h5(str(cached))

        if resolved_kind == "mpco":
            return cls.from_mpco(
                path, fem=fem, model_h5=cached,
                merge_partitions=merge_partitions,
            )
        if resolved_kind == "ladruno":
            return cls.from_ladruno(
                path, fem=fem, model_h5=cached,
                merge_partitions=merge_partitions,
            )
        # native — pass the rehydrated neutral-only model + its path so
        # the subprocess viewer can forward --model-h5.
        from ..opensees.opensees_model import OpenSeesModel

        return cls.from_native(
            path, fem=fem, model=OpenSeesModel.from_h5(cached),
            model_path=cached,
        )

    # ------------------------------------------------------------------
    # FEM access & binding
    # ------------------------------------------------------------------

    @property
    def fem(self) -> "Optional[FEMData]":
        """The bound FEMData snapshot, or None if not bound.

        Raises :class:`SchemaVersionError` on a native results file whose
        embedded ``/model`` is below its floor and no ``fem=`` was
        supplied (ADR 0113 D9): the zone is never read silently, and
        the message says to pass ``fem=`` or call :meth:`bind`.
        """
        if self._fem_unavailable is not None:
            from ..opensees._internal.schema_version import SchemaVersionError

            raise SchemaVersionError(
                f"{self._fem_unavailable} The embedded neutral zone is "
                "unavailable; only /stages reads (ADR 0113 D9). Pass "
                "fem= to Results.from_native, or call results.bind(fem), "
                "with the FEMData this run used."
            )
        return self._fem

    @property
    def model(self) -> "OpenSeesModel":
        """The bound :class:`OpenSeesModel` broker.

        Phase 8 (ADR 0020 INV-1) — always non-None on a constructed
        :class:`Results`.  The chain-forward handle from which the
        OpenSeesModel and its embedded FEMData can be reached.
        """
        return self._model

    # ------------------------------------------------------------------
    # Units & unit-system translation
    # ------------------------------------------------------------------

    @property
    def units(self) -> "Optional[str]":
        """The active consistent unit system ID (e.g. ``"kN_m_s"``), or
        ``None`` when no unit system was declared."""
        from ._units import resolve_results_units
        return resolve_results_units(self)

    @property
    def source_units(self) -> "Optional[str]":
        """The native (as-recorded) unit system ID of the underlying file
        or model, or ``None`` when undeclared and no ``source=`` was given."""
        from ._units import resolve_results_source_units
        return resolve_results_source_units(self)

    def unit_factor(
        self,
        component: str,
        *,
        level: "str | Any" = "nodes",
        target: "Any" = None,
        source: "Any" = None,
    ) -> float:
        """Return the multiplicative conversion factor for ``component`` at
        ``level`` (``1.0`` when ``target=None``, or converting to ``target``).

        Raises :class:`ValueError` when ``target`` is supplied on an
        undeclared ``Results`` without an explicit ``source=``.
        """
        from ._units import compute_results_unit_factor
        return compute_results_unit_factor(
            self, component, level=level, target=target, source=source,
        )

    def unit_symbol(
        self,
        component: str,
        *,
        level: "str | Any" = "nodes",
    ) -> str:
        """Return the display unit symbol for ``component`` in :attr:`units`
        (e.g. ``"kN"``, ``"MPa"``), or ``""`` when :attr:`units` is ``None``
        or the component is dimensionless."""
        from ._units import compute_results_unit_symbol
        return compute_results_unit_symbol(self, component, level=level)

    def to_units(
        self,
        target: "Any",
        *,
        source: "Any" = None,
    ) -> "Results":
        """Return a new :class:`Results` view whose slabs and bound
        :attr:`fem` coordinates are scaled to ``target`` consistent units.

        * When :attr:`units` is declared on the model or results file,
          ``source`` is inferred automatically and may be omitted
          (passing a conflicting ``source`` raises :class:`ValueError`).
        * When :attr:`units` is ``None`` (undeclared), ``source`` is
          **required** — omitting it raises :class:`ValueError` asking the
          caller to specify the source unit system explicitly.
        """
        from ._units import build_scaled_results
        return build_scaled_results(self, target, source=source)

    def energy(
        self,
        *,
        region: "Optional[int]" = None,
        stage: Optional[str] = None,
    ) -> "Any":
        """Energy-balance time history — **Ladruno-recorder feature**.

        Returns a :class:`pandas.DataFrame` of the closure components
        ``KE`` / ``IE`` / ``DW`` / ``ULW`` / ``RES`` / ``ERR`` indexed by
        simulation time, written by the recorder's ``-G energy`` verb.

        * ``region=None`` → whole-domain balance (``ON_DOMAIN``).
        * ``region=<tag>`` → the per-region balance (``ON_REGIONS``) for
          the OpenSees region tag.

        ``ERR`` (the normalized energy-balance error %) is the headline
        solution-quality diagnostic for explicit runs. Raises
        :class:`TypeError` on a non-Ladruno results object (MPCO / native
        carry no energy balance) and ``ValueError`` if energy was not
        recorded / the region is unknown.

        ``results.plot.energy(...)`` renders this as a matplotlib
        time-history figure.
        """
        read_energy = getattr(self._reader, "read_energy", None)
        if read_energy is None:
            raise TypeError(
                "Results.energy() is a Ladruno-recorder feature. Open a "
                ".ladruno via Results.from_ladruno(...) recorded with the "
                "'-G energy' verb; MPCO / native results carry no energy "
                "balance."
            )
        sid = self._resolve_stage(stage)
        cols, values, time = read_energy(sid, region=region)
        import pandas as pd
        return pd.DataFrame(
            values, columns=cols, index=pd.Index(time, name="time"),
        )

    def energy_regions(self, *, stage: Optional[str] = None) -> "list[int]":
        """OpenSees region tags with a recorded per-region energy balance.

        The bridge auto-allocates an integer region tag when a Ladruno
        recorder is given ``energy_pg=`` (or a value filter + ``energy``);
        that tag is opaque to the author. This lists the tags actually
        present in ``ON_REGIONS/energyBalance`` so you can pick one to pass
        to :meth:`energy` — e.g. ``r.energy(region=r.energy_regions()[0])``.

        Returns ``[]`` when only the whole-model balance was recorded
        (read it with ``energy()``, no ``region=``). Ladruno-recorder
        feature; raises :class:`TypeError` on MPCO / native results.
        """
        available = getattr(self._reader, "available_energy_regions", None)
        if available is None:
            raise TypeError(
                "Results.energy_regions() is a Ladruno-recorder feature. "
                "Open a .ladruno via Results.from_ladruno(...); MPCO / "
                "native results carry no energy balance."
            )
        sid = self._resolve_stage(stage)
        return available(sid)

    def node_envelope(
        self,
        component: str,
        *,
        stage: Optional[str] = None,
    ) -> "Any":
        """Per-node time-reduced extremes — **Ladruno ``-envelope`` feature**.

        When a ``.ladruno`` is recorded with the recorder's ``-envelope``
        flag, each node channel stores componentwise running extremes
        (``MIN``/``MAX``/``ABSMAX`` and the step at which the abs-extreme
        occurred) *instead of* a time series — the cheap way to capture peak
        response over a long run without keeping every step.

        Returns a :class:`pandas.DataFrame` indexed by node id with columns
        ``min`` / ``max`` / ``absmax`` / ``arg_step`` for ``component`` (a
        neutral name like ``"displacement_x"``). Raises :class:`TypeError`
        on a non-Ladruno results object, and :class:`ValueError` if the file
        was not recorded with ``-envelope`` or the component is absent.

        ``results.plot.node_envelope(...)`` paints a chosen measure on
        the mesh as a matplotlib figure.
        """
        read_node_envelope = getattr(self._reader, "read_node_envelope", None)
        if read_node_envelope is None:
            raise TypeError(
                "Results.node_envelope() is a Ladruno-recorder feature. Open "
                "a single-file .ladruno via Results.from_ladruno(...) recorded "
                "with the '-envelope' flag. (MPCO / native results and "
                "partitioned .ladruno envelope merges are not supported.)"
            )
        sid = self._resolve_stage(stage)
        env = read_node_envelope(sid, component)
        import pandas as pd
        return pd.DataFrame(
            {
                "min": env.min,
                "max": env.max,
                "absmax": env.absmax,
                "arg_step": env.arg_step,
            },
            index=pd.Index(env.node_ids, name="node_id"),
        )

    @property
    def lineage(self) -> "Lineage":
        """Phase-6 lineage chain — git-style ``fem → model → results``.

        ADR 0021 defines a three-link hash chain ``fem_hash →
        model_hash → results_hash`` where each layer's hash includes
        its parent's hash (one-directional, tamper-evident).
        Mismatches between stored and recomputed hashes surface as
        ``[lineage] ...`` warnings in :attr:`Lineage.warnings`; they
        never raise from this property (INV-2).

        Phase-8 derivation order:

        1. Inherit ``fem_hash`` + ``model_hash`` + accumulated
           warnings from :attr:`model.lineage` (the broker
           recomputes against the same file).
        2. Read the stored ``/meta/lineage/results_hash`` via the
           reader's ``results_lineage_attrs`` helper and recompute
           from ``/stages/...`` via ``recompute_results_hash``;
           append a drift warning on mismatch.

        Readers that don't implement the Phase-6 result-layer
        protocol methods are tolerated via ``getattr`` cushions:
        their lineage stays at the model layer, no warning emitted.
        """
        from ..opensees._internal.lineage import (
            WARNING_PREFIX,
            Lineage,
        )
        from ..opensees.emitter.h5_reader import records_program_decl

        # Layer 1: inherit from the bound OpenSeesModel.
        base = self._model.lineage
        warnings = list(base.warnings)
        fem_hash = base.fem_hash
        model_hash = base.model_hash

        # Layer 2: layer the results_hash on top.  Reader protocol
        # extension is optional — older readers (MPCO) silently lack
        # the helpers; their lineage stays at the model layer.
        results_hash: "str | None" = None
        get_stored = getattr(self._reader, "results_lineage_attrs", None)
        recompute = getattr(self._reader, "recompute_results_hash", None)
        if get_stored is not None and recompute is not None:
            try:
                stored = get_stored()
            except Exception:
                stored = (None, None, None)
            stored_fem, stored_model, stored_results = stored
            # Drift checks at the result-layer envelope.
            if stored_fem is not None and fem_hash and stored_fem != fem_hash:
                warnings.append(
                    WARNING_PREFIX
                    + "fem hash mismatch between Results and OpenSeesModel: "
                    f"results-file stamp {stored_fem!r} != "
                    f"OpenSeesModel.lineage.fem_hash {fem_hash!r}"
                )
            if (
                stored_model is not None
                and model_hash is not None
                and stored_model != model_hash
            ):
                message = (
                    WARNING_PREFIX
                    + "model hash mismatch between Results and OpenSeesModel: "
                    f"results-file stamp {stored_model!r} != "
                    f"OpenSeesModel.lineage.model_hash {model_hash!r}"
                )
                # Name the declaration-id hash change when exactly one
                # side's hash includes the ids, per the completeness
                # marker: records_program_decl on the model,
                # /opensees/decls@program_decl on the results file.
                # Still a warning; a results file with no /opensees zone
                # keeps the text.
                results_records = self._results_records_program_decl()
                model_records = records_program_decl(self._model.declarations)
                cause = _model_hash_era_cause(results_records, model_records)
                if cause is None and results_records and model_records:
                    # Both markers "complete": the side-channel step,
                    # when exactly one side's side-channel calls carry
                    # their declaration ids.
                    cause = _side_channel_hash_cause(
                        self._results_side_channel_proof(),
                        _side_channel_proof(
                            (run.method, run.decl)
                            for run in self._model.program()
                        ),
                    )
                if cause is not None:
                    message += "; " + cause
                warnings.append(message)
            # Recompute results_hash from /stages/ and compare to
            # the stamped value.  When the model_hash link is
            # absent (bridge-only files, recorder-only runs) we
            # chain on empty — same as the writer.
            try:
                recomputed_results = recompute(model_hash or "")
            except Exception as exc:
                recomputed_results = None
                warnings.append(
                    WARNING_PREFIX
                    + f"results_hash recompute failed: {exc!r}"
                )
            results_hash = recomputed_results
            if (
                stored_results is not None
                and recomputed_results is not None
                and stored_results != recomputed_results
            ):
                warnings.append(
                    WARNING_PREFIX
                    + "results layer drift: stored results_hash="
                    f"{stored_results!r} != recomputed "
                    f"{recomputed_results!r}"
                )

        return Lineage(
            fem_hash=fem_hash,
            model_hash=model_hash,
            results_hash=results_hash,
            warnings=tuple(warnings),
        )

    def _results_records_program_decl(self) -> "bool | None":
        """Whether the results file's embedded ``/opensees`` records its
        program declarations: ``/opensees/decls@program_decl`` is
        ``"complete"`` (the zone is copied verbatim from the source
        model.h5). A missing ``decls`` group or marker is the old hash
        (``False``). ``None`` for a non-native reader, a file with no
        ``/opensees`` zone, or a closed handle. Reads the reader's open
        handle; called only on a ``model_hash`` mismatch.
        """
        from ..opensees.emitter.h5_reader import PROGRAM_DECL_COMPLETE
        from .readers._native import NativeReader

        reader = getattr(self._reader, "base_reader", self._reader)
        if not isinstance(reader, NativeReader):
            return None
        h5 = reader._h5
        try:
            if "opensees" not in h5:
                return None
            ops = h5["opensees"]
            if "decls" not in ops:
                return False
            marker = ops["decls"].attrs.get("program_decl")
        except (KeyError, OSError, ValueError):
            return None
        if isinstance(marker, bytes):
            marker = marker.decode("utf-8", "replace")
        return marker is not None and str(marker) == PROGRAM_DECL_COMPLETE

    def _results_side_channel_proof(self) -> "bool | None":
        """:func:`_side_channel_proof` of the results file's embedded
        ``/opensees/program`` (copied verbatim from the source model.h5).
        ``None`` for a non-native reader, no program, a program whose
        columns are not ``method`` / ``decl`` coded as the reader writes
        them, or a closed handle: the cause is then never guessed. Called
        only on a ``model_hash`` mismatch whose markers agree.
        """
        from .readers._native import NativeReader

        reader = getattr(self._reader, "base_reader", self._reader)
        if not isinstance(reader, NativeReader):
            return None
        h5 = reader._h5
        try:
            if "opensees" not in h5 or "program" not in h5["opensees"]:
                return None
            ds = h5["opensees"]["program"]
            names = tuple(ds.dtype.names or ())
            if "method" not in names or "decl" not in names:
                return None
            methods = [
                m.decode("utf-8", "replace") if isinstance(m, bytes) else str(m)
                for m in ds.attrs["methods"]
            ]
            table = ds[()]
            codes = [int(m) for m in table["method"]]
            if any(not 0 <= m < len(methods) for m in codes):
                return None
            pairs = [
                (methods[m], int(d)) for m, d in zip(codes, table["decl"])
            ]
        except (KeyError, OSError, ValueError):
            return None
        return _side_channel_proof(pairs)

    def bind(self, fem: "FEMData") -> "Results":
        """Re-bind to ``fem``.

        Useful when you've re-built the same mesh in a fresh session
        and want labels / Parts that the embedded snapshot doesn't
        carry. No hash validation is performed — pairing the FEMData
        with a results file from the same run is the user's
        responsibility.
        """
        from ._units import resolve_results_units

        bound = _resolve_fem(self._reader, fem)
        derived = self._derive(fem=bound)
        if (
            self._units_override is not None
            and bound is not None
            and getattr(bound, "units", None) is not None
            and bound.units not in (self.units, self.source_units)
        ):
            raise ValueError(
                f"Conflicting unit declarations on Results.bind: "
                f"Results has units={self.units!r}, but fem declares "
                f"units={bound.units!r}."
            )
        resolve_results_units(derived)
        return derived

    def _element_reads_fem_keyed(self) -> bool:
        """True when element-level reads return a fem_eid-keyed index.

        Consumed (duck-typed) by the viewer's
        ``resolve_orientation_source`` to decide which id space the
        scene must be built in. A merely *attached* translator is NOT
        enough: a deliberately-unrelated stub ``model_h5=`` (the
        sanctioned test-suite pattern) attaches a pairing that does not
        describe the files, and reads stay in the file's own ops-tag
        space. Reads relabel exactly when the translator says the
        archive describes the files it was attached to
        (:meth:`ElementTagTranslator.describes_files`, #2066): every
        recorded ops tag is a paired element or one emitted without a
        mesh id. The second kind keeps its ops tag, so judging by the
        paired tags alone would call a file with an interface
        ``zeroLength`` ops-keyed while its reads come back fem-keyed.
        Cached: the answer is fixed at open time.
        """
        cached = getattr(self, "_fem_keyed_reads_cache", None)
        if cached is not None:
            return cached
        tag_map = getattr(self._reader, "_tag_map", None)
        result = tag_map is not None and tag_map.describes_files()
        self._fem_keyed_reads_cache = result
        return result

    def _element_tag_pairing_missing(self) -> bool:
        """True when element-level reads cannot be relabelled fem↔ops.

        The unreliable combination (ADR 0043 slice 1.3 follow-up): the
        reader keys element results by OpenSees ops tags and its reads
        do not relabel into ``fem_eid`` space, while the bound
        ``FEMData`` was supplied *externally* (``fem=`` / ``.bind(fem)``)
        and therefore speaks gmsh ``fem_eid``s. Judged by
        :meth:`_element_reads_fem_keyed` — NOT by mere translator
        attachment: an attached pairing that does not cover the file's
        recorded element ids (an unrelated stub ``model_h5=``) never
        fires, so reads stay in ops space and the cross-id-space join
        deserves the same warning as no pairing at all. When the fem is
        the reader's own embedded snapshot the two id spaces coincide by
        construction, so that case is fine; native results are fem-keyed
        already (their reader has no ``attach_tag_map``).
        """
        reader = self._reader
        if not hasattr(reader, "attach_tag_map"):
            return False  # native reader — element results are fem-keyed
        if self._element_reads_fem_keyed():
            return False  # covering pairing — reads relabel to fem_eids
        if self._fem is None:
            return False  # no fem, no fem_eid-space expectations
        try:
            embedded = reader.fem()
        except Exception:
            embedded = None
        return self._fem is not embedded

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    @property
    def stages(self) -> list[StageInfo]:
        """All stages in the file (scoped instances also list them)."""
        return list(self._all_stages())

    def stage(self, name_or_id: str) -> "Results":
        """Return a Results scoped to a stage (matched by id, name or alias)."""
        info = self._lookup_stage(name_or_id)
        return self._derive(stage_id=info.id)

    @property
    def modes(self) -> list["Results"]:
        """Stages with ``kind='mode'`` as a list of mode-scoped Results.

        Order is the order the modes were written (typically by
        ascending mode_index). For a stable lookup by index, sort:
        ``sorted(results.modes, key=lambda m: m.mode_index)``.
        """
        return [
            self._derive(stage_id=s.id)
            for s in self._all_stages() if s.kind == "mode"
        ]

    @property
    def eigen_modes(self) -> list[EigenMode]:
        """Mode-kind stages as lightweight :class:`EigenMode` snapshots.

        Each :class:`EigenMode` carries only the four scalar fields
        (``mode_index``, ``eigenvalue``, ``frequency_hz``, ``period_s``)
        — no file handle, no mode-shape arrays.  Use this when you
        need the eigenvalue spectrum but not the per-node shapes
        (e.g. an LTB Mcr probe, a pickle-able report, or a return
        value from a function whose Results context is about to be
        closed).

        For the per-node mode shape arrays, use the mode-scoped
        :class:`Results` from :attr:`modes` instead and query via
        ``mode.nodes.get(component="displacement_x", ...)``.

        Order matches :attr:`modes`.  For a stable lookup by index,
        sort: ``sorted(results.eigen_modes, key=lambda m: m.mode_index)``.
        """
        out: list[EigenMode] = []
        for s in self._all_stages():
            if s.kind != "mode":
                continue
            # Mode-kind stages always have these four fields populated
            # (capture_modes guarantees it); guard with asserts so the
            # type checker accepts the float coercion.
            assert s.eigenvalue is not None
            assert s.frequency_hz is not None
            assert s.period_s is not None
            assert s.mode_index is not None
            out.append(EigenMode(
                mode_index=int(s.mode_index),
                eigenvalue=float(s.eigenvalue),
                frequency_hz=float(s.frequency_hz),
                period_s=float(s.period_s),
            ))
        return out

    # ------------------------------------------------------------------
    # Stage-scoped properties (raise on unscoped or wrong-kind access)
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return self._require_scoped().kind

    @property
    def name(self) -> str:
        return self._require_scoped().name

    @property
    def n_steps(self) -> int:
        return self._require_scoped().n_steps

    @property
    def time(self) -> ndarray:
        info = self._require_scoped()
        return self._reader.time_vector(info.id)

    @property
    def eigenvalue(self) -> float:
        info = self._require_mode()
        assert info.eigenvalue is not None
        return info.eigenvalue

    @property
    def frequency_hz(self) -> float:
        info = self._require_mode()
        assert info.frequency_hz is not None
        return info.frequency_hz

    @property
    def period_s(self) -> float:
        info = self._require_mode()
        assert info.period_s is not None
        return info.period_s

    @property
    def mode_index(self) -> Optional[int]:
        return self._require_mode().mode_index

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying reader (releases the HDF5 file handle)."""
        if hasattr(self._reader, "close"):
            self._reader.close()

    def __enter__(self) -> "Results":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Demo / sample data
    # ------------------------------------------------------------------

    @classmethod
    def demo(cls, **kwargs) -> "Results":
        """Return a ready-to-view demo :class:`Results` (cantilever pushover).

        Zero-setup sample data so ``Results.demo().show_web()`` (or
        ``.viewer()``) renders without supplying an ``.mpco`` /
        ``model.h5`` pair — handy for docs, smoke tests, and trying the
        viewer. A real ``apeSees``-emitted model with a synthetic, ramped
        cantilever deflection (no OpenSees solve). See
        :func:`apeGmsh.results.make_demo_results` for the keyword options
        (``length`` / ``n_elements`` / ``n_steps`` / ``tip_drift`` /
        ``path``).
        """
        from .demo import make_demo_results
        return make_demo_results(**kwargs)

    # ------------------------------------------------------------------
    # Static plotting (matplotlib)
    # ------------------------------------------------------------------

    @property
    def plot(self) -> "ResultsPlot":
        """``results.plot`` — static matplotlib renderer.

        Mirrors the interactive viewer's diagram catalog as headless,
        publication-ready matplotlib figures::

            results.plot.contour("displacement_z", step=-1)
            results.plot.deformed(step=-1, scale=50, component="stress_xx")
            results.plot.history(node=412, component="displacement_x")

        Requires the ``[plot]`` extra (matplotlib).
        """
        if self._plot is None:
            from .plot import ResultsPlot
            self._plot = ResultsPlot(self)
        return self._plot

    # ------------------------------------------------------------------
    # Assess (ADR 0094 S2)
    # ------------------------------------------------------------------

    def assess(
        self,
        *,
        figures: bool = False,
        out_dir: "str | Path | None" = None,
    ) -> "AssessmentReport":
        """Compile a v1 :class:`~apeGmsh.assess.AssessmentReport`.

        ``figures=True`` calls :meth:`render_pack`. Default is ``False``.
        """
        from apeGmsh.assess import assess_results
        return assess_results(self, figures=figures, out_dir=out_dir)

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def session(self):
        """The presentation session for these results (ADR 0098 §1).

        Presentation with no window: a ``ResultsSession`` (from
        ``apeGmsh.results.session``) bound to this broker, booted with
        the default picture — ONE empty mesh view (grey analysis mesh,
        no slots, no legends). Configure it (slots, deform, time), then
        ``s.render("a.png")`` for a still; the Qt client (``s.show()``)
        arrives at S2 and ``viewer()`` flips onto it at S6.

        **Persisted section cuts boot as view clips** (ADR 0098 S6b).
        The retired ``section_cut`` diagram kind took its auto-load
        contract with it, but not the contract itself: cuts persisted
        under ``/opensees/cuts/`` come back on the booted view as
        clips. Only the ones that translate honestly do — a cut that
        named a strict subset of the model's elements, or that carries
        a bounding polygon, cuts LESS than a view clip does, so it is
        skipped with one ``[session]`` line rather than silently
        widening what disappears from the screen. Reading the cuts can
        never fail this call: a bad zone is a line, not a traceback.
        """
        from .session import ResultsSession
        from .session._cuts import attach_persisted_cuts

        s = ResultsSession(results=self)
        view = s.add_view()
        try:
            notices = attach_persisted_cuts(self, view)
        except Exception as exc:  # noqa: BLE001 - the session always boots
            notices = (
                f"persisted section cuts could not be loaded as view "
                f"clips: {type(exc).__name__}: {exc}",
            )
        for notice in notices:
            print(f"[session] {notice}")
        return s

    def viewer(
        self,
        *,
        blocking: "Optional[bool]" = None,
        title: Optional[str] = None,
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
    ):
        """Open the post-solve results window on a ``ResultsSession``.

        Sugar for :meth:`session` + ``show()`` (ADR 0098 §1, flipped at
        S6a). The one-liner is unchanged; what it opens is not. A
        :class:`~apeGmsh.results.session.ResultsSession` is the
        document — tiled mesh and plot panes, each with the closed §4
        slot catalog — and the window is a client that projects it. The
        retired Geometry / Composition / Diagram window is gone from
        this door. Everything the window does, a script can do to the
        same object::

            s = results.session()   # the document, no window
            s.render("a.png")       # a still, no Qt
            results.viewer()        # the human one-liner

        Parameters
        ----------
        blocking
            ``None`` (default) — auto: ``True`` in scripts and the
            plain CLI, ``False`` inside a Jupyter / IPython ZMQ kernel,
            where the blocking Qt loop would freeze (often kill) the
            kernel. A hand-built, in-memory Results (no backing file)
            in a notebook cannot spawn a subprocess and raises
            :class:`RuntimeError` naming the ``from_*`` constructors to
            open it from a file instead. The notebook path announces
            itself with one line.
            ``True`` — open the window in-process and block the calling
            thread until it closes. Matches the signature of
            :meth:`g.mesh.viewer` and :meth:`g.model.viewer`.
            ``False`` — spawn a subprocess via
            ``python -m apeGmsh.viewers <path>`` so the notebook /
            kernel can keep running. Requires that the Results was
            opened from disk (``self._path`` is set); raises
            :class:`RuntimeError` for in-memory Results.
        title
            Optional window title; defaults to ``"Results — <filename>"``.
        restore_session
            What to do with a session snapshot saved beside the results
            file. ``True`` restores silently, ``False`` ignores it,
            ``"prompt"`` (default) asks. No effect for in-memory
            Results, which have no file to sit beside.
        save_session
            If ``True`` (default), the session — panes, slots, pose,
            time link, selection — is written to
            ``<results>.viewer-session.json`` when the window closes.
            ``False`` disables auto-save. Auto-save also disarms itself
            for a window that could not read an existing file there, so
            the unreadable file survives (INV-SESSION-OPEN, see
            ``apeGmsh.results.session._boot``).

        Returns
        -------
        ResultsSession
            The session the window projected, **after** the window
            closes (blocking). Still live: query it, render stills off
            it, snapshot it.
        subprocess.Popen
            The spawned process handle (non-blocking). Deliberately not
            unified with the blocking return — a session in this
            process is not what the child window is showing.
        None
            If ``APEGMSH_SKIP_VIEWER`` is set in the environment. This
            lets the same cell run under ``jupyter nbconvert --execute``
            or in CI without spawning a GUI window.

        Raises
        ------
        RuntimeError
            Auto mode (``blocking=None``) inside a notebook kernel on an
            in-memory Results (``_path is None``): there is no file for
            ``python -m apeGmsh.viewers`` to open. Open the results with
            :meth:`from_native`, :meth:`from_mpco`, :meth:`from_ladruno`,
            :meth:`from_recorders` or :meth:`from_fem`. Nothing is
            converted to a temporary file: the reader protocol does not
            carry everything a native ``.h5`` needs, so a conversion
            would lose data (#1664).

        Notes
        -----
        The v13 ``<results>.viewer-session.json`` written by the retired
        window is not restorable (ADR 0098 Consequences). The first
        flipped open says so in one line and renames it aside to
        ``.legacy`` — never overwriting it, and never overwriting an
        aside that is already there.

        ``cuts=`` is retired with the diagram ontology (§1): a cut plane
        is clip state on a view. Build cuts with :mod:`apeGmsh.cuts` and
        add them as clips — ``results.session()`` then ``view.add_clip``.
        """
        import os
        if os.environ.get("APEGMSH_SKIP_VIEWER"):
            print("[skip viewer] APEGMSH_SKIP_VIEWER set")
            return None
        if blocking is None:
            if not _in_notebook_kernel():
                blocking = True
            elif self._path is None:
                raise RuntimeError(
                    "Results.viewer(): a notebook kernel needs the viewer "
                    "in a separate process, and this Results is in-memory "
                    "(no backing file) so there is nothing for that "
                    "process to open. Open it from a file with "
                    "Results.from_native(path), Results.from_mpco(path), "
                    "Results.from_recorders(spec, output_dir, fem=fem), "
                    "Results.from_fem(fem, path) or another "
                    "Results.from_* constructor, then call viewer() again. "
                    "(blocking=True opens the Qt window in-process, but "
                    "that kills a Jupyter kernel.)"
                )
            else:
                print(
                    "[viewer] notebook kernel detected — spawning the "
                    "viewer as a separate process so the kernel keeps "
                    "running (pass blocking=True to open it in-process)."
                )
                blocking = False
        if not blocking:
            handle = self._spawn_viewer_subprocess(
                title=title,
                restore_session=restore_session,
                save_session=save_session,
            )
            # The subprocess opens its own NativeReader against the
            # path; the parent kernel's reader is no longer needed for
            # rendering. Close it here so the user can re-run a capture
            # script (which deletes / recreates the same .h5) without
            # hitting ``PermissionError: file is being used by another
            # process`` — Windows refuses to unlink a file that any
            # process has open, even read-only.
            #
            # If the user wants to keep querying ``results`` after the
            # spawn, they can re-bind via ``Results.from_native(path)``.
            try:
                self.close()
            except Exception:
                pass
            return handle
        return self._show_session_window(
            title=title,
            restore_session=restore_session,
            save_session=save_session,
        )

    def _show_session_window(
        self,
        *,
        title: Optional[str],
        restore_session: "bool | str",
        save_session: bool,
    ):
        """The blocking half of :meth:`viewer` — boot, show, save.

        Split out so the S6a open policy has one body whether it is
        reached from ``viewer()`` or from the ``python -m
        apeGmsh.viewers`` child, and so a test can drive the policy
        without going through the notebook / subprocess branching
        above.

        Save-on-close needs no Qt hook: ``show(blocking=True)`` returns
        once the window has closed, and closing disposes WIDGETS — the
        session IR is untouched — so writing here writes what was on
        screen.
        """
        from .session._boot import announce, boot_session, save_on_close

        confirm = None
        if restore_session == "prompt":
            from ..viewers.session._restore_prompt import confirm_restore
            confirm = confirm_restore

        boot = boot_session(
            self,
            restore=restore_session,
            save=save_session,
            confirm=confirm,
        )
        announce(boot)
        boot.session.show(blocking=True, title=title)
        save_on_close(boot)
        return boot.session

    def render(
        self,
        path: "str | Path",
        *,
        view: str = "contour",
        component: Optional[str] = None,
        step: int = -1,
        deform: "Optional[Any]" = None,
        camera: "Optional[str]" = None,
        window_size: tuple[int, int] = (1280, 720),
    ) -> "Optional[Path]":
        """Write one offscreen still (ADR 0094 S1).

        Drawn by :meth:`ResultsSession.render` (X2-d): ``view`` is
        translated into one session mesh view (slots, pose, instant)
        and rendered VTK-offscreen — no Qt window, no event loop, no
        ``setup(plotter, director)``. ``view`` is a closed set:
        ``mesh`` / ``contour`` / ``deformed`` / ``reactions``.

        The stage drawn is the scope of ``results.stage(s)`` (or
        ``results.modes[i]``), else the only stage; an unscoped
        multi-stage Results draws its LAST stage. ``step`` indexes that
        stage.

        ``camera=`` defaults to ``xy`` for a planar model, ``iso``
        otherwise (ADR 0094 Amendment 3); pass it explicitly to
        override.

        Returns the written :class:`~pathlib.Path`, or ``None`` (and
        prints the ``[skip viewer]`` notice) under
        ``APEGMSH_SKIP_VIEWER=1`` or with no GL.
        """
        from apeGmsh.viewers.session._results_stills import (
            render_results_still,
        )
        return render_results_still(
            self, path,
            view=view, component=component, step=step,
            deform=deform, camera=camera, window_size=window_size,
        )

    def render_pack(
        self,
        out_dir: "str | Path",
        *,
        camera: "Optional[str]" = None,
        window_size: tuple[int, int] = (1280, 720),
    ) -> tuple[Path, ...]:
        """Write the canned report pack (ADR 0094 S3).

        Each still is a :meth:`render` call, so the pack is drawn by
        :meth:`ResultsSession.render` too. Returns the tuple of written paths, or ``()`` under
        ``APEGMSH_SKIP_VIEWER=1`` / no GL (and prints the
        ``[skip viewer]`` notice). Closed ``view=`` set only; no
        ``setup()``. There is no ``fem.render_pack``. The stage is
        :meth:`render`'s: a stage-scoped Results packs that stage.

        ``camera=`` defaults to ``xy`` for a planar model, ``iso``
        otherwise (ADR 0094 Amendment 3); pass it explicitly to
        override.
        """
        from apeGmsh.viewers.session._results_stills import (
            render_results_pack,
        )
        return render_results_pack(
            self, out_dir, camera=camera, window_size=window_size,
        )

    def export_animation(
        self,
        path: "str | Any",
        *,
        fps: int = 30,
        step_stride: int = 1,
        stage: "Optional[str]" = None,
        deform: "Optional[Any]" = None,
        camera: "Optional[Any]" = None,
        window_size: "Optional[tuple[int, int]]" = (1280, 720),
        setup: "Optional[Any]" = None,
    ):
        """Render the time history to a video / GIF without a GUI session.

        Re-steps one :class:`~apeGmsh.results.session.ResultsSession`
        mesh view (the ``results.session()`` picture) through an
        offscreen plotter: the scene is realized once, then each
        captured step moves the session instant and re-steps the pane
        in place. No Qt window and no event loop. Frames are encoded to
        the format chosen by ``path``'s suffix: ``.mp4`` (H.264, needs
        the ``apegmsh[animation]`` extra) or ``.gif`` (Pillow, no
        extra). A failed frame leaves no file at ``path``.

        Parameters
        ----------
        path
            Output file. Suffix selects the format (``.mp4`` / ``.gif``).
        fps
            Frames per second of the output.
        step_stride
            Capture every N-th step (plus always the last). Useful to
            keep long histories short.
        stage
            Stage id/name to animate. Defaults to this Results' own
            stage (``results.stage(...)``), else the only stage; with
            several stages and none picked it raises.
        deform
            Deformed-shape scaling. A number applies that scale to the
            ``"displacement"`` field; a ``(field, scale)`` pair selects
            another field. ``None`` (default) renders the undeformed
            mesh.
        camera
            ``"iso"`` / ``"xy"`` / ``"xz"`` / ``"yz"``, or an explicit
            value assigned to ``plotter.camera_position``. ``None``
            uses the still rule: ``xy`` for a planar model, ``iso``
            otherwise.
        window_size
            ``(width, height)`` of the rendered frames. ``None`` uses
            ``(1280, 720)``.
        setup
            Optional ``callback(plotter, view)`` invoked once after the
            first frame's scene is built and the camera applied, before
            capture. ``view`` is the session's
            :class:`~apeGmsh.results.session.MeshView`: fill a slot
            (``view.contour = Contour("displacement_x")``) and the
            pane is realized again before the first frame. Before
            X2-b (#1663) the second argument was the retired
            ``ResultsDirector``; a callback still written against it
            fails loudly.

        Returns
        -------
        pathlib.Path
            The resolved output path, or ``None`` when
            ``APEGMSH_SKIP_VIEWER`` is set or no GL context is
            available.
        """
        import os
        if os.environ.get("APEGMSH_SKIP_VIEWER"):
            print("[skip viewer] APEGMSH_SKIP_VIEWER set")
            return None
        from apeGmsh.viewers.session._animation import (
            export_session_animation,
        )

        from .session import Deform

        if stage == "__all__":
            # The director's combined stage (every stage on one
            # concatenated time axis). The session has one instant per
            # (stage, step), so there is no combined animation to draw.
            names = [s.name for s in self._all_stages()]
            raise ValueError(
                "export_animation: stage='__all__' (all stages on one "
                "time axis) is no longer supported. Animate one stage "
                f"at a time with stage=<id or name> (have: {names})."
            )
        # stage= wins, then this Results' own scope (``.stage()``), then
        # the only stage; several stages and no pick refuse and name
        # ``stage=`` (the director animated nothing there either).
        try:
            stage_id = self._resolve_stage(stage)
        except RuntimeError as exc:
            raise RuntimeError(
                f"export_animation: no stage to animate — {exc}"
            ) from exc
        session = self.session()
        view = session.panes[0]
        if deform is not None:
            if isinstance(deform, (tuple, list)):
                d_field, d_scale = deform[0], float(deform[1])
            else:
                d_field, d_scale = "displacement", float(deform)
            view.deform = Deform(field=d_field, scale=d_scale)
        return export_session_animation(
            session, view, path,
            stage_id=stage_id, fps=fps, step_stride=step_stride,
            camera=camera, window_size=window_size, setup=setup,
        )

    def show_web(
        self,
        *,
        stage: "Optional[str]" = None,
        show: bool = True,
        controls: bool = True,
        render_mode: str = "client",
    ):
        """Open the view-only web / Jupyter results viewer (ADR 0042 R-C).

        Renders the FEM substrate plus any diagrams the director holds
        through a ``pyvista.trame`` backend — the kernel-safe path that
        replaces the blocking Qt :meth:`viewer` in a notebook. View-only
        (picking is deferred to R-D), but with a step slider + per-layer
        visibility checkboxes when ``ipywidgets`` is available.

        No results file handy? ``Results.demo().show_web()`` renders a
        zero-setup cantilever-pushover sample.

        Parameters
        ----------
        stage
            Stage id or name to activate; defaults to the first stage.
        show
            When ``True`` (default), display inline immediately. When
            ``False``, return the :class:`~apeGmsh.viewers.web_viewer.WebViewer`
            unshown so diagrams can be added via ``viewer.director`` first.
        controls
            When ``True`` (default), stack an ``ipywidgets`` control panel
            (step slider + layer toggles) above the view. Degrades to a
            bare view if ``ipywidgets`` is absent.
        render_mode
            ``"client"`` (default) renders in the browser via WebGL — fast
            camera interaction. ``"server"`` renders on the kernel and
            streams images (laggy, most VTK-feature-complete; for very
            large models). ``"hybrid"`` is pyvista's ``trame`` backend with
            a local/remote toggle in the toolbar.

        Returns
        -------
        WebViewer
            The viewer handle (``.director`` / ``.set_step`` / ``.show``).
        """
        from ..viewers.web_viewer import show_web as _show_web
        return _show_web(
            self, stage=stage, show=show, controls=controls,
            render_mode=render_mode,
        )

    def serve_web(
        self,
        *,
        stage: "Optional[str]" = None,
        render_mode: str = "client",
        port: "Optional[int]" = None,
        open_browser: bool = True,
        title: str = "apeGmsh",
        **start_kwargs,
    ):
        """Serve the results as a standalone trame web app (ADR 0042 R-C).

        The non-Jupyter counterpart of :meth:`show_web`: builds a vuetify3
        single-page app (the FEM view plus a step slider and per-layer
        switches) and serves it at a local URL, opening a browser tab and
        blocking until stopped (Ctrl-C). In a notebook use :meth:`show_web`
        instead.

        Parameters
        ----------
        stage
            Stage id or name to activate; defaults to the first stage.
        render_mode
            ``"client"`` (default), ``"server"``, or ``"hybrid"`` — see
            :meth:`show_web`.
        port
            Port to serve on; ``None`` lets trame pick one.
        open_browser
            Open a browser tab at the served URL.
        title
            App title shown in the toolbar.
        **start_kwargs
            Passed through to the trame ``server.start`` (e.g.
            ``exec_mode``).

        Returns
        -------
        WebViewer
            The viewer handle.
        """
        from ..viewers.web_viewer import serve_web as _serve_web
        return _serve_web(
            self, stage=stage, render_mode=render_mode, port=port,
            open_browser=open_browser, title=title, **start_kwargs,
        )

    # ------------------------------------------------------------------
    # Custom scalar definitions (ADR 0076) — cross-process transport
    # ------------------------------------------------------------------

    def _definitions_payload(self) -> "list[dict]":
        """Serialize every composite's custom scalar registry to plain
        dicts (name / expr / domain / label / units), in definition
        order so dependencies precede dependents on replay."""
        nodes = getattr(self, "nodes", None)
        elements = getattr(self, "elements", None)
        gauss = getattr(elements, "gauss", None) if elements is not None else None
        payload: "list[dict]" = []
        for domain, comp in (("node", nodes), ("gauss", gauss)):
            if comp is None:
                continue
            for d in comp.definitions.values():
                payload.append({
                    "name": d.name, "expr": d.expr, "domain": domain,
                    "label": d.label, "units": d.units,
                })
        return payload

    def _apply_definitions_payload(self, payload: "list[dict]") -> int:
        """Re-register custom scalars from a :meth:`_definitions_payload`
        dump onto this Results' composites. Idempotent (a name already
        registered is skipped) and best-effort (a definition that no
        longer resolves — stale operand, e.g. a sidecar from a richer
        run — warns and is skipped, not fatal). Returns the count newly
        applied."""
        import warnings
        targets = {"node": self.nodes, "gauss": self.elements.gauss}
        applied = 0
        for item in payload:
            comp = targets.get(item.get("domain"))
            if comp is None:      # unknown domain from a newer writer — skip
                continue
            name = item.get("name")
            if name in comp.definitions:      # idempotent
                continue
            try:
                comp.define(
                    name, item["expr"],
                    label=item.get("label"), units=item.get("units"),
                )
                applied += 1
            except Exception as exc:          # noqa: BLE001 — best-effort overlay
                warnings.warn(
                    f"skipped custom scalar {name!r}: {exc}", stacklevel=2,
                )
        return applied

    def save_definitions(self, path: "Optional[str | Path]" = None) -> Path:
        """Persist this Results' custom scalar definitions to a JSON
        sidecar (default ``<results>.defs.json``).

        Reloaded automatically by :meth:`from_native` / :meth:`from_mpco`
        / :meth:`from_ladruno`, and carried to the subprocess viewer.
        Raises for an in-memory Results with no ``path=`` given.
        """
        import json
        if path is None:
            if self._path is None:
                raise RuntimeError(
                    "in-memory Results has no default sidecar path; "
                    "pass save_definitions(path=...)."
                )
            path = self._default_defs_path(self._path)
        path = Path(path)
        path.write_text(
            json.dumps(self._definitions_payload(), indent=2), encoding="utf-8",
        )
        return path

    def load_definitions(self, path: "Optional[str | Path]" = None) -> int:
        """Load and register custom scalar definitions from a JSON sidecar
        (default ``<results>.defs.json``). Returns the count applied; a
        missing file is a no-op returning 0. Idempotent / best-effort —
        see :meth:`_apply_definitions_payload`."""
        import json
        if path is None:
            if self._path is None:
                return 0
            path = self._default_defs_path(self._path)
        path = Path(path)
        if not path.exists():
            return 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._apply_definitions_payload(payload)

    def _with_autoloaded_definitions(self) -> "Results":
        """Best-effort auto-load of the default definitions sidecar at
        open time; returns ``self`` for chaining off a constructor. A
        malformed sidecar warns rather than failing the open."""
        try:
            self.load_definitions()
        except Exception as exc:  # noqa: BLE001 — a bad sidecar never blocks open
            import warnings
            warnings.warn(
                f"could not load custom scalar definitions sidecar: {exc}",
                stacklevel=2,
            )
        return self

    @staticmethod
    def _default_defs_path(results_path: "str | Path") -> Path:
        """Sidecar path carrying custom definitions to the subprocess —
        deliberately NOT ``<results>.viewer-session.json`` (owned by the
        diagram session machinery, overwritten on close)."""
        return Path(str(results_path) + ".defs.json")

    def _build_viewer_argv(
        self,
        *,
        title: Optional[str],
        python_exe: Optional[str] = None,
        defs_path: "Optional[Path]" = None,
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
    ) -> list[str]:
        """Build the argv for ``python -m apeGmsh.viewers``.

        Factored out of :meth:`_spawn_viewer_subprocess` so the wire
        contract between parent and child is testable without
        launching a real subprocess.  Emits ``--model-h5`` iff
        :attr:`_model_path` is set (MPCO-opened Results) — the
        child's ``__main__.py`` exits(2) on ``.mpco`` paths without
        that flag.  Emits ``--defs`` iff ``defs_path`` is supplied
        (custom scalar definitions, ADR 0076).

        ``--restore-session`` / ``--no-save-session`` carry the S6a
        open policy across the argv hop. Before the flip they did not
        exist and the child silently used the defaults, so
        ``viewer(blocking=False, save_session=False)`` saved anyway —
        a documented promise the subprocess path did not keep. Only
        the non-default values are emitted, so the common argv is
        unchanged.
        """
        import sys
        if self._path is None:
            raise RuntimeError(
                "In-memory Results cannot launch in a subprocess. Open "
                "the Results from disk via Results.from_native(path) or "
                "Results.from_mpco(path), or open the window in-process "
                "with results.viewer(blocking=True) outside a notebook "
                "kernel."
            )
        args = [
            python_exe or sys.executable,
            "-m", "apeGmsh.viewers",
            str(self._path),
        ]
        if title is not None:
            args.extend(["--title", title])
        if self._model_path is not None:
            args.extend(["--model-h5", str(self._model_path)])
        if defs_path is not None:
            args.extend(["--defs", str(defs_path)])
        if restore_session != "prompt":
            args.extend([
                "--restore-session",
                {True: "yes", False: "no"}[bool(restore_session)],
            ])
        if not save_session:
            args.append("--no-save-session")
        return args

    def _spawn_viewer_subprocess(
        self,
        *,
        title: Optional[str],
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
    ):
        """Launch ``python -m apeGmsh.viewers <path>`` and return the Popen.

        A daemon thread monitors the child's exit code and surfaces
        non-zero exits to stderr (see :func:`_monitor_subprocess`).
        Without this, a child failure (corrupt file, missing dep,
        ``sys.exit(2)`` from ``__main__``) is invisible to the
        parent — the fire-and-forget ``Popen`` swallows the failure
        entirely.  PR1 fixed the specific MPCO-without-model-h5
        case; this monitor catches the general case.
        """
        import subprocess
        defs_path: "Optional[Path]" = None
        if self._definitions_payload():
            defs_path = self.save_definitions()
        args = self._build_viewer_argv(
            title=title,
            defs_path=defs_path,
            restore_session=restore_session,
            save_session=save_session,
        )
        handle = subprocess.Popen(args)
        _start_subprocess_monitor(handle, args)
        return handle

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return self.inspect.summary()

    # ------------------------------------------------------------------
    # Internal helpers (used by composites and inspect)
    # ------------------------------------------------------------------

    def _all_stages(self) -> list[StageInfo]:
        if self._stages_cache is None:
            self._stages_cache = self._reader.stages()
        return self._stages_cache

    def _lookup_stage(self, name_or_id: str) -> StageInfo:
        """Resolve a stage by exact id, then by name, then by alias.

        Three passes, not one first-match: the viewers hand back
        ``StageInfo.id`` (``stage_<k>``), and a program whose stages
        are named ``stage_1`` / ``stage_2`` would otherwise send
        ``stage("stage_1")`` to ``stage_0`` by name (#1393).
        """
        stages = self._all_stages()
        for pick in (
            lambda s: s.id == name_or_id,
            lambda s: s.name == name_or_id,
            lambda s: name_or_id in s.aliases,
        ):
            for s in stages:
                if pick(s):
                    return s
        names = sorted({s.name for s in stages} |
                        {s.id for s in stages} |
                        {a for s in stages for a in s.aliases})
        raise KeyError(
            f"No stage matches {name_or_id!r}. Available: {names}"
        )

    def _resolve_stage(self, requested: str | None) -> str:
        """Pick the stage_id for a read.

        Resolution order:
          1. explicit ``stage=`` argument
          2. ``self._stage_id`` (set by ``.stage()`` / ``.modes``)
          3. the only stage when there is exactly one
          4. raise (multiple stages, none picked)
        """
        if requested is not None:
            return self._lookup_stage(requested).id
        if self._stage_id is not None:
            return self._stage_id
        stages = self._all_stages()
        if len(stages) == 1:
            return stages[0].id
        if not stages:
            raise RuntimeError("No stages in this results file.")
        names = [s.name for s in stages]
        raise RuntimeError(
            f"Multiple stages present ({names}). Pick one with "
            f"results.stage(name_or_id) or pass stage=."
        )

    def _require_scoped(self) -> StageInfo:
        if self._stage_id is None:
            raise AttributeError(
                "This attribute is only available on a stage-scoped Results "
                "(use results.stage(...) or results.modes[i])."
            )
        # Refresh from the cache.
        for s in self._all_stages():
            if s.id == self._stage_id:
                return s
        raise KeyError(
            f"Scoped stage_id={self._stage_id!r} not found in reader."
        )

    def _require_mode(self) -> StageInfo:
        info = self._require_scoped()
        if info.kind != "mode":
            raise AttributeError(
                f"Stage {info.id!r} has kind={info.kind!r}, not 'mode'. "
                f"This attribute is only available on mode-scoped Results."
            )
        return info

    def _reader_path(self) -> str:
        return str(self._path) if self._path else "(in-memory)"

    def _derive(
        self,
        *,
        fem=_SENTINEL,
        stage_id=_SENTINEL,
        model=_SENTINEL,
    ) -> "Results":
        """Create a copy of this Results with a few fields overridden.

        Sharing the underlying reader (and stages cache) avoids
        re-opening the file or re-listing stages. Propagates
        ``_model`` through stage / mode derivation so
        ``results.stage("grav").model`` returns the same broker as
        ``results.model``.
        """
        new = Results.__new__(Results)
        new._reader = self._reader
        new._fem = self._fem if isinstance(fem, _Sentinel) else fem
        # A supplied fem (``bind``) clears the D9 refusal; stage / mode
        # derivation carries it.
        new._fem_unavailable = (
            self._fem_unavailable if isinstance(fem, _Sentinel) else None
        )
        new._stage_id = (
            self._stage_id if isinstance(stage_id, _Sentinel) else stage_id
        )
        new._path = self._path
        new._model = (
            self._model if isinstance(model, _Sentinel) else model
        )
        new._model_path = self._model_path
        new._stages_cache = self._stages_cache
        new._units_override = getattr(self, "_units_override", None)
        new._source_units_override = getattr(self, "_source_units_override", None)
        new._base_fem = (
            getattr(self, "_base_fem", self._fem)
            if isinstance(fem, _Sentinel)
            else fem
        )
        new.nodes = NodeResultsComposite(new)
        new.elements = ElementResultsComposite(new)
        new.inspect = ResultsInspect(new)
        new._plot = None
        # Custom scalar definitions (ADR 0076) follow stage / mode
        # derivation — a scalar defined on ``results`` is readable on
        # ``results.stage("grav")`` too. ExprDefs are frozen, so a shallow
        # copy of the registry dict is enough.
        new.nodes._registry = dict(self.nodes._registry)
        new.elements.gauss._registry = dict(self.elements.gauss._registry)
        return new
