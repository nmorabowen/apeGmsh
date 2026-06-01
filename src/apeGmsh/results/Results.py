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
from typing import TYPE_CHECKING, Any, Optional

from numpy import ndarray

from ._bind import _resolve_fem, resolve_bound_model
from ._composites import (
    ElementResultsComposite,
    NodeResultsComposite,
)
from ._inspect import ResultsInspect
from .readers._protocol import EigenMode, ResultsReader, StageInfo

if TYPE_CHECKING:
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
    "model= is required. Pass model=OpenSeesModel.from_h5(...) at "
    "construction. Required since the Phase 8 prune of the major "
    "architectural refactor."
)


_MODEL_H5_REQUIRED_MESSAGE = (
    "model_h5= is required. Pass model_h5='model.h5' (sibling model "
    "archive) at construction. Required since the Phase 8 prune of "
    "the major architectural refactor."
)


def _model_is_composed(model: Any) -> bool:
    """True when the bound model carries compose provenance (ADR 0038).

    Read-time compose-detection (ADR 0043 slice 1.3): the fem_eid↔ops-tag
    relabel only applies to composed models, where base-tag offsets push
    ``fem_eid`` off the allocator-assigned ops tag.  Defensive — any
    failure to reach ``model.fem.composed_from`` is treated as not
    composed (no relabel).
    """
    try:
        return len(model.fem.composed_from) > 0
    except Exception:
        return False


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

        ``model_path`` records the on-disk archive the ``model`` was read
        from, for when it is *not* ``path`` itself — e.g. results whose
        embedded ``/model`` zone is not independently readable. The
        non-blocking subprocess viewer forwards it as ``--model-h5`` so the
        child re-reads the model from there instead of from ``path``.
        """
        if model is None:
            raise TypeError(_MODEL_REQUIRED_MESSAGE)
        from .readers._native import NativeReader
        reader = NativeReader(path)
        bound_fem = _resolve_fem(reader, fem)
        bound_model = resolve_bound_model(reader, model)
        # ``resolve_bound_model`` always returns ``model`` here since
        # we just asserted it is non-None, but route through the helper
        # to keep the resolution semantics in one place.
        assert bound_model is not None
        return cls(
            reader, fem=bound_fem, path=Path(path), model=bound_model,
            model_path=Path(model_path) if model_path is not None else None,
        )

    @classmethod
    def from_recorders(
        cls,
        spec,
        output_dir: str | Path,
        *,
        fem: "FEMData",
        cache_root: str | Path | None = None,
        stage_name: str = "analysis",
        stage_kind: str = "transient",
        file_format: str = "out",
        stage_id: str | None = None,
        model: "Optional[OpenSeesModel]" = None,
    ) -> "Results":
        """Open the result of an OpenSees run driven by Tcl/Py recorders.

        Phase 8 (ADR 0020 INV-1) — ``model=`` is required. Missing
        supply raises :class:`TypeError`. The model's ``/opensees/``
        zone is embedded into the transcoded native h5 (the
        Composed-file pattern); downstream
        :meth:`Results.from_native` then auto-resolves the broker
        from the same file.

        Parses the ``.out`` / ``.xml`` files emitted at
        ``output_dir`` (matching what ``spec.emit_recorders(...)`` or
        the apeGmsh OpenSees bridge's Tcl/Py emit produced) into an
        apeGmsh native HDF5, caches the result at
        ``cache_root``, and opens it through ``NativeReader``.

        Caching: subsequent calls with unchanged input files return
        the cached HDF5 directly (file mtime + size + spec
        ``snapshot_id`` form the cache key). See
        ``writers/_cache.py``.

        ``stage_id`` matches the per-stage filename prefix used by
        :meth:`ResolvedRecorderSpec.emit_recorders` together with
        ``begin_stage(stage_id, ...)``. When set, only files prefixed
        with ``<stage_id>__`` are read; ``stage_name`` defaults to
        ``stage_id`` if not overridden. ``None`` (default) keeps the
        legacy flat-naming used by Tcl/Py exports.

        Phase 6 v1 supports nodal records only; element-level records
        in the spec are skipped with a note. The capture flow
        (Phase 7) handles modal recorders.
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
        if stage_id is not None and stage_name == "analysis":
            stage_name = stage_id

        out_dir = Path(output_dir)
        cache_dir = _cache.resolve_cache_root(cache_root)

        source_files = _cache.list_source_files(
            spec, out_dir, file_format=file_format, stage_id=stage_id,
        )
        key = _cache.compute_cache_key(
            source_files,
            parser_version=PARSER_VERSION,
            fem_snapshot_id=fem.snapshot_id,
        )
        cached_h5, _ = _cache.cache_paths(cache_dir, key)

        if not cached_h5.exists():
            # Materialise the model's ``/opensees/`` zone alongside the
            # transcoded results.  ``OpenSeesModel.to_h5`` (the public,
            # schema-authority-respecting writer) shapes the source;
            # NativeWriter copies the ``/opensees/`` group at open
            # time (Composed-file pattern, ADR 0020 INV-3 preserved).
            model_h5_src: Path = cached_h5.with_suffix(".model.h5")
            model.to_h5(model_h5_src)
            transcoder = RecorderTranscoder(
                spec, out_dir, cached_h5, fem,
                stage_name=stage_name,
                stage_kind=stage_kind,
                file_format=file_format,
                stage_id=stage_id,
                model_h5_src=model_h5_src,
            )
            transcoder.run()

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

        Single-file mode (default for non-partitioned analyses): pass
        the path of one ``.mpco`` file. Synthesizes a partial FEMData
        from the MPCO ``MODEL/`` group if ``fem`` is omitted.

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
        bound_fem = _resolve_fem(reader, fem)
        # Per INV-3, this is an in-memory rehydrate from the sibling
        # file; we never copy the zone into a derived h5.
        from ..opensees.opensees_model import OpenSeesModel
        bound_model = OpenSeesModel.from_h5(model_h5)
        # ADR 0043 slice 1.3 — MPCO buckets key element results by the
        # OpenSees ops tag; over a composed model (ADR 0038) that differs
        # from the fem_eid the results API speaks, so the reader must
        # relabel ops↔fem through the model's element_meta. We attach the
        # translator ONLY for a composed model: that is the only case the
        # gap exists (an uncomposed model's ops tag == fem_eid by
        # allocator construction, so translation would be a no-op), and
        # gating on compose-provenance is the read-time compose-detection
        # that lets a deliberately-unrelated stub ``model_h5=`` (common in
        # tests) pass through untranslated rather than be mis-relabelled.
        # (A non-composed but sparsely-renumbered model also has
        # ops != fem_eid; that case is a deferred follow-up — see ADR 0043
        # §slice 1.3.)
        if _model_is_composed(bound_model):
            from .readers._tag_translation import ElementTagTranslator
            reader.attach_tag_map(
                ElementTagTranslator.from_model(bound_model),
            )
        return cls(
            reader, fem=bound_fem, path=anchor, model=bound_model,
            model_path=Path(model_h5),
        )

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
          transcode.
        * supplied → the richer broker is loaded via
          :meth:`OpenSeesModel.from_h5` (full bridge records + lineage),
          and — for a composed model (ADR 0043) — the fem_eid↔ops-tag
          translator is attached.

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
        bound_fem = _resolve_fem(reader, fem)
        bound_model: "Optional[OpenSeesModel]"
        if model_h5 is not None:
            from ..opensees.opensees_model import OpenSeesModel
            bound_model = OpenSeesModel.from_h5(model_h5)
            if _model_is_composed(bound_model):
                from .readers._tag_translation import ElementTagTranslator
                reader.attach_tag_map(
                    ElementTagTranslator.from_model(bound_model),
                )
            model_path: "Optional[Path]" = Path(model_h5)
        else:
            # Self-sufficient path — minimal broker from the file itself.
            bound_model = resolve_bound_model(reader, None)
            model_path = None
        assert bound_model is not None
        return cls(
            reader, fem=bound_fem, path=anchor, model=bound_model,
            model_path=model_path,
        )

    # ------------------------------------------------------------------
    # FEM access & binding
    # ------------------------------------------------------------------

    @property
    def fem(self) -> "Optional[FEMData]":
        """The bound FEMData snapshot, or None if not bound."""
        return self._fem

    @property
    def model(self) -> "OpenSeesModel":
        """The bound :class:`OpenSeesModel` broker.

        Phase 8 (ADR 0020 INV-1) — always non-None on a constructed
        :class:`Results`.  The chain-forward handle from which the
        OpenSeesModel and its embedded FEMData can be reached.
        """
        return self._model

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
                warnings.append(
                    WARNING_PREFIX
                    + "model hash mismatch between Results and OpenSeesModel: "
                    f"results-file stamp {stored_model!r} != "
                    f"OpenSeesModel.lineage.model_hash {model_hash!r}"
                )
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

    def bind(self, fem: "FEMData") -> "Results":
        """Re-bind to ``fem``.

        Useful when you've re-built the same mesh in a fresh session
        and want labels / Parts that the embedded snapshot doesn't
        carry. No hash validation is performed — pairing the FEMData
        with a results file from the same run is the user's
        responsibility.
        """
        bound = _resolve_fem(self._reader, fem)
        return self._derive(fem=bound)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    @property
    def stages(self) -> list[StageInfo]:
        """All stages in the file (scoped instances also list them)."""
        return list(self._all_stages())

    def stage(self, name_or_id: str) -> "Results":
        """Return a Results scoped to a stage (matched by id or name)."""
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
    # Viewer
    # ------------------------------------------------------------------

    def viewer(
        self,
        *,
        blocking: bool = True,
        title: Optional[str] = None,
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
        cuts: "Optional[Any]" = None,
    ):
        """Open the post-solve results viewer.

        Parameters
        ----------
        blocking
            ``True`` (default) — open the viewer in-process and block
            the calling thread until the window closes. Matches the
            signature of :meth:`g.mesh.viewer` and :meth:`g.model.viewer`.
            ``False`` — spawn a subprocess via
            ``python -m apeGmsh.viewers <path>`` so the notebook /
            kernel can keep running. Requires that the Results was
            opened from disk (``self._path`` is set); raises
            :class:`RuntimeError` for in-memory Results.
        title
            Optional window title; defaults to ``"Results — <filename>"``.
        restore_session
            How to handle a previously-saved session JSON next to the
            results file. ``True`` restores silently, ``False`` ignores,
            ``"prompt"`` (default) opens a yes/no dialog if a matching
            session exists. No effect for in-memory Results.
        save_session
            If ``True`` (default), the active set of diagrams + scrubber
            position is saved to ``<results>.viewer-session.json`` when
            the window closes. ``False`` disables auto-save.
        cuts
            Optional sequence of :class:`apeGmsh.cuts.SectionCutDef`
            instances to render as Layers at boot. Each cut becomes a
            new ``SectionCutDiagram`` in the active geometry's
            ``"Section cuts"`` composition (created if absent).
            Subprocess launches (``blocking=False``) currently ignore
            this argument; build cuts programmatically via
            ``director.add_section_cut`` on the spawned viewer once it
            exposes IPC.

        Returns
        -------
        ResultsViewer
            The viewer instance after the window closes (blocking).
        subprocess.Popen
            The spawned process handle (non-blocking).
        None
            If ``APEGMSH_SKIP_VIEWER`` is set in the environment. This
            lets the same cell run under ``jupyter nbconvert --execute``
            or in CI without spawning a GUI window.
        """
        import os
        if os.environ.get("APEGMSH_SKIP_VIEWER"):
            print("[skip viewer] APEGMSH_SKIP_VIEWER set")
            return None
        if not blocking:
            handle = self._spawn_viewer_subprocess(title=title)
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
        from ..viewers.results_viewer import ResultsViewer
        return ResultsViewer(
            self,
            title=title,
            restore_session=restore_session,
            save_session=save_session,
            cuts=cuts,
        ).show()

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

    def _build_viewer_argv(
        self,
        *,
        title: Optional[str],
        python_exe: Optional[str] = None,
    ) -> list[str]:
        """Build the argv for ``python -m apeGmsh.viewers``.

        Factored out of :meth:`_spawn_viewer_subprocess` so the wire
        contract between parent and child is testable without
        launching a real subprocess.  Emits ``--model-h5`` iff
        :attr:`_model_path` is set (MPCO-opened Results) — the
        child's ``__main__.py`` exits(2) on ``.mpco`` paths without
        that flag.
        """
        import sys
        if self._path is None:
            raise RuntimeError(
                "In-memory Results cannot launch in a subprocess. Open "
                "the Results from disk via Results.from_native(path) or "
                "Results.from_mpco(path), or call results.viewer() with "
                "the default blocking=True."
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
        return args

    def _spawn_viewer_subprocess(
        self,
        *,
        title: Optional[str],
    ):
        """Launch ``python -m apeGmsh.viewers <path>`` and return the Popen.

        ``cuts=`` is *not* forwarded (live ``SectionCutDef`` objects
        don't survive an argv hop; that needs real IPC — separate
        future work, see ``viewer()`` docstring).

        A daemon thread monitors the child's exit code and surfaces
        non-zero exits to stderr (see :func:`_monitor_subprocess`).
        Without this, a child failure (corrupt file, missing dep,
        ``sys.exit(2)`` from ``__main__``) is invisible to the
        parent — the fire-and-forget ``Popen`` swallows the failure
        entirely.  PR1 fixed the specific MPCO-without-model-h5
        case; this monitor catches the general case.
        """
        import subprocess
        args = self._build_viewer_argv(title=title)
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
        for s in self._all_stages():
            if s.id == name_or_id or s.name == name_or_id:
                return s
        names = sorted({s.name for s in self._all_stages()} |
                        {s.id for s in self._all_stages()})
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
        new._stage_id = (
            self._stage_id if isinstance(stage_id, _Sentinel) else stage_id
        )
        new._path = self._path
        new._model = (
            self._model if isinstance(model, _Sentinel) else model
        )
        new._model_path = self._model_path
        new._stages_cache = self._stages_cache
        new.nodes = NodeResultsComposite(new)
        new.elements = ElementResultsComposite(new)
        new.inspect = ResultsInspect(new)
        new._plot = None
        return new
