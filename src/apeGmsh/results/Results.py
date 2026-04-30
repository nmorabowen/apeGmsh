"""Results — backend-agnostic FEM post-processing container.

Mirrors the ``FEMData`` composite shape: ``results.nodes``,
``results.elements.gauss``, ``results.elements.fibers`` etc., with
the same ``pg=`` / ``label=`` / ``ids=`` selection vocabulary.

Construction
------------
::

    # Native HDF5 (apeGmsh-produced or domain capture)
    results = Results.from_native("run.h5")
    results = Results.from_native("run.h5", fem=fem)   # explicit bind

    # MPCO HDF5 (Phase 3)
    results = Results.from_mpco("run.mpco")

    # Recorder transcoder (Phase 6)
    results = Results.from_recorders(spec, output_dir="out/", fem=fem)

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
validates the candidate's hash matches and swaps it in (useful when
re-using session-side labels and Parts).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from numpy import ndarray

from ._bind import resolve_bound_fem
from ._composites import (
    ElementResultsComposite,
    NodeResultsComposite,
)
from ._inspect import ResultsInspect
from .readers._protocol import ResultsReader, StageInfo

if TYPE_CHECKING:
    from ..mesh.FEMData import FEMData


# Sentinel for ``Results._derive`` so we can distinguish
# "not passed" from "None" in the fem= / stage_id= overrides.
class _Sentinel:
    pass


_SENTINEL = _Sentinel()


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
    ) -> None:
        self._reader = reader
        self._fem = fem
        self._stage_id = stage_id
        self._path = path
        self._stages_cache: Optional[list[StageInfo]] = None

        # Composites
        self.nodes = NodeResultsComposite(self)
        self.elements = ElementResultsComposite(self)
        self.inspect = ResultsInspect(self)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_native(
        cls,
        path: str | Path,
        *,
        fem: "Optional[FEMData]" = None,
    ) -> "Results":
        """Open an apeGmsh native HDF5 results file.

        If ``fem`` is omitted, the embedded ``/model/`` snapshot is
        used as the bound FEMData. If ``fem`` is provided and the file
        embeds a snapshot, the two ``snapshot_id`` hashes must match.
        """
        from .readers._native import NativeReader
        reader = NativeReader(path)
        bound_fem = resolve_bound_fem(reader, fem)
        return cls(reader, fem=bound_fem, path=Path(path))

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
    ) -> "Results":
        """Open the result of an OpenSees run driven by Tcl/Py recorders.

        Parses the ``.out`` / ``.xml`` files emitted at
        ``output_dir`` (matching what ``g.opensees.export.tcl(...,
        recorders=spec)`` produced) into an apeGmsh native HDF5,
        caches the result at ``cache_root``, and opens it through
        ``NativeReader``.

        Caching: subsequent calls with unchanged input files return
        the cached HDF5 directly (file mtime + size + spec
        ``snapshot_id`` form the cache key). See
        ``writers/_cache.py``.

        Phase 6 v1 supports nodal records only; element-level records
        in the spec are skipped with a note. The capture flow
        (Phase 7) handles modal recorders.
        """
        from .schema._versions import PARSER_VERSION
        from .transcoders import RecorderTranscoder
        from .writers import _cache

        if fem is None:
            raise TypeError(
                "Results.from_recorders(...) requires fem= "
                "(the spec's snapshot_id must match)."
            )

        out_dir = Path(output_dir)
        cache_dir = _cache.resolve_cache_root(cache_root)

        source_files = _cache.list_source_files(
            spec, out_dir, file_format=file_format,
        )
        key = _cache.compute_cache_key(
            source_files,
            parser_version=PARSER_VERSION,
            fem_snapshot_id=fem.snapshot_id,
        )
        cached_h5, _ = _cache.cache_paths(cache_dir, key)

        if not cached_h5.exists():
            transcoder = RecorderTranscoder(
                spec, out_dir, cached_h5, fem,
                stage_name=stage_name,
                stage_kind=stage_kind,
                file_format=file_format,
            )
            transcoder.run()

        return cls.from_native(cached_h5, fem=fem)

    @classmethod
    def from_mpco(
        cls,
        path: "str | Path | list[str | Path]",
        *,
        fem: "Optional[FEMData]" = None,
        merge_partitions: bool = True,
    ) -> "Results":
        """Open a STKO ``.mpco`` HDF5 results file.

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
        bound_fem = resolve_bound_fem(reader, fem)
        return cls(reader, fem=bound_fem, path=anchor)

    # ------------------------------------------------------------------
    # FEM access & binding
    # ------------------------------------------------------------------

    @property
    def fem(self) -> "Optional[FEMData]":
        """The bound FEMData snapshot, or None if not bound."""
        return self._fem

    def bind(self, fem: "FEMData") -> "Results":
        """Re-bind to ``fem`` after validating ``snapshot_id`` matches.

        Useful when you've re-built the same mesh in a fresh session
        and want labels / Parts that the embedded snapshot doesn't
        carry. Raises ``BindError`` on hash mismatch.
        """
        bound = resolve_bound_fem(self._reader, fem)
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
    # Viewer
    # ------------------------------------------------------------------

    def viewer(
        self,
        *,
        blocking: bool = True,
        title: Optional[str] = None,
        restore_session: "bool | str" = "prompt",
        save_session: bool = True,
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

        Returns
        -------
        ResultsViewer
            The viewer instance after the window closes (blocking).
        subprocess.Popen
            The spawned process handle (non-blocking).
        """
        if not blocking:
            return self._spawn_viewer_subprocess(title=title)
        from ..viewers.results_viewer import ResultsViewer
        return ResultsViewer(
            self,
            title=title,
            restore_session=restore_session,
            save_session=save_session,
        ).show()

    def _spawn_viewer_subprocess(self, *, title: Optional[str]):
        """Launch ``python -m apeGmsh.viewers <path>`` and return the Popen."""
        import subprocess
        import sys
        if self._path is None:
            raise RuntimeError(
                "In-memory Results cannot launch in a subprocess. Open "
                "the Results from disk via Results.from_native(path) or "
                "Results.from_mpco(path), or call results.viewer() with "
                "the default blocking=True."
            )
        args = [sys.executable, "-m", "apeGmsh.viewers", str(self._path)]
        if title is not None:
            args.extend(["--title", title])
        return subprocess.Popen(args)

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
    ) -> "Results":
        """Create a copy of this Results with a few fields overridden.

        Sharing the underlying reader (and stages cache) avoids
        re-opening the file or re-listing stages.
        """
        new = Results.__new__(Results)
        new._reader = self._reader
        new._fem = self._fem if isinstance(fem, _Sentinel) else fem
        new._stage_id = (
            self._stage_id if isinstance(stage_id, _Sentinel) else stage_id
        )
        new._path = self._path
        new._stages_cache = self._stages_cache
        new.nodes = NodeResultsComposite(new)
        new.elements = ElementResultsComposite(new)
        new.inspect = ResultsInspect(new)
        return new
