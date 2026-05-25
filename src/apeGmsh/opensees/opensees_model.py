"""
``OpenSeesModel`` — read-side broker for one ``model.h5`` archive.

ADR 0019 introduces a third role on the OpenSees side that is
distinct from both :class:`apeSees` (full bridge, write-only) and
:class:`ModelData` (vanilla side-feeder, orientation-only).
:class:`OpenSeesModel` reads an archived ``model.h5`` into a frozen
Python object carrying typed record collections and an embedded
:class:`FEMData`.

Surface (per ADR 0019 §Decision):

.. code-block:: python

    om = OpenSeesModel.from_h5(path)                # rehydrate
    om.to_h5(path)                                   # round-trip
    om.build(target)                                 # re-emit:
                                                     # 'tcl' | 'py' | 'live' | 'h5'

    om.fem                                           # embedded FEMData
    om.materials() / om.sections() / om.transforms() / om.beam_integration()
    om.time_series() / om.patterns() / om.recorders() / om.cuts() / om.sweeps()

    om.lineage                                       # ADR 0021 lineage chain
    om.snapshot_id                                   # composite model hash

Invariants honored
==================

**INV-1.**  :class:`apeSees` does NOT gain ``from_h5`` — the read side
lives in *this* class.  Importing :mod:`apeGmsh.opensees.apesees`
never reaches this module.

**INV-2.**  :class:`OpenSeesModel` is not a unification of
:class:`apeSees` and :class:`ModelData`.  Three roles, three classes;
the write-side asymmetry (full bridge vs orientation-only enrichment)
stays.  This class is read-mostly; :meth:`build` re-emits the
rehydrated record graph but introduces no new mutable surface.

**INV-3.**  No ``h5py`` write surface on this class.  :meth:`to_h5`
delegates to :func:`apeGmsh.opensees._internal.compose._compose_model_h5`,
which owns the schema authority.  Enforced by an AST scan in the
companion test suite.

**INV-4.**  :class:`FEMData` is lazy-imported.  Import-time graph for
:mod:`apeGmsh.opensees` gains no eager edge to
:mod:`apeGmsh.mesh`; the embedded ``FEMData`` is bound inside
:meth:`from_h5`.

**INV-5.**  ``build('live')`` (and the other re-emit targets) may
produce tag identity that diverges from a fresh
``apeSees(fem).run()``.  The bridge's :class:`TagAllocator`
allocations are lost across H5 round-trip; this class replays the
stored tags exactly, which matches the bridge's first run but is not
guaranteed to match a subsequent fresh build.  Downstream tooling
that depends on bridge-time tag stability must capture the
:class:`BuiltModel` from :meth:`apeSees.build` directly and avoid the
H5 round-trip.

See also
========

- :doc:`/architecture/decisions/0019-opensees-model-read-side-broker`
- :doc:`/architecture/decisions/0018-modeldata-vanilla-opensees-enrichment`
- :doc:`/architecture/decisions/0011-h5-as-fourth-emit-target`
- :doc:`/architecture/decisions/0020-results-carries-opensees-model`
- :doc:`/architecture/decisions/0021-lineage-chain-replaces-snapshot-id`
- :mod:`apeGmsh.opensees._internal.typed_records` — the record types
  this class exposes through its accessors.
- :mod:`apeGmsh.opensees._internal.compose` — the single composer +
  the new ``_replay_into`` helper used by :meth:`build`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from ._internal.lineage import Lineage
from ._internal.typed_records import (
    BeamIntegrationRecord,
    DeclContext,
    ElementRecord,
    FixRecord,
    MassRecord,
    MaterialRecord,
    PatternRecord,
    RecorderRecord,
    SectionComplexRecord,
    SectionSimpleRecord,
    TimeSeriesRecord,
    TransformRecord,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from apeGmsh.cuts import SectionCutDef, SectionSweepDef
    from apeGmsh.mesh.FEMData import FEMData

    from .emitter.h5 import H5Emitter


__all__ = ["OpenSeesModel"]


# Type alias for the read-side section discriminated union.
SectionRecord = SectionSimpleRecord | SectionComplexRecord


@dataclass(frozen=True, slots=True)
class OpenSeesModel:
    """Read-side broker for one ``model.h5`` archive.

    Construct via :meth:`from_h5` (loads an archived file) or
    :meth:`from_compose_buffers` (used by the bridge to materialise
    an instance from its just-built buffers without a
    write-then-read round trip).  The class is frozen: every record
    collection is exposed as a read-only view, and there is no
    public setter for any field.

    The :meth:`build` method re-emits the rehydrated graph to a
    target ('tcl' / 'py' / 'live' / 'h5' / a path).  Tag identity
    may diverge from a fresh :meth:`apeSees.run` for the same FEM —
    see ADR 0019 INV-5.
    """

    # -- Stored fields (private; surface via accessors / properties) -----
    _fem: "FEMData"
    _model_name: str
    _ndm: int
    _ndf: int
    _snapshot_id: str
    _materials_by_family: Mapping[str, tuple[MaterialRecord, ...]]
    _sections: tuple[SectionRecord, ...]
    _transforms: tuple[TransformRecord, ...]
    _beam_integration: tuple[BeamIntegrationRecord, ...]
    _time_series: tuple[TimeSeriesRecord, ...]
    _elements: tuple[ElementRecord, ...]
    _fixes: tuple[FixRecord, ...]
    _masses: tuple[MassRecord, ...]
    _patterns: tuple[PatternRecord, ...]
    _recorders: tuple[RecorderRecord, ...]
    _analysis_attrs: Mapping[str, Any]
    _analyze_call: "tuple[int, float | None] | None"
    _cuts: tuple["SectionCutDef", ...]
    _sweeps: tuple["SectionSweepDef", ...]
    _lineage: Lineage = field(default_factory=Lineage)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        path: str | Path,
        *,
        fem_root: str = "/",
        opensees_root: str = "/opensees",
    ) -> "OpenSeesModel":
        """Load an archived ``model.h5`` into a typed broker.

        Reads the neutral zone via :meth:`FEMData.from_h5` and the
        ``/opensees/...`` zone via the typed accessors on
        :class:`H5Model`.  Cuts and sweeps come from
        :func:`apeGmsh.cuts._h5_io.read_cuts_and_sweeps`.

        Parameters
        ----------
        path
            File system path to the ``model.h5`` archive.
        fem_root
            Sub-group root for the embedded :class:`FEMData` rich
            neutral zone.  Default ``"/"`` rehydrates from a
            standalone ``model.h5``.  Per ADR 0020 (Phase 4 cleanup),
            composed ``results.h5`` files carry the same rich layout
            under ``/model/``; pass ``fem_root="/model"`` together
            with ``opensees_root="/opensees"`` to rehydrate from a
            composed file.
        opensees_root
            HDF5 path to the bridge ``/opensees`` zone.  Default
            ``"/opensees"`` works for both standalone model.h5 files
            and composed results.h5 files because the composed file
            copies the bridge zone verbatim at root (no nested
            sub-namespace).  Surfaced so callers can override for
            non-default layouts.

        Raises
        ------
        FileNotFoundError
            ``path`` does not exist.
        apeGmsh.opensees.emitter.h5_reader.SchemaVersionError
            Schema major mismatch.
        apeGmsh.opensees.emitter.h5_reader.MalformedH5Error
            ``/meta`` is missing or unparseable.
        """
        # Lazy FEMData import — ADR 0019 INV-4 (no module-level edge
        # from ``apeGmsh.opensees`` to ``apeGmsh.mesh``).
        from apeGmsh.mesh.FEMData import FEMData

        from apeGmsh.cuts._h5_io import read_cuts_and_sweeps

        from .emitter import h5_reader

        spath = str(path)
        fem = FEMData.from_h5(spath, root=fem_root)

        # Locate the bridge meta — it lives at ``{fem_root}/meta`` so
        # ``h5_reader.open`` can validate the bridge schema major at
        # the right group (the file root ``/meta`` of a composed
        # results.h5 is the results envelope, not the bridge meta).
        if fem_root in ("", "/"):
            meta_path = "meta"
        else:
            meta_path = f"{fem_root.strip('/')}/meta"

        cuts, sweeps = read_cuts_and_sweeps(spath, meta_path=meta_path)

        with h5_reader.open(spath, meta_path=meta_path) as model:
            meta = model.meta()
            model_name = str(meta.get("model_name", "model"))
            # Broker-stamped ``/meta.ndm`` reflects element-type
            # dimensionality, which can be less than the bridge's
            # spatial ndm (a 3-D frame composed of 1-D line elements
            # has broker ndm=1 but bridge ndm=3).  For ``build()`` we
            # need the bridge's spatial ndm so the re-emitted deck
            # validates against ``ops.model(ndm=N, ndf=ndf)``.
            # ADR 0019 INV-5: tag identity may diverge across
            # round-trip; this ndm-inference branch documents that the
            # spatial dimension also has to be reconstituted.  Phase 6
            # (ADR 0021 lineage) is the right place to surface this
            # explicitly; Phase 3 derives via the transform vecxz
            # vector length (the bridge writes a vecxz of length 3 in
            # 3D and length 0 in pure 2D — a non-empty vecxz with N
            # components implies ndm >= N).
            ndm = max(
                int(meta.get("ndm", 0)),
                _infer_ndm_from_transforms(model.handle),
            )
            ndf = int(meta.get("ndf", 0))
            snapshot_id = str(meta.get("snapshot_id", ""))

            materials_by_family = model.materials_by_family()
            sections = tuple(model.sections())
            transforms = tuple(model.transforms())
            beam_integration = tuple(model.beam_integration())
            time_series = tuple(model.time_series())
            patterns = tuple(model.patterns())
            recorders = tuple(model.recorders())

            elements = cls._load_elements(model)
            fixes, masses = cls._load_bcs(model)
            analysis_attrs, analyze_call = cls._load_analysis(model)

        # ADR 0021 lineage chain.  Read the stamped ``/meta/lineage``
        # attrs and recompute them from the loaded zones; on mismatch,
        # surface drift warnings — never raise (INV-2).
        lineage = _resolve_lineage(spath, fem, fem_root)

        # S2 (ADR 0033): the stored envelope must cover every
        # per-node declaration in the rehydrated FEM.  Catches
        # hand-corrupted H5 files that claim ``ndf=6`` on a node in
        # an ``ndf=3`` envelope.
        from ._internal.build import validate_envelope_covers_broker_ndf
        validate_envelope_covers_broker_ndf(fem, ndf)

        return cls(
            _fem=fem,
            _model_name=model_name,
            _ndm=ndm,
            _ndf=ndf,
            _snapshot_id=snapshot_id,
            _materials_by_family=MappingProxyType(
                {fam: tuple(recs) for fam, recs in materials_by_family.items()}
            ),
            _sections=sections,
            _transforms=transforms,
            _beam_integration=beam_integration,
            _time_series=time_series,
            _elements=elements,
            _fixes=fixes,
            _masses=masses,
            _patterns=patterns,
            _recorders=recorders,
            _analysis_attrs=MappingProxyType(dict(analysis_attrs)),
            _analyze_call=analyze_call,
            _cuts=tuple(cuts),
            _sweeps=tuple(sweeps),
            _lineage=lineage,
        )

    @classmethod
    def from_compose_buffers(
        cls,
        fem: "FEMData",
        emitter: "H5Emitter",
        *,
        snapshot_id: str,
        cuts: "Sequence[SectionCutDef]" = (),
        sweeps: "Sequence[SectionSweepDef]" = (),
    ) -> "OpenSeesModel":
        """Materialise an :class:`OpenSeesModel` from a populated
        :class:`H5Emitter`'s buffers.

        Used by :meth:`apeSees.h5` (and by future
        :meth:`apeSees.compose_model` flows) to publish an
        :class:`OpenSeesModel` for the just-built file without doing
        a write-then-read round-trip — the buffers already carry the
        same typed records this class wraps.

        Parameters mirror the emitter's accumulator state directly;
        the helper builds the immutable views.
        """
        materials_by_family: dict[str, tuple[MaterialRecord, ...]] = {}
        if emitter._uniaxial:
            materials_by_family["uniaxial"] = tuple(emitter._uniaxial)
        if emitter._nd:
            materials_by_family["nd"] = tuple(emitter._nd)
        sections: tuple[SectionRecord, ...] = (
            tuple(emitter._sections_simple) + tuple(emitter._sections_complex)
        )

        # ADR 0021 lineage — for the in-memory compose path the
        # broker's own ``snapshot_id`` is authoritative (no h5 round-
        # trip yet).  ``model_hash`` carries the bridge-stamped value
        # opaquely; Phase 6 doesn't recompute over the in-flight
        # emitter buffers (the canonical-bytes contract is on disk).
        try:
            fem_hash = str(fem.snapshot_id)
        except Exception:
            fem_hash = ""
        lineage = Lineage(
            fem_hash=fem_hash,
            model_hash=snapshot_id or None,
        )

        # S2 (ADR 0033): the envelope ``ndf`` carried on the emitter
        # buffers must cover every per-node declaration on the broker
        # FEM.  ``apeSees.model()`` already runs the same check at
        # call time, but ``from_compose_buffers`` is reachable from
        # programmatic compose flows that bypass ``apeSees.model``;
        # rechecking here is cheap and guarantees the published
        # broker is consistent.
        from ._internal.build import validate_envelope_covers_broker_ndf
        validate_envelope_covers_broker_ndf(fem, int(emitter._ndf or 0))

        return cls(
            _fem=fem,
            _model_name=emitter._model_name,
            _ndm=int(emitter._ndm or 0),
            _ndf=int(emitter._ndf or 0),
            _snapshot_id=snapshot_id,
            _materials_by_family=MappingProxyType(materials_by_family),
            _sections=sections,
            _transforms=tuple(emitter._transforms),
            _beam_integration=tuple(emitter._beam_integrations),
            _time_series=tuple(emitter._time_series),
            _elements=tuple(emitter._elements),
            _fixes=tuple(emitter._fixes),
            _masses=tuple(emitter._masses),
            _patterns=tuple(emitter._patterns_complete),
            _recorders=tuple(emitter._recorders),
            _analysis_attrs=MappingProxyType(dict(emitter._analysis_attrs)),
            _analyze_call=emitter._analyze_call,
            _cuts=tuple(cuts),
            _sweeps=tuple(sweeps),
            _lineage=lineage,
        )

    # ------------------------------------------------------------------
    # Output — write / build
    # ------------------------------------------------------------------

    def to_h5(self, path: str | Path) -> None:
        """Write the broker contents back to a ``model.h5`` file.

        Delegates to the shared
        :func:`apeGmsh.opensees._internal.compose._compose_model_h5`
        composer (the same one :meth:`apeSees.h5` and
        :meth:`ModelData.write` use).  No ``h5py`` write surface
        lives on this class — INV-3.

        Output is byte-equivalent to the source ``model.h5``
        (modulo ``/meta/created_iso``, which the bridge always
        re-stamps with the current UTC timestamp).

        Parameters
        ----------
        path
            Destination path.
        """
        from .emitter.h5 import H5Emitter

        emitter = self._build_h5_emitter()
        self._compose_h5(emitter, str(path))

    def build(
        self,
        target: Literal["tcl", "py", "live", "h5"] | str,
        *,
        out: "str | Path | None" = None,
    ) -> "str | None":
        """Re-emit the rehydrated graph to a build target.

        Supported targets:

        - ``"tcl"`` — return the Tcl deck as a string (or write it
          when ``out=`` is given).
        - ``"py"`` — return the openseespy deck as a string (or
          write it when ``out=`` is given).
        - ``"live"`` — drive an in-process
          :class:`LiveOpsEmitter` (requires openseespy);
          ``out=`` is ignored.  Returns ``None``.
        - ``"h5"`` — write a new ``model.h5`` to ``out=`` (which is
          required for the ``h5`` target).  Returns ``None``.

        .. note::

            **Tag identity may diverge from a fresh
            ``apeSees(fem).run()``** for the same FEM.  ADR 0019
            INV-5 — the bridge's :class:`TagAllocator` allocations
            are not preserved across an H5 round-trip; this method
            replays the stored tags as-is.  Callers needing
            bridge-time tag stability should hold the
            :class:`BuiltModel` directly.

        Raises
        ------
        ValueError
            ``target`` is unrecognised.
        TypeError
            ``out=`` missing for the ``"h5"`` target.
        """
        if target == "tcl":
            return self._build_text("tcl", out)
        if target == "py":
            return self._build_text("py", out)
        if target == "live":
            self._build_live()
            return None
        if target == "h5":
            if out is None:
                raise TypeError(
                    "OpenSeesModel.build('h5'): the 'out=' kwarg is "
                    "required for the h5 target."
                )
            self.to_h5(out)
            return None
        raise ValueError(
            f"OpenSeesModel.build: unknown target {target!r}. "
            "Supported targets are 'tcl', 'py', 'live', 'h5'."
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def fem(self) -> "FEMData":
        """The embedded :class:`FEMData` (lazy-imported per INV-4)."""
        return self._fem

    @property
    def model_name(self) -> str:
        """``/meta/model_name`` value from the source archive."""
        return self._model_name

    @property
    def ndm(self) -> int:
        """``/meta/ndm`` value from the source archive."""
        return self._ndm

    @property
    def ndf(self) -> int:
        """``/meta/ndf`` value from the source archive."""
        return self._ndf

    @property
    def snapshot_id(self) -> str:
        """Composite hash (``/meta/snapshot_id`` on the source archive).

        Phase 3 surfaces the byte string exactly as the bridge or
        :meth:`ModelData.write` stamped it.  Phase 6 (ADR 0021)
        replaces this with a recomputed canonical-bytes hash plus a
        drift warning when the recomputed value disagrees with the
        stamped one.
        """
        return self._snapshot_id

    @property
    def lineage(self) -> Lineage:
        """The Phase-6 lineage chain (Phase-3 stub).

        Only ``fem_hash`` and ``model_hash`` are populated in Phase
        3; ``results_hash`` is set by :class:`Results` in Phase 4
        and ``warnings`` are emitted by the Phase-6 lineage
        verifier.
        """
        return self._lineage

    # ------------------------------------------------------------------
    # Typed-record accessors
    # ------------------------------------------------------------------

    def materials(
        self, *, family: "str | None" = None,
    ) -> tuple[MaterialRecord, ...]:
        """Return the material records.

        ``family=None`` returns every material across both families
        (``uniaxial`` and ``nd``) in their write-time order.
        ``family="uniaxial"`` / ``"nd"`` returns just that subset.

        The returned tuple is immutable; mutating the underlying
        record dataclasses raises ``FrozenInstanceError`` (every
        typed record is frozen — see
        :mod:`apeGmsh.opensees._internal.typed_records`).
        """
        if family is None:
            out: list[MaterialRecord] = []
            for recs in self._materials_by_family.values():
                out.extend(recs)
            return tuple(out)
        return tuple(self._materials_by_family.get(family, ()))

    def materials_by_family(
        self,
    ) -> Mapping[str, tuple[MaterialRecord, ...]]:
        """Return ``{family: (recs, ...)}`` as a read-only mapping."""
        return self._materials_by_family

    def sections(self) -> tuple[SectionRecord, ...]:
        """Return every section record.

        Complex (Fiber) sections are :class:`SectionComplexRecord`
        with populated ``patches`` / ``fibers`` / ``layers``; simple
        sections are :class:`SectionSimpleRecord`.
        """
        return self._sections

    def transforms(self) -> tuple[TransformRecord, ...]:
        """Return every transform record.

        Schema deviation: one record per ``geomTransf`` call (not
        per spec) — see :class:`TransformRecord` docstring.
        """
        return self._transforms

    def beam_integration(self) -> tuple[BeamIntegrationRecord, ...]:
        """Return every ``beamIntegration`` declaration."""
        return self._beam_integration

    def time_series(self) -> tuple[TimeSeriesRecord, ...]:
        """Return every ``timeSeries`` declaration."""
        return self._time_series

    def patterns(self) -> tuple[PatternRecord, ...]:
        """Return every load / motion pattern.

        ``Plain`` and ``MultiSupport`` records carry populated
        ``loads`` / ``sps`` / ``ele_loads``; single-line patterns
        (``UniformExcitation``) keep their body in ``args``.
        """
        return self._patterns

    def recorders(self) -> tuple[RecorderRecord, ...]:
        """Return every ``recorder`` call.

        Schema 2.3.0 unifies typed and declared recorders; the
        :attr:`RecorderRecord.kind_label` property returns ``"typed"``
        / ``"declared"`` without forcing the caller to introspect
        ``decl_context``.
        """
        return self._recorders

    def elements(self) -> tuple[ElementRecord, ...]:
        """Return every ``element`` call.

        For records loaded from an archive (via :meth:`from_h5`),
        ``connectivity`` is empty ``()`` — the writer drops the
        connectivity prefix (the broker zone carries it).
        """
        return self._elements

    def fixes(self) -> tuple[FixRecord, ...]:
        """Return every model-level ``fix`` directive."""
        return self._fixes

    def masses(self) -> tuple[MassRecord, ...]:
        """Return every model-level ``mass`` directive."""
        return self._masses

    def analysis(self) -> Mapping[str, Any]:
        """Return the analysis-chain attrs as a read-only mapping.

        Keys mirror the H5 emitter accumulator: ``handler`` /
        ``numberer`` / ``system`` / ``test`` / ``algorithm`` /
        ``integrator`` / ``analysis``; trailing ``_args`` entries
        carry the positional arg tuples when present.
        """
        return self._analysis_attrs

    def cuts(self) -> tuple["SectionCutDef", ...]:
        """Return the apeGmsh.cuts ``SectionCutDef`` records."""
        return self._cuts

    def sweeps(self) -> tuple["SectionSweepDef", ...]:
        """Return the apeGmsh.cuts ``SectionSweepDef`` records."""
        return self._sweeps

    # ==================================================================
    # Private — internal construction helpers
    # ==================================================================

    @staticmethod
    def _load_elements(model: Any) -> tuple[ElementRecord, ...]:
        """Walk ``/opensees/element_meta`` into typed :class:`ElementRecord`.

        Mirrors :meth:`ModelData.from_h5`'s element rehydrate path:
        the writer drops the connectivity prefix so the stored args
        are just the tail; on re-emit we keep ``connectivity=()`` so
        a future ``to_h5`` is byte-stable (the same ``arity=0`` /
        same tail land at the same datasets).
        """
        import numpy as np

        f = model.handle
        if "opensees" not in f or "element_meta" not in f["opensees"]:
            return ()
        em_grp = f["opensees/element_meta"]
        out: list[ElementRecord] = []
        for type_group_name in em_grp:
            g = em_grp[type_group_name]
            if "ids" not in g or "fem_eids" not in g:
                continue
            ids = np.asarray(g["ids"][...]).reshape(-1)
            fem_eids = np.asarray(g["fem_eids"][...]).reshape(-1)
            args_arr = (
                np.asarray(g["args"][...])
                if "args" in g else np.zeros((len(ids), 0))
            )
            args_str = g["args_str"][...] if "args_str" in g else None
            type_token = type_group_name
            if "type" in g.attrs:
                attr_v = g.attrs["type"]
                if isinstance(attr_v, bytes):
                    type_token = attr_v.decode("utf-8", "replace")
                else:
                    type_token = str(attr_v)
            n_rows = min(len(ids), len(fem_eids))
            for i in range(n_rows):
                row_args: list[float | int | str] = []
                if args_arr.ndim == 2 and args_arr.shape[1] > 0:
                    for j in range(args_arr.shape[1]):
                        if args_str is not None:
                            s = args_str[i, j]
                            if isinstance(s, bytes):
                                s = s.decode("utf-8", "replace")
                            if s != "":
                                row_args.append(str(s))
                                continue
                        v = args_arr[i, j]
                        if np.isnan(v):
                            row_args.append(float("nan"))
                            continue
                        # Recover int when whole, mirroring the writer.
                        if float(v).is_integer():
                            row_args.append(int(v))
                        else:
                            row_args.append(float(v))
                out.append(ElementRecord(
                    type_token=type_token,
                    tag=int(ids[i]),
                    args=tuple(row_args),
                    # Writer drops connectivity prefix — keep it
                    # empty so to_h5 re-emits the same args tail.
                    connectivity=(),
                    fem_eid=int(fem_eids[i]),
                ))
        return tuple(out)

    @staticmethod
    def _load_bcs(
        model: Any,
    ) -> "tuple[tuple[FixRecord, ...], tuple[MassRecord, ...]]":
        """Walk ``/opensees/bcs`` into typed :class:`FixRecord` /
        :class:`MassRecord` sequences."""
        f = model.handle
        if "opensees" not in f or "bcs" not in f["opensees"]:
            return ((), ())
        bcs = f["opensees/bcs"]
        fixes: list[FixRecord] = []
        masses: list[MassRecord] = []
        if "fix" in bcs:
            for row in bcs["fix"][:]:
                tag = int(_decode(row["target"]))
                dofs = tuple(int(d) for d in row["dofs"])
                fixes.append(FixRecord(tag=tag, dofs=dofs))
        if "mass" in bcs:
            for row in bcs["mass"][:]:
                tag = int(_decode(row["target"]))
                # The writer stores zero-padded slots; preserve the
                # padding so byte-equivalence on re-emit holds.
                values = tuple(float(v) for v in row["values"])
                masses.append(MassRecord(tag=tag, values=values))
        return tuple(fixes), tuple(masses)

    @staticmethod
    def _load_analysis(
        model: Any,
    ) -> "tuple[dict[str, Any], tuple[int, float | None] | None]":
        """Read ``/opensees/analysis`` into a flat attrs dict.

        Returns the attrs dict (mirroring ``H5Emitter._analysis_attrs``)
        plus the ``(steps, dt)`` tuple from ``analyze_steps`` /
        ``analyze_dt`` when present.
        """
        f = model.handle
        if "opensees" not in f or "analysis" not in f["opensees"]:
            return {}, None
        a = f["opensees/analysis"]
        attrs: dict[str, Any] = {}
        analyze_call: "tuple[int, float | None] | None" = None
        for key, value in a.attrs.items():
            decoded = _decode(value)
            if key == "analyze_steps":
                steps = int(decoded)
                # Preserve dt if also stamped.
                dt: float | None = None
                if "analyze_dt" in a.attrs:
                    dt_raw = a.attrs["analyze_dt"]
                    dt_dec = _decode(dt_raw)
                    if isinstance(dt_dec, (int, float)):
                        dt = float(dt_dec)
                analyze_call = (steps, dt)
                continue
            if key == "analyze_dt":
                continue  # consumed alongside analyze_steps
            # Tuple-shaped attrs (handler_args, system_args, ...) come
            # back as numpy arrays; keep them as tuples in the dict so
            # the replay path can splat them.
            try:
                import numpy as np

                if isinstance(decoded, np.ndarray):
                    decoded = tuple(decoded.tolist())
            except ImportError:  # pragma: no cover - numpy is a hard dep
                pass
            attrs[key] = decoded
        return attrs, analyze_call

    # ==================================================================
    # Private — emit helpers
    # ==================================================================

    def _build_h5_emitter(self) -> "H5Emitter":
        """Construct a populated :class:`H5Emitter` from the record graph.

        Used by :meth:`to_h5` to delegate the actual file-shaping
        work to the schema-owning emitter — INV-3 (no h5py write
        surface on this class).
        """
        from .emitter.h5 import H5Emitter

        emitter = H5Emitter(
            model_name=self._model_name,
            snapshot_id=self._snapshot_id,
        )
        self._populate_emitter(emitter)
        return emitter

    def _populate_emitter(self, emitter: Any) -> None:
        """Walk the record graph and drive ``emitter`` through the
        schema-relevant Protocol methods.

        This is the body of :meth:`build`, factored so :meth:`to_h5`
        can reuse it.  Delegates to
        :func:`apeGmsh.opensees._internal.compose._replay_into`
        which centralises the protocol-call order.

        For non-H5 targets we rehydrate each :class:`ElementRecord`'s
        connectivity from the broker FEM (the H5 writer dropped the
        connectivity prefix per
        :meth:`H5Emitter._write_element_argstack`).
        """
        from ._internal.compose import _replay_into

        uniaxial = self._materials_by_family.get("uniaxial", ())
        nd = self._materials_by_family.get("nd", ())
        simple_sections = tuple(
            s for s in self._sections if isinstance(s, SectionSimpleRecord)
        )
        complex_sections = tuple(
            s for s in self._sections if isinstance(s, SectionComplexRecord)
        )
        # S2 (ADR 0033): widen per-node tuple to carry the broker's
        # per-node ndf so the build path emits ``-ndf K`` for nodes
        # declared via ``g.node_ndf`` (shell-on-solid).  ``ndf_for``
        # raises ``LookupError`` for undeclared nodes; we pass ``None``
        # in that case so the envelope ``ops.model(ndm, ndf=K)`` wins.
        def _ndf_or_none(nid: int) -> int | None:
            try:
                return int(self._fem.nodes.ndf_for(int(nid)))
            except LookupError:
                return None

        nodes = tuple(
            (
                int(nid),
                (float(c[0]), float(c[1]), float(c[2])),
                _ndf_or_none(int(nid)),
            )
            for nid, c in zip(self._fem.nodes.ids, self._fem.nodes.coords)
        )

        # Rehydrate connectivity per element from the broker FEM —
        # without this the tcl/py/live targets emit invalid element
        # calls (missing iNode / jNode prefix).
        elements_with_conn = self._rehydrate_element_connectivity(self._elements)

        _replay_into(
            emitter,
            ndm=self._ndm,
            ndf=self._ndf,
            nodes=nodes,
            uniaxial_materials=uniaxial,
            nd_materials=nd,
            simple_sections=simple_sections,
            complex_sections=complex_sections,
            transforms=self._transforms,
            beam_integrations=self._beam_integration,
            time_series=self._time_series,
            elements=elements_with_conn,
            fixes=self._fixes,
            masses=self._masses,
            patterns=self._patterns,
            recorders=self._recorders,
            analysis_attrs=dict(self._analysis_attrs),
            analyze_call=self._analyze_call,
        )

    def _rehydrate_element_connectivity(
        self, records: tuple[ElementRecord, ...],
    ) -> tuple[ElementRecord, ...]:
        """Re-attach connectivity prefix to records loaded from h5.

        The H5 writer drops the per-element connectivity prefix
        (``_write_element_argstack`` writes ``args[arity:]`` only).
        Tcl / Py / Live emitters need the full call shape:
        ``element <type> <tag> <iNode> <jNode> ... <transfTag>
        <integrationTag>``.  This helper looks up each FEM element
        id in ``self._fem.elements`` and prepends the connectivity to
        ``args``, returning fresh :class:`ElementRecord` instances.

        Records emitted outside a bridge fan-out
        (``fem_eid == MISSING_FEM_ELEMENT_ID``, i.e. ``-1``) are
        returned unchanged — their connectivity is unrecoverable
        from the FEM, and the H5 path tolerates the empty prefix.
        """
        if not records:
            return records
        try:
            fem_elem_index = self._build_fem_element_index()
        except Exception:
            # The broker doesn't expose ``.elements`` iteration the way
            # we expect — fall back to passing records through
            # unchanged (the H5 round-trip path doesn't need the
            # prefix anyway).
            return records
        out: list[ElementRecord] = []
        for rec in records:
            if rec.connectivity or rec.fem_eid <= 0:
                out.append(rec)
                continue
            conn = fem_elem_index.get(int(rec.fem_eid))
            if conn is None:
                out.append(rec)
                continue
            out.append(ElementRecord(
                type_token=rec.type_token,
                tag=rec.tag,
                args=(*conn, *rec.args),
                connectivity=conn,
                fem_eid=rec.fem_eid,
            ))
        return tuple(out)

    def _build_fem_element_index(self) -> dict[int, tuple[int, ...]]:
        """Return ``{fem_element_id: connectivity_tuple}`` over the FEM."""
        index: dict[int, tuple[int, ...]] = {}
        for group in self._fem.elements:
            ids = group.ids
            conn = group.connectivity
            for i in range(len(ids)):
                index[int(ids[i])] = tuple(int(c) for c in conn[i])
        return index

    def _populate_emitter_h5(self, emitter: "H5Emitter") -> None:
        """H5-only population: skip ``model``+``node`` emit so the
        compose path keeps producing byte-equivalent output.

        The H5 emitter records the ``model``/``node`` calls in
        ``self._ndm`` / ``self._node_*`` buffers, but the broker
        side (:func:`_compose_model_h5`) writes the canonical neutral
        zone from ``self._fem`` directly — the bridge's own ``/meta``
        / ``/nodes`` are written only as a fallback when the broker
        path is unavailable.  Driving ``model`` / ``node`` here would
        leave duplicate or shadowed state on the emitter; the records
        we need on the H5 side are everything *after* nodes.
        """
        from ._internal.compose import _replay_into

        uniaxial = self._materials_by_family.get("uniaxial", ())
        nd = self._materials_by_family.get("nd", ())
        simple_sections = tuple(
            s for s in self._sections if isinstance(s, SectionSimpleRecord)
        )
        complex_sections = tuple(
            s for s in self._sections if isinstance(s, SectionComplexRecord)
        )
        # Set ndm / ndf on the H5 emitter so ``/meta`` carries the
        # right values when the broker zone is unavailable (fallback
        # path).  When the broker zone IS written, _compose_model_h5
        # overwrites ``/meta`` with the broker-stamped attrs and the
        # bridge's model() call is harmless.
        emitter.model(ndm=int(self._ndm), ndf=int(self._ndf))

        _replay_into(
            emitter,
            ndm=self._ndm,
            ndf=self._ndf,
            # nodes empty — the H5 emitter does NOT write /nodes
            # (the broker does); driving emitter.node() here would
            # leave dead state on the emitter.
            nodes=(),
            uniaxial_materials=uniaxial,
            nd_materials=nd,
            simple_sections=simple_sections,
            complex_sections=complex_sections,
            transforms=self._transforms,
            beam_integrations=self._beam_integration,
            time_series=self._time_series,
            elements=self._elements,
            fixes=self._fixes,
            masses=self._masses,
            patterns=self._patterns,
            recorders=self._recorders,
            analysis_attrs=dict(self._analysis_attrs),
            analyze_call=self._analyze_call,
        )

    def _compose_h5(self, emitter: "H5Emitter", path: str) -> None:
        """Compose the H5 file at ``path`` using the shared composer.

        Note: this method is the **only** entry point on the class
        that opens the H5 file for writing, and even then the actual
        ``h5py.File("w", ...)`` call happens inside
        :func:`_compose_model_h5` — not here.  INV-3 enforcement
        target.
        """
        from ._internal.compose import _compose_model_h5

        # Re-populate the emitter with the H5-shaped record graph so
        # the file's /opensees/... zone matches the source archive.
        # ``self._build_h5_emitter()`` already called _populate_emitter
        # which drove nodes through emitter.node(); H5Emitter buffers
        # them but doesn't write /nodes (the broker zone does).
        # For byte-stable round-trip we instead build a fresh emitter
        # here using the H5-only population path.
        emitter_fresh: "H5Emitter" = type(emitter)(
            model_name=self._model_name,
            snapshot_id=self._snapshot_id,
        )
        self._populate_emitter_h5(emitter_fresh)
        _compose_model_h5(
            self._fem,
            emitter_fresh,
            path,
            model_name=self._model_name,
            ndf=int(self._ndf),
            cuts=self._cuts,
            sweeps=self._sweeps,
            snapshot_id=self._snapshot_id or None,
        )

    def _build_text(
        self, kind: Literal["tcl", "py"], out: "str | Path | None",
    ) -> str:
        """Render the deck for the ``tcl`` or ``py`` target.

        Returns the deck as a string; if ``out=`` is given, writes it
        to that path as well.
        """
        if kind == "tcl":
            from .emitter.tcl import TclEmitter

            emitter = TclEmitter()
        else:
            from .emitter.py import PyEmitter

            emitter = PyEmitter()
        self._populate_emitter(emitter)
        text = "\n".join(emitter.lines()) + "\n"
        if out is not None:
            Path(out).write_text(text, encoding="utf-8")
        return text

    def _build_live(self) -> None:
        """Drive an in-process :class:`LiveOpsEmitter` over the deck.

        ADR 0019 INV-5: tag identity may diverge from a fresh
        :meth:`apeSees.run` for the same FEM.  Documented; not a bug.
        """
        from .emitter.live import LiveOpsEmitter

        emitter = LiveOpsEmitter(wipe=True)
        self._populate_emitter(emitter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(value: Any) -> Any:
    """Decode an h5py compound-field bytes value (mirrors
    ``h5_reader._decode_bytes``)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _resolve_lineage(
    path: str,
    fem: "FEMData",
    fem_root: str,
) -> Lineage:
    """Read ``/meta/lineage`` + recompute; build the Phase-6 chain.

    ADR 0021 INV-2 — never raises.  On any divergence between stored
    and recomputed hashes, appends a warning to :attr:`Lineage.warnings`
    with the :data:`LINEAGE_WARNING_PREFIX` prefix.  When ``/meta/
    lineage`` is absent (legacy file), the function recomputes the
    chain and emits a "lineage absent — legacy file" warning so the
    surface stays uniform.

    The function imports the lineage helpers lazily so the existing
    import-DAG polarity (apeGmsh.opensees ↛ apeGmsh.mesh at import
    time) stays intact.
    """
    import h5py

    from ._internal.lineage import (
        LINEAGE_GROUP,
        WARNING_PREFIX,
        compute_fem_hash,
        compute_model_hash,
        read_stored_lineage,
    )

    # Embedded FEM's snapshot_id is the authoritative ``fem_hash``
    # source — INV-1 byte-identity is the contract.
    try:
        fem_snapshot = str(fem.snapshot_id)
    except Exception:
        fem_snapshot = ""

    warnings_list: list[str] = []
    stored_fem: "str | None" = None
    stored_model: "str | None" = None
    recomputed_model: "str | None" = None

    try:
        with h5py.File(path, "r") as f:
            meta_path = _resolve_meta_path(fem_root)
            meta_grp = f.get(meta_path) if meta_path in f else None
            if meta_grp is not None and LINEAGE_GROUP in meta_grp:
                stored_fem, stored_model, _ = read_stored_lineage(meta_grp)
            else:
                warnings_list.append(
                    WARNING_PREFIX
                    + "lineage absent — legacy file; recomputed without "
                    "stored hashes to compare against."
                )

            # Recompute model_hash from the loaded /opensees zone.
            if "opensees" in f:
                try:
                    recomputed_model = compute_model_hash(
                        fem_snapshot, f["opensees"],
                    )
                except Exception as exc:
                    warnings_list.append(
                        WARNING_PREFIX
                        + f"model_hash recompute failed: {exc!r}"
                    )

            # Recompute fem_hash from the on-disk neutral zone — sanity
            # for INV-1.  Compare against fem.snapshot_id below.
            recomputed_fem: "str | None" = None
            neutral_grp = _resolve_neutral_group(f, fem_root)
            if neutral_grp is not None:
                try:
                    recomputed_fem = compute_fem_hash(neutral_grp)
                except Exception as exc:
                    warnings_list.append(
                        WARNING_PREFIX
                        + f"fem_hash recompute failed: {exc!r}"
                    )
    except OSError as exc:
        # File no longer accessible (deleted, locked).  Surface as a
        # warning; the broker is otherwise usable from already-loaded
        # in-memory state.
        warnings_list.append(
            WARNING_PREFIX
            + f"could not reopen file for lineage recompute: {exc!r}"
        )
        recomputed_fem = None

    # Compare stored vs recomputed.
    if stored_fem is not None and recomputed_fem is not None:
        if stored_fem != recomputed_fem:
            warnings_list.append(
                WARNING_PREFIX
                + "fem layer drift: stored fem_hash="
                f"{stored_fem!r} != recomputed {recomputed_fem!r}"
            )
    if stored_model is not None and recomputed_model is not None:
        if stored_model != recomputed_model:
            warnings_list.append(
                WARNING_PREFIX
                + "opensees layer drift: stored model_hash="
                f"{stored_model!r} != recomputed {recomputed_model!r}"
            )

    # The Lineage published to consumers carries the *recomputed*
    # values: they're the authoritative current state of the bytes on
    # disk.  Warnings record the divergence from what was stamped.
    return Lineage(
        fem_hash=fem_snapshot,
        model_hash=recomputed_model,
        warnings=tuple(warnings_list),
    )


def _resolve_meta_path(fem_root: str) -> str:
    """Return the relative path to the ``meta`` group under ``fem_root``."""
    if fem_root in ("", "/"):
        return "meta"
    return f"{fem_root.strip('/')}/meta"


def _resolve_neutral_group(f: Any, fem_root: str) -> Any:
    """Return the neutral-zone group, or ``None`` when absent."""
    if fem_root in ("", "/"):
        # Standalone model.h5 — neutral zone is at root.
        if "nodes" in f:
            return f
        return None
    key = fem_root.lstrip("/")
    if key not in f:
        return None
    return f[key]


def _infer_ndm_from_transforms(f: Any) -> int:
    """Best-effort spatial dimension from ``/opensees/transforms/*/per_element_vecxz``.

    Returns 0 when no transforms are present (caller's ``max(broker_ndm,
    inferred)`` falls back to the broker value).  The H5 emitter
    writes ``per_element_vecxz`` as ``(N, 3)`` even in 2D, so this
    can't distinguish 2D from 3D — but distinguishes "has bridge
    output at all" from "broker only" which is the case worth
    salvaging at read time.
    """
    if "opensees" not in f:
        return 0
    if "transforms" not in f["opensees"]:
        return 0
    for tname in f["opensees/transforms"]:
        g = f[f"opensees/transforms/{tname}"]
        if "per_element_vecxz" in g:
            shape = g["per_element_vecxz"].shape
            if len(shape) >= 2 and shape[1] >= 3:
                return 3
            if len(shape) >= 2 and shape[1] == 2:
                return 2
    return 0
