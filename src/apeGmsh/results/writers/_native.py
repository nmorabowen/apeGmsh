"""NativeWriter — produces apeGmsh native HDF5 result files.

Bulk-write API. The user typically calls higher-level entry points
(``Results.from_recorders`` in Phase 6, ``DomainCapture`` in Phase 7),
which use this writer internally.

Usage
-----
::

    with NativeWriter(path) as w:
        w.open(fem=fem, source_type="domain_capture")

        # Stage 1 — transient
        sid = w.begin_stage(name="gravity", kind="transient",
                            time=time_grav)
        w.write_nodes(sid, "partition_0",
                      node_ids=ids,
                      components={"displacement_x": ux, ...})
        w.write_gauss_group(sid, "partition_0", "group_0",
                            class_tag=4, int_rule=1,
                            element_index=eidx,
                            natural_coords=nc,
                            components={"stress_xx": sxx})
        w.end_stage()

        # Stage 2 — mode shape (T=1, kind="mode")
        sid = w.begin_stage(name="mode_1", kind="mode",
                            time=np.array([0.0]),
                            eigenvalue=158.7,
                            frequency_hz=2.005,
                            period_s=0.499,
                            mode_index=1)
        w.write_nodes(sid, "partition_0",
                      node_ids=ids,
                      components={"displacement_x": shape_x[None, :], ...})
        w.end_stage()

Composed-file pattern (Phase 4, ADR 0020)
-----------------------------------------
``open(..., model_h5_src=path)`` embeds a copy of the ``/opensees/``
zone from an existing ``model.h5`` into the same results file. The
copy happens at open time, **not at close**: h5py file fragmentation
is materially worse when the bulk lands during the close fsync, and
the resulting layout is ``/meta`` → ``/model`` (neutral) →
``/opensees/`` → ``/stages/...`` (results data appended during the
run). The caller (typically :meth:`Results.from_recorders` with
``model=`` or :class:`DomainCapture` with ``bridge=``) is responsible
for materialising the source model file; ``NativeWriter`` does not
import :mod:`apeGmsh.opensees` and stays purely h5py-side.

When ``model_h5_src`` is ``None`` (today's behavior), the file
carries only ``/meta`` + ``/model/`` + ``/stages/...`` — exactly the
shape pre-Phase-4 readers expect.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ..schema import _native, _versions

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData


class NativeWriter:
    """Bulk writer for apeGmsh native HDF5 result files.

    The writer holds an open ``h5py.File`` for its lifetime; use as a
    context manager or call ``open()`` / ``close()`` explicitly.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._h5: Optional["h5py.File"] = None
        self._current_stage: Optional[str] = None
        self._stage_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "NativeWriter":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def open(
        self,
        *,
        fem: Optional["FEMData"] = None,
        source_type: str = _native.SOURCE_DOMAIN_CAPTURE,
        source_path: str = "",
        analysis_label: str = "",
        model_h5_src: Optional[str | Path] = None,
    ) -> None:
        """Create the file, write root attrs, embed FEMData if provided.

        ``model_h5_src`` (Phase 4, ADR 0020) — when supplied, points at
        an existing apeGmsh-produced ``model.h5`` whose ``/opensees/``
        zone is copied into this results file at open time. The
        resulting Composed file carries both ``/model/`` (neutral
        FEMData snapshot) and ``/opensees/`` (the OpenSees-specific
        broker zone) alongside ``/stages/...``. Downstream readers
        auto-resolve :attr:`Results.model` from the same file.

        Raises
        ------
        FileNotFoundError
            ``model_h5_src`` is supplied but does not exist.
        RuntimeError
            ``model_h5_src`` does not carry a ``/opensees/`` group.
        """
        import h5py

        if self._h5 is not None:
            raise RuntimeError(f"NativeWriter for {self._path} already open.")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self._path, "w")

        h5 = self._h5
        h5.attrs[_native.ATTR_SCHEMA_VERSION] = _versions.SCHEMA_VERSION
        # ADR 0023 — per-zone marker; the envelope above bumps only on
        # partition-shape changes, this one tracks the results-zone
        # content shape independently. Phase 7a wires the two-version
        # read window against this attr.
        h5.attrs[_native.ATTR_RESULTS_SCHEMA_VERSION] = (
            _versions.RESULTS_SCHEMA_VERSION
        )
        # ADR 0023 §"Three per-zone version stamps" — composed result
        # files also carry the neutral + opensees per-zone keys at root
        # so a reader probing the results envelope's meta can validate
        # all three zones from one attr-map.  ``write_model`` /
        # ``write_opensees_from`` are still optional; the corresponding
        # per-zone keys are dropped when those zones are not present
        # (see below — they're set lazily after the embedding calls).
        h5.attrs[_native.ATTR_SOURCE_TYPE] = source_type
        h5.attrs[_native.ATTR_SOURCE_PATH] = source_path
        h5.attrs[_native.ATTR_CREATED_AT] = (
            datetime.now(tz=timezone.utc).isoformat()
        )
        h5.attrs[_native.ATTR_APEGMSH_VERSION] = _apegmsh_version()
        h5.attrs[_native.ATTR_ANALYSIS_LABEL] = analysis_label

        # Empty stages container
        h5.create_group(_native.STAGES_GROUP[1:])

        if fem is not None:
            self.write_model(fem)

        # Composed-file embedding — perform copy at open time, before
        # any stage / partition data lands. h5py file fragmentation is
        # materially worse when the bulk arrives during the close
        # fsync (see ADR 0020 §"Negative consequences"). The natural
        # group ordering ``/meta`` → ``/model`` → ``/opensees/`` →
        # ``/stages/...`` is preserved by writing the OpenSees zone
        # here, sandwiched between the broker write and the begin /
        # write / end-stage flow that follows.
        if model_h5_src is not None:
            self.write_opensees_from(model_h5_src)

        # ADR 0023 — forward the per-zone version stamps to the results
        # envelope's root attrs so a reader probing the file root sees
        # all three zones' versions in one attr-map.  Zone presence is
        # inferred from the embedded groups (no ``/model/`` ⇒ no
        # neutral key; no ``/opensees/`` ⇒ no opensees key).  Sources
        # the versions from the embedded ``/model/meta`` per-zone keys
        # (stamped by :func:`write_meta`); falls back to the current
        # writer constants when the source lacks them (legacy
        # ``to_native_h5`` fixtures).
        self._stamp_root_per_zone_versions()

        # ADR 0021 — stamp ``fem_hash`` + ``model_hash`` at open time,
        # before any stage data lands.  ``results_hash`` chains in at
        # close time (see :meth:`close`).  All three live under
        # ``/meta/lineage/``.  Standalone-results files (no embedded
        # ``/opensees/`` zone) carry only ``fem_hash`` here; the
        # ``model_hash`` link stays absent.
        self._stamp_open_lineage()

    def write_opensees_from(self, model_h5_src: str | Path) -> None:
        """Embed the bridge zone from a sibling ``model.h5`` at root.

        Phase 4 cleanup (ADR 0020) — copies ``/opensees/`` verbatim
        from the source into the Composed results file at root.  The
        composed file shape becomes::

            /meta                 (results envelope)
            /model/               (rich FEMData neutral zone)
            /opensees/            (bridge zone — copied here)
            /stages/...           (results data)

        Both the viewer (via :mod:`h5_reader`) and
        :meth:`OpenSeesModel.from_h5(path, opensees_root="/opensees")`
        read the bridge zone at root; pairing it with ``/model/`` (the
        rich neutral zone written by :meth:`write_model`) gives the
        ``OpenSeesModel`` rehydration path everything it needs without
        the legacy ``/opensees_archive/`` mirror or the temp-file
        extract dance the reader used to perform.

        Idempotency: calling this twice on the same writer raises
        :class:`RuntimeError` because ``/opensees/`` already exists.
        Schema authority stays with :class:`apeGmsh.opensees.H5Emitter`
        (ADR 0019 INV-3 unchanged — this method neither inspects nor
        rewrites the zone's content; it copies bytes).

        Raises
        ------
        FileNotFoundError
            ``model_h5_src`` does not exist.
        RuntimeError
            ``/opensees/`` is missing from the source, or the writer
            already holds a ``/opensees/`` group.
        """
        import h5py

        h5 = self._require_open()
        if "opensees" in h5:
            raise RuntimeError(
                f"NativeWriter for {self._path}: /opensees/ is "
                f"already present in the file; cannot copy from "
                f"{model_h5_src!s}."
            )
        src_path = Path(model_h5_src)
        if not src_path.is_file():
            raise FileNotFoundError(
                f"NativeWriter.write_opensees_from: source model.h5 "
                f"not found at {src_path!s}."
            )
        with h5py.File(str(src_path), "r") as src:
            if "opensees" not in src:
                raise RuntimeError(
                    f"NativeWriter.write_opensees_from: source "
                    f"{src_path!s} has no /opensees/ group."
                )
            # Bridge zone copy — pairs with the rich /model/ neutral
            # zone (written by write_model) for the OpenSeesModel
            # rehydrate path.  /opensees/cuts and /opensees/sweeps
            # come along for the ride; the viewer's probe surface is
            # the same /opensees/transforms + /opensees/element_meta
            # pair it always was.
            src.copy(src["opensees"], h5, name="opensees")
            # Forward bridge-stamped meta attrs onto /model/meta.  The
            # source model.h5's /meta carries the bridge's spatial ndf
            # and the user's model_name; NativeWriter.write_model
            # passed defaults (ndf=0, model_name="") to the broker
            # because the broker doesn't know either.  Without this
            # enrichment OpenSeesModel.from_h5(fem_root="/model") would
            # surface those defaults instead of the bridge's values,
            # losing parity with the standalone ``apeSees(fem).h5()``
            # rehydrate.  ndm is inferred from transforms by the read
            # side, so it's not forwarded here.
            if "meta" in src and "/model/meta" in h5:
                src_meta = src["meta"].attrs
                dst_meta = h5["/model/meta"].attrs
                if "ndf" in src_meta:
                    dst_meta["ndf"] = int(src_meta["ndf"])
                if "model_name" in src_meta:
                    name = src_meta["model_name"]
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", "replace")
                    dst_meta["model_name"] = str(name)
                # ADR 0023 — forward the source's per-zone opensees
                # version onto /model/meta so the composed file's
                # bridge zone has a recoverable version under the same
                # meta scope OpenSeesModel.from_h5(fem_root="/model")
                # probes.  Falls back to the envelope value for
                # pre-Phase-7a source files.
                if "opensees_schema_version" in src_meta:
                    dst_meta["opensees_schema_version"] = str(
                        src_meta["opensees_schema_version"]
                    )
                elif "schema_version" in src_meta:
                    dst_meta["opensees_schema_version"] = str(
                        src_meta["schema_version"]
                    )

            # Forward the source's stamped lineage (ADR 0021) onto
            # ``/model/meta/lineage`` so :meth:`OpenSeesModel.from_h5`
            # called against the composed file (with
            # ``fem_root="/model"``) finds the same stored hashes
            # ``_compose_model_h5`` wrote into the source.  Without
            # this forward, the model-layer broker recomputes from
            # the embedded zones but has nothing to compare against,
            # producing a "lineage absent — legacy file" warning on
            # every Composed-file read.
            if (
                "meta" in src
                and "lineage" in src["meta"]
                and "/model/meta" in h5
            ):
                src_lineage = src["meta/lineage"]
                dst_meta_grp = h5["/model/meta"]
                if "lineage" in dst_meta_grp:
                    del dst_meta_grp["lineage"]
                src.copy(src_lineage, dst_meta_grp, name="lineage")

    def close(self) -> None:
        if self._h5 is None:
            return
        # ADR 0021 — stamp ``results_hash`` after every stage has
        # landed but before the file is closed.  The chain depends on
        # the previously-stamped ``model_hash``; bridge-only files
        # (no ``/opensees/`` zone) skip the link silently.
        try:
            self._stamp_close_lineage()
        except Exception:
            # Lineage drift is warn-not-raise (INV-2).  A failure here
            # means the file's lineage triple stays partial — that's
            # the same shape a pre-Phase-6 file has, so readers cope.
            pass
        self._h5.close()
        self._h5 = None
        self._current_stage = None

    # ------------------------------------------------------------------
    # Per-zone schema stamping (ADR 0023)
    # ------------------------------------------------------------------

    def _stamp_root_per_zone_versions(self) -> None:
        """Forward per-zone version stamps from embedded zones to root.

        Per ADR 0023 §"Three per-zone version stamps + one envelope":
        composed result files carry three zones (neutral via ``/model/``,
        opensees via ``/opensees/``, results via ``/stages/``); the
        per-zone keys live at the file root so a reader probing
        ``f.attrs`` sees all three independently.  Sources versions
        from the embedded ``/model/meta`` group's per-zone keys
        (stamped by :func:`apeGmsh.mesh._femdata_h5_io.write_meta` and
        the source ``model.h5``'s bridge ``_write_meta``); falls back
        to the current writer constants when those keys are absent
        (legacy fixtures, pre-Phase-7a source files).
        """
        from ...mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION
        from ...opensees.emitter.h5 import SCHEMA_VERSION as OPENSEES_VERSION
        from ...opensees._internal.schema_version import (
            NEUTRAL_KEY,
            OPENSEES_KEY,
        )

        h5 = self._require_open()
        if "model" in h5 and "meta" in h5["model"]:
            model_meta_attrs = h5["model/meta"].attrs
            if NEUTRAL_KEY in model_meta_attrs:
                h5.attrs[NEUTRAL_KEY] = str(model_meta_attrs[NEUTRAL_KEY])
            elif "schema_version" in model_meta_attrs:
                # Legacy single-stamp source — fall back to the envelope.
                h5.attrs[NEUTRAL_KEY] = str(model_meta_attrs["schema_version"])
            else:
                h5.attrs[NEUTRAL_KEY] = NEUTRAL_SCHEMA_VERSION
            # Source's bridge-stamped per-zone opensees key (when present).
            if OPENSEES_KEY in model_meta_attrs:
                h5.attrs[OPENSEES_KEY] = str(model_meta_attrs[OPENSEES_KEY])
        if "opensees" in h5 and OPENSEES_KEY not in h5.attrs:
            # /opensees/ was embedded but the source's /model/meta didn't
            # carry the per-zone key (legacy/composed source).  Stamp the
            # current writer's version — this matches the
            # whichever-wrote-last envelope semantics for legacy readers.
            h5.attrs[OPENSEES_KEY] = OPENSEES_VERSION

    # ------------------------------------------------------------------
    # Lineage stamping helpers (ADR 0021)
    # ------------------------------------------------------------------

    def _stamp_open_lineage(self) -> None:
        """Stamp ``fem_hash`` + ``model_hash`` at open time.

        Run after ``write_model`` and ``write_opensees_from`` so both
        source zones are present.  Bridge-only files (no ``/model/``,
        no ``/opensees/``) skip the corresponding links — the resulting
        :class:`Lineage` carries empty strings / ``None`` for the
        missing layers, matching the ADR 0021 surface.
        """
        from ...opensees._internal.lineage import (
            compute_fem_hash,
            compute_model_hash,
            write_lineage_attrs,
        )

        h5 = self._require_open()
        fem_hash = ""
        model_hash: str | None = None
        if "model" in h5:
            try:
                fem_hash = compute_fem_hash(h5["model"])
            except Exception:
                # Recompute failed (e.g. partial neutral zone).
                # Lineage stays empty; readers will surface
                # "lineage absent" warnings rather than crashing.
                fem_hash = ""
        if "opensees" in h5:
            model_hash = compute_model_hash(fem_hash, h5["opensees"])
        if not fem_hash and model_hash is None:
            return
        lineage_meta = self._require_lineage_meta_group()
        write_lineage_attrs(
            lineage_meta,
            fem_hash=fem_hash if fem_hash else None,
            model_hash=model_hash,
        )

    def _stamp_close_lineage(self) -> None:
        """Stamp ``results_hash`` at close time.

        Reads back the previously-stamped ``model_hash`` and chains it
        with the canonical bytes of ``/stages/``.  When no
        ``model_hash`` was stamped (bridge-only files), ``results_hash``
        chains on an empty string — consistent with the
        :func:`compute_results_hash` contract.
        """
        from ...opensees._internal.lineage import (
            LINEAGE_GROUP,
            compute_results_hash,
            read_stored_lineage,
            write_lineage_attrs,
        )

        h5 = self._require_open()
        if _native.STAGES_GROUP[1:] not in h5:
            return
        stages = h5[_native.STAGES_GROUP[1:]]
        # No model_hash was written ⇒ chain on empty (still
        # tamper-evident at the results layer alone).
        prior_fem = ""
        prior_model = ""
        if "meta" in h5 and LINEAGE_GROUP in h5["meta"]:
            stored = read_stored_lineage(h5["meta"])
            prior_fem = stored[0] or ""
            prior_model = stored[1] or ""
        results_hash = compute_results_hash(prior_model, stages)
        lineage_meta = self._require_lineage_meta_group()
        # Preserve the open-time fem / model hashes; only add the
        # results link.
        write_lineage_attrs(
            lineage_meta,
            fem_hash=prior_fem if prior_fem else None,
            model_hash=prior_model if prior_model else None,
            results_hash=results_hash,
        )

    def _require_lineage_meta_group(self):
        """Return the ``/meta`` group for lineage stamping, creating it.

        NativeWriter stores envelope info as root attrs (no ``/meta``
        group); the lineage triple needs a structured sub-group for
        ``read_stored_lineage`` to consume it like the standalone
        ``model.h5`` flow does.  Idempotent — repeated calls return
        the same group.
        """
        h5 = self._require_open()
        if "meta" in h5:
            grp = h5["meta"]
        else:
            grp = h5.create_group("meta")
        return grp

    # ------------------------------------------------------------------
    # Embedded FEMData snapshot
    # ------------------------------------------------------------------

    def write_model(self, fem: "FEMData") -> None:
        h5 = self._require_open()
        if _native.MODEL_GROUP[1:] in h5:
            raise RuntimeError("/model/ already written.")
        model_grp = h5.create_group(_native.MODEL_GROUP[1:])
        fem.to_native_h5(model_grp)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def begin_stage(
        self,
        *,
        name: str,
        kind: str,
        time: ndarray,
        stage_id: Optional[str] = None,
        eigenvalue: Optional[float] = None,
        frequency_hz: Optional[float] = None,
        period_s: Optional[float] = None,
        mode_index: Optional[int] = None,
    ) -> str:
        """Open a new stage. Returns the stage_id (auto-generated if omitted)."""
        h5 = self._require_open()
        if self._current_stage is not None:
            raise RuntimeError(
                f"Stage {self._current_stage!r} still open — call end_stage()."
            )
        if kind not in _native.ALL_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_native.ALL_KINDS)} (got {kind!r})."
            )

        if stage_id is None:
            stage_id = _native.stage_id(self._stage_count)
        self._stage_count += 1

        stage_grp = h5.create_group(_native.stage_path(stage_id)[1:])
        stage_grp.attrs[_native.ATTR_STAGE_NAME] = name
        stage_grp.attrs[_native.ATTR_STAGE_KIND] = kind

        time_arr = np.asarray(time, dtype=np.float64)
        stage_grp.create_dataset(_native.DSET_TIME, data=time_arr)

        if kind == _native.KIND_MODE:
            if eigenvalue is None or frequency_hz is None or period_s is None:
                raise ValueError(
                    "kind='mode' requires eigenvalue, frequency_hz, period_s."
                )
            stage_grp.attrs[_native.ATTR_EIGENVALUE] = float(eigenvalue)
            stage_grp.attrs[_native.ATTR_FREQUENCY_HZ] = float(frequency_hz)
            stage_grp.attrs[_native.ATTR_PERIOD_S] = float(period_s)
            if mode_index is not None:
                stage_grp.attrs[_native.ATTR_MODE_INDEX] = int(mode_index)

        # Pre-create the partitions container so writes can require_group it.
        stage_grp.create_group(_native.GROUP_PARTITIONS)

        self._current_stage = stage_id
        return stage_id

    def end_stage(self) -> None:
        if self._current_stage is None:
            raise RuntimeError("No stage open.")
        self._current_stage = None

    # ------------------------------------------------------------------
    # Bulk writes — nodes
    # ------------------------------------------------------------------

    def write_nodes(
        self,
        stage_id: str,
        partition_id: str,
        *,
        node_ids: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        """Write nodal results for one partition.

        ``components[name]`` must have shape ``(T, N)`` matching the
        stage's time vector length and the node count.
        """
        nodes_grp = self._require_partition(stage_id, partition_id).require_group(
            _native.GROUP_NODES,
        )
        node_ids = np.asarray(node_ids, dtype=np.int64)
        _write_or_validate_ids(nodes_grp, _native.DSET_IDS, node_ids)
        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != node_ids.size:
                raise ValueError(
                    f"Component {comp_name!r} has {arr.shape[1]} nodes but "
                    f"node_ids has {node_ids.size}."
                )
            nodes_grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — Gauss points
    # ------------------------------------------------------------------

    def write_gauss_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        class_tag: int,
        int_rule: int = 0,
        custom_rule_idx: int = 0,
        element_index: ndarray,
        natural_coords: ndarray,
        components: dict[str, ndarray],
        local_axes_quaternion: Optional[ndarray] = None,
    ) -> None:
        """Write one ``(class_tag, int_rule)`` Gauss group.

        Shapes:
        - ``element_index``: ``(E_g,)``
        - ``natural_coords``: ``(n_GP_g, dim)``
        - ``components[name]``: ``(T, E_g, n_GP_g)``
        - ``local_axes_quaternion``: ``(E_g, 4)``, optional (shells)
        """
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_GAUSS_POINTS, group_id,
        )
        grp.attrs[_native.ATTR_CLASS_TAG] = int(class_tag)
        grp.attrs[_native.ATTR_INT_RULE] = int(int_rule)
        grp.attrs[_native.ATTR_CUSTOM_RULE_IDX] = int(custom_rule_idx)

        eidx = np.asarray(element_index, dtype=np.int64)
        nc = np.asarray(natural_coords, dtype=np.float64)
        grp.create_dataset(_native.DSET_ELEMENT_INDEX, data=eidx)
        grp.create_dataset(_native.DSET_NATURAL_COORDS, data=nc)
        if local_axes_quaternion is not None:
            grp.create_dataset(
                _native.DSET_LOCAL_AXES_QUATERNION,
                data=np.asarray(local_axes_quaternion, dtype=np.float64),
            )

        n_gp = nc.shape[0]
        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != eidx.size or arr.shape[2] != n_gp:
                raise ValueError(
                    f"Component {comp_name!r} has shape {arr.shape}; expected "
                    f"(T, {eidx.size}, {n_gp})."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — fibers
    # ------------------------------------------------------------------

    def write_fibers_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        section_tag: int,
        section_class: str,
        element_index: ndarray,
        gp_index: ndarray,
        y: ndarray,
        z: ndarray,
        area: ndarray,
        material_tag: ndarray,
        components: dict[str, ndarray],
        station_natural_coord: "ndarray | None" = None,
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_FIBERS, group_id,
        )
        grp.attrs[_native.ATTR_SECTION_TAG] = int(section_tag)
        grp.attrs[_native.ATTR_SECTION_CLASS] = section_class

        eidx = np.asarray(element_index, dtype=np.int64)
        n = eidx.size
        # ``station_natural_coord`` is optional: per-fiber TRUE station
        # ξ ∈ [-1, +1] (NaN where the capture's geometry probe failed).
        # Omitted = pre-station caller; readers return None and
        # consumers fall back per element.
        index_datasets: list[tuple[str, ndarray, type]] = [
            (_native.DSET_ELEMENT_INDEX, eidx, np.int64),
            (_native.DSET_GP_INDEX, gp_index, np.int64),
            (_native.DSET_Y, y, np.float64),
            (_native.DSET_Z, z, np.float64),
            (_native.DSET_AREA, area, np.float64),
            (_native.DSET_MATERIAL_TAG, material_tag, np.int64),
        ]
        if station_natural_coord is not None:
            index_datasets.append((
                _native.DSET_STATION_NATURAL_COORD,
                station_natural_coord, np.float64,
            ))
        for name, arr, dtype in index_datasets:
            a: ndarray = np.asarray(arr, dtype=dtype)
            if a.size != n:
                raise ValueError(
                    f"Fiber index dataset {name!r} has size {a.size}; "
                    f"expected {n}."
                )
            grp.create_dataset(name, data=a)

        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != n:
                raise ValueError(
                    f"Fiber component {comp_name!r} has {arr.shape[1]} fibers; "
                    f"expected {n}."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — layers
    # ------------------------------------------------------------------

    def write_layers_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        element_index: ndarray,
        gp_index: ndarray,
        layer_index: ndarray,
        sub_gp_index: ndarray,
        thickness: ndarray,
        local_axes_quaternion: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_LAYERS, group_id,
        )
        eidx = np.asarray(element_index, dtype=np.int64)
        n = eidx.size
        for name, arr, dtype in [
            (_native.DSET_ELEMENT_INDEX, eidx, np.int64),
            (_native.DSET_GP_INDEX, gp_index, np.int64),
            (_native.DSET_LAYER_INDEX, layer_index, np.int64),
            (_native.DSET_SUB_GP_INDEX, sub_gp_index, np.int64),
            (_native.DSET_THICKNESS, thickness, np.float64),
        ]:
            a = np.asarray(arr, dtype=dtype)
            if a.size != n:
                raise ValueError(
                    f"Layer index dataset {name!r} has size {a.size}; "
                    f"expected {n}."
                )
            grp.create_dataset(name, data=a)

        quat = np.asarray(local_axes_quaternion, dtype=np.float64)
        if quat.shape != (n, 4):
            raise ValueError(
                f"local_axes_quaternion must have shape ({n}, 4); "
                f"got {quat.shape}."
            )
        grp.create_dataset(_native.DSET_LOCAL_AXES_QUATERNION, data=quat)

        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != n:
                raise ValueError(
                    f"Layer component {comp_name!r} has {arr.shape[1]} entries; "
                    f"expected {n}."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — line stations
    # ------------------------------------------------------------------

    def write_line_stations_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        class_tag: int,
        int_rule: int = 0,
        element_index: ndarray,
        station_natural_coord: ndarray,
        components: dict[str, ndarray],
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_LINE_STATIONS, group_id,
        )
        grp.attrs[_native.ATTR_CLASS_TAG] = int(class_tag)
        grp.attrs[_native.ATTR_INT_RULE] = int(int_rule)

        eidx = np.asarray(element_index, dtype=np.int64)
        snc = np.asarray(station_natural_coord, dtype=np.float64)
        grp.create_dataset(_native.DSET_ELEMENT_INDEX, data=eidx)
        grp.create_dataset(_native.DSET_STATION_NATURAL_COORD, data=snc)

        n_stations = snc.size
        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != eidx.size or arr.shape[2] != n_stations:
                raise ValueError(
                    f"Line-station component {comp_name!r} has shape "
                    f"{arr.shape}; expected (T, {eidx.size}, {n_stations})."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Bulk writes — nodal forces (per-element-node)
    # ------------------------------------------------------------------

    def write_nodal_forces_group(
        self,
        stage_id: str,
        partition_id: str,
        group_id: str,
        *,
        class_tag: int,
        frame: str,                        # "global" or "local"
        element_index: ndarray,
        components: dict[str, ndarray],    # (T, E_g, npe_g)
    ) -> None:
        grp = self._require_element_subgroup(
            stage_id, partition_id, _native.GROUP_NODAL_FORCES, group_id,
        )
        grp.attrs[_native.ATTR_CLASS_TAG] = int(class_tag)
        grp.attrs[_native.ATTR_FRAME] = frame

        eidx = np.asarray(element_index, dtype=np.int64)
        grp.create_dataset(_native.DSET_ELEMENT_INDEX, data=eidx)

        for comp_name, values in components.items():
            arr = np.asarray(values)
            self._validate_time_axis(stage_id, arr)
            if arr.shape[1] != eidx.size:
                raise ValueError(
                    f"Nodal-force component {comp_name!r} has {arr.shape[1]} "
                    f"elements; expected {eidx.size}."
                )
            grp.create_dataset(comp_name, data=arr)

    # ------------------------------------------------------------------
    # Element ID writing (per partition)
    # ------------------------------------------------------------------

    def write_element_ids(
        self,
        stage_id: str,
        partition_id: str,
        ids: ndarray,
    ) -> None:
        """Write the partition's flat element ID list at ``elements/_ids``."""
        grp = self._require_partition(stage_id, partition_id).require_group(
            _native.GROUP_ELEMENTS,
        )
        _write_or_validate_ids(
            grp, _native.DSET_IDS, np.asarray(ids, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> "h5py.File":
        if self._h5 is None:
            raise RuntimeError(
                f"NativeWriter for {self._path} is not open. "
                f"Call open() or use as context manager."
            )
        return self._h5

    def _require_partition(self, stage_id: str, partition_id: str) -> Any:
        h5 = self._require_open()
        partitions_grp = h5.require_group(
            _native.partitions_path(stage_id)[1:],
        )
        return partitions_grp.require_group(partition_id)

    def _require_element_subgroup(
        self,
        stage_id: str,
        partition_id: str,
        category: str,
        group_id: str,
    ) -> Any:
        elem_grp = self._require_partition(stage_id, partition_id).require_group(
            _native.GROUP_ELEMENTS,
        )
        cat_grp = elem_grp.require_group(category)
        if group_id in cat_grp:
            raise RuntimeError(
                f"Group {group_id!r} already exists under "
                f"{stage_id}/{partition_id}/elements/{category}/."
            )
        return cat_grp.create_group(group_id)

    def _validate_time_axis(self, stage_id: str, arr: ndarray) -> None:
        h5 = self._require_open()
        time = h5[_native.stage_time_path(stage_id)]
        n_steps = time.shape[0]
        if arr.ndim < 1 or arr.shape[0] != n_steps:
            raise ValueError(
                f"Component array has shape {arr.shape}; expected leading "
                f"dim {n_steps} (matching stage time vector)."
            )


# =====================================================================
# Module helpers
# =====================================================================

def _write_or_validate_ids(group: Any, key: str, ids: ndarray) -> None:
    """Write ``ids`` to ``group[key]`` if absent; otherwise validate equal."""
    if key in group:
        existing = group[key][...]
        if not np.array_equal(np.asarray(existing, dtype=np.int64), ids):
            raise ValueError(
                f"{group.name}/{key} mismatch: existing {existing} vs new {ids}."
            )
        return
    group.create_dataset(key, data=ids)


def _apegmsh_version() -> str:
    try:
        from apeGmsh import __version__ as v
        return str(v)
    except Exception:
        return ""
