"""MPCOReader — reads STKO ``.mpco`` files through the unified protocol.

Phase 3 covered nodal results (DISPLACEMENT, ROTATION, VELOCITY,
ACCELERATION, REACTION_FORCE, …) translated to canonical apeGmsh
names. Phase 11a wires Gauss-level reads through the shared response
catalog in :mod:`apeGmsh.opensees._response_catalog`; fibers, layers,
line stations, and per-element-node forces remain stubbed (empty
slabs) until their catalog entries land.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Sequence

import numpy as np
from numpy import ndarray

from .._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
    SpringSlab,
)
from .._time import resolve_time_slice
from . import _mpco_element_io as _melem
from . import _mpco_fiber_io as _mfiber
from . import _mpco_layer_io as _mlayer
from . import _mpco_line_io as _mline
from . import _mpco_local_force_io as _mlocal
from . import _mpco_material_io as _mmat
from . import _mpco_nodal_io as _mnodal
from . import _mpco_spring_io as _mspring
from . import _mpco_translation as _mtr
from ._protocol import ResultLevel, StageInfo, TimeSlice
from ._tag_translation import ElementTagTranslator

if TYPE_CHECKING:
    import h5py

    from ...mesh.FEMData import FEMData


# Stage groups in MPCO are ``MODEL_STAGE[<stamp>]``
_STAGE_PREFIX = "MODEL_STAGE["

#: The kinds a program stage takes (:meth:`MPCOReader.attach_stage_names`).
_PROGRAM_STAGE_KINDS = frozenset({"static", "transient"})


class MPCOBucketSkippedWarning(UserWarning):
    """A material-state, spring or line-station bucket the MPCO reader
    leaves unread.

    Its decoder refused the bucket (malformed ``META``, a ``NUM_COLUMNS``
    mismatch, a missing ``GP_X``, a section code with no canonical), so
    the elements it holds return no values. One warning per read names
    the file, every such bucket and the decoder's reason. The layer level
    names its unread buckets in
    :class:`~apeGmsh.results.readers._mpco_layer_io.LayerBucketUnreadWarning`.
    """


def _warn_skipped_buckets(
    skipped: "list[tuple[str, str]]", *, what: str, source: str,
) -> None:
    """One :class:`MPCOBucketSkippedWarning` naming every skipped bucket.

    ``skipped`` holds ``(bucket path, reason)`` pairs; an empty list
    warns nothing. ``source`` is the file name: a partition set repeats
    the same bucket path in every part. ``stacklevel`` points at the
    reader method's caller.
    """
    if not skipped:
        return
    listing = "; ".join(f"{path}: {reason}" for path, reason in skipped)
    warnings.warn(
        f"MPCO {what} in {source} left {len(skipped)} bucket(s) "
        f"unread, so their elements return no values: {listing}.",
        MPCOBucketSkippedWarning, stacklevel=3,
    )


def _check_layer_indices(
    bucket_grp: "h5py.Group",
    layout: Any,
    component: str,
    *,
    element_ids: Optional[ndarray],
    gp_indices: Optional[ndarray],
    layer_indices: Optional[ndarray],
    where: str,
) -> None:
    """Raise ``ValueError`` on a ``gp_indices`` / ``layer_indices`` the
    bucket does not have.

    A bad caller argument is refused, not skipped. The check runs where
    :func:`_mpco_layer_io.read_layer_bucket_slab` would apply the filter:
    the bucket holds the component and a requested element.
    """
    if component not in layout.component_layout:
        return
    if element_ids is not None:
        ids = np.asarray(bucket_grp["ID"][...]).ravel().astype(np.int64)
        if not np.isin(ids, np.asarray(element_ids, dtype=np.int64)).any():
            return
    for name, sel, n in (
        ("gp_indices", gp_indices, layout.n_surface_gp),
        ("layer_indices", layer_indices, layout.n_layers),
    ):
        if sel is None:
            continue
        arr = np.asarray(sel, dtype=np.int64).ravel()
        if arr.size and (int(arr.max()) >= n or int(arr.min()) < 0):
            raise ValueError(
                f"{where}: {name} {arr.tolist()} out of range [0, {n})."
            )


def _stage_sort_key(name: str) -> "tuple[int, str]":
    """Numeric-stamp sort key for ``MODEL_STAGE[<k>]`` group names."""
    try:
        return (int(name[len(_STAGE_PREFIX):].rstrip("]")), name)
    except ValueError:
        return (2**31, name)


def _child(group: "h5py.Group", path: str):
    """Return the child at a ``/``-separated path, or ``None`` if any
    segment is absent.

    Uses membership (``H5Lexists``) at every level instead of
    ``Group.get(path)``. A multi-segment ``Group.get("A/B")`` returns
    ``None`` indistinguishably whether ``A`` or ``B`` is missing — and
    can mis-resolve a broken/soft link to ``None`` — masking a malformed
    file as "absent" (project ``h5py`` optional-child hazard rule).
    """
    node = group
    for segment in path.split("/"):
        if segment not in node:
            return None
        node = node[segment]
    return node


def _node_step_groups(stage_grp: "h5py.Group") -> "Iterator[tuple[str, h5py.Group]]":
    """``(name, DATA group)`` of each node result on the stage cadence."""
    on_nodes = _child(stage_grp, "RESULTS/ON_NODES")
    if on_nodes is None:
        return
    for result_name in on_nodes:
        if _mtr.is_modal_result_name(result_name):
            # Modal groups record one STEP per eigen solution, not on
            # the stage cadence; they are read as mode stages.
            continue
        data_grp = _child(on_nodes[result_name], "DATA")
        if data_grp is not None:
            yield result_name, data_grp


def _element_step_groups(stage_grp: "h5py.Group") -> "Iterator[tuple[str, h5py.Group]]":
    """``(token/bucket, DATA group)`` of each element bucket in the stage."""
    on_elements = _child(stage_grp, "RESULTS/ON_ELEMENTS")
    if on_elements is None:
        return
    for token in on_elements:
        for bucket_key in on_elements[token]:
            data_grp = _child(on_elements[token][bucket_key], "DATA")
            if data_grp is not None:
                yield f"{token}/{bucket_key}", data_grp


def _cadence_time(
    groups: "Iterable[tuple[str, h5py.Group]]",
) -> Optional[ndarray]:
    """TIME of each step of the first group with steps, or ``None``.

    Every other group must hold the same step count: STKO records all
    results on one cadence, and a differing count means step-indexed
    reads would misalign, so it raises.
    """
    ref_time: Optional[ndarray] = None
    ref_name = ""
    for name, data_grp in groups:
        step_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("STEP_")],
            key=lambda s: int(s.split("_", 1)[1]),
        )
        if not step_keys:
            continue
        if ref_time is None:
            ref_time = np.array(
                [float(_attr_scalar(data_grp[k].attrs.get("TIME", 0.0)))
                 for k in step_keys],
                dtype=np.float64,
            )
            ref_name = name
        elif len(step_keys) != ref_time.size:
            raise ValueError(
                f"MPCO stage step-count mismatch: result "
                f"{name!r} has {len(step_keys)} steps but "
                f"{ref_name!r} has {ref_time.size}. STKO records all "
                f"results on one cadence; this file is inconsistent "
                f"and step-indexed reads would misalign."
            )
    return ref_time


class MPCOReader:
    """Reader for STKO ``.mpco`` HDF5 files."""

    def __init__(self, path: str | Path) -> None:
        import h5py

        self._path = Path(path)
        self._h5: "h5py.File" = h5py.File(self._path, "r")
        self._stage_cache: Optional[list[StageInfo]] = None
        # MPCO: each apeGmsh stage_id maps to one MPCO group name
        self._stage_to_mpco: dict[str, str] = {}
        # Mode stage id -> (host MODEL_STAGE group, its ModeEntry).
        # Mode stages are not MODEL_STAGE groups, so they never enter
        # ``_stage_to_mpco``; reads branch on this map first.
        self._mode_refs: dict[str, tuple[str, _mnodal.ModeEntry]] = {}
        self._host_ids: list[str] = []
        self._fem_cache: "Optional[FEMData] | _Sentinel" = _SENTINEL
        # ADR 0043 slice 1.3 — MPCO buckets are keyed by the OpenSees ops
        # tag; the results API speaks fem_eid. Over a composed model the
        # two differ. ``from_mpco`` attaches an ElementTagTranslator built
        # from the bound model's element_meta; ``None`` = no translation
        # (direct constructions / pre-attach reads behave as before).
        self._tag_map: "Optional[ElementTagTranslator]" = None
        self._rotation_mask = _mnodal.RotationMask(self._path.name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def __enter__(self) -> "MPCOReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        # Release the h5py handle when the reader is garbage-collected
        # — otherwise re-running a notebook cell that recreates the
        # Results object can leave the previous reader holding a file
        # lock on Windows until the kernel restarts.
        self.close()

    # ------------------------------------------------------------------
    # Tag translation (ADR 0043 slice 1.3)
    # ------------------------------------------------------------------

    def attach_tag_map(self, tag_map: "ElementTagTranslator") -> None:
        """Attach the bound-model fem_eid↔ops-tag translator and declare
        to it the ops tags this file records (#2066).

        Called by :meth:`Results.from_mpco` once the model is loaded.
        """
        tag_map.declare_recorded(self._recorded_element_tags())
        self._tag_map = tag_map

    def _recorded_element_tags(self) -> ndarray:
        """Ops tags of every element the file records, over all stages:
        the ``MODEL/ELEMENTS`` listing (a recorder ``region`` writes only
        its elements there) and each ``RESULTS/ON_ELEMENTS`` bucket ``ID``.
        """
        import h5py

        self._ensure_stages()
        parts = [np.empty(0, dtype=np.int64)]
        for name in self._stage_to_mpco.values():
            stage = self._h5[name]
            elements = _child(stage, "MODEL/ELEMENTS")
            for ds_name in elements if elements is not None else ():
                ds = elements[ds_name]
                if (isinstance(ds, h5py.Dataset) and ds.ndim == 2
                        and ds.shape[0] and ds.shape[1]):
                    parts.append(np.asarray(ds[:, 0], dtype=np.int64))
            on_elements = _child(stage, "RESULTS/ON_ELEMENTS")
            for token in on_elements.values() if on_elements is not None else ():
                for bucket in token.values() if isinstance(token, h5py.Group) else ():
                    if isinstance(bucket, h5py.Group) and "ID" in bucket:
                        parts.append(np.asarray(
                            bucket["ID"][...], dtype=np.int64).ravel())
        return np.unique(np.concatenate(parts))

    def attach_nodes_ndf(
        self, nodes_ndf: "dict[int, int] | None", ndm: int,
    ) -> None:
        """Store the model's per-node ndf: rotation reads then give NaN at
        a node whose DOF layout has no such rotation (#1971). ``None`` is
        an archive older than nodes_ndf.

        Called by :meth:`Results.from_mpco` once the model is loaded.
        """
        self._ensure_stages()
        self._rotation_mask.attach(nodes_ndf, ndm, _mnodal.model_node_ids(
            [self._h5[name] for name in self._stage_to_mpco.values()],
        ))

    def _element_read_translation(
        self, element_ids: "Optional[ndarray]",
    ) -> "tuple[Optional[ndarray], Any]":
        """Per-read ``(bucket_filter, index_to_fem_fn)`` — consistent pair.

        Both directions of one read are decided together (see
        :meth:`ElementTagTranslator.read_translation`): each filter id
        translates on its own, and an id the files cannot return warns.
        The relabel is element-wise and order-preserving, so collocated
        per-element data (natural coords, local-axes quaternions) stays
        aligned.
        """
        if self._tag_map is None:
            return element_ids, (lambda index: index)
        return self._tag_map.read_translation(element_ids)

    # ------------------------------------------------------------------
    # Stage discovery
    # ------------------------------------------------------------------

    def attach_stage_names(
        self, names: "Sequence[str]", kinds: "Sequence[str]",
    ) -> None:
        """Name and kind the capture stages after the program's stages.

        An ``.mpco`` names its stages ``MODEL_STAGE[<k>]`` and does not
        say whether a stage was static or transient; the user's
        ``ops.stage(name=...)`` names and each stage's ``analysis``
        record live only in the sibling ``model.h5`` (#1324, #1948).
        :meth:`Results.from_mpco` calls this with the program names and
        kinds, in registration order, once it has checked that the
        counts agree — the pairing is positional, so a count mismatch is
        the caller's decision, not a guess made here.  Stage ids are
        untouched (reads key on them); the ``MODEL_STAGE[<k>]`` name
        moves to :attr:`StageInfo.aliases`.
        """
        current = self.stages()
        hosts = [s for s in current if s.kind != "mode"]
        if not len(names) == len(kinds) == len(hosts):
            raise ValueError(
                f"attach_stage_names: {len(names)} names and "
                f"{len(kinds)} kinds for {len(hosts)} MODEL_STAGE groups "
                f"in {self._path.name}."
            )
        bad = sorted({str(k) for k in kinds} - _PROGRAM_STAGE_KINDS)
        if bad:
            raise ValueError(
                f"attach_stage_names: stage kinds {bad} are not program "
                f"stage kinds {sorted(_PROGRAM_STAGE_KINDS)}."
            )
        # Mode stages keep their own names and kind: they are not
        # program stages, and the pairing is positional over the hosts.
        new = {s.id: (str(n), str(k)) for s, n, k in zip(hosts, names, kinds)}
        self._stage_cache = [
            replace(s, name=new[s.id][0], kind=new[s.id][1],
                    aliases=(s.name, *s.aliases))
            if s.id in new else s
            for s in current
        ]

    def stages(self) -> list[StageInfo]:
        if self._stage_cache is not None:
            return list(self._stage_cache)

        hosts: list[StageInfo] = []
        # Map MPCO group names to apeGmsh stage_ids in stamp order —
        # sort by the numeric stamp inside ``MODEL_STAGE[<k>]``
        # (lexicographic sorting would put MODEL_STAGE[10] before
        # MODEL_STAGE[2] once a run reaches ten stages).
        names = sorted(
            (k for k in self._h5.keys() if k.startswith(_STAGE_PREFIX)),
            key=_stage_sort_key,
        )
        host_sets = []
        for name in names:
            grp = self._h5[name]
            time = self._build_time_vector_for_mpco_stage(grp)
            stage_id = f"stage_{len(hosts)}"
            self._stage_to_mpco[stage_id] = name
            hosts.append(StageInfo(
                id=stage_id,
                name=name,
                # An .mpco does not tag its stages: "transient" until
                # from_mpco(model_h5=) kinds them (attach_stage_names).
                kind="transient",
                n_steps=int(time.size),
            ))
            on_nodes = _child(grp, "RESULTS/ON_NODES")
            host_sets.append(
                [] if on_nodes is None else _mnodal.discover_modes(
                    on_nodes, source=f"{self._path.name} {name}",
                )
            )
        self._host_ids = [h.id for h in hosts]
        # Each recorded mode is its own kind="mode" stage, listed
        # right after the MODEL_STAGE that holds it (the layout of a
        # native results file).
        modes_after: dict[int, list[StageInfo]] = {}
        for h, sid, mname, entry in _mnodal.mode_stage_layout(
            self._host_ids, names, host_sets,
        ):
            self._mode_refs[sid] = (names[h], entry)
            modes_after.setdefault(h, []).append(StageInfo(
                id=sid, name=mname, kind="mode", n_steps=1,
                eigenvalue=entry.eigenvalue,
                frequency_hz=entry.frequency_hz,
                period_s=entry.period_s,
                mode_index=entry.mode_index,
            ))
        out: list[StageInfo] = []
        for h, host in enumerate(hosts):
            out.append(host)
            out.extend(modes_after.get(h, []))
        self._stage_cache = out
        return list(out)

    def _build_time_vector_for_mpco_stage(
        self, stage_grp: "h5py.Group",
    ) -> ndarray:
        """Aggregate TIME attrs from all STEP_<k> datasets in the stage.

        Reads the TIME attribute on each step of the first
        ``RESULTS/ON_NODES/<X>`` group with STEP data. STKO records every
        result on one shared step cadence, so the remaining node-result
        groups are cross-checked and a mismatch raises — a differing step
        count means later reads would index this time vector against the
        wrong steps.

        A recorder with only ``-E`` channels writes no node results; the
        cadence is then read the same way from the element buckets
        (``RESULTS/ON_ELEMENTS/<token>/<bucket>``), cross-checked among
        themselves.
        """
        time = _cadence_time(_node_step_groups(stage_grp))
        if time is None:
            time = _cadence_time(_element_step_groups(stage_grp))
        return time if time is not None else np.array([], dtype=np.float64)

    def time_vector(self, stage_id: str) -> ndarray:
        self._ensure_stages()
        if stage_id in self._mode_refs:
            return np.array([0.0], dtype=np.float64)
        mpco_name = self._stage_to_mpco[stage_id]
        return self._build_time_vector_for_mpco_stage(self._h5[mpco_name])

    def partitions(self, stage_id: str) -> list[str]:
        # MPCO file = single partition. Multi-partition runs use multiple
        # ``.part-<n>.mpco`` files; merging across them lands in a later
        # phase (open MPCOReader per file then stitch at the composite
        # layer).
        return ["partition_0"]

    # ------------------------------------------------------------------
    # FEM access
    # ------------------------------------------------------------------

    def spatial_dim(self) -> int:
        """Column count of the capture's ``MODEL/NODES/COORDINATES`` (2 or 3).

        The recorder stores ``ndm`` columns; :meth:`fem` pads two to
        three (``FEMData`` is always ``(N, 3)``), so the padded ``z``
        is not the model's.  :func:`results._bind._resolve_fem_via_model`
        compares coordinates over this many columns when the archive
        declares no ``ndm`` (#1393).  ``3`` when the file has no stage
        or no ``MODEL`` group (nothing to compare then).
        """
        self._ensure_stages()
        if not self._host_ids:
            return 3
        mpco_name = self._stage_to_mpco[self._host_ids[-1]]
        coords = _child(self._h5[mpco_name], "MODEL/NODES/COORDINATES")
        if coords is None or coords.ndim != 2 or coords.shape[1] not in (2, 3):
            return 3
        return int(coords.shape[1])

    def fem(self) -> "Optional[FEMData]":
        if not isinstance(self._fem_cache, _Sentinel):
            return self._fem_cache
        self._ensure_stages()
        if not self._host_ids:
            self._fem_cache = None
            return None
        # Use the LAST stage's MODEL — most up-to-date geometry.
        mpco_name = self._stage_to_mpco[self._host_ids[-1]]
        model_grp = _child(self._h5[mpco_name], "MODEL")
        if model_grp is None:
            self._fem_cache = None
            return None
        from ...mesh.FEMData import FEMData
        self._fem_cache = FEMData.from_mpco_model(model_grp)
        return self._fem_cache

    def opensees_model(self):
        """MPCO files carry no ``/opensees/`` zone — always ``None``.

        Per the ``project_mpco_no_vecxz`` memory and ADR 0020's
        MPCO-third-party-file case: the broker handle on a Results
        opened from MPCO is wired through the ``model_h5=`` kwarg on
        :meth:`Results.from_mpco`, not synthesised from the MPCO file.
        """
        return None

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        self._ensure_stages()
        if stage_id in self._mode_refs:
            if level.value != "nodes":
                return []
            on_nodes = _child(
                self._h5[self._mode_refs[stage_id][0]], "RESULTS/ON_NODES",
            )
            return [] if on_nodes is None else _mnodal.modal_components(on_nodes)
        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return []

        if level.value == "nodes":
            on_nodes = _child(self._h5[mpco_name], "RESULTS/ON_NODES")
            if on_nodes is None:
                return []
            out: set[str] = set()
            for result_name in on_nodes:
                grp = on_nodes[result_name]
                comps = _parse_components_attr(grp.attrs.get("COMPONENTS"))
                for comp_label in comps:
                    canonical = _mtr.canonical_node_component(
                        result_name, comp_label,
                    )
                    if canonical is not None:
                        out.add(canonical)
            return sorted(out)

        if level.value == "gauss":
            return self._gauss_available_components(mpco_name)

        if level.value == "line_stations":
            return self._line_stations_available_components(mpco_name)

        if level.value == "elements":
            return self._elements_available_components(mpco_name)

        if level.value == "fibers":
            return self._fibers_available_components(mpco_name)

        if level.value == "layers":
            return self._layers_available_components(mpco_name)

        if level.value == "springs":
            return self._springs_available_components(mpco_name)

        return []

    def _gauss_available_components(self, mpco_name: str) -> list[str]:
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        # Tensor canonicals (catalog-driven, fixed shape): any axis
        # suffix discovers the same buckets, so probe with ``_xx``.
        for prefix in ("stress", "strain"):
            _, buckets = _melem.discover_gauss_buckets(
                on_elements, canonical_component=f"{prefix}_xx",
            )
            for b in buckets:
                out.update(b.layout.component_layout)
        # Material-state canonicals (META-driven, variable shape).
        # Probe each parent token; surface every per-segment canonical
        # the bucket's META declares.
        skipped: list[tuple[str, str]] = []
        for parent in ("damage", "equivalent_plastic_strain", "plastic_strain"):
            _, mat_buckets = _mmat.discover_material_state_buckets(
                on_elements, canonical_component=parent,
            )
            for mb in mat_buckets:
                bucket_grp = on_elements[mb.mpco_group_name][mb.bracket_key]
                try:
                    canonicals = _mmat.material_state_canonicals_in_bucket(
                        bucket_grp, mb,
                    )
                except ValueError as exc:
                    # A refused bucket (malformed META) leaves the listing
                    # and is named in the one warning below.
                    skipped.append((bucket_grp.name, str(exc)))
                    continue
                out.update(canonicals)
        _warn_skipped_buckets(
            skipped, what="gauss material-state component listing",
            source=self._path.name,
        )
        return sorted(out)

    def _elements_available_components(
        self, mpco_name: str,
    ) -> list[str]:
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        # Probe each frame via a representative canonical name; the
        # discover helper handles the canonical→MPCO group-name
        # routing and bucket filtering against NODAL_FORCE_CATALOG.
        for probe in (
            "nodal_resisting_force_x",       # globalForce
            "nodal_resisting_force_local_x", # localForce
        ):
            _, buckets = _mnodal.discover_nodal_force_buckets(
                on_elements, canonical_component=probe,
            )
            for b in buckets:
                out.update(b.layout.component_layout)
        return sorted(out)

    def _fibers_available_components(self, mpco_name: str) -> list[str]:
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        for canonical in ("fiber_stress", "fiber_strain"):
            _, buckets = _mfiber.discover_fiber_buckets(
                on_elements, canonical_component=canonical,
            )
            if buckets:
                out.add(canonical)
        return sorted(out)

    def _layers_available_components(self, mpco_name: str) -> list[str]:
        stage_grp = self._h5[mpco_name]
        on_elements = _child(stage_grp, "RESULTS/ON_ELEMENTS")
        section_assignments = _child(stage_grp, "MODEL/SECTION_ASSIGNMENTS")
        if on_elements is None or section_assignments is None:
            return []
        out: set[str] = set()
        unread: list[tuple[str, str]] = []
        # For each parent token, walk every catalogued bucket and
        # surface its META-resolved per-cell canonicals (single-
        # component buckets keep the bare name; multi-component
        # buckets expose ``fiber_stress_<i>`` per index). A bucket the
        # read would refuse (the same resolve + NUM_COLUMNS check) is
        # left out of the listing and named in one warning.
        for parent in ("fiber_stress", "fiber_strain"):
            _, buckets, scan_unread = _mlayer.scan_layer_buckets(
                on_elements, canonical_component=parent,
            )
            unread.extend(scan_unread)
            for b in buckets:
                bucket_grp = on_elements[b.mpco_group_name][b.bracket_key]
                try:
                    layout = _mlayer.resolve_layer_bucket_layout(
                        section_assignments, bucket_grp, b,
                    )
                    _mlayer.validate_layer_bucket_meta(
                        bucket_grp, layout, bracket_key=b.bracket_key,
                    )
                except ValueError as exc:
                    unread.append((
                        f"ON_ELEMENTS/{b.mpco_group_name}/{b.bracket_key}",
                        f"its layout is refused ({exc})",
                    ))
                    continue
                out.update(layout.component_layout)
        _mlayer.warn_unread_layer_buckets(
            unread, source=self._path.name, stacklevel=3,
        )
        return sorted(out)

    def _line_stations_available_components(
        self, mpco_name: str,
    ) -> list[str]:
        stage_grp = self._h5[mpco_name]
        on_elements = _child(stage_grp, "RESULTS/ON_ELEMENTS")
        model_elements = _child(stage_grp, "MODEL/ELEMENTS")
        if on_elements is None or model_elements is None:
            return []
        out: set[str] = set()
        skipped: list[tuple[str, str]] = []

        # ── section.force / Custom-rule path (force-based beams) ─────
        # Discover line-stations buckets via any line-station canonical
        # (the routing only depends on the topology, not the specific
        # component name); ``axial_force`` is always present in v1.
        if model_elements is not None:
            _, buckets = _mline.discover_line_station_buckets(
                on_elements, canonical_component="axial_force",
            )
            for bucket in buckets:
                try:
                    token_grp = on_elements["section.force"][bucket.bracket_key]
                    layout, _ = _mline.resolve_bucket_layout(
                        model_elements, token_grp, bucket,
                    )
                except (ValueError, KeyError) as exc:
                    # A section code with no canonical (KeyError) or a
                    # missing GP_X / malformed META (ValueError): the read
                    # raises on it, and the listing names it.
                    skipped.append((
                        f"ON_ELEMENTS/section.force/{bucket.bracket_key}",
                        f"{type(exc).__name__}: {exc}",
                    ))
                    continue
                out.update(layout.component_layout)
        _warn_skipped_buckets(
            skipped, what="line-stations component listing",
            source=self._path.name,
        )

        # ── localForce path (stiffness beams: ElasticBeam2d/3d) ──────
        out.update(_mlocal.available_components_in_local_force(on_elements))

        # ── basicForce end moments (rank 2). globalForce and every
        # other token list nothing: they are never a station source. ──
        for tok in ("basicForce", "basicForces"):
            for bf in _mlocal.discover_basic_force_buckets(
                on_elements, token=tok,
            ):
                out.update(bf.layout)

        return sorted(out)

    def _springs_available_components(self, mpco_name: str) -> list[str]:
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return []
        out: set[str] = set()
        skipped: list[tuple[str, str]] = []
        for root in ("spring_force", "spring_deformation"):
            _, buckets = _mspring.discover_spring_buckets(
                on_elements, canonical_component=root,
            )
            for b in buckets:
                bucket_grp = on_elements[b.mpco_group_name][b.bracket_key]
                try:
                    canonicals = _mspring.spring_canonicals_in_bucket(
                        bucket_grp, b,
                    )
                except ValueError as exc:
                    skipped.append((bucket_grp.name, str(exc)))
                    continue
                out.update(canonicals)
        _warn_skipped_buckets(
            skipped, what="springs component listing", source=self._path.name,
        )
        return sorted(out)

    # ------------------------------------------------------------------
    # Slab reads — nodes
    # ------------------------------------------------------------------

    def node_partition_reduction(
        self, stage_id: str, component: str,
    ) -> Optional[str]:
        """How the partition copies of ``component`` combine.

        ``.mpco`` carries no ``PARTITION_REDUCTION`` attribute, so the
        kind comes from the result group name (reactions and unbalanced
        loads sum; the rest is the same in every copy). ``None`` when the
        file does not record the component.
        """
        self._ensure_stages()
        if stage_id in self._mode_refs:
            # A mode shape is the same at every copy of a node.
            comps = self.available_components(stage_id, ResultLevel.NODES)
            return _mtr.PARTITION_REDUCTION_NONE if component in comps else None
        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return None
        on_nodes = _child(self._h5[mpco_name], "RESULTS/ON_NODES")
        if on_nodes is None:
            return None
        loc = _locate_canonical_in_mpco(on_nodes, component)
        if loc is None:
            return None
        return _mtr.partition_reduction_from_name(loc[0])

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        slab = self._read_nodes_as_recorded(
            stage_id, component, node_ids=node_ids, time_slice=time_slice,
        )
        host = (
            self._mode_refs[stage_id][0] if stage_id in self._mode_refs
            else self._stage_to_mpco[stage_id]
        )
        return self._rotation_mask.apply(
            slab, _child(self._h5[host], "MODEL/ELEMENTS"),
        )

    def _read_nodes_as_recorded(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        if stage_id in self._mode_refs:
            return self._read_mode_nodes(
                stage_id, component, node_ids, time, t_idx,
            )
        mpco_name = self._stage_to_mpco[stage_id]
        on_nodes = _child(self._h5[mpco_name], "RESULTS/ON_NODES")
        if on_nodes is None:
            return _empty_node_slab(component, time, t_idx)

        # Find the (mpco_result_name, column_index) for the requested canonical.
        loc = _locate_canonical_in_mpco(on_nodes, component)
        if loc is None:
            return _empty_node_slab(component, time, t_idx)
        mpco_result_name, col_idx = loc

        result_grp = on_nodes[mpco_result_name]
        data_grp = _child(result_grp, "DATA")
        if data_grp is None:
            return _empty_node_slab(component, time, t_idx)

        # Read the result's ID array (subset of model nodes for this result).
        all_ids = (
            np.asarray(result_grp["ID"][...], dtype=np.int64).flatten()
            if "ID" in result_grp
            else self._fallback_node_ids(mpco_name)
        )

        # Apply node_ids filter
        if node_ids is None:
            sel = np.arange(all_ids.size)
            sel_ids = all_ids
        else:
            requested = np.asarray(node_ids, dtype=np.int64)
            mask = np.isin(all_ids, requested)
            if not mask.any():
                return NodeSlab(
                    component=component,
                    values=np.zeros((t_idx.size, 0), dtype=np.float64),
                    node_ids=np.array([], dtype=np.int64),
                    time=time[t_idx],
                )
            sel = np.where(mask)[0]
            sel_ids = all_ids[sel]

        # Read selected steps and selected node rows.
        step_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("STEP_")],
            key=lambda s: int(s.split("_", 1)[1]),
        )
        values = np.empty((t_idx.size, sel_ids.size), dtype=np.float64)
        for i, k in enumerate(t_idx):
            step_ds = data_grp[step_keys[int(k)]]
            row = step_ds[:, col_idx]                 # (nNodes,)
            values[i, :] = row[sel]

        return NodeSlab(
            component=component, values=values,
            node_ids=sel_ids, time=time[t_idx],
        )

    def _read_mode_nodes(
        self, stage_id: str, component: str,
        node_ids: Optional[ndarray], time: ndarray, t_idx: ndarray,
    ) -> NodeSlab:
        """One mode's shape as a single-station node slab."""
        host, entry = self._mode_refs[stage_id]
        on_nodes = _child(self._h5[host], "RESULTS/ON_NODES")
        got = (
            None if on_nodes is None
            else _mnodal.read_mode_shape(
                on_nodes, entry, component, node_ids=node_ids,
            )
        )
        if got is None:
            return _empty_node_slab(component, time, t_idx)
        values, ids = got
        return NodeSlab(
            component=component,
            values=np.repeat(values[None, :], t_idx.size, axis=0),
            node_ids=ids, time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Element-level reads — Phase 3 stubs
    # ------------------------------------------------------------------

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_element_slab(component, time, time_slice)
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return _empty_element_slab(component, time, time_slice)

        element_ids, index_to_fem = self._element_read_translation(element_ids)

        token, buckets, skipped = _mnodal.scan_nodal_force_buckets(
            on_elements, canonical_component=component,
        )
        if skipped:
            assert token is not None    # a skip is a bucket under token
            _mnodal.warn_skipped_nodal_buckets(
                on_elements[token], skipped, component=component,
                element_ids=element_ids, source=self._path.name,
            )
        if not buckets:
            return _empty_element_slab(component, time, time_slice)
        token_grp = on_elements[token]

        # Read per-bucket slabs and stitch on the element axis.
        values_parts: list[ndarray] = []
        element_id_parts: list[ndarray] = []

        for bucket in buckets:
            bucket_grp = token_grp[bucket.bracket_key]
            result = _mnodal.read_nodal_bucket_slab(
                bucket_grp, bucket, component,
                t_idx=t_idx, element_ids=element_ids,
            )
            if result is None:
                continue
            values, eids = result
            values_parts.append(values)
            element_id_parts.append(eids)

        if not values_parts:
            return _empty_element_slab(component, time, time_slice)

        return ElementSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_ids=index_to_fem(np.concatenate(element_id_parts)),
            time=time[t_idx],
        )

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_line_station_slab(component, time, t_idx)
        stage_grp = self._h5[mpco_name]
        on_elements = _child(stage_grp, "RESULTS/ON_ELEMENTS")
        model_elements = _child(stage_grp, "MODEL/ELEMENTS")
        if on_elements is None or model_elements is None:
            return _empty_line_station_slab(component, time, t_idx)

        # Read per-bucket slabs and stitch on the station axis. Each
        # element takes one source, the first by rank:
        # ``section.force`` (DispBeamColumn3d writes it beside
        # ``localForce``, the IP-level detail beside an end-force
        # summary), then ``localForce`` (ElasticBeam2d/3d), then the
        # ``basicForce`` end moments. ``globalForce`` and every other
        # token are never a source; a requested element they alone name
        # is warned about by bucket.

        element_ids, index_to_fem = self._element_read_translation(element_ids)

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        station_coord_parts: list[ndarray] = []
        section_force_eids: set[int] = set()

        # ── section.force / Custom-rule path ────────────────────────
        token, buckets = _mline.discover_line_station_buckets(
            on_elements, canonical_component=component,
        )
        if buckets and token is not None and token in on_elements:
            token_grp = on_elements[token]
            for bucket in buckets:
                bucket_grp = token_grp[bucket.bracket_key]
                result = _mline.read_line_bucket_slab(
                    bucket_grp, model_elements, bucket, component,
                    t_idx=t_idx, element_ids=element_ids,
                )
                if result is None:
                    continue
                values, element_index, station_coord = result
                values_parts.append(values)
                element_index_parts.append(element_index)
                station_coord_parts.append(station_coord)
                section_force_eids.update(int(e) for e in element_index)

        # ── localForce (rank 1), then basicForce end moments (rank 2),
        # each skipping every element a higher rank or an earlier bucket
        # covers. A plural token reads as its singular, after it. ──
        covered = set(section_force_eids)
        skipped: list[tuple[str, str]] = []

        def _bare(bucket_grp: "h5py.Group") -> ndarray:
            """Requested ids in the bucket that no source covers yet."""
            ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)
            if element_ids is not None:
                ids = ids[np.isin(ids, np.asarray(element_ids, dtype=np.int64))]
            return np.unique(ids[~np.isin(ids, list(covered))])

        for src_token, ranked in (
            (tok, discover(on_elements, token=tok))
            for tok, discover in (
                ("localForce", _mlocal.discover_local_force_buckets),
                ("localForces", _mlocal.discover_local_force_buckets),
                ("basicForce", _mlocal.discover_basic_force_buckets),
                ("basicForces", _mlocal.discover_basic_force_buckets),
            )
        ):
            for lf in ranked:
                bucket_grp = on_elements[src_token][lf.bracket_key]
                if lf.refusal is not None:
                    # A refused layout names no component reliably. It
                    # blocks only requested elements no higher-ranked
                    # source covers, as on .ladruno.
                    if "ID" not in bucket_grp:
                        raise ValueError(lf.refusal)
                    bare = _bare(bucket_grp)
                    if bare.size:
                        raise ValueError(
                            f"{lf.refusal} (requested element(s) "
                            f"{bare[:10].tolist()} are in this bucket, with "
                            f"no higher-ranked stations for {component!r})"
                        )
                    continue
                for comp, reason in lf.refused:
                    if comp != component or "ID" not in bucket_grp:
                        continue
                    bare = _bare(bucket_grp)
                    if bare.size:
                        skipped.append((
                            f"ON_ELEMENTS/{src_token}/{lf.bracket_key}",
                            f"{reason} Element(s) {bare[:10].tolist()} get "
                            f"no {component!r} stations",
                        ))
                result = _mlocal.read_local_force_bucket_slab(
                    bucket_grp, lf, component,
                    t_idx=t_idx, element_ids=element_ids,
                )
                if result is None:
                    continue
                values, element_index, station_coord = result
                keep_mask = ~np.isin(element_index, list(covered))
                if not keep_mask.any():
                    continue
                values_parts.append(values[:, keep_mask])
                element_index_parts.append(element_index[keep_mask])
                station_coord_parts.append(station_coord[keep_mask])
                covered.update(int(e) for e in element_index[keep_mask])

        # ── Never a source: globalForce and every other token.
        # Name the ones that were a requested element's only candidate.
        unranked: list[tuple[str, str]] = []
        for where, bucket_grp in _mlocal.unranked_station_buckets(
            on_elements, component,
        ):
            bare = _bare(bucket_grp)
            if bare.size:
                unranked.append((f"ON_ELEMENTS/{where}", (
                    f"names {component!r} for element(s) "
                    f"{bare[:10].tolist()}, but {where.split('/', 1)[0]!r} "
                    f"is never a line-station source (section.force, then "
                    f"localForce, then basicForce are; globalForce end "
                    f"forces are global, not station values)"
                )))
        _warn_skipped_buckets(
            skipped + unranked, what="line-stations read",
            source=self._path.name,
        )

        if not values_parts:
            return _empty_line_station_slab(component, time, t_idx)

        return LineStationSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=index_to_fem(
                np.concatenate(element_index_parts)
            ),
            station_natural_coord=np.concatenate(station_coord_parts),
            time=time[t_idx],
        )

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_gauss_slab(component, time, t_idx)
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return _empty_gauss_slab(component, time, t_idx)

        element_ids, index_to_fem = self._element_read_translation(element_ids)

        # Dispatch on canonical type: material-state tokens
        # (``damage`` / ``equivalent_plastic_strain`` and their
        # ``_tension``/``_compression`` variants) take the META-driven
        # path; everything else uses the catalog-driven path.
        if _mmat.parent_token_for_canonical(component) is not None:
            return self._read_gauss_material_state(
                on_elements, component, time, t_idx, element_ids,
                index_to_fem,
            )

        token, buckets = _melem.discover_gauss_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_gauss_slab(component, time, t_idx)

        # Read per-bucket slabs and stitch on the GP/element axis.
        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        natural_coords_parts: list[ndarray] = []

        for bucket in buckets:
            # Buckets may live under different alias group names
            # (e.g. legacy ``stresses`` vs. modern ``material.stress``)
            # — use each bucket's own source group.
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            result = _melem.read_bucket_slab(
                bucket_grp, bucket, component,
                t_idx=t_idx, element_ids=element_ids,
            )
            if result is None:
                continue
            values, element_index, natural_coords = result
            values_parts.append(values)
            element_index_parts.append(element_index)
            natural_coords_parts.append(natural_coords)

        if not values_parts:
            return _empty_gauss_slab(component, time, t_idx)

        return GaussSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=index_to_fem(
                np.concatenate(element_index_parts)
            ),
            natural_coords=np.concatenate(natural_coords_parts, axis=0),
            local_axes_quaternion=None,
            time=time[t_idx],
        )

    def _read_gauss_material_state(
        self,
        on_elements: "h5py.Group",
        component: str,
        time: ndarray,
        t_idx: ndarray,
        element_ids: Optional[ndarray],
        index_to_fem: Any,
    ) -> GaussSlab:
        """META-driven material-state read — separate from the catalog
        path because ``n_components_per_gp`` depends on the assigned
        constitutive model, not just the element class."""
        token, buckets = _mmat.discover_material_state_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_gauss_slab(component, time, t_idx)

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        natural_coords_parts: list[ndarray] = []

        skipped: list[tuple[str, str]] = []
        for bucket in buckets:
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            try:
                result = _mmat.read_material_state_bucket_slab(
                    bucket_grp, bucket, component,
                    t_idx=t_idx, element_ids=element_ids,
                )
            except ValueError as exc:
                # A refused bucket (malformed META) is named in the one
                # warning below; MaterialStateComponentRefused is a
                # LookupError and reaches the caller.
                skipped.append((bucket_grp.name, str(exc)))
                continue
            if result is None:
                continue
            values, element_index, natural_coords = result
            values_parts.append(values)
            element_index_parts.append(element_index)
            natural_coords_parts.append(natural_coords)
        _warn_skipped_buckets(
            skipped, what=f"gauss material-state read of {component!r}",
            source=self._path.name,
        )

        if not values_parts:
            return _empty_gauss_slab(component, time, t_idx)

        return GaussSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=index_to_fem(
                np.concatenate(element_index_parts)
            ),
            natural_coords=np.concatenate(natural_coords_parts, axis=0),
            local_axes_quaternion=None,
            time=time[t_idx],
        )

    def read_fibers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> FiberSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_fiber_slab(component, time, t_idx)
        stage_grp = self._h5[mpco_name]
        on_elements = _child(stage_grp, "RESULTS/ON_ELEMENTS")
        model_elements = _child(stage_grp, "MODEL/ELEMENTS")
        section_assignments = _child(stage_grp, "MODEL/SECTION_ASSIGNMENTS")
        if (
            on_elements is None
            or model_elements is None
            or section_assignments is None
        ):
            return _empty_fiber_slab(component, time, t_idx)

        element_ids, index_to_fem = self._element_read_translation(element_ids)

        token, buckets = _mfiber.discover_fiber_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_fiber_slab(component, time, t_idx)
        token_grp = on_elements[token]

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        gp_index_parts: list[ndarray] = []
        station_xi_parts: list[ndarray] = []
        y_parts: list[ndarray] = []
        z_parts: list[ndarray] = []
        area_parts: list[ndarray] = []
        material_tag_parts: list[ndarray] = []

        for bucket in buckets:
            bucket_grp = token_grp[bucket.bracket_key]
            result = _mfiber.read_fiber_bucket_slab(
                bucket_grp, model_elements, section_assignments, bucket,
                t_idx=t_idx,
                element_ids=element_ids,
                gp_indices=gp_indices,
            )
            if result is None:
                continue
            values, ei, gpi, station_xi, y, z, area, mtag = result
            values_parts.append(values)
            element_index_parts.append(ei)
            gp_index_parts.append(gpi)
            station_xi_parts.append(station_xi)
            y_parts.append(y)
            z_parts.append(z)
            area_parts.append(area)
            material_tag_parts.append(mtag)

        if not values_parts:
            return _empty_fiber_slab(component, time, t_idx)

        return FiberSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=index_to_fem(
                np.concatenate(element_index_parts)
            ),
            gp_index=np.concatenate(gp_index_parts),
            y=np.concatenate(y_parts),
            z=np.concatenate(z_parts),
            area=np.concatenate(area_parts),
            material_tag=np.concatenate(material_tag_parts),
            time=time[t_idx],
            station_natural_coord=np.concatenate(station_xi_parts),
        )

    def read_layers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        layer_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LayerSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_layer_slab(component, time, t_idx)
        stage_grp = self._h5[mpco_name]
        on_elements = _child(stage_grp, "RESULTS/ON_ELEMENTS")
        section_assignments = _child(stage_grp, "MODEL/SECTION_ASSIGNMENTS")
        if on_elements is None or section_assignments is None:
            return _empty_layer_slab(component, time, t_idx)
        local_axes = _child(stage_grp, "MODEL/LOCAL_AXES")  # may be None

        element_ids, index_to_fem = self._element_read_translation(element_ids)

        token, buckets, unread = _mlayer.scan_layer_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            _mlayer.warn_unread_layer_buckets(
                unread, source=self._path.name, stacklevel=2,
            )
            return _empty_layer_slab(component, time, t_idx)

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []
        gp_index_parts: list[ndarray] = []
        layer_index_parts: list[ndarray] = []
        sub_gp_index_parts: list[ndarray] = []
        thickness_parts: list[ndarray] = []
        quaternion_parts: list[ndarray] = []

        for bucket in buckets:
            # Buckets may live under either ``material.fiber.<X>``
            # (swapped) or ``section.fiber.<X>`` (unswapped). Use
            # each bucket's own source group.
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            bucket_path = (
                f"ON_ELEMENTS/{bucket.mpco_group_name}/{bucket.bracket_key}"
            )
            if gp_indices is not None or layer_indices is not None:
                try:
                    layout = _mlayer.resolve_layer_bucket_layout(
                        section_assignments, bucket_grp, bucket,
                    )
                    _mlayer.validate_layer_bucket_meta(
                        bucket_grp, layout, bracket_key=bucket.bracket_key,
                    )
                except ValueError as exc:
                    unread.append((bucket_path, f"its read is refused ({exc})"))
                    continue
                _check_layer_indices(
                    bucket_grp, layout, component,
                    element_ids=element_ids, gp_indices=gp_indices,
                    layer_indices=layer_indices,
                    where=f"{self._path.name} {bucket_path}",
                )
            try:
                result = _mlayer.read_layer_bucket_slab(
                    bucket_grp, section_assignments, local_axes, bucket,
                    component,
                    t_idx=t_idx,
                    element_ids=element_ids,
                    gp_indices=gp_indices,
                    layer_indices=layer_indices,
                )
            except ValueError as exc:
                # A refused bucket (mixed layer counts, several sub-GPs
                # per layer, malformed META) is named, with the
                # buckets discovery left out, in one warning. An index
                # filter beyond the bucket raised above, before this try.
                unread.append((bucket_path, f"its read is refused ({exc})"))
                continue
            if result is None:
                continue
            (values, ei, gpi, lyri, subi,
             thick, quat) = result
            values_parts.append(values)
            element_index_parts.append(ei)
            gp_index_parts.append(gpi)
            layer_index_parts.append(lyri)
            sub_gp_index_parts.append(subi)
            thickness_parts.append(thick)
            quaternion_parts.append(quat)
        _mlayer.warn_unread_layer_buckets(
            unread, source=self._path.name, stacklevel=2,
        )

        if not values_parts:
            return _empty_layer_slab(component, time, t_idx)

        return LayerSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=index_to_fem(
                np.concatenate(element_index_parts)
            ),
            gp_index=np.concatenate(gp_index_parts),
            layer_index=np.concatenate(layer_index_parts),
            sub_gp_index=np.concatenate(sub_gp_index_parts),
            thickness=np.concatenate(thickness_parts),
            local_axes_quaternion=np.concatenate(quaternion_parts, axis=0),
            time=time[t_idx],
        )

    def read_springs(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> SpringSlab:
        self._ensure_stages()
        time = self.time_vector(stage_id)
        t_idx = resolve_time_slice(time_slice, time)

        mpco_name = self._stage_to_mpco.get(stage_id)
        if mpco_name is None:
            return _empty_spring_slab(component, time, t_idx)
        on_elements = _child(self._h5[mpco_name], "RESULTS/ON_ELEMENTS")
        if on_elements is None:
            return _empty_spring_slab(component, time, t_idx)

        element_ids, index_to_fem = self._element_read_translation(element_ids)

        _, buckets = _mspring.discover_spring_buckets(
            on_elements, canonical_component=component,
        )
        if not buckets:
            return _empty_spring_slab(component, time, t_idx)

        values_parts: list[ndarray] = []
        element_index_parts: list[ndarray] = []

        skipped: list[tuple[str, str]] = []
        for bucket in buckets:
            bucket_grp = on_elements[bucket.mpco_group_name][bucket.bracket_key]
            try:
                result = _mspring.read_spring_bucket_slab(
                    bucket_grp, bucket, component,
                    t_idx=t_idx,
                    element_ids=element_ids,
                )
            except ValueError as exc:
                skipped.append((bucket_grp.name, str(exc)))
                continue
            if result is None:
                continue
            values, ei = result
            values_parts.append(values)
            element_index_parts.append(ei)
        _warn_skipped_buckets(
            skipped, what=f"springs read of {component!r}",
            source=self._path.name,
        )

        if not values_parts:
            return _empty_spring_slab(component, time, t_idx)

        return SpringSlab(
            component=component,
            values=np.concatenate(values_parts, axis=1),
            element_index=index_to_fem(
                np.concatenate(element_index_parts)
            ),
            time=time[t_idx],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_stages(self) -> None:
        if self._stage_cache is None:
            self.stages()

    def _fallback_node_ids(self, mpco_stage_name: str) -> ndarray:
        """When a result group has no ID, fall back to MODEL/NODES/ID."""
        return np.asarray(
            self._h5[mpco_stage_name]["MODEL/NODES/ID"][...],
            dtype=np.int64,
        ).flatten()


# =====================================================================
# Module helpers
# =====================================================================

def attach_model_nodes_ndf(
    reader: object, model_h5: "str | Path", ndm: int,
) -> None:
    """Attach ``model_h5``'s per-node ndf to a one-file reader (#1971);
    see :func:`_mpco_nodal_io.attach_nodes_ndf`."""
    _mnodal.attach_nodes_ndf(reader, MPCOReader, model_h5, ndm)


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


def _attr_scalar(value):
    """Coerce an h5py attribute value (scalar or 1-element array) to a scalar."""
    if hasattr(value, "size") and getattr(value, "size", 0) == 1:
        return value.item() if hasattr(value, "item") else value[0]
    if hasattr(value, "__len__") and len(value) == 1:
        return value[0]
    return value


def _parse_components_attr(value) -> list[str]:
    """Parse the COMPONENTS attribute into a list of component labels.

    The attribute is typically a 1-element array of fixed-length bytes,
    e.g. ``array([b'Ux,Uy,Uz'])``. Some files may store it as a plain
    bytes / str.
    """
    if value is None:
        return []
    raw = _attr_scalar(value)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _locate_canonical_in_mpco(
    on_nodes_grp, canonical_component: str,
) -> tuple[str, int] | None:
    """Find ``(mpco_result_name, column_index)`` for a canonical component.

    Iterates the MPCO result groups under ``ON_NODES``, parsing each
    group's COMPONENTS attribute and translating each column to its
    canonical name. Returns the first match.
    """
    for mpco_result_name in on_nodes_grp:
        grp = on_nodes_grp[mpco_result_name]
        comps = _parse_components_attr(grp.attrs.get("COMPONENTS"))
        for col_idx, comp_label in enumerate(comps):
            canonical = _mtr.canonical_node_component(
                mpco_result_name, comp_label,
            )
            if canonical == canonical_component:
                return (mpco_result_name, col_idx)
    return None


def _empty_node_slab(component: str, time: ndarray, t_idx) -> NodeSlab:
    if t_idx is None:
        t_idx = np.arange(time.size, dtype=np.int64)
    return NodeSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        node_ids=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_element_slab(component: str, time: ndarray, time_slice) -> ElementSlab:
    t_idx = resolve_time_slice(time_slice, time)
    return ElementSlab(
        component=component,
        values=np.zeros((t_idx.size, 0, 0), dtype=np.float64),
        element_ids=np.array([], dtype=np.int64),
        time=time[t_idx],
    )


def _empty_gauss_slab(component: str, time: ndarray, t_idx) -> GaussSlab:
    return GaussSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        natural_coords=np.zeros((0, 3), dtype=np.float64),
        local_axes_quaternion=None,
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_line_station_slab(
    component: str, time: ndarray, t_idx,
) -> LineStationSlab:
    return LineStationSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        station_natural_coord=np.array([], dtype=np.float64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_fiber_slab(component: str, time: ndarray, t_idx) -> FiberSlab:
    return FiberSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        gp_index=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.float64),
        z=np.array([], dtype=np.float64),
        area=np.array([], dtype=np.float64),
        material_tag=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_layer_slab(component: str, time: ndarray, t_idx) -> LayerSlab:
    return LayerSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        gp_index=np.array([], dtype=np.int64),
        layer_index=np.array([], dtype=np.int64),
        sub_gp_index=np.array([], dtype=np.int64),
        thickness=np.array([], dtype=np.float64),
        local_axes_quaternion=np.zeros((0, 4), dtype=np.float64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )


def _empty_spring_slab(component: str, time: ndarray, t_idx) -> SpringSlab:
    return SpringSlab(
        component=component,
        values=np.zeros((np.size(t_idx), 0), dtype=np.float64),
        element_index=np.array([], dtype=np.int64),
        time=time[t_idx] if time.size else np.array([], dtype=np.float64),
    )
