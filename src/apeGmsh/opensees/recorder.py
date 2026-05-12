"""
Typed ``recorder`` primitives.

Phase 3B ships three concrete recorder classes mirroring the OpenSees
``recorder`` command:

  * :class:`Node`    — ``recorder Node ...``
  * :class:`Element` — ``recorder Element ...``
  * :class:`MPCO`    — ``recorder mpco ...`` (HDF5)

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``;
the matching :class:`apeGmsh.opensees._internal.ns.recorder._RecorderNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Recorders never compose other primitives (``dependencies()`` returns
``()``). They are leaves in the dependency graph; the build pipeline
emits them after the topology + analysis chain so that each ``recorder``
command sees fully-allocated node and element tags.

The ``pg=`` form (physical-group fan-out into node/element tags) is
declared on the type signatures for forward-compatibility but
:meth:`_emit` raises :class:`NotImplementedError` until the Phase 4
build pipeline materializes the FEM-snapshot lookup. Recorders
constructed today supply explicit ``nodes=`` / ``elements=`` lists.

OpenSees command shapes
-----------------------

::

    recorder Node    -file fname [-time] [-dT dT] [-node n...]
                                 -dof d... response
    recorder Element -file fname [-time] [-dT dT] [-ele e...]
                                 response_tokens...
    recorder mpco    fname.mpco  [-N nodal_responses...]
                                 [-E elem_responses...]  [-T dT_or_nsteps]

The ``-time`` flag (when ``time_format="dt"``) instructs OpenSees to
include the simulation-time column in the output file. The default
``time_format="step"`` writes only the response columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._vocabulary import (
    DERIVED_SCALARS,
    FIBER,
    LINE_DIAGRAMS,
    MATERIAL_STATE,
    NODAL_FORCES,
    NODAL_KINEMATICS,
    PER_ELEMENT_NODAL_FORCES,
    STRAIN,
    STRESS,
    is_canonical,
)
from ._internal.types import Primitive, Recorder

if TYPE_CHECKING:
    from .emitter.base import Emitter


__all__ = [
    # Typed primitives (Phase 3B)
    "Node",
    "Element",
    "MPCO",
    # Unified declaration (Phase 9)
    "RecorderRecord",
    "RecorderDeclaration",
    "ALL_RECORDER_CATEGORIES",
]


# ---------------------------------------------------------------------------
# Node — ``recorder Node ...``
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Node(Recorder):
    """``recorder Node`` — record nodal response history.

    OpenSees command::

        recorder Node -file fname [-time] [-dT dT]
                      (-node n1 n2 ... | -nodeRange first last)
                      -dof d1 d2 ... response

    Exactly one of ``nodes=`` (explicit list) or ``pg=`` (physical-group
    label) must be supplied; the build pipeline (Phase 4) materializes
    the ``pg=`` form into a concrete node-tag list. Until then, the
    ``pg=`` path raises :class:`NotImplementedError` from :meth:`_emit`.

    Parameters
    ----------
    file
        Output file path.
    response
        OpenSees response token (``"disp"``, ``"vel"``, ``"accel"``,
        ``"reaction"``, ``"unbalance"``, ...).
    nodes
        Explicit tuple of node tags. Mutually exclusive with ``pg``.
    pg
        Physical-group label whose nodes the recorder targets.
        Mutually exclusive with ``nodes``. Build-pipeline only.
    dofs
        DOF indices (1-based, OpenSees convention). At least one
        required.
    dT
        Optional cadence — record only every ``dT`` simulation
        seconds. ``None`` records every step.
    time_format
        ``"step"`` (default) writes only response columns;
        ``"dt"`` emits the OpenSees ``-time`` flag, prepending the
        simulation-time column.
    """

    file: str
    response: str
    nodes: tuple[int, ...] | None = None
    pg: str | None = None
    dofs: tuple[int, ...]
    dT: float | None = None
    time_format: str = "step"

    def __post_init__(self) -> None:
        if (self.nodes is None) == (self.pg is None):
            raise ValueError(
                "Node recorder: supply exactly one of nodes= or pg= "
                f"(got nodes={self.nodes!r}, pg={self.pg!r})."
            )
        if not self.dofs:
            raise ValueError(
                "Node recorder: at least one dof required."
            )
        if self.time_format not in ("step", "dt"):
            raise ValueError(
                "Node recorder: time_format must be 'step' or 'dt', "
                f"got {self.time_format!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[int | float | str] = ["-file", self.file]
        if self.dT is not None:
            args += ["-dT", self.dT]
        if self.time_format == "dt":
            args += ["-time"]
        if self.nodes is not None:
            args += ["-node", *self.nodes]
        else:
            # pg → node fan-out is build-pipeline territory (Phase 4).
            raise NotImplementedError(
                "Node recorder pg= deferred to Phase 4 build pipeline; "
                "supply explicit nodes= for now."
            )
        args += ["-dof", *self.dofs, self.response]
        emitter.recorder("Node", *args)


# ---------------------------------------------------------------------------
# Element — ``recorder Element ...``
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Element(Recorder):
    """``recorder Element`` — record element-level response history.

    OpenSees command::

        recorder Element -file fname [-time] [-dT dT]
                         (-ele e1 e2 ... | -eleRange first last)
                         response_tokens...

    ``response`` is a tuple of OpenSees response tokens — the simplest
    case is ``("globalForce",)`` or ``("stresses",)``; element types
    that nest responses (e.g. fiber sections) take multi-token forms
    such as ``("section", "1", "force")``.

    Exactly one of ``elements=`` (explicit list) or ``pg=`` (physical-
    group label) must be supplied; ``pg=`` is deferred to Phase 4.

    Parameters
    ----------
    file
        Output file path.
    response
        Tuple of OpenSees response tokens (at least one).
    elements
        Explicit tuple of element tags. Mutually exclusive with ``pg``.
    pg
        Physical-group label whose elements the recorder targets.
        Mutually exclusive with ``elements``. Build-pipeline only.
    dT
        Optional cadence — record only every ``dT`` simulation
        seconds. ``None`` records every step.
    time_format
        ``"step"`` (default) writes only response columns;
        ``"dt"`` emits the OpenSees ``-time`` flag.
    """

    file: str
    response: tuple[str, ...]
    elements: tuple[int, ...] | None = None
    pg: str | None = None
    dT: float | None = None
    time_format: str = "step"

    def __post_init__(self) -> None:
        if (self.elements is None) == (self.pg is None):
            raise ValueError(
                "Element recorder: supply exactly one of elements= or "
                f"pg= (got elements={self.elements!r}, pg={self.pg!r})."
            )
        if not self.response:
            raise ValueError(
                "Element recorder: response token required."
            )
        if self.time_format not in ("step", "dt"):
            raise ValueError(
                "Element recorder: time_format must be 'step' or "
                f"'dt', got {self.time_format!r}."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[int | float | str] = ["-file", self.file]
        if self.dT is not None:
            args += ["-dT", self.dT]
        if self.time_format == "dt":
            args += ["-time"]
        if self.elements is not None:
            args += ["-ele", *self.elements]
        else:
            # pg → element fan-out is build-pipeline territory (Phase 4).
            raise NotImplementedError(
                "Element recorder pg= deferred to Phase 4 build "
                "pipeline; supply explicit elements= for now."
            )
        args += list(self.response)
        emitter.recorder("Element", *args)


# ---------------------------------------------------------------------------
# MPCO — ``recorder mpco ...`` (HDF5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class MPCO(Recorder):
    """``recorder mpco`` — write a single HDF5 ``.mpco`` file.

    OpenSees command::

        recorder mpco fname.mpco [-N nodal_responses...]
                                 [-E elem_responses...]
                                 [-T dT_or_nsteps]

    The MPCO recorder captures the full response tensor for each
    requested token (no per-DOF selection at write time); STKO /
    apeGmsh consumers filter at read time. At least one of
    ``nodal_responses`` or ``elem_responses`` must be non-empty.

    Cadence is selected by exactly one of ``dT`` (seconds) or
    ``nsteps`` (analysis steps). Supplying both raises ``ValueError``;
    supplying neither records every analysis step.

    Parameters
    ----------
    file
        Output ``.mpco`` (HDF5) file path.
    nodal_responses
        Tuple of MPCO ``-N`` tokens (e.g. ``("displacement",
        "reactionForce")``). Empty tuple means no nodal recording.
    elem_responses
        Tuple of MPCO ``-E`` tokens (e.g. ``("stresses",
        "section.fiber.stress")``). Empty tuple means no element
        recording.
    dT
        Optional time-based cadence (seconds). Mutually exclusive
        with ``nsteps``.
    nsteps
        Optional step-based cadence (every N analysis steps).
        Mutually exclusive with ``dT``.
    """

    file: str
    nodal_responses: tuple[str, ...] = ()
    elem_responses: tuple[str, ...] = ()
    dT: float | None = None
    nsteps: int | None = None

    def __post_init__(self) -> None:
        if not (self.nodal_responses or self.elem_responses):
            raise ValueError(
                "MPCO: at least one of nodal_responses or "
                "elem_responses required."
            )
        if self.dT is not None and self.nsteps is not None:
            raise ValueError(
                "MPCO: supply only one of dT or nsteps "
                f"(got dT={self.dT!r}, nsteps={self.nsteps!r})."
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[int | float | str] = [self.file]
        if self.nodal_responses:
            args += ["-N", *self.nodal_responses]
        if self.elem_responses:
            args += ["-E", *self.elem_responses]
        if self.dT is not None:
            args += ["-T", self.dT]
        elif self.nsteps is not None:
            args += ["-T", self.nsteps]
        emitter.recorder("mpco", *args)


# ---------------------------------------------------------------------------
# RecorderDeclaration — unified recorder spec (Phase 9)
# ---------------------------------------------------------------------------
#
# Two recorder declaration systems coexist on `main` until Phase 9:
#   - `Node` / `Element` / `MPCO` typed primitives (above)
#   - `apeGmsh.results.spec.Recorders` fluent helper (legacy)
#
# Phase 9 introduces a single bridge-side declaration that subsumes
# both surfaces. The typed primitives stay as the "exact OpenSees
# command shape" path; `RecorderDeclaration` is the "broad canonical
# vocabulary" path that supports nodes / elements / line_stations /
# gauss / fibers / layers / modal categories with shorthand expansion.
#
# This commit (Phase 9 commit 2) defines the types only; emit-time
# fan-out lands in commit 3 (`_internal/build.py::emit_recorder_spec`).

ALL_RECORDER_CATEGORIES: tuple[str, ...] = (
    "nodes",
    "elements",
    "line_stations",
    "gauss",
    "fibers",
    "layers",
    "modal",
)

# Per-category canonical component sets. Components outside the
# category's allowed set fail validation unless supplied via ``raw=``
# (the escape hatch for non-catalogued OpenSees tokens).
_CATEGORY_CANONICALS: dict[str, frozenset[str]] = {
    "nodes":         frozenset(NODAL_KINEMATICS + NODAL_FORCES),
    "elements":      frozenset(PER_ELEMENT_NODAL_FORCES),
    "line_stations": frozenset(LINE_DIAGRAMS),
    "gauss":         frozenset(STRESS + STRAIN + DERIVED_SCALARS + MATERIAL_STATE),
    "fibers":        frozenset(FIBER + MATERIAL_STATE),
    "layers":        frozenset(FIBER + MATERIAL_STATE),
    "modal":         frozenset(),  # no per-component vocabulary; n_modes only
}


@dataclass(frozen=True, kw_only=True, slots=True)
class RecorderRecord:
    """One category-level declaration entry within a RecorderDeclaration.

    Stores already-expanded canonical components (or raw OpenSees
    tokens via the ``raw=`` escape hatch). Shorthand expansion
    (``"displacement"`` → ``displacement_x/y/z``) happens at
    construction in the namespace method (Phase 9 commit 3), not in
    this dataclass — by the time a record is built, components are
    fully expanded.

    Parameters
    ----------
    category
        One of :data:`ALL_RECORDER_CATEGORIES`.
    components
        Tuple of canonical component names. Validated against
        :data:`_CATEGORY_CANONICALS` per category, plus indexed
        canonicals (``state_variable_<n>``, ``fiber_stress_<n>``,
        ``spring_force_<n>``) recognized via :func:`is_canonical`.
    raw
        Escape hatch for non-canonical OpenSees tokens (e.g. a
        custom recorder response). Bypasses canonical validation.
    pg / label / selection / ids
        Target selectors. ``ids=`` is mutually exclusive with the
        named selectors. Resolution against FEMData happens at
        emit time (commit 3).
    dt / n_steps
        Recording cadence. At most one may be set; both ``None``
        records every step.
    name
        Optional user-supplied name for this record; auto-generated
        when ``None``.
    n_modes
        Required for ``category="modal"``; rejected for other
        categories.
    element_class_name
        Optional OpenSees C++ class name override for element-level
        records. Used by the .out transcoder to disambiguate
        elements that share a flat response size (e.g. tri31 vs
        SSPquad). Carried from the legacy ``Recorders.elements``
        contract.
    """

    category: str
    components: tuple[str, ...] = ()
    raw: tuple[str, ...] = ()
    pg: tuple[str, ...] = ()
    label: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()
    ids: tuple[int, ...] | None = None
    dt: float | None = None
    n_steps: int | None = None
    name: str | None = None
    n_modes: int | None = None
    element_class_name: str | None = None

    def __post_init__(self) -> None:
        # Category
        if self.category not in ALL_RECORDER_CATEGORIES:
            raise ValueError(
                f"RecorderRecord: unknown category {self.category!r}. "
                f"Allowed: {ALL_RECORDER_CATEGORIES}."
            )

        # Components — must be canonical and category-allowed (or
        # match an indexed-canonical pattern). Raw tokens bypass
        # validation.
        allowed = _CATEGORY_CANONICALS.get(self.category, frozenset())
        for comp in self.components:
            if comp in allowed:
                continue
            # Allow indexed canonicals (state_variable_N, fiber_stress_N,
            # spring_force_N etc.) in element-level categories.
            if is_canonical(comp) and self.category in (
                "elements", "line_stations", "gauss", "fibers", "layers",
            ):
                continue
            raise ValueError(
                f"RecorderRecord(category={self.category!r}): component "
                f"{comp!r} not in canonical vocabulary or category-allowed "
                f"set. Use raw= for non-canonical OpenSees tokens."
            )

        # Cadence: at most one of dt / n_steps
        if self.dt is not None and self.n_steps is not None:
            raise ValueError(
                "RecorderRecord: supply at most one of dt= or n_steps= "
                f"(got dt={self.dt!r}, n_steps={self.n_steps!r})."
            )

        # Selectors: ids is mutually exclusive with named selectors
        named_used = bool(self.pg or self.label or self.selection)
        if self.ids is not None and named_used:
            raise ValueError(
                "RecorderRecord: ids= is mutually exclusive with "
                f"pg=/label=/selection= (got ids={self.ids!r}, "
                f"pg={self.pg!r}, label={self.label!r}, "
                f"selection={self.selection!r})."
            )

        # Modal: requires n_modes; rejects components / selectors
        if self.category == "modal":
            if self.n_modes is None or self.n_modes < 1:
                raise ValueError(
                    "RecorderRecord(category='modal'): n_modes >= 1 required."
                )
        elif self.n_modes is not None:
            raise ValueError(
                f"RecorderRecord(category={self.category!r}): "
                "n_modes is only valid for category='modal'."
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class RecorderDeclaration(Recorder):
    """A bundle of recorder records, registered as a single Primitive.

    Captures the bridge's ``ndm`` and ``ndf`` at construction time
    (Phase 9 D8 — implicit source-of-truth binding). Drives the
    file-emit path via :func:`emit_recorder_spec` in
    :mod:`apeGmsh.opensees._internal.build`.

    Parameters
    ----------
    records
        Tuple of :class:`RecorderRecord` entries. Each is one
        category-level declaration; emit fans them out into one or
        more concrete OpenSees ``recorder`` commands.
    name
        Identifier for this declaration (defaults to ``"default"``).
        Multiple named declarations can coexist on one bridge.
    ndm, ndf
        Snapshot of the bridge's ``ndm``/``ndf`` at construction time.
        Used downstream for shorthand expansion and validation. The
        bridge passes these in (Phase 9 D8 — user never repeats
        ``ops.model(ndm=, ndf=)`` values).
    """

    records: tuple[RecorderRecord, ...]
    name: str = "default"
    ndm: int = 3
    ndf: int = 6

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        # Emit fan-out lives in emit_recorder_spec (Phase 9 commit 3).
        # Direct ._emit calls are not supported because per-record
        # translation needs FEM context (for pg/label/selection
        # resolution) that the bridge fan-out supplies.
        raise NotImplementedError(
            "RecorderDeclaration._emit is driven by emit_recorder_spec "
            "in apeGmsh.opensees._internal.build; call it via the "
            "bridge build pipeline rather than directly."
        )
