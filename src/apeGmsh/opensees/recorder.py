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

from ._internal.types import Primitive, Recorder

if TYPE_CHECKING:
    from .emitter.base import Emitter


__all__ = [
    "Node",
    "Element",
    "MPCO",
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
