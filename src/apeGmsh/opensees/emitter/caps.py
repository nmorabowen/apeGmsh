"""Target capabilities: what an emit target can do, declared once (ADR 0114 D6).

``TargetCaps`` is the typed home of the flags the bridge used to probe
with ``getattr(emitter, "<flag>", default)``, set with ``emitter.<flag> =
...`` behind ``type: ignore[attr-defined]``, or sniff with
``type(emitter).__name__ == "H5Emitter"``. Every emitter declares one
``caps`` value; the bridge reads ``emitter.caps.<field>`` and never asks
an emitter for its class. The quirk rule ``emitter-sniff`` in
``scripts/check_quirks.py`` bans the sniffs outside this package.

``SolveStamp`` is the other half of the seam: what the archive records
about the solve it was built for (``/opensees@will_solve``,
``@solve_refusals``, ``@requires``; opensees schema 2.24.0), so a replay
can fail closed on a target that lacks what the model needs.

Data only: this module imports nothing from apeGmsh.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetCaps:
    """What one emit target can do; frozen, one value per emitter class.

    The defaults are the values the bridge's former ``getattr`` probes
    took for an emitter that declared nothing.

    ``archival``
        The target stores the model instead of driving a solve
        (``H5Emitter``): solve-time gates do not enforce, and the mass
        stream stays in the neutral zone under ``mass_from_model()``.
    ``supports_partitions``
        The target consumes the per-rank ``partition_open`` /
        ``partition_close`` brackets. A single-process target (live) has
        a partitioned model flattened into one domain instead.
    ``per_rank_fragments``
        The target writes one fragment per rank (``TclEmitter`` under
        ``split=True, per_rank=True``), so the bridge emits each rank's
        block into its own stream.
    ``suppress_analysis_chain_auto_emit``
        The bridge must not auto-emit a numberer / system for a
        partitioned flat deck (the FEAST eigen deck owns its own).
    ``model_reissue_purges``
        A ``model BasicBuilder`` re-issue on this target purges the
        builder-scoped registries (timeSeries / geomTransf /
        beamIntegration / damping), so a staged replay re-declares them
        at bracket close. True for the classic Tcl interpreter only.
    ``emit_stage_markers``
        ``stage_open`` / ``stage_close`` print a runtime
        ``APEGMSH_STAGE open|close <name>`` marker so solver-stats
        blocks can be attributed to their stage.
    ``supports_stages``
        The target accepts ``stage_open`` / ``stage_close``. The live
        in-process target raises on them, so a staged replay refuses
        before its first call instead of dying mid-domain.

    The ``archival`` target also carries the archive's side channels
    (``mark_mass_from_model``, ``set_solve_stamp``); the bridge reaches
    them through ``apesees._archive_side_channel``.
    """

    archival: bool = False
    supports_partitions: bool = True
    per_rank_fragments: bool = False
    suppress_analysis_chain_auto_emit: bool = False
    model_reissue_purges: bool = False
    emit_stage_markers: bool = False
    supports_stages: bool = True


#: The partition modes a solve verdict is stored for: the archive's own
#: mode (``"partitioned"`` when the FEM is partitioned and the target
#: consumes the brackets, else ``"flat"``) and always ``"flat"``, the mode
#: every replay target emits today.
SOLVE_MODES: tuple[str, ...] = ("flat", "partitioned")

#: The capability tokens this bridge writes into ``/opensees@requires``.
#: ``"fork"`` comes from the ``VERBS`` rows (the Ladruno build). The other
#: two depend on argument values (opensees 2.33.0), so the stamp writer
#: derives them, not ``VERBS`` (:func:`value_requires`):
#: ``"mpco"``, any ``recorder mpco`` (pip openseespy ships no MPCO
#: recorder); ``"mpi"``, a Mumps-family ``system`` or a partitioned
#: ``solve_mode`` (the deck needs a multi-rank MPI run). A reader accepts
#: any token; a solving target that does not know one refuses it.
REQUIRES_TOKENS: tuple[str, ...] = ("fork", "mpco", "mpi")


def is_mpi_system(s_type: str) -> bool:
    """Whether ``system <s_type>`` is a Mumps-family solver: the one family
    that needs an MPI build (``Mumps``, ``MumpsParallel``, any case)."""
    return s_type.casefold().startswith("mumps")


def value_requires(
    *, recorder_kinds: "tuple[str, ...]", systems: "tuple[str, ...]",
    solve_mode: str,
) -> frozenset[str]:
    """The requirement tokens a model's argument values imply: ``"mpco"``
    for a ``recorder mpco``, ``"mpi"`` for a Mumps-family ``system`` or a
    partitioned ``solve_mode`` (see :data:`REQUIRES_TOKENS`)."""
    if solve_mode not in SOLVE_MODES:
        raise ValueError(
            f"value_requires: solve_mode must be one of {SOLVE_MODES}, got "
            f"{solve_mode!r}"
        )
    out: set[str] = set()
    if any(k.casefold() == "mpco" for k in recorder_kinds):
        out.add("mpco")
    if solve_mode == "partitioned" or any(is_mpi_system(s) for s in systems):
        out.add("mpi")
    return frozenset(out)


@dataclass(frozen=True, slots=True)
class SolveStamp:
    """What ``model.h5`` records about the solve it was emitted for.

    Written as five attributes on ``/opensees`` (opensees 2.24.0) and
    read back by ``H5Model.solve_stamp()``.

    ``will_solve``
        ``staged or any(Analysis)`` at emit: the model carries a solve.
    ``solve_refusals``
        The ids of the solve-time gates that refused at emit **under the
        archive's own partition mode** (``solve_mode``), in gate order.
    ``requires``
        The sorted union of the capability tokens the ``VERBS`` rows of
        every emitted verb require (``"fork"``: the Ladruno build),
        ledger rows included, and, from opensees 2.33.0, the tokens the
        emitted argument values imply (``"mpco"``, ``"mpi"``:
        :func:`value_requires`), so ``build('live')`` can refuse on a
        backend that lacks one. The writer derives it from its own
        program (``/opensees/program@methods``) and stores. An archive
        below 2.33.0 never names the value tokens: its ``requires`` keeps
        the meaning it was written with (verbs only).
    ``solve_mode``
        The partition mode ``solve_refusals`` was judged under: one of
        :data:`SOLVE_MODES`.
    ``solve_refusals_flat``
        The same gates' verdict under the flat (single-domain) mode. A
        gate that allows a partitioned deck (no ``system`` rides the
        ADR 0027 auto-emit) can refuse its flat replay, so the flat
        verdict is stored on its own and a flat replay fails closed on
        it (:meth:`refusals_for`). Equal to ``solve_refusals`` when
        ``solve_mode`` is ``"flat"``; ``None`` at construction copies it
        then.
    """

    will_solve: bool
    solve_refusals: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    solve_mode: str = "flat"
    solve_refusals_flat: "tuple[str, ...] | None" = None

    def __post_init__(self) -> None:
        if self.solve_mode not in SOLVE_MODES:
            raise ValueError(
                f"SolveStamp.solve_mode must be one of {SOLVE_MODES}, got "
                f"{self.solve_mode!r}"
            )
        if self.solve_refusals_flat is None:
            if self.solve_mode != "flat":
                raise ValueError(
                    "SolveStamp: a partitioned stamp needs its flat verdict "
                    "(solve_refusals_flat)"
                )
            object.__setattr__(self, "solve_refusals_flat", self.solve_refusals)
        for name in ("solve_refusals", "solve_refusals_flat", "requires"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                    isinstance(v, str) and v for v in value):
                raise TypeError(
                    f"SolveStamp.{name} must be a tuple of non-empty "
                    f"strings, got {value!r}"
                )
        if self.solve_mode == "flat" and (
                self.solve_refusals_flat != self.solve_refusals):
            raise ValueError(
                "SolveStamp: a flat stamp's solve_refusals_flat must equal "
                f"solve_refusals, got {self.solve_refusals_flat!r} vs "
                f"{self.solve_refusals!r}"
            )
        if tuple(sorted(set(self.requires))) != self.requires:
            raise ValueError(
                f"SolveStamp.requires must be sorted and unique, got "
                f"{self.requires!r}"
            )

    def refusals_for(self, mode: str) -> tuple[str, ...]:
        """The stored verdict for the partition mode a replay emits.

        ``mode`` is the archive's own mode or ``"flat"``; any other mode
        has no stored verdict and raises, so a caller fails closed
        instead of guessing.
        """
        if mode == self.solve_mode:
            return self.solve_refusals
        if mode == "flat":
            assert self.solve_refusals_flat is not None
            return self.solve_refusals_flat
        raise ValueError(
            f"SolveStamp holds no verdict for partition mode {mode!r}; it "
            f"stores {self.solve_mode!r} and 'flat'"
        )


__all__ = [
    "REQUIRES_TOKENS", "SOLVE_MODES", "SolveStamp", "TargetCaps",
    "is_mpi_system", "value_requires",
]
