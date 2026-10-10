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


@dataclass(frozen=True, slots=True)
class SolveStamp:
    """What ``model.h5`` records about the solve it was emitted for.

    Written as three attributes on ``/opensees`` (opensees 2.24.0) and
    read back by ``H5Model.solve_stamp()``.

    ``will_solve``
        ``staged or any(Analysis)`` at emit: the model carries a solve.
    ``solve_refusals``
        The ids of the solve-time gates that refused at emit, in order.
        A replay to a deck fails closed on a non-empty tuple.
    ``requires``
        The sorted union of the capability tokens the ``VERBS`` rows of
        every emitted verb require (``"fork"``: the Ladruno build),
        ledger rows included, so ``build('live')`` can refuse on a
        backend that lacks one. The writer derives it from its own
        program (``/opensees/program@methods``).
    """

    will_solve: bool
    solve_refusals: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("solve_refusals", "requires"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                    isinstance(v, str) and v for v in value):
                raise TypeError(
                    f"SolveStamp.{name} must be a tuple of non-empty "
                    f"strings, got {value!r}"
                )
        if tuple(sorted(set(self.requires))) != self.requires:
            raise ValueError(
                f"SolveStamp.requires must be sorted and unique, got "
                f"{self.requires!r}"
            )


__all__ = ["SolveStamp", "TargetCaps"]
