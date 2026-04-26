"""Inspection helpers for ``Results``.

Provides the ``ResultsInspect`` composite — what's in the file,
what stages exist, what components are available where.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .readers._protocol import ResultLevel

if TYPE_CHECKING:
    from .Results import Results


class ResultsInspect:
    """``results.inspect`` — what's available."""

    def __init__(self, results: "Results") -> None:
        self._r = results

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        r = self._r
        lines = [f"Results: {r._reader_path()!s}"]

        fem = r.fem
        if fem is not None:
            lines.append(
                f"  FEM: {len(fem.nodes.ids)} nodes, "
                f"{sum(len(g) for g in fem.elements)} elements "
                f"(snapshot_id={fem.snapshot_id})"
            )
        else:
            lines.append("  FEM: not bound")

        stages = r.stages
        if not stages:
            lines.append("  Stages: (none)")
        else:
            lines.append(f"  Stages ({len(stages)}):")
            for s in stages:
                detail = f"steps={s.n_steps}, kind={s.kind}"
                if s.kind == "mode":
                    detail += (
                        f", f={s.frequency_hz:.4g} Hz, "
                        f"T={s.period_s:.4g} s, "
                        f"mode_index={s.mode_index}"
                    )
                lines.append(f"    - {s.id} ({s.name}): {detail}")

        return "\n".join(lines)

    def components(
        self, *, stage: str | None = None,
    ) -> dict[str, list[str]]:
        """Available components per topology level for one stage.

        If no stage is given, defaults to the only stage when there is
        exactly one; otherwise raises.
        """
        sid = self._r._resolve_stage(stage)
        return {
            level.value: self._r._reader.available_components(sid, level)
            for level in ResultLevel
        }

    def __repr__(self) -> str:
        return self.summary()
