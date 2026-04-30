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

    def diagnose(
        self,
        component: str,
        *,
        stage: str | None = None,
    ) -> str:
        """Explain where a component lives (or doesn't) in this stage.

        When a viewer or downstream consumer asks for a component and
        gets nothing back, this is the routing-side answer to "why
        is the slab empty?". Walks every topology, calls each
        composite's ``available_components()``, and returns a
        human-readable report that shows where ``component`` was
        found and what's actually available at each level.

        Parameters
        ----------
        component
            Canonical component name (e.g. ``"axial_force"``,
            ``"displacement_z"``, ``"stress_xx"``).
        stage
            Stage id or name. Defaults to the only stage when there
            is exactly one.

        Returns
        -------
        str
            Multi-line report. Print it or include it in an error
            message.
        """
        try:
            sid = self._r._resolve_stage(stage)
        except Exception as exc:
            return f"diagnose({component!r}): could not resolve stage: {exc}"

        lines = [
            f"diagnose({component!r}) — stage={sid!r}",
        ]
        per_level: list[tuple[str, list[str], bool]] = []
        errors: list[tuple[str, str]] = []
        found: list[str] = []

        for level in ResultLevel:
            try:
                comps = self._r._reader.available_components(sid, level)
            except Exception as exc:
                errors.append((level.value, f"{type(exc).__name__}: {exc}"))
                continue
            is_match = component in comps
            per_level.append((level.value, comps, is_match))
            if is_match:
                found.append(level.value)

        if found:
            lines.append(f"  FOUND in: {', '.join(found)}")
        else:
            lines.append("  NOT FOUND in any topology level.")

        # Per-level preview — same for found and not-found, so the user
        # always sees what's actually present at each level.
        for level_value, comps, is_match in per_level:
            preview = ", ".join(comps[:6])
            if len(comps) > 6:
                preview += f", … (+{len(comps) - 6} more)"
            available = preview if comps else "(empty — no buckets present)"
            marker = "✓" if is_match else " "
            lines.append(
                f"    {marker} {level_value:16s}  available: {available}"
            )

        for level_value, msg in errors:
            lines.append(f"      {level_value:16s}  error: {msg}")

        if not found:
            lines.append("")
            lines.append(
                "  If you expected the component above, try:"
            )
            lines.append(
                "    * Check spelling against ``results.inspect.components()``."
            )
            lines.append(
                "    * Check the recorder declared this component in this stage."
            )
            lines.append(
                "    * For MPCO files: confirm the underlying recorder "
                "wrote a bucket the reader knows about (section.force, "
                "localForce, etc.)."
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
