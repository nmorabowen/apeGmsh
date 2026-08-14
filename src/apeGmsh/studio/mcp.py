"""``python -m apeGmsh.studio.mcp`` — Cursor stdio adapter (ADR 0095 S4a–S4d).

Tools: ``status``, ``get_selection``, ``run_until``, ``assess``,
``render``, ``animate(kind=history|yield)``, ``results_pin``,
``emit_report(format=markdown)``, ``highlight``, ``promote_selection``.
Requires the optional extra ``pip install mcp`` (or ``apeGmsh[mcp]``).

Cursor ``mcp.json`` (workspace cwd is the model root)::

    {
      "mcpServers": {
        "apegmsh-studio": {
          "command": "python",
          "args": ["-m", "apeGmsh.studio.mcp"]
        }
      }
    }
"""

from __future__ import annotations

from typing import Any


def _require_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "apeGmsh.studio.mcp requires the MCP SDK: pip install mcp"
        ) from exc
    return FastMCP


def build_server() -> Any:
    """Return a FastMCP server wired to the S4a–S4d tool bodies."""
    FastMCP = _require_fastmcp()
    from apeGmsh.studio._mcp import animate as _animate
    from apeGmsh.studio._mcp import assess as _assess
    from apeGmsh.studio._mcp import emit_report as _emit_report
    from apeGmsh.studio._mcp import get_selection as _get_selection
    from apeGmsh.studio._mcp import highlight as _highlight
    from apeGmsh.studio._mcp import promote_selection as _promote_selection
    from apeGmsh.studio._mcp import render as _render
    from apeGmsh.studio._mcp import results_pin as _results_pin
    from apeGmsh.studio._mcp import run_until as _run_until
    from apeGmsh.studio._mcp import status as _status

    mcp = FastMCP(
        "apeGmsh.studio",
        instructions=(
            "apeGmsh.studio habitat (ADR 0095 S4a–S4d). "
            "Identity is labels / physical groups / phase, not tags. "
            "Do not wrap g.model.* or apeSees primitives. "
            "animate kind=history|yield; no setup=. formation is later. "
            "kind=yield is a von Mises contour on auto-scaled deform, "
            "not an iso-clip. "
            "emit_report format=markdown; html/canvas are later skins. "
            "highlight writes .apegmsh/highlight.json only (file poll, "
            "not a Qt highlight(names) mutator); "
            "promote_selection suggests g.model.select(None, dim=)"
            ".in_box(lo, hi).to_label() and does not write the .py. "
            "Authored chapters go in docs/, not .apegmsh/."
        ),
    )

    @mcp.tool()
    def status() -> dict[str, Any]:
        """Last run, names, and pick from .apegmsh/ (no replay, no Qt)."""
        return _status()

    @mcp.tool()
    def get_selection() -> dict[str, Any]:
        """Names-first pick envelope written by the Qt host."""
        return _get_selection()

    @mcp.tool()
    def run_until(script: str, phase: str = "model") -> dict[str, Any]:
        """Replay script up to phase (model|mesh|results). No Qt window."""
        return _run_until(script, phase=phase)

    @mcp.tool()
    def assess(
        path: str,
        figures: bool = False,
        model_h5: str | None = None,
    ) -> dict[str, Any]:
        """Verdict + markdown from model.h5 or a results file. Not Qt."""
        return _assess(path, figures=figures, model_h5=model_h5)

    @mcp.tool()
    def render(
        path: str,
        output: str | None = None,
        view: str = "contour",
        component: str | None = None,
        step: int = -1,
        camera: str = "iso",
        pack: bool = False,
        deform: str | None = None,
        model_h5: str | None = None,
    ) -> dict[str, Any]:
        """Write a still (or canned pack) under .apegmsh/visors/. Closed view=."""
        return _render(
            path,
            output,
            view=view,
            component=component,
            step=step,
            camera=camera,
            pack=pack,
            deform=deform,
            model_h5=model_h5,
        )

    @mcp.tool()
    def animate(
        path: str,
        output: str | None = None,
        kind: str = "history",
        model_h5: str | None = None,
        fps: int = 30,
        step_stride: int = 1,
    ) -> dict[str, Any]:
        """kind=history|yield. yield = von Mises contour on auto-scaled deform. No setup=."""
        return _animate(
            path,
            output,
            kind=kind,
            model_h5=model_h5,
            fps=fps,
            step_stride=step_stride,
        )

    @mcp.tool()
    def results_pin(
        model_h5: str | None = None,
        results: str | None = None,
    ) -> dict[str, Any]:
        """Stamp model.h5 / results path+hash into the ledger. No file copy."""
        return _results_pin(model_h5, results)

    @mcp.tool()
    def emit_report(
        format: str = "markdown",
        output: str | None = None,
        pin_id: str | None = None,
    ) -> dict[str, Any]:
        """Write docs/ Markdown from the ReportBundle. html/canvas not shipped."""
        return _emit_report(format=format, output=output, pin_id=pin_id)

    @mcp.tool()
    def highlight(names: list[str]) -> dict[str, Any]:
        """Point at named faces/groups. Writes highlight.json only. No Gmsh."""
        return _highlight(names)

    @mcp.tool()
    def promote_selection() -> dict[str, Any]:
        """Suggested select(None, dim=).in_box(lo, hi).to_label() edits. Does not write .py."""
        return _promote_selection()

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
