"""``python -m apeGmsh.studio.mcp`` — Cursor stdio adapter (ADR 0095 S4a–S4e / S5a).

Tools: ``status``, ``get_selection``, ``lookup``, ``run_until``, ``assess``,
``render``, ``animate(kind=history|yield)``, ``results_pin``,
``emit_report(format=markdown|html|canvas)``, ``highlight``,
``promote_selection``.
Requires the optional extra ``pip install mcp`` (or ``apeGmsh[mcp]``).

Every tool accepts optional ``root`` (INV-15): explicit project directory
for ``.apegmsh/``. Resolution: ``root=`` → ``APEGMSH_ROOT`` → nearest
ancestor with ``.apegmsh/`` → process cwd. One MCP process may serve
multiple projects by passing different ``root`` values.

Cursor ``mcp.json``. The office venv's ``ladruno_opensees.pth`` prints a
banner to **stdout** at interpreter startup — set
``LADRUNO_OPENSEES_QUIET=1`` *before* Python starts or JSON-RPC is
corrupted. ``scripts/studio-mcp.ps1`` does that. ``APEGMSH_QUIET=1``
keeps the apeGmsh banner off stderr. Optionally set ``APEGMSH_ROOT`` to
the model project (do not rely on process cwd)::

    {
      "mcpServers": {
        "apegmsh-studio": {
          "command": "powershell.exe",
          "args": [
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/studio-mcp.ps1"
          ],
          "env": {
            "LADRUNO_OPENSEES_QUIET": "1",
            "APEGMSH_QUIET": "1"
          }
        }
      }
    }
"""

from __future__ import annotations

from typing import Any


def _require_fastmcp() -> Any:
    """``FastMCP`` (mcp 1.x) or ``MCPServer`` (mcp 2.x). Same ``tool`` / ``run``."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        FastMCP = None
    if FastMCP is not None:
        return FastMCP
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise SystemExit(
            "apeGmsh.studio.mcp requires the MCP SDK: pip install mcp"
        ) from exc
    return MCPServer


def build_server() -> Any:
    """Return a FastMCP server wired to the S4a–S4e / 0096 S3 / S5a tool bodies."""
    Server = _require_fastmcp()
    from apeGmsh.studio._mcp import animate as _animate
    from apeGmsh.studio._mcp import assess as _assess
    from apeGmsh.studio._mcp import emit_report as _emit_report
    from apeGmsh.studio._mcp import get_selection as _get_selection
    from apeGmsh.studio._mcp import highlight as _highlight
    from apeGmsh.studio._mcp import lookup as _lookup
    from apeGmsh.studio._mcp import promote_selection as _promote_selection
    from apeGmsh.studio._mcp import render as _render
    from apeGmsh.studio._mcp import results_pin as _results_pin
    from apeGmsh.studio._mcp import run_until as _run_until
    from apeGmsh.studio._mcp import status as _status
    from apeGmsh.studio._mcp_log import logged

    mcp = Server(
        "apeGmsh.studio",
        instructions=(
            "apeGmsh.studio habitat (ADR 0095 S4a–S4e / S5a, 0096 S3). "
            "Identity is labels / physical groups / phase, not tags. "
            "Do not wrap g.model.* or apeSees primitives. "
            "Optional root= on every tool (INV-15): project directory for "
            ".apegmsh/; else APEGmSH_ROOT; else nearest .apegmsh/ ancestor; "
            "else cwd. One server may serve multiple projects via root=. "
            "lookup(symbol) is See-family inspect of the generated index "
            "(~20 lines); it is not CAD and does not grep src/. "
            "animate kind=history|yield; no setup=. formation is later. "
            "kind=yield is a von Mises contour on auto-scaled deform, "
            "not an iso-clip. "
            "emit_report format=markdown|html|canvas; markdown is the archive; "
            "html is docs/ print; canvas is IDE canvases/ (not git-portable). "
            "highlight writes .apegmsh/highlight.json only (file poll, "
            "not a Qt highlight(names) mutator); "
            "promote_selection suggests g.model.select(None, dim=)"
            ".in_box(lo, hi).to_label() and does not write the .py. "
            "Authored chapters go in docs/, not .apegmsh/. "
            "Do not add a profiler MCP tool; python -m apeGmsh.studio.profile "
            "is the sidecar; --promote prints eligibility and writes nothing."
        ),
    )

    @mcp.tool()
    def status(root: str | None = None) -> dict[str, Any]:
        """Last run, names, and pick from .apegmsh/ (no replay, no Qt)."""
        return logged("status", _status, root=root)

    @mcp.tool()
    def get_selection(root: str | None = None) -> dict[str, Any]:
        """Names-first pick envelope written by the Qt host."""
        return logged("get_selection", _get_selection, root=root)

    @mcp.tool()
    def lookup(symbol: str, root: str | None = None) -> dict[str, Any]:
        """Public composite signature from the generated index. ~20 lines. Not CAD."""
        return logged("lookup", _lookup, symbol=symbol, root=root)

    @mcp.tool()
    def run_until(
        script: str | None = None,
        phase: str = "model",
        root: str | None = None,
    ) -> dict[str, Any]:
        """Replay script up to phase (model|mesh|results). Omitting script uses project.json."""
        return logged("run_until", _run_until, script=script, phase=phase, root=root)

    @mcp.tool()
    def assess(
        path: str,
        figures: bool = False,
        model_h5: str | None = None,
        root: str | None = None,
    ) -> dict[str, Any]:
        """Verdict + markdown from model.h5 or a results file. Not Qt."""
        return logged(
            "assess", _assess, path, figures=figures, model_h5=model_h5, root=root,
        )

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
        root: str | None = None,
    ) -> dict[str, Any]:
        """Write a still (or canned pack) under .apegmsh/visors/. Closed view=."""
        return logged(
            "render",
            _render,
            path,
            output,
            view=view,
            component=component,
            step=step,
            camera=camera,
            pack=pack,
            deform=deform,
            model_h5=model_h5,
            root=root,
        )

    @mcp.tool()
    def animate(
        path: str,
        output: str | None = None,
        kind: str = "history",
        model_h5: str | None = None,
        fps: int = 30,
        step_stride: int = 1,
        root: str | None = None,
    ) -> dict[str, Any]:
        """kind=history|yield. yield = von Mises contour on auto-scaled deform. No setup=."""
        return logged(
            "animate",
            _animate,
            path,
            output,
            kind=kind,
            model_h5=model_h5,
            fps=fps,
            step_stride=step_stride,
            root=root,
        )

    @mcp.tool()
    def results_pin(
        model_h5: str | None = None,
        results: str | None = None,
        root: str | None = None,
    ) -> dict[str, Any]:
        """Stamp model.h5 / results path+hash into the ledger. No file copy."""
        return logged(
            "results_pin", _results_pin, model_h5=model_h5, results=results, root=root,
        )

    @mcp.tool()
    def emit_report(
        format: str = "markdown",
        output: str | None = None,
        pin_id: str | None = None,
        root: str | None = None,
    ) -> dict[str, Any]:
        """Write a ReportBundle skin. format=markdown|html|canvas; markdown is the archive."""
        return logged(
            "emit_report",
            _emit_report,
            format=format,
            output=output,
            pin_id=pin_id,
            root=root,
        )

    @mcp.tool()
    def highlight(names: list[str], root: str | None = None) -> dict[str, Any]:
        """Point at named faces/groups. Writes highlight.json only. No Gmsh."""
        return logged("highlight", _highlight, names, root=root)

    @mcp.tool()
    def promote_selection(root: str | None = None) -> dict[str, Any]:
        """Suggested select(None, dim=).in_box(lo, hi).to_label() edits. Does not write .py."""
        return logged("promote_selection", _promote_selection, root=root)

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
