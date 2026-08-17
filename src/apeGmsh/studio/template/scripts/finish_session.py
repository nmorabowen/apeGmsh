#!/usr/bin/env python3
"""Finish an APE Studio habitat session.

1. Report apeCAD / apeSketch ports (does not kill Cursor MCP)
2. Run Studio token profiler (metrics JSON) and separately --promote
3. Scaffold postmortem from templates + optimization checklist
4. Mark session finished
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _habitat import (  # noqa: E402
    HABITAT,
    SESSION_FILE,
    ensure_pythonpath,
    read_session,
    utc_now,
    write_session,
)

BRIDGE_PORTS = (8765, 9966, 9967)
MCP_SERVER_ID = "apegmsh-studio-habitat"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def _report_listeners() -> list[str]:
    notes: list[str] = []
    for port in BRIDGE_PORTS:
        if _port_open(port):
            notes.append(
                f"port {port} still listening — stop apeCAD/apeSketch "
                f"(Ctrl+C in that terminal) if this session started it"
            )
        else:
            notes.append(f"port {port} closed")
    return notes


def _run_profile(*, promote: bool) -> tuple[int, str]:
    ensure_pythonpath()
    cmd = [
        sys.executable,
        "-m",
        "apeGmsh.studio.profile",
        "--root",
        str(HABITAT),
        "--json",
    ]
    if promote:
        cmd.append("--promote")
    proc = subprocess.run(
        cmd,
        cwd=str(HABITAT),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _fill_template(text: str, mapping: dict[str, str]) -> str:
    out = text
    for key, val in mapping.items():
        out = out.replace("{{" + key + "}}", val)
    return out


def _write_metrics(
    path: Path,
    *,
    day: str,
    slug: str,
    profile_text: str,
    port_notes: list[str],
    force: bool,
) -> None:
    if path.is_file() and not force:
        return
    tmpl_path = HABITAT / "postmortem" / "templates" / "01_metrics.md"
    if tmpl_path.is_file():
        body = _fill_template(
            tmpl_path.read_text(encoding="utf-8"),
            {
                "DATE": day,
                "SLUG": slug,
                "HABITAT_NAME": HABITAT.name,
                "AGENT": "(fill)",
                "What we set out to do": "(fill)",
            },
        )
        # Append auto blocks after template body
        snippet = (
            profile_text[:6000]
            if profile_text.strip()
            else (
                "(no .apegmsh/mcp_calls.jsonl under this habitat — "
                "MCP may have written elsewhere, or no tools were called)"
            )
        )
        body += f"""

---

## Auto (finish_session)

**Closed (UTC):** {utc_now()}  
**MCP server id (expected):** `{MCP_SERVER_ID}`  
**Profile artifact:** `profile_studio.json`  
**Promote artifact:** `profile_promote.json`

### Profiler JSON (excerpt)

```json
{snippet}
```

### Bridge ports at finish

{chr(10).join('- ' + n for n in port_notes)}

### Live MCP root

Agent: call `{MCP_SERVER_ID}` `status` and record `status.root` here.
Expected: `{HABITAT.resolve()}`
"""
    else:
        body = f"# Session metrics — {day}_{slug}\n\n(fill; template missing)\n"
    path.write_text(body, encoding="utf-8")


def _scaffold_postmortem(
    slug: str,
    *,
    profile_text: str,
    port_notes: list[str],
    refresh_metrics: bool,
) -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    folder = HABITAT / "postmortem" / "sessions" / f"{day}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)

    _write_metrics(
        folder / "01_metrics.md",
        day=day,
        slug=slug,
        profile_text=profile_text,
        port_notes=port_notes,
        force=refresh_metrics,
    )

    friction = folder / "02_friction.md"
    if not friction.is_file():
        tmpl = HABITAT / "postmortem" / "templates" / "02_friction.md"
        body = (
            tmpl.read_text(encoding="utf-8")
            if tmpl.is_file()
            else "# Friction\n\n(fill)\n"
        )
        body = _fill_template(
            body,
            {
                "DATE": day,
                "SLUG": slug,
                "HABITAT_NAME": HABITAT.name,
            },
        )
        friction.write_text(body, encoding="utf-8")

    opt = folder / "03_optimization_revisions.md"
    backlog = HABITAT / "postmortem" / "backlog" / "open.md"
    backlog_txt = (
        backlog.read_text(encoding="utf-8") if backlog.is_file() else "(missing)"
    )
    opt.write_text(
        f"""# Optimization revisions — {day}_{slug}

Auto-generated at session finish. Review and burn down P0/P1.

## Checks run

- [x] Template re-check (`scripts/check_template.py`)
- [x] Studio token profiler (`python -m apeGmsh.studio.profile --json`)
- [x] Promote eligibility (`python -m apeGmsh.studio.profile --promote`)
- [ ] Human: triage friction -> `postmortem/backlog/open.md`
- [ ] Human: promote upstream ADRs/issues if needed
- [ ] Next session starts from backlog P0/P1

## Current backlog snapshot

```markdown
{backlog_txt}
```

## Continuous improvement map

See `APE/instructions/continuous-improvement.md` and `analysis-profiling.md`.
""",
        encoding="utf-8",
    )
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        default=None,
        help="postmortem slug (default: active session slug or 'session')",
    )
    args = parser.parse_args()

    print("=== APE habitat FINISH ===")
    print(f"root: {HABITAT}")

    sess = read_session()
    slug = args.slug or sess.get("slug") or "session"

    trc = subprocess.call(
        [sys.executable, str(HABITAT / "scripts" / "check_template.py")],
        cwd=str(HABITAT),
    )
    print(f"template check exit={trc}")

    port_notes = _report_listeners()
    for n in port_notes:
        print(f"  {n}")

    folder = _scaffold_postmortem(
        slug, profile_text="", port_notes=port_notes, refresh_metrics=False
    )

    metrics_path = folder / "profile_studio.json"
    promote_path = folder / "profile_promote.json"

    prc, ptext = _run_profile(promote=False)
    metrics_path.write_text(ptext if ptext.strip() else "{}\n", encoding="utf-8")
    print(f"studio.profile (metrics) exit={prc} -> {metrics_path}")

    prc2, ptext2 = _run_profile(promote=True)
    promote_path.write_text(ptext2 if ptext2.strip() else "{}\n", encoding="utf-8")
    print(f"studio.profile (promote) exit={prc2} -> {promote_path}")

    _scaffold_postmortem(
        slug,
        profile_text=ptext,
        port_notes=port_notes,
        refresh_metrics=True,
    )

    write_session(
        {
            **sess,
            "state": "finished",
            "finished_at": utc_now(),
            "slug": slug,
            "mcp_server_id": sess.get("mcp_server_id", MCP_SERVER_ID),
            "postmortem_session": str(folder.relative_to(HABITAT)).replace("\\", "/"),
            "note": (
                f"Session finished. Reload/disable `{MCP_SERVER_ID}` in Cursor "
                "to drop the stdio MCP process if desired."
            ),
        }
    )
    print(f"OK: session marker -> {SESSION_FILE} (state=finished)")
    print(f"OK: postmortem -> {folder}")
    print()
    print("Complete 02_friction.md and update postmortem/backlog/open.md.")
    print("=== FINISH complete ===")
    return 0 if trc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
