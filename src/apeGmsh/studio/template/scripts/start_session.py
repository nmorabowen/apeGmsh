#!/usr/bin/env python3
"""Start an APE Studio habitat session.

1. Validate folder template
2. Align APEGMSH_STUDIO_ROOT / mcp.json (project file)
3. Ensure .apegmsh exists; import-check apeGmsh Studio
4. Record session marker + AGENT ACTION to verify live MCP status.root
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _habitat import (  # noqa: E402
    HABITAT,
    SESSION_FILE,
    ensure_pythonpath,
    mcp_json_studio_root,
    utc_now,
    write_session,
)

MCP_SERVER_ID = "apegmsh-studio-habitat"


def _run_check_template() -> int:
    return subprocess.call(
        [sys.executable, str(HABITAT / "scripts" / "check_template.py"), "--strict"],
        cwd=str(HABITAT),
    )


def _import_studio_ok() -> tuple[bool, str]:
    """Confirm apeGmsh Studio imports; does NOT verify the live Cursor MCP root."""
    ensure_pythonpath()
    try:
        import apeGmsh.studio  # noqa: F401

        return True, "apeGmsh.studio import OK"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        default="session",
        help="short session slug for later postmortem folder naming",
    )
    parser.add_argument(
        "--skip-template",
        action="store_true",
        help="do not fail on template gaps (still prints check)",
    )
    args = parser.parse_args()

    print("=== APE habitat START ===")
    print(f"root: {HABITAT}")

    rc = _run_check_template()
    if rc != 0 and not args.skip_template:
        print("Abort: fix template gaps, or pass --skip-template.")
        return rc

    mcp_root = mcp_json_studio_root()
    habitat_res = str(HABITAT.resolve())
    if mcp_root is None:
        print("WARN: .cursor/mcp.json missing or has no APEGMSH_STUDIO_ROOT")
    elif mcp_root != habitat_res:
        print("FAIL: mcp.json APEGMSH_STUDIO_ROOT mismatch")
        print(f"  mcp.json -> {mcp_root}")
        print(f"  habitat  -> {habitat_res}")
        print("  Fix .cursor/mcp.json then reload MCP in Cursor.")
        return 2
    else:
        print(f"OK: project mcp.json studio root -> {mcp_root}")
        print(f"OK: expected MCP server id -> {MCP_SERVER_ID}")

    ensure_pythonpath()
    (HABITAT / ".apegmsh").mkdir(parents=True, exist_ok=True)

    ok, msg = _import_studio_ok()
    if not ok:
        print(f"FAIL: Studio import: {msg}")
        print("  Check APEGMSH_PYTHON / APEGMSH_SRC (apeGmsh src).")
        return 3
    print(f"OK: {msg}")

    # Live MCP root cannot be verified from this process (Cursor owns stdio).
    # Self-echo status(root=HABITAT) is theater — do not pretend it checks the server.
    print()
    print("AGENT ACTION (required — live MCP root check):")
    print(f"  1. Ensure Cursor MCP server `{MCP_SERVER_ID}` is enabled for this workspace.")
    print("  2. Call tool `status` on that server (not apegmsh-studio-lib).")
    print(f"  3. Confirm status.root == {habitat_res}")
    print("  4. If mismatch: reload MCP / disable colliding apegmsh-studio* servers.")
    print("  See APE/instructions/studio-mcp.md")
    print()

    write_session(
        {
            "schema": 1,
            "state": "active",
            "started_at": utc_now(),
            "slug": args.slug,
            "habitat": habitat_res,
            "mcp_server_id": MCP_SERVER_ID,
            "mcp_json_root": mcp_root,
            "live_status_root": None,
            "live_status_verified": False,
            "note": (
                f"stdio MCP is owned by Cursor. Agent must call `{MCP_SERVER_ID}` "
                "status and record status.root in postmortem metrics."
            ),
        }
    )
    print(f"OK: session marker -> {SESSION_FILE}")
    print()
    print("Next: verify MCP status.root (above), then open APE/README.md.")
    print("      Tools: python tools/apeCAD/open_interface.py")
    print("             python tools/apeSketch/open_interface.py")
    print("=== START complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
