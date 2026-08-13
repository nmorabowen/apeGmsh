"""Child process — mesh one continuum ``SectionDocument`` (ADR 0080 B6).

Run as ``python -m apeGmsh.sections._mesh_worker <out_path>`` with the
document JSON on stdin. Meshes it and pickles an envelope to
``out_path``::

    {"ok": True,  "fem": <FEMData>, "error": None}
    {"ok": False, "fem": None,      "error": "SectionDocumentError: …"}

The envelope goes to a **file**, not stdout, because Gmsh writes
progress chatter to stdout and would corrupt a binary stream on it.

Why a child process at all: Gmsh is one process-global C++ runtime,
neither thread-safe nor reentrant. Running it here means the parent's
properties worker never touches Gmsh, so it neither races the main
thread (which aborts the interpreter) nor has to queue behind a session
the user left open. See :mod:`apeGmsh.sections._mesh_proc` for the
parent half.
"""
from __future__ import annotations

import json
import pickle
import sys
from typing import Any


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(
            "usage: python -m apeGmsh.sections._mesh_worker <out_path> "
            "(document JSON on stdin)",
            file=sys.stderr,
        )
        return 2
    out_path = argv[0]
    payload: "dict[str, Any]" = {"ok": False, "fem": None, "error": None}
    try:
        data = json.loads(sys.stdin.read())
        from apeGmsh.sections._document import SectionDocument

        payload["fem"] = SectionDocument(data).build_fem()
        payload["ok"] = True
    except BaseException as exc:      # noqa: BLE001 - reported, not raised
        payload["error"] = f"{type(exc).__name__}: {exc}"
    try:
        with open(out_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException as exc:      # noqa: BLE001 - last resort
        print(f"could not write the mesh envelope: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":       # pragma: no cover - process entry point
    raise SystemExit(main())
