"""Out-of-process meshing for the properties worker (ADR 0080 B6).

The parent half of :mod:`apeGmsh.sections._mesh_worker`: hand a document
to a child interpreter, get its ``FEMData`` back.

**Why.** Gmsh is one process-global C++ runtime, neither thread-safe nor
reentrant. `_session.py`'s runtime lock makes concurrent Gmsh *within*
one process safe by serializing it — but serializing is not enough for
the live-properties worker, because a session the user left open holds
that lock for its whole lifetime and the worker would simply never get
a turn. Meshing in a child process removes the contention instead of
managing it: the worker's Gmsh work happens somewhere with no other
Gmsh user at all.

Only the meshing moves. The analyzer is pure NumPy over the snapshot
(``FEMData`` pickles in ~90 KB for a typical section), so it is rebuilt
and solved back in the parent — which keeps a *live*
:class:`~apeGmsh.sections.SectionProperties` for the inspector panel to
drive interactively (``analysis.stress(**loads)`` on user-entered
loads), something a "return the numbers" split would have lost.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Any


__all__ = ["MESH_SUBPROCESS_TIMEOUT", "MeshSubprocessError", "mesh_document"]


#: Wall-clock cap on one child mesh. Generous — a section mesh is
#: sub-second and this only has to catch a wedged child, not police
#: slow ones.
MESH_SUBPROCESS_TIMEOUT: float = 120.0


class MeshSubprocessError(RuntimeError):
    """The child interpreter failed to produce a mesh."""


def _child_env() -> "dict[str, str]":
    """Environment for the child, pinned to *this* apeGmsh.

    The child must import the same code the parent is running — in a git
    worktree or a source checkout, a bare ``python -m apeGmsh...`` would
    otherwise resolve through whatever the installed distribution points
    at.
    """
    import apeGmsh

    package_root = os.path.dirname(os.path.dirname(
        os.path.abspath(apeGmsh.__file__)
    ))
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join([package_root, existing]) if existing
        else package_root
    )
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    return env


def mesh_document(
    doc_dict: "dict[str, Any]",
    *,
    timeout: float | None = None,
) -> Any:
    """Mesh a continuum document in a child interpreter and return its
    ``FEMData``.

    Raises :class:`MeshSubprocessError` when the child cannot be run,
    times out, dies, or reports a document error — the message carries
    the child's own diagnosis so the panel can show it.
    """
    handle, out_path = tempfile.mkstemp(suffix=".fem.pkl", prefix="apegmsh_")
    os.close(handle)
    try:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "apeGmsh.sections._mesh_worker",
                 out_path],
                input=json.dumps(doc_dict),
                capture_output=True,
                text=True,
                timeout=MESH_SUBPROCESS_TIMEOUT if timeout is None else timeout,
                env=_child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise MeshSubprocessError(
                f"meshing did not finish within "
                f"{exc.timeout:g} s and was stopped."
            ) from exc
        except OSError as exc:
            raise MeshSubprocessError(
                f"could not start a mesh process: {exc}"
            ) from exc

        try:
            with open(out_path, "rb") as fh:
                payload = pickle.load(fh)
        except (OSError, EOFError, pickle.UnpicklingError) as exc:
            # the child died before writing a usable envelope — the
            # abort case this whole module exists to keep out of the
            # parent
            tail = (completed.stderr or "").strip().splitlines()
            detail = tail[-1] if tail else f"exit code {completed.returncode}"
            raise MeshSubprocessError(
                f"the mesh process failed without reporting: {detail}"
            ) from exc

        if not payload.get("ok"):
            raise MeshSubprocessError(
                payload.get("error") or "the mesh process reported no result."
            )
        return payload["fem"]
    finally:
        try:
            os.unlink(out_path)
        except OSError:             # pragma: no cover - already gone
            pass
