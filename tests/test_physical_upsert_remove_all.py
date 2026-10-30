"""``g.physical.remove_all()`` after a targeted removal exits 0 (#2608).

In gmsh 4.15.2 the argument-less ``removePhysicalGroups()`` corrupts
the heap at finalize once any targeted ``removePhysicalGroups([(dim,
tag)])`` ran in the same model: the process dies at ``g.end()`` with
no traceback (``0xc0000374``, exit 127 under bash on Windows).  The
``g.physical.add`` upsert (remove the group, re-add it with the union
of entities) is such a targeted removal, so ``add`` on an existing
``(dim, name)``, then ``remove_all()``, then ``g.end()`` killed the
process 3 of 3.  Plain-gmsh probes narrowed it: the targeted removal
alone, the re-add alone, the rename route and an explicit-list
removal of every group all exit 0; only the argument-less form after
a targeted removal dies.  ``remove_all`` now lists the groups and
removes them by ``(dim, tag)``.

Each case runs the session in a fresh interpreter: the failure is a
process kill, so it can never run in the shared pytest process.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.subprocess

_ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = """
import gmsh
from apeGmsh import apeGmsh
g = apeGmsh(model_name="pg_remove_all", verbose=False)
g.begin()
occ = gmsh.model.occ
pts = [occ.addPoint(float(i), 0.0, 0.0) for i in range(3)]
a = occ.addLine(pts[0], pts[1])
b = occ.addLine(pts[1], pts[2])
occ.synchronize()
"""

_EPILOGUE = """
assert gmsh.model.getPhysicalGroups() == [], gmsh.model.getPhysicalGroups()
assert g.physical.get_tag(1, "grp") is None
g.end()
print("ENDED-CLEANLY")
"""

# The card's reproducer: the upsert route, then remove_all, then end.
_UPSERT = """
t = g.physical.add(1, [a], name="grp")
assert g.physical.add(1, [b], name="grp") == t
assert sorted(g.physical.get_entities(1, t)) == sorted([a, b])
g.physical.remove_all()
"""

# The minimal trigger through the public API: one targeted remove.
_REMOVE_ONE = """
t = g.physical.add(1, [a], name="grp")
t2 = g.physical.add(1, [b], name="other")
g.physical.remove([(1, t2)])
g.physical.remove_all()
"""


def _run_session(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # The worktree's sources, never the editable install's checkout.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    env["APEGMSH_ARTIFACT_DIR"] = str(tmp_path)
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + body + _EPILOGUE],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=300, check=False,
    )


@pytest.mark.parametrize(
    "body", [_UPSERT, _REMOVE_ONE], ids=["upsert", "remove_one"])
def test_remove_all_after_a_targeted_removal_exits_0(
        tmp_path: Path, body: str) -> None:
    proc = _run_session(tmp_path, body)
    assert proc.returncode == 0, (
        f"exit {proc.returncode} (0x{proc.returncode & 0xFFFFFFFF:08x})\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-4000:]}")
    assert "ENDED-CLEANLY" in proc.stdout, proc.stdout[-2000:]
