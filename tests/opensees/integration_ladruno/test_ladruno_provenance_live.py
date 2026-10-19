"""#2099 — ``stamp_provenance`` on a real fork ``.ladruno`` (ADR 0122 D3).

One 2-D cantilever, two ``ops.stage`` blocks (``gravity``, Static, then
``shake``, Transient). The bridge saves the model with ``ops.h5`` and emits
a Py deck that records a ``.ladruno``; the deck runs on the fork module in
a fresh process and closes the recorder (``remove recorders``) before it
exits. ``stamp_provenance`` then stamps the closed file.

Oracles:

* the fork's own executable spec, ``ladruno_format.validate()``, fetched
  from the fork repository at the ``FORK_PIN`` commit, returns no
  violation on the stamped file (§7b.1: ``validate()`` does not inspect
  ``PROVENANCE``, so a stamped file stays conformant);
* ``model_hash`` / ``fem_hash`` equal the model's ``/meta/lineage``;
* ``stage_names`` holds the program's names, because the recorder's
  ``KIND`` per stage matches each stage's ``analysis`` record;
* stamped with the ``ladrunoBuild()`` the deck's own engine printed,
  passed explicitly as the recording caller would, ``opensees_backend`` is
  ``"ladruno"`` and ``opensees_build_sha`` is that hash and, in the
  ``live-fork`` lane (``APEGMSH_FORK_PIN_ENFORCE=1``), the ``FORK_PIN``
  sha; a copy stamped with defaults from this process reads ``"unknown"``
  (#2196, #2208 F4, §7b.1 at fork PR #997).

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import pytest

import apeGmsh
from apeGmsh.results.writers import stamp_provenance

from tests.opensees.integration_ladruno.test_ladruno_modes_live import modal_bridge
from tests.opensees.live.test_readback_live import P_TIP, build_cantilever_fem

pytestmark = pytest.mark.ladruno_fork

_PIN = Path(__file__).resolve().parents[3] / "FORK_PIN"
_SPEC_DIR = "Ladruno_scripts/ladruno_recorder_tests"
_SPEC_FILES = ("ladruno_format.py", "ladruno_basis.py")


def _pin(key: str) -> str:
    for line in _PIN.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == key:
            return v.strip()
    raise RuntimeError(f"no {key}= line in {_PIN}")


def _fresh_env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])],
    )
    return env


def _chain(ops: Any, integrator: Any, analysis: Any) -> "dict[str, Any]":
    return dict(
        test=ops.test.NormDispIncr(tol=1e-12, max_iter=10),
        algorithm=ops.algorithm.Linear(), integrator=integrator,
        constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
        system=ops.system.BandGeneral(), analysis=analysis,
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    tmp = tmp_path_factory.mktemp("provenance_live")
    ladruno, model_h5, deck = tmp / "r.ladruno", tmp / "model.h5", tmp / "deck.py"
    ops = modal_bridge(build_cantilever_fem())
    ops.recorder.Ladruno(file=str(ladruno), nodal_responses=("displacement",))
    tip = ops.nodes.one(label="tip")
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=tip, forces=(0.0, -P_TIP, 0.0))
        s.analysis(**_chain(ops, ops.integrator.LoadControl(dlam=0.25),
                            ops.analysis.Static()))
        s.run(n_increments=4)
    with ops.stage(name="shake") as s:
        with s.pattern(series=ops.timeSeries.Constant()) as p:
            p.load(node=tip, forces=(P_TIP, 0.0, 0.0))
        s.analysis(**_chain(ops, ops.integrator.Newmark(gamma=0.5, beta=0.25),
                            ops.analysis.Transient()))
        s.run(n_increments=10, dt=0.01)
    ops.h5(str(model_h5))
    ops.py(str(deck), progress=False)

    text = deck.read_text(encoding="utf-8")
    stock_import = "import openseespy.opensees as ops"
    assert stock_import in text
    text = text.replace(
        stock_import,
        "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n",
    )
    # Close the recorder, then prove the deck ran to its end (a Tcl or
    # Python error can still exit 0 on some engines).
    text += ("\nops.remove('recorders')\n"
             "print('ENGINE-BUILD=' + str(ops.ladrunoBuild()))\n"
             "print('DECK-DONE')\n")
    deck.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], cwd=tmp, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0 and "DECK-DONE" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-2000:])
    assert ladruno.is_file()
    build = next(line.split("=", 1)[1].strip()
                 for line in proc.stdout.splitlines()
                 if line.startswith("ENGINE-BUILD="))
    # A copy stamped with defaults, from this (pytest) process: it did not
    # record the file, so it cannot know the build (#2208 F4).
    foreign = tmp / "foreign" / "r.ladruno"
    foreign.parent.mkdir()
    shutil.copy(ladruno, foreign)
    stamp_provenance(foreign, model_h5)
    # The caller that ran the analysis passes the build it ran on.
    stamp_provenance(ladruno, model_h5, opensees_build_sha=build)
    return {"ladruno": ladruno, "model_h5": model_h5, "tmp": tmp,
            "build": build, "foreign": foreign}


def _fork_validate(tmp: Path) -> Any:
    """``ladruno_format.validate`` from the fork repository at ``FORK_PIN``."""
    sha, repo = _pin("sha"), _pin("repo")
    spec = tmp / "fork_spec"
    spec.mkdir(exist_ok=True)
    for name in _SPEC_FILES:
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{_SPEC_DIR}/{name}"
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            (spec / name).write_bytes(resp.read())
    return spec


def test_fork_validate_passes_on_the_stamped_file(run, monkeypatch) -> None:
    spec = _fork_validate(run["tmp"])
    monkeypatch.syspath_prepend(str(spec))
    saved = {k: sys.modules[k] for k in ("ladruno_format", "ladruno_basis")
             if k in sys.modules}
    try:
        ladruno_format = importlib.import_module("ladruno_format")
        with h5py.File(run["ladruno"], "r") as f:
            assert "model_hash" in f["INFO/PROVENANCE"].attrs  # it is stamped
        assert ladruno_format.validate(str(run["ladruno"])) == []
    finally:
        for k in ("ladruno_format", "ladruno_basis"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_stamped_hashes_equal_the_model_lineage(run) -> None:
    with h5py.File(run["model_h5"], "r") as m, h5py.File(run["ladruno"], "r") as f:
        lineage, prov = m["meta/lineage"].attrs, f["INFO/PROVENANCE"].attrs
        for key in ("model_hash", "fem_hash"):
            assert lineage[key]
            assert prov[key] == lineage[key]


def test_stage_names_are_the_program_names(run) -> None:
    with h5py.File(run["ladruno"], "r") as f:
        kinds = [f[g].attrs["KIND"] for g in sorted(
            (k for k in f if k.startswith("MODEL_STAGE[")),
            key=lambda n: int(n[len("MODEL_STAGE["):-1]))]
        names = json.loads(f["INFO/PROVENANCE"].attrs["stage_names"])
    assert [k.tobytes().decode() if hasattr(k, "tobytes") else k
            for k in kinds] == ["static", "transient"]
    assert names == ["gravity", "shake"]


def test_the_engine_is_the_fork_build_that_ran(run) -> None:
    """The caller's stamp names the build the deck ran on (§7b.1 names)."""
    build = run["build"]
    assert re.fullmatch(r"[0-9a-f]{40}", build), build
    if os.environ.get("APEGMSH_FORK_PIN_ENFORCE") == "1":
        assert build == _pin("sha")
    with h5py.File(run["ladruno"], "r") as f:
        prov = f["INFO/PROVENANCE"].attrs
        assert prov["opensees_backend"] == "ladruno"
        assert prov["opensees_build_sha"] == build


def test_a_default_stamp_from_another_process_is_unknown(run) -> None:
    """The pytest process did not record the file: it may not claim a build."""
    with h5py.File(run["foreign"], "r") as f:
        prov = f["INFO/PROVENANCE"].attrs
        assert prov["opensees_backend"] == "ladruno"
        assert prov["opensees_build_sha"] == "unknown"
