"""The app pairs a ``stamp_provenance``-stamped ``.ladruno`` (ADR 0122 D10 row 4, #2129).

The two halves of ADR 0122 D3 live on either side of the ADR 0112 D4
wall: Python's ``stamp_provenance`` copies ``/meta/lineage`` of a
bridge-written ``model.h5`` into ``INFO/PROVENANCE`` of a closed
``.ladruno``, and the app's ``state/results.ts::pairResults`` reads it back
through ``reader/ladruno.ts`` to decide whether the run belongs to the open
model. Each half has its own tests against hand-written stamps; this one
runs both on the same files.

Python writes the files: a model through the real bridge, and copies of
the committed fork-written ``tests/fixtures/ladruno/truss2d.ladruno``
stamped with the real ``stamp_provenance``. Then it spawns ``node`` on
``apeGmshViewer/scripts/pair_oracle.ts``, which opens them with the app's
own main-process path (``ResultsHost.open``, ``openLineage``,
``openModel``) and prints ``pairResults``' verdict. The oracle is the
D3 rule itself:

* stamped against this model: ``paired`` by ``model_hash``, the hash the
  app read equal to the model's ``/meta/lineage@model_hash`` byte for byte;
* unstamped (the recorder's file as written): ``unpairable``, the reason
  naming ``stamp_provenance``;
* stamped against another model on the same mesh (``fem_hash`` equal,
  ``model_hash`` not): ``mismatch``, never a ``fem_hash`` pairing.

Nothing crosses the wall: this file imports nothing from the app, and the
node side imports nothing from ``src/apeGmsh``. It needs node >= 22.18
(the app's ``engines``; it runs the TypeScript by type stripping, as
``npm test`` does) and the app's ``npm ci`` (h5wasm); without either it
skips with the reason, shown under ``-rs``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.opensees import apeSees
from apeGmsh.results.writers import stamp_provenance

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "apeGmshViewer"
ORACLE = APP / "scripts" / "pair_oracle.ts"
TRUSS = REPO / "tests" / "fixtures" / "ladruno" / "truss2d.ladruno"
#: package.json "engines": the first node that strips TypeScript types unflagged.
NODE_MIN = (22, 18)


def _node_or_skip_reason() -> "tuple[str | None, str | None]":
    """``(node, None)`` when the oracle can run, else ``(None, why not)``."""
    node = shutil.which("node")
    if node is None:
        return None, "node is not on PATH (the oracle needs node >= 22.18)"
    out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30)
    m = re.fullmatch(r"v(\d+)\.(\d+)\.\d+", out.stdout.strip())
    if m is None:
        return None, f"cannot read the node version from {out.stdout.strip()!r}"
    if (int(m.group(1)), int(m.group(2))) < NODE_MIN:
        return None, f"node {out.stdout.strip()} is older than 22.18 (no unflagged type stripping)"
    if not (APP / "node_modules" / "h5wasm").is_dir():
        return None, "apeGmshViewer/node_modules has no h5wasm: run `npm ci` in apeGmshViewer/"
    return node, None


NODE, SKIP_REASON = _node_or_skip_reason()
pytestmark = pytest.mark.skipif(NODE is None, reason=SKIP_REASON or "")


def _truss_fem() -> FEMData:
    """Two bars on nodes 1-2-3 along x, as ``truss2d.ladruno`` records them."""
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    group = ElementGroup(
        element_type=info,
        ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
    )
    pg = {
        (1, 1): {"name": "Bars", "node_ids": node_ids, "node_coords": coords, "element_ids": group.ids},
        (0, 2): {"name": "Base", "node_ids": node_ids[:1], "node_coords": coords[:1]},
    }
    nodes = NodeComposite(node_ids=node_ids, node_coords=coords,
                          physical=PhysicalGroupSet(pg), labels=LabelSet({}))
    elements = ElementComposite(groups={1: group}, physical=PhysicalGroupSet(pg), labels=LabelSet({}))
    return FEMData(nodes=nodes, elements=elements,
                   info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[info]))


def _write_model(path: Path, *, E: float) -> Path:
    """A bridge-written ``model.h5``: one static stage of 4 LoadControl steps, as the fixture ran."""
    ops = apeSees(_truss_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
    ops.element.Truss(pg="Bars", A=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(pg="Bars", forces=(10.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
            algorithm=ops.algorithm.Linear(),
            integrator=ops.integrator.LoadControl(dlam=0.25),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.BandGeneral(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=4)
    ops.h5(str(path))
    return path


def _lineage(model_h5: Path) -> "dict[str, str]":
    with h5py.File(model_h5, "r") as f:
        attrs = f["meta/lineage"].attrs
        return {k: str(attrs[k]) for k in ("model_hash", "fem_hash")}


def _ladruno(tmp: Path, name: str, stamp_with: "Path | None") -> Path:
    path = tmp / name
    shutil.copyfile(TRUSS, path)
    if stamp_with is not None:
        assert stamp_provenance(path, stamp_with) == [path]
    return path


def _app_pairing(model_h5: Path, ladruno: Path) -> "dict[str, object]":
    """Run the app's own read + ``pairResults`` on the two files; its JSON line."""
    assert NODE is not None
    out = subprocess.run(
        [NODE, str(ORACLE), str(model_h5), str(ladruno)],
        cwd=APP, capture_output=True, encoding="utf-8", timeout=120,
    )
    assert out.returncode == 0, f"pair_oracle.ts exited {out.returncode}:\n{out.stderr}"
    lines = out.stdout.strip().splitlines()
    assert len(lines) == 1, out.stdout
    return json.loads(lines[0])


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Path]":
    d = tmp_path_factory.mktemp("pairing_oracle")
    this = _write_model(d / "truss.h5", E=1000.0)
    # The same mesh with another material: fem_hash equal, model_hash not.
    other = _write_model(d / "other.h5", E=2000.0)
    a, b = _lineage(this), _lineage(other)
    assert a["fem_hash"] == b["fem_hash"], "precondition: one mesh, one fem_hash"
    assert a["model_hash"] != b["model_hash"], "precondition: another model, another model_hash"
    return {"this": this, "other": other}


def test_a_ladruno_stamped_against_the_model_pairs_by_model_hash(models, tmp_path) -> None:
    run = _ladruno(tmp_path, "truss.ladruno", models["this"])
    got = _app_pairing(models["this"], run)
    assert got["pairing"] == {"state": "paired", "by": "model_hash"}, got
    want = _lineage(models["this"])
    results = got["results"]
    assert isinstance(results, dict)
    assert results["sourceType"] == "ladruno"
    # The app read the stamp stamp_provenance wrote: the model's hashes, byte for byte.
    assert results["lineage"] == {"modelHash": want["model_hash"], "femHash": want["fem_hash"], "resultsHash": None}
    assert results["lineageReason"] is None


def test_an_unstamped_ladruno_is_unpairable_and_names_the_remedy(models, tmp_path) -> None:
    run = _ladruno(tmp_path, "truss.ladruno", None)
    got = _app_pairing(models["this"], run)
    pairing = got["pairing"]
    assert isinstance(pairing, dict)
    assert pairing["state"] == "unpairable", got
    assert "not stamped for pairing" in pairing["reason"]
    assert "stamp_provenance(path, model_h5)" in pairing["reason"]


def test_a_stamp_from_another_model_on_the_same_mesh_is_a_mismatch(models, tmp_path) -> None:
    run = _ladruno(tmp_path, "truss.ladruno", models["other"])
    got = _app_pairing(models["this"], run)
    pairing = got["pairing"]
    assert isinstance(pairing, dict)
    # Not paired by fem_hash: a stamp that names a model_hash is judged by it alone.
    assert pairing["state"] == "mismatch", got
    other = _lineage(models["other"])["model_hash"]
    this = _lineage(models["this"])["model_hash"]
    assert pairing["reason"] == (
        "model_hash differs: the results were written from another model "
        f"(results {other[:8]}…, model {this[:8]}…)"
    )
