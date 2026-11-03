"""make_demo_results / Results.demo — zero-setup sample results.

Headless: builds the demo (mesh + apeSees model emit + synthetic
pushover, no OpenSees solve, no render) and checks the shape of the
result — one pushover stage, the right step count, and a deflection
that ramps from zero to the requested tip drift. The viewer rendering
itself is covered by the WebViewer / viewer tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# apeSees model emit needs the opensees bridge; mesh needs gmsh (base dep).
pytest.importorskip("openseespy.opensees", reason="apeSees bridge import")


def test_make_demo_results_shape():
    from apeGmsh.results import make_demo_results

    r = make_demo_results(length=10.0, n_elements=6, n_steps=5, tip_drift=2.0)
    slab = r.nodes.get(component="displacement_x")
    vals = np.asarray(slab.values, dtype=np.float64)   # (n_steps, n_nodes)
    assert vals.shape[0] == 5
    # Step 0 is the undeformed reference.
    np.testing.assert_allclose(vals[0], 0.0, atol=1e-12)
    # Last step is non-trivially deflected.
    assert np.abs(vals[-1]).max() > 0.0
    # Monotone ramp at the most-deflected node.
    tip_col = int(np.argmax(np.abs(vals[-1])))
    series = vals[:, tip_col]
    assert np.all(np.diff(series) >= -1e-12)
    # Tip drift lands near the requested amplitude (analytic shape == 1.0
    # at the free end, ramped to tip_drift).
    np.testing.assert_allclose(np.abs(series[-1]), 2.0, rtol=0.05)
    r.close()


def test_results_demo_classmethod():
    from apeGmsh.results import Results

    r = Results.demo(n_steps=3, n_elements=4)
    slab = r.nodes.get(component="displacement_x")
    assert np.asarray(slab.values).shape[0] == 3
    r.close()


def test_demo_rejects_bad_args():
    from apeGmsh.results import make_demo_results

    with pytest.raises(ValueError):
        make_demo_results(n_steps=0)
    with pytest.raises(ValueError):
        make_demo_results(n_elements=0)


def _lineage(model_h5: Path) -> "tuple[str, str]":
    import h5py

    with h5py.File(model_h5, "r") as f:
        attrs = f["meta/lineage"].attrs
        return tuple(  # type: ignore[return-value]
            v.decode() if isinstance(v, bytes) else str(v)
            for v in (attrs["model_hash"], attrs["fem_hash"]))


def test_demo_writes_a_stamped_ladruno(tmp_path: Path):
    """ADR 0131 D10 W2 (#2778): the demo is a ``from_ladruno`` read of a
    ``.ladruno`` stamped against its bridge archive (ADR 0122 D3).

    Oracles: the stamp's ``model_hash`` / ``fem_hash`` are byte-equal to
    ``demo_model.h5``'s ``/meta/lineage``; the displacements are the
    closed-form tip-load shape ``u(z) = a z^2 (3L - z) / (2 L^3)``. On
    origin/main the demo wrote a native ``demo_results.h5`` instead.
    """
    import h5py

    from apeGmsh.results import make_demo_results
    from apeGmsh.results.readers._ladruno import LadrunoReader

    length, drift, n_steps = 10.0, 2.0, 4
    r = make_demo_results(path=tmp_path, length=length, n_elements=5,
                          n_steps=n_steps, tip_drift=drift)
    try:
        assert isinstance(r._reader, LadrunoReader)
        assert r._path == tmp_path / "demo_results.ladruno"
        assert r._model_path == tmp_path / "demo_model.h5"
        assert not (tmp_path / "demo_results.h5").exists()
        assert [(s.name, s.kind) for s in r.stages] == [
            ("MODEL_STAGE[1]", "static")]

        slab = r.nodes.get(component="displacement_x")
        z = np.asarray(r.fem.nodes.coords, dtype=float)[:, 2]
        nid = [int(i) for i in r.fem.nodes.ids]
        col = {int(t): j for j, t in enumerate(slab.node_ids)}
        shape = z ** 2 * (3.0 * length - z) / (2.0 * length ** 3)
        want = np.outer(np.linspace(0.0, drift, n_steps), shape)
        got = np.asarray(slab.values)[:, [col[t] for t in nid]]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
        for comp in ("displacement_y", "displacement_z"):
            np.testing.assert_array_equal(
                r.nodes.get(component=comp).values, 0.0)
    finally:
        r.close()

    model_hash, fem_hash = _lineage(tmp_path / "demo_model.h5")
    with h5py.File(tmp_path / "demo_results.ladruno", "r") as f:
        assert f["INFO"].attrs["GENERATOR"].tolist() == [b"Ladruno"]
        prov = dict(f["INFO/PROVENANCE"].attrs)
    assert prov["model_hash"] == model_hash
    assert prov["fem_hash"] == fem_hash
    # No element bucket is written (ADR 0131 D6), and says so; nothing
    # solved the values, so no engine is named.
    assert json.loads(prov["unsupported"]) == ["elements"]
    assert prov["opensees_backend"] == "unknown"
    assert prov["opensees_build_sha"] == "unknown"
    assert prov["writer"].startswith("apeGmsh demo->ladruno ")


@pytest.mark.ladruno_fork
def test_demo_ladruno_passes_fork_validate(tmp_path: Path):
    """The demo's file passes the fork's ``ladruno_format.validate()`` at
    ``FORK_PIN`` (ADR 0122 D8), fetched as the live-fork lane fetches it."""
    from apeGmsh.results import make_demo_results
    from tests.opensees.integration_ladruno import _fork_spec

    make_demo_results(path=tmp_path / "demo", n_elements=4, n_steps=3).close()
    spec = _fork_spec.fetch(tmp_path / "spec")
    with _fork_spec.ladruno_format(spec) as module:
        assert module.validate(
            str(tmp_path / "demo" / "demo_results.ladruno")) == []


@pytest.mark.parametrize("results, match", [
    ({"VELOCITY": np.zeros((2, 2, 3))}, "not a nodal group"),
    ({"DISPLACEMENT": np.zeros((2, 2, 2))}, "has shape"),
    ({}, "no result"),
])
def test_write_nodal_history_refuses(tmp_path: Path, results, match):
    """A group the writer does not know the recorder's layout of, or a
    block of the wrong shape, refuses before any file exists."""
    from apeGmsh.results.readers._ladruno_writer import write_nodal_history

    out = tmp_path / "x.ladruno"
    with pytest.raises(ValueError, match=match):
        write_nodal_history(
            out, model_h5=tmp_path / "absent.h5", node_ids=[1, 2],
            coords=np.zeros((2, 3)), ndm=3, kind="static",
            time=[0.0, 1.0], results=results, source="test")
    assert list(tmp_path.iterdir()) == []


def test_demo_model_archive_is_readable():
    """The model archive the demo binds reads back as the bound model:
    ``OpenSeesModel.from_h5(demo_model.h5)`` has the demo's nodes and
    lineage (the composed native file it used to embed is gone)."""
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import make_demo_results

    r = make_demo_results(n_elements=4, n_steps=3)
    try:
        model = OpenSeesModel.from_h5(str(r._model_path))
        assert model.model_name == "demo_model"
        assert model.snapshot_id
        assert model.snapshot_id == r.fem.snapshot_id
        assert model.fem.nodes.ids.size == r.fem.nodes.ids.size == 5
    finally:
        r.close()


def test_demo_records_model_path_in_viewer_argv():
    """The non-blocking subprocess viewer must forward --model-h5 to the
    sibling demo_model.h5: the .ladruno carries no bridge records."""
    from apeGmsh.results import Results

    r = Results.demo(n_steps=3, n_elements=4)
    try:
        assert r._model_path is not None
        assert r._model_path.name == "demo_model.h5"
        argv = r._build_viewer_argv(title=None)
        assert "--model-h5" in argv
        assert str(r._model_path) in argv
    finally:
        r.close()


def test_demo_subprocess_viewer_opens():
    """Regression: `Results.demo().viewer(blocking=False)` crashed the child
    with MalformedH5Error (/model zone empty) because --model-h5 was never
    forwarded. Run the real child argv (APEGMSH_SKIP_VIEWER so it opens the
    results then skips the GUI) and assert it exits 0."""
    import os
    import subprocess
    from apeGmsh.results import Results

    r = Results.demo(n_steps=3, n_elements=4)
    argv = r._build_viewer_argv(title=None)
    r.close()  # release the file handle before the child opens it (Windows)

    env = dict(os.environ, APEGMSH_SKIP_VIEWER="1")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_demo_viewer_data_reads_model_archive():
    """The blocking desktop viewer's read seam — `ViewerData.from_h5` →
    `h5_reader.open` — opens the demo's model archive: the `.ladruno`'s
    element reads are fem-keyed against `demo_model.h5`, so the scene
    source is that archive (formerly the composed native results file)."""
    from apeGmsh.results import make_demo_results
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source
    from apeGmsh.viewers.data._viewer_data import ViewerData

    r = make_demo_results(n_elements=4, n_steps=3)
    try:
        source = resolve_orientation_source(r)
        assert source is not None and str(source) == str(r._model_path)
        vd = ViewerData.from_h5(str(source))
        assert vd.nodes.ids.size > 0
        assert vd.snapshot_id
    finally:
        r.close()
