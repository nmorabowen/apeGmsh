"""VA6e (#2794): a stored Gauss component contour draws through ``apegmsh-view/1``.

The parity plan's row R16b (ruled a repoint on 2026-11-02) names a stored Gauss
component, e.g. ``stress_xx``, as ``contour: {field, component, source: "element",
gauss}``. The app names a ``.ladruno`` Gauss column ``<token>/<label>``: the
ON_ELEMENTS token and the file's own COMP_NAMES label
(``apeGmshViewer/src/reader/ladrunoElements.ts``, ``state/contour.ts``
``gaussFields``). The still keeps v1's averaged picture: ``gauss_cell_averaged`` at
one Gauss point, ``gauss_node`` (extrapolated to the corners, averaged) otherwise.

The pair is the fork's ``quad2d.ladruno`` (one FourNodeQuad, four Gauss points,
two steps) beside a ``<stem>.h5`` holding the same node ids, coordinates and
cell as a gmsh ``Quadrilateral 4``, stamped as the recorder's run would be.

Oracles: the spec values are the file's own names (literals read off the file
above); the live test (``APEGMSH_VIEWER`` set to a source build, never the
installed app) compares the app's manifest with the closed-form bilinear
extrapolation of the Gauss values Python's ``Results`` reads.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import stamp_provenance

from tests.assess.test_assess import _make_fem

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno" / "quad2d.ladruno"


class _FakeApp:
    """``agent_view.render_stills`` without the app: records each spec and writes
    a stand-in PNG per shot and an all-ok manifest, as the app does on exit 0."""

    def __init__(self) -> None:
        self.specs: list[dict] = []

    def __call__(self, spec, out, **kwargs):
        self.specs.append(json.loads(json.dumps(spec)))
        out = Path(out)
        out.mkdir(parents=True)
        shots = []
        for shot in spec["shots"]:
            (out / f"{shot['name']}.png").write_bytes(b"\x89PNG stand-in")
            shots.append({"name": shot["name"], "status": "ok", "png": f"{shot['name']}.png"})
        manifest = {"spec": spec["spec"], "shots": shots}
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


@pytest.fixture
def app_backend(monkeypatch):
    """The Windows path, on any platform, with a fake app."""
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    fake = _FakeApp()
    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "app")
    monkeypatch.setattr(agent_view, "render_stills", fake)
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    return fake


def _quad4_info():
    from apeGmsh.mesh._element_types import make_type_info

    return make_type_info(code=3, gmsh_name="Quadrilateral 4", dim=2, order=1, npe=4, count=1)


@pytest.fixture
def quad_set(tmp_path: Path) -> Path:
    """``quad.ladruno`` beside ``quad.h5``: the pair the app opens.

    ``quad.h5`` holds the file's own nodes and its one cell as a gmsh quad4, so
    the app can extrapolate (a FEMData read back from the ``.ladruno`` types the
    cell by OpenSees class). ``stamp_provenance`` writes the pairing hashes; it
    leaves ``element_index_space`` absent for an archive with no ops records, so
    it is set here: element 1 is FEM element 1 by construction.
    """
    src = tmp_path / "quad.ladruno"
    shutil.copy(_FIXTURE, src)
    with Results.from_ladruno(src) as r:
        (group,) = list(r.fem.elements)
        fem = _make_fem(
            node_ids=np.asarray(r.fem.nodes.ids).tolist(),
            coords=[tuple(p) for p in np.asarray(r.fem.nodes.coords, dtype=float).tolist()],
            groups=[(_quad4_info(), np.asarray(group.ids).tolist(), np.asarray(group.connectivity).tolist())],
        )
    fem.to_h5(str(tmp_path / "quad.h5"))
    stamp_provenance(src, tmp_path / "quad.h5")
    with h5py.File(src, "r+") as f:
        f["INFO/PROVENANCE"].attrs["element_index_space"] = "fem"
    return src


# ---------------------------------------------------------------------
# The spec (no app; runs on every platform)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("component, column", [
    ("stress_xx", "stress/sigma11"),
    ("stress_xy", "stress/sigma12"),
    ("strain_yy", "strain/eta22"),
])
def test_stored_gauss_component_is_the_app_element_field(app_backend, quad_set, tmp_path, component, column):
    """R16b: the file's token and its own label; four Gauss points draw as gauss_node."""
    with Results.from_ladruno(quad_set) as r:
        out = r.render(tmp_path / "g.png", view="contour", component=component)
    assert out == tmp_path / "g.png" and out.is_file()
    (spec,) = app_backend.specs
    (shot,) = spec["shots"]
    assert shot["stage"] == 0 and shot["step"] == 1
    assert shot["contour"] == {
        "field": column.split("/")[0], "component": column, "source": "element", "gauss": "node",
    }


def test_deformed_gauss_still_warps_by_displacement(app_backend, quad_set, tmp_path):
    with Results.from_ladruno(quad_set) as r:
        r.render(tmp_path / "d.png", view="deformed", component="stress_xx", deform=5.0)
    (shot,) = app_backend.specs[0]["shots"]
    assert shot["contour"]["component"] == "stress/sigma11"
    assert shot["deform"] == {"scale": 5.0}


def test_derived_scalar_stays_refused(app_backend, quad_set, tmp_path):
    """R16c (G10): a derived scalar is not a recorded column; refused by name."""
    with Results.from_ladruno(quad_set) as r:
        with pytest.raises(ValueError, match="von_mises_stress"):
            r.render(tmp_path / "x.png", view="contour", component="von_mises_stress")
    assert app_backend.specs == []


def test_a_column_in_two_tokens_is_refused(app_backend, quad_set, tmp_path):
    """Fail closed: two app columns for one component is no answer, never a pick."""
    with h5py.File(quad_set, "r+") as f:
        on = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
        on.copy(on["stress"], "material.stress")
    with Results.from_ladruno(quad_set) as r:
        with pytest.raises(ValueError, match=r"stress/sigma11.*R16b|R16b"):
            r.render(tmp_path / "x.png", view="contour", component="stress_xx")
    assert app_backend.specs == []


_SHELL_LABELS = ("p11", "p22", "p12", "m11", "m22", "m12", "q1", "q2")


def _add_shell_resultants(path: Path) -> None:
    """The quad as a ShellMITC4 whose only element result is ``stresses``: the shell
    resultant family the fork's shells tag per Gauss point (``p11..q2``), which the
    reader renames in place to the catalog's names."""
    shell, quad = "53-ShellMITC4[201:0", "31-FourNodeQuad[201:0"
    with h5py.File(path, "r+") as f:
        model = f["MODEL_STAGE[1]/MODEL/ELEMENTS"]
        model.move(quad + "]", shell + "]")
        on = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
        on.move("stress", "stresses")
        on["stresses"].move(quad + ":0]", shell + ":0]")
        for token in ("strain", "force"):
            del on[token]
        bucket = on["stresses"][shell + ":0]"]
        bucket["COLUMN_MAP"].attrs["COMP_NAMES"] = np.bytes_("\n".join([",".join(_SHELL_LABELS)] * 4))
        bucket["COLUMN_MAP/NUM_COMP"][...] = len(_SHELL_LABELS)
        bucket.attrs["NUM_COLUMNS"] = np.array([4 * len(_SHELL_LABELS)], dtype=np.int32)
        del bucket["DATA"]
        bucket["DATA"] = np.arange(2 * 4 * len(_SHELL_LABELS), dtype=float).reshape(2, 1, -1)


@pytest.mark.parametrize("component, label", [("membrane_force_xx", "p11"), ("transverse_shear_yz", "q2")])
def test_shell_resultant_is_the_files_own_label(app_backend, quad_set, tmp_path, component, label):
    """Review F1: the reader names a shell block by the catalog in place; the app reads
    the file's label at that position (``stresses/p11``), so the shot names it."""
    _add_shell_resultants(quad_set)
    with Results.from_ladruno(quad_set) as r:
        assert component in r.inspect.components(stage=r.stages[0].id)["gauss"]
        r.render(tmp_path / "s.png", view="contour", component=component)
    (shot,) = app_backend.specs[0]["shots"]
    assert shot["contour"] == {"field": "stresses", "component": f"stresses/{label}", "source": "element", "gauss": "node"}


def _strip_elements(path: Path) -> None:
    with h5py.File(path, "r+") as f:
        on = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
        for token in list(on):
            del on[token]


@pytest.mark.parametrize("opened", [0, 1])
def test_a_partitioned_set_is_scanned_whole(app_backend, quad_set, tmp_path, opened):
    """Review F2: the app opens every part of the set; the Gauss buckets sit in part 1
    only, and either part names the same column."""
    from apeGmsh.viewers.session._results_stills import _app_contour

    parts = [tmp_path / f"q.part-{i}.ladruno" for i in (0, 1)]
    for i, p in enumerate(parts):
        shutil.copy(_FIXTURE, p)
        with h5py.File(p, "r+") as f:
            f["INFO"].attrs.update({"NUM_PARTITIONS": [2], "PARTITION_ID": [i], "PARTITIONED": [1]})
    _strip_elements(parts[0])
    with Results.from_ladruno(parts[opened]) as r:
        assert _app_contour(r, r.stages[0].id, "stress_xx") == {
            "field": "stress", "component": "stress/sigma11", "source": "element", "gauss": "node",
        }


def test_shot_for_a_source_the_app_cannot_open_is_refused(quad_set, tmp_path, monkeypatch):
    """Review F3: ``app_shot_for_view`` is public; without the skip, a Gauss shot of a
    source that is not a ``.ladruno`` is refused, never a made-up contour."""
    from apeGmsh.viewers.session._results_stills import app_shot_for_view

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with Results.from_ladruno(quad_set) as r:
        monkeypatch.setattr(type(r), "path", property(lambda self: tmp_path / "quad.h5"))
        with pytest.raises(ValueError, match="R22b"):
            app_shot_for_view(r, name="s", view="contour", component="stress_xx", step=-1, deform=None, camera="iso")


def test_skip_viewer_with_a_native_source_draws_nothing_and_does_not_refuse(app_backend, quad_set, tmp_path, monkeypatch, capsys):
    """Under APEGMSH_SKIP_VIEWER=1 any source skips (VA6b); a Gauss shot of a source the
    app cannot open is validated by Python's recorded set and never resolved against a file."""
    from apeGmsh.viewers.session import _results_stills as stills

    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    with Results.from_ladruno(quad_set) as r:
        monkeypatch.setattr(type(r), "path", property(lambda self: tmp_path / "quad.h5"))
        assert r.render(tmp_path / "x.png", view="contour", component="stress_xx") is None
        assert stills._app_gauss_contour(r, "stress_xx")["source"] == "element"
    assert "[skip viewer]" in capsys.readouterr().out
    assert app_backend.specs == []


# ---------------------------------------------------------------------
# Drawn by the app (a source build named by APEGMSH_VIEWER)
# ---------------------------------------------------------------------


def _corner_values(natural: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The bilinear field through the four Gauss values, at the corners in quad4
    order (-1,-1), (1,-1), (1,1), (-1,1): gauss_node on one element (no averaging)."""
    xi, eta = natural[:, 0], natural[:, 1]
    basis = np.column_stack([np.ones_like(xi), xi, eta, xi * eta])
    coef = np.linalg.solve(basis, values)
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return np.column_stack([np.ones(4), corners[:, 0], corners[:, 1], corners[:, 0] * corners[:, 1]]) @ coef


def _built_app() -> str:
    named = os.environ.get("APEGMSH_VIEWER")
    if not named or not Path(named).is_file():
        pytest.skip("APEGMSH_VIEWER does not name a source build of apeGmshViewer")
    return named


def test_gauss_contour_draws_through_the_app_and_matches_python(quad_set, tmp_path, monkeypatch):
    """The still is drawn, and the manifest's numbers are Python's: at the last step
    the extrema are the extrapolated corner values, and the history range spans both steps."""
    _built_app()
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    real, manifests = agent_view.render_stills, []

    def spy(spec, out, **kwargs):
        manifests.append(real(spec, out, **kwargs))
        return manifests[-1]

    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "app")
    monkeypatch.setattr(agent_view, "render_stills", spy)
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with Results.from_ladruno(quad_set) as r:
        out = r.render(tmp_path / "stress.png", view="contour", component="stress_xx")
        stage = r.stage(r.stages[0].id)
        conn = np.asarray(list(r.fem.elements)[0].connectivity)[0]
        steps = []
        for k in (0, 1):
            slab = stage.elements.gauss.get(component="stress_xx", time=k)
            steps.append(_corner_values(np.asarray(slab.natural_coords), np.asarray(slab.values)[0]))
    assert out.is_file() and out.stat().st_size > 0
    (entry,) = manifests[0]["shots"]
    assert entry["status"] == "ok", entry["reason"]
    contour = entry["contour"]
    assert contour["drawn"] is True
    assert (contour["source"], contour["field"], contour["component"]) == ("element", "stress", "stress/sigma11")
    assert contour["topology"] == "gauss_node"
    last = steps[1]
    rec = contour["extrema"]["recorded"]
    assert rec["min"]["value"] == pytest.approx(last.min(), rel=1e-9)
    assert rec["max"]["value"] == pytest.approx(last.max(), rel=1e-9)
    assert rec["min"]["node"] == int(conn[int(np.argmin(last))])
    assert rec["max"]["node"] == int(conn[int(np.argmax(last))])
    both = np.concatenate(steps)
    assert contour["range"]["mode"] == "history"
    assert contour["range"]["min"] == pytest.approx(both.min(), rel=1e-9)
    assert contour["range"]["max"] == pytest.approx(both.max(), rel=1e-9)
