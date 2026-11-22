"""VA6e (#2794): a stored Gauss component contour draws through ``apegmsh-view/1``.

The parity plan's row R16b (ruled a repoint on 2026-11-02) names a stored Gauss
component, e.g. ``stress_xx``, as ``contour: {field, component, source: "element",
gauss}``. The app names a ``.ladruno`` Gauss column ``<token>/<label>``: the
ON_ELEMENTS token and the file's own COMP_NAMES label
(``apeGmshViewer/src/reader/ladrunoElements.ts``, ``state/contour.ts``
``gaussFields``). The still keeps v1's averaged picture: ``gauss_node`` (extrapolated
to the corners, averaged) when a group the app has natural coordinates for has several
Gauss points, ``gauss_cell_averaged`` otherwise. Since #2706 the app reads a ``"bary"``
GP_PARAM, so BezierTet10 and BezierTri6 ask ``node`` (#3085, retiring #2942 and #2950's
class-name exclusion); a group with no GP_PARAM still draws its element mean, per group,
as the owner ruled on #2924 item 2. Row R16c names
``von_mises_stress`` / ``von_mises_strain`` as a component of the element field
that holds the tensor.

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


@pytest.mark.parametrize("component, field", [("von_mises_stress", "stress"), ("von_mises_strain", "strain")])
def test_derived_scalar_is_a_component_of_the_tensor_field(app_backend, quad_set, tmp_path, component, field):
    """R16c (G10, #2942): the app offers ``von_mises_<family>`` as a component of the
    element field whose columns make the tensor (``compile.test.ts`` VA6d), at the
    tensor columns' mode; on origin/main the Python still refused it."""
    with Results.from_ladruno(quad_set) as r:
        r.render(tmp_path / "v.png", view="contour", component=component)
    (shot,) = app_backend.specs[0]["shots"]
    assert shot["contour"] == {"field": field, "component": component, "source": "element", "gauss": "node"}


def test_a_derived_scalar_the_app_does_not_draw_stays_refused(app_backend, quad_set, tmp_path):
    """Python derives tresca_stress; the app serves von Mises only, so it is refused by name."""
    with Results.from_ladruno(quad_set) as r:
        with pytest.raises(ValueError, match="tresca_stress.*von_mises_stress / von_mises_strain"):
            r.render(tmp_path / "x.png", view="contour", component="tresca_stress")
    assert app_backend.specs == []


def test_a_derived_scalar_of_a_split_tensor_is_refused(quad_set, monkeypatch):
    """Fail closed: tensor columns in two element fields give the app no field whose
    whole tensor it derives from; refused, never one field's part."""
    from apeGmsh.viewers.session import _results_stills as stills

    real = stills._app_gauss_contour

    def split(results, component):
        shot = real(results, component)
        return {**shot, "field": "other"} if component == "stress_xy" else shot

    monkeypatch.setattr(stills, "_app_gauss_contour", split)
    with Results.from_ladruno(quad_set) as r:
        with pytest.raises(ValueError, match=r"von_mises_stress.*R16c.*'stress_xy', 'stress/sigma12'"):
            stills._app_contour(r, r.stages[0].id, "von_mises_stress")


def test_a_nodal_derived_scalar_is_refused(quad_set, monkeypatch):
    """The app derives von Mises from an element field only: a stage that derives it
    at the nodes is refused, never sent as an element contour."""
    from apeGmsh.viewers.session import _results_stills as stills

    monkeypatch.setattr(stills, "_require_contour_data", lambda *a: "nodes")
    with Results.from_ladruno(quad_set) as r:
        with pytest.raises(ValueError, match="no apegmsh-view/1 contour"):
            stills._app_contour(r, r.stages[0].id, "von_mises_stress")


def test_a_derived_shot_of_a_source_the_app_cannot_open_is_refused(quad_set, tmp_path, monkeypatch):
    from apeGmsh.viewers.session._results_stills import app_shot_for_view

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with Results.from_ladruno(quad_set) as r:
        monkeypatch.setattr(type(r), "path", property(lambda self: tmp_path / "quad.h5"))
        with pytest.raises(ValueError, match="R22b"):
            app_shot_for_view(r, name="s", view="contour", component="von_mises_stress", step=-1, deform=None, camera="iso")


_TET10 = _FIXTURE.with_name("bezier_tet10.ladruno")
_TRI6 = _FIXTURE.with_name("bezier_tri6.ladruno")
#: fixture -> its one stress bucket (the fork's class key).
_BEZIER = {_TET10: "33001-BezierTet10[1000:1:0]", _TRI6: "33000-BezierTri6[1000:1:0]"}


@pytest.mark.parametrize("fixture", [_TET10, _TRI6], ids=["BezierTet10", "BezierTri6"])
@pytest.mark.parametrize("component, contour", [
    ("stress_xx", {"field": "stress", "component": "stress/sigma_xx"}),
    ("von_mises_stress", {"field": "stress", "component": "von_mises_stress"}),
])
def test_bezier_gauss_still_asks_node(tmp_path, fixture, component, contour):
    """#3085: BezierTet10's four and BezierTri6's three Gauss points are barycentric
    (PARAM_DOMAIN "bary"), which the app converts to natural coordinates since #2706, so
    a field of only these groups asks ``node``, as any extrapolable field does. From #2942
    (tet10) and #2950 (tri6) until #3085 the still kept them out of the vote by class name
    and asked ``cell_averaged``."""
    from apeGmsh.viewers.session._results_stills import _app_contour

    src = tmp_path / fixture.name
    shutil.copy(fixture, src)
    with h5py.File(src, "r") as f:
        grp = f[f"MODEL_STAGE[1]/MODEL/ELEMENTS/{_BEZIER[fixture][:-3]}]"]
        assert grp.attrs["PARAM_DOMAIN"].tobytes() == b"bary" and "GP_PARAM" in grp["QUADRATURE"]
    with Results.from_ladruno(src) as r:
        assert {len(g.ids) for g in r.fem.elements} == {1}
        assert _app_contour(r, r.stages[0].id, component) == {**contour, "source": "element", "gauss": "node"}


def _add_quad_bucket(path: Path) -> None:
    """The quad's four-point ``stress`` bucket and its MODEL/ELEMENTS group (a ``[-1,1]``
    GP_PARAM) beside the Bezier one, labelled as the Bezier classes label their columns
    (``sigma_xx``, ``sigma_yy``, ``sigma_xy``), so ``stress_xx`` is one app column held by
    a Bezier group and a FourNodeQuad group."""
    key = "31-FourNodeQuad[201:0:0]"
    with h5py.File(_FIXTURE, "r") as quad, h5py.File(path, "r+") as f:
        stress = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/stress"]
        quad.copy(quad["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/stress"][key], stress, name=key)
        stress[key]["COLUMN_MAP"].attrs["COMP_NAMES"] = np.bytes_("\n".join(["sigma_xx,sigma_yy,sigma_xy"] * 4))
        quad.copy(quad["MODEL_STAGE[1]/MODEL/ELEMENTS/31-FourNodeQuad[201:0]"], f["MODEL_STAGE[1]/MODEL/ELEMENTS"])


def _drop_gp_param(path: Path, bucket_key: str) -> None:
    """The group of ``bucket_key`` without its QUADRATURE/GP_PARAM, as the fork writes a class
    its recorder has no rule for (the app then offers no natural coordinates)."""
    with h5py.File(path, "r+") as f:
        del f[f"MODEL_STAGE[1]/MODEL/ELEMENTS/{bucket_key[:-3]}]/QUADRATURE/GP_PARAM"]


@pytest.mark.parametrize("fixture", [_TET10, _TRI6], ids=["BezierTet10", "BezierTri6"])
@pytest.mark.parametrize("gp_param, mixed, mode", [
    (True, False, "node"),
    (True, True, "node"),
    (False, False, "cell_averaged"),
    (False, True, "node"),
], ids=["coords", "coords+quad", "none", "none+quad"])
def test_the_gauss_mode_vote_follows_natural_coordinates(tmp_path, fixture, gp_param, mixed, mode):
    """#3085: a group votes for ``node`` when the app has natural coordinates for it, whatever
    its class. A Bezier group with its "bary" GP_PARAM asks ``node``, alone or beside a quad.
    Without GP_PARAM (the 2026-11-13 rulings on #2924 item 2, unchanged): alone it asks
    ``cell_averaged``; beside a four-point quad group it asks ``node``, and the app's per-group
    fallback draws the Bezier group's element mean. The scan reads only ``path``."""
    from types import SimpleNamespace

    from apeGmsh.viewers.session._results_stills import _app_gauss_contour

    src = tmp_path / "set.ladruno"
    shutil.copy(fixture, src)
    if mixed:
        _add_quad_bucket(src)
    if not gp_param:
        _drop_gp_param(src, _BEZIER[fixture])
    with h5py.File(src, "r") as f:
        assert sorted(f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/stress"]) == (
            ["31-FourNodeQuad[201:0:0]", _BEZIER[fixture]] if mixed else [_BEZIER[fixture]]
        )
    shot = _app_gauss_contour(SimpleNamespace(path=src), "stress_xx")
    assert shot == {"field": "stress", "component": "stress/sigma_xx", "source": "element", "gauss": mode}


_SIMPLEX = Path(__file__).resolve().parents[1] / "fixtures" / "simplex_gp"


@pytest.mark.parametrize("name, without", [
    # The app's own reading of these files: test/contour/extrapolation.test.ts "VF2 #2950 / #3071"
    # (tri: the SixNodeTri's naturalCoords null, the BezierTri6's and the quad's offered) and
    # test/results/ladrunoGauss.test.ts "Python and the viewer agree" (#2706: every "bary" group of both files).
    ("tri.ladruno", {"209-SixNodeTri[0:0:0]", "33016-LadrunoLST[0:0:0]"}),
    ("tet.ladruno", set()),
])
def test_natural_coords_are_the_apps_on_the_fork_fixtures(name, without):
    """#3085: the groups the stills find without natural coordinates are the ones the app's
    reader gives none, on the fork's simplex solves (``tests/fixtures/simplex_gp``): only the
    classes that write no GP_PARAM; every "bary" group (Tri31, BezierTri6, the three tets) has them."""
    from apeGmsh.viewers.session._results_stills import _app_has_natural_coords

    with h5py.File(_SIMPLEX / name, "r") as f:
        stage = f["MODEL_STAGE[1]"]
        keys = {k for t in stage["RESULTS/ON_ELEMENTS"] for k in stage["RESULTS/ON_ELEMENTS"][t]}
        assert len(keys) >= 4
        assert {k for k in keys if not _app_has_natural_coords(stage, k)} == without


@pytest.mark.parametrize("attr, value, expected", [
    (None, None, True),
    ("PARAM_DOMAIN", "[-1,1]", True),
    ("PARAM_DOMAIN", "natural", False),
    ("TOPOLOGY", "triangle", False),
    ("FAMILY", "serendipity", False),
    ("GP_PARAM", [[0.5, 0.5, 0.0]] * 3, False),
    ("GP_PARAM", [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [0.7, 0.4]], False),
    ("GP_PARAM", [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [-0.1, 2 / 3]], False),
    ("GP_PARAM", [[1, 0], [0, 1], [0, 0]], False),
    ("model", None, False),
], ids=["bary", "[-1,1]", "unknown-domain", "tims-triangle", "unknown-family", "wrong-width",
        "sum-above-1", "negative-weight", "integer-rows", "no-model-group"])
def test_natural_coords_refuse_what_the_app_refuses(tmp_path, attr, value, expected):
    """#3085: the reader's rule, restated (``reader/ladrunoElements.ts`` ``bucket`` and
    ``baryToGmsh``): a group has natural coordinates for a ``"[-1,1]"`` or a convertible
    ``"bary"`` GP_PARAM of floats, and none for an unknown domain, TOPOLOGY or FAMILY, the
    wrong width, a row outside the simplex, or no MODEL/ELEMENTS group. Each of those draws
    its element mean in a ``node`` field, as the app does."""
    from apeGmsh.viewers.session._results_stills import _app_has_natural_coords

    key = _BEZIER[_TRI6]
    src = tmp_path / "tri6.ladruno"
    shutil.copy(_TRI6, src)
    with h5py.File(src, "r+") as f:
        elements = f["MODEL_STAGE[1]/MODEL/ELEMENTS"]
        grp = elements[key[:-3] + "]"]
        if attr == "GP_PARAM":
            del grp["QUADRATURE/GP_PARAM"]
            grp["QUADRATURE"]["GP_PARAM"] = np.asarray(value)
        elif attr == "model":
            elements.move(grp.name, f"{elements.name}/33000-BezierTri6[1001:1]")
        elif attr is not None:
            grp.attrs[attr] = np.array([value.encode()])
    with h5py.File(src, "r") as f:
        assert _app_has_natural_coords(f["MODEL_STAGE[1]"], key) is expected


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


def _app_manifests(monkeypatch) -> list[dict]:
    """The Windows path through the source build: every manifest the app writes."""
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
    return manifests


def test_von_mises_still_draws_through_the_app_and_matches_python(quad_set, tmp_path, monkeypatch):
    """R16c (#2942): ``component="von_mises_stress"`` is drawn by the app, and its
    numbers are Python's: the corner values are the bilinear extrapolation of v1's
    von Mises at the four Gauss points (``gauss.get``), the range spans both steps."""
    manifests = _app_manifests(monkeypatch)
    with Results.from_ladruno(quad_set) as r:
        out = r.render(tmp_path / "von_mises.png", view="contour", component="von_mises_stress")
        stage = r.stage(r.stages[0].id)
        steps = []
        for k in (0, 1):
            slab = stage.elements.gauss.get(component="von_mises_stress", time=k)
            steps.append(_corner_values(np.asarray(slab.natural_coords), np.asarray(slab.values)[0]))
    assert out.is_file() and out.stat().st_size > 0
    (entry,) = manifests[0]["shots"]
    assert entry["status"] == "ok", entry.get("reason")
    contour = entry["contour"]
    assert (contour["source"], contour["field"], contour["component"]) == ("element", "stress", "von_mises_stress")
    assert contour["topology"] == "gauss_node"
    rec = contour["extrema"]["recorded"]
    assert (rec["min"]["value"], rec["max"]["value"]) == pytest.approx((steps[1].min(), steps[1].max()), rel=1e-9)
    both = np.concatenate(steps)
    assert (contour["range"]["min"], contour["range"]["max"]) == pytest.approx((both.min(), both.max()), rel=1e-9)


#: fixture -> the gmsh cell an apeGmsh mesh gives its one element: (code, name, dim, npe).
_GMSH_CELL = {
    "fournodequad": (3, "Quadrilateral 4", 2, 1, 4),
    "sixnodetri": (9, "Triangle 6", 2, 2, 6),
    "beziertri6": (9, "Triangle 6", 2, 2, 6),
    "beziertet10": (11, "Tetrahedron 10", 3, 2, 10),
}


def _pair(src: Path) -> Path:
    """``<stem>.h5`` beside ``src``: the file's nodes and its elements as the gmsh cells an
    apeGmsh mesh gives them (one group per cell type), stamped as the recorder's run would be."""
    from apeGmsh.mesh._element_types import make_type_info

    cells: dict[int, tuple[tuple, list, list]] = {}
    with Results.from_ladruno(src) as r:
        for group in r.fem.elements:
            cell = _GMSH_CELL[group.type_name]
            _, ids, conn = cells.setdefault(cell[0], (cell, [], []))
            ids += np.asarray(group.ids).tolist()
            conn += np.asarray(group.connectivity).tolist()
        fem = _make_fem(
            node_ids=np.asarray(r.fem.nodes.ids).tolist(),
            coords=[tuple(p) for p in np.asarray(r.fem.nodes.coords, dtype=float).tolist()],
            groups=[(make_type_info(code=c, gmsh_name=n, dim=d, order=o, npe=npe, count=len(ids)), ids, conn)
                    for (c, n, d, o, npe), ids, conn in cells.values()],
        )
    fem.to_h5(str(src.with_suffix(".h5")))
    stamp_provenance(src, src.with_suffix(".h5"))
    with h5py.File(src, "r+") as f:
        f["INFO/PROVENANCE"].attrs["element_index_space"] = "fem"
    return src


@pytest.fixture(params=[_TET10, _TRI6], ids=["BezierTet10", "BezierTri6"])
def bezier_set(request, tmp_path: Path) -> Path:
    """``<stem>.ladruno`` (one BezierTet10 with four, or one BezierTri6 with three,
    barycentric Gauss points, one step) beside ``<stem>.h5``: its nodes and the cell as
    the gmsh ``Tetrahedron 10`` or ``Triangle 6`` an apeGmsh mesh gives it."""
    src = tmp_path / request.param.name
    shutil.copy(request.param, src)
    return _pair(src)


def _linear_corners(natural: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The linear field through a simplex's Gauss values (as many points as corners), at its
    corners in gmsh order: the corner shape functions ``(1 - sum, u, v[, w])`` at the points,
    inverted. gauss_node on one element (no averaging)."""
    basis = np.column_stack([1.0 - natural.sum(axis=1), natural])
    return np.linalg.solve(basis, values)


def test_bezier_still_draws_node_through_the_app(bezier_set, tmp_path, monkeypatch):
    """#3085: the still asks ``node`` and the app extrapolates the Bezier group from its
    "bary" natural coordinates, naming no fallback. One element, so each corner holds the
    closed-form linear extrapolation of the Gauss values Python reads at the points Python
    places (BezierTri6: 15, 15, -15 from 10, 10, -5); from #2942/#2950 until #3085 every
    node held their mean."""
    manifests = _app_manifests(monkeypatch)
    with Results.from_ladruno(bezier_set) as r:
        out = r.render(tmp_path / f"{bezier_set.stem}.png", view="contour", component="stress_xx")
        stage = r.stage(r.stages[0].id)
        slab = stage.elements.gauss.get(component="stress_xx", time=0)
        corners = _linear_corners(np.asarray(slab.natural_coords, dtype=float), np.asarray(slab.values, dtype=float)[0])
        conn = np.asarray(list(r.fem.elements)[0].connectivity)[0][: len(corners)].tolist()
    assert out.is_file() and out.stat().st_size > 0
    (entry,) = manifests[0]["shots"]
    assert entry["status"] == "ok", entry.get("reason")
    contour = entry["contour"]
    assert (contour["field"], contour["component"]) == ("stress", "stress/sigma_xx")
    assert contour["topology"] == "gauss_node"
    assert contour.get("topology_fallback") is None
    assert corners.max() - corners.min() > 1.0, "the field varies over the element"
    rec = contour["extrema"]["recorded"]
    assert (rec["min"]["value"], rec["max"]["value"]) == pytest.approx((corners.min(), corners.max()), rel=1e-9)
    for end in ("min", "max"):
        assert rec[end]["value"] == pytest.approx(corners[conn.index(rec[end]["node"])], rel=1e-9)


@pytest.fixture
def mixed_tri_set(tmp_path: Path) -> Path:
    """``tri.ladruno`` (the fork's simplex solve: u_i = x_i^2 / 2, so eps_ii at a point is
    its x_i) holding the ``strains`` of a FourNodeQuad (``[-1,1]``), a SixNodeTri (no
    GP_PARAM) and a BezierTri6 (``"bary"``), beside its ``<stem>.h5``. The quad's and the
    SixNodeTri's ``eta`` labels become the BezierTri6's (``eps_xx``, ...), so ``strain_xx`` is
    one app column; the LadrunoLST (an element-level ``C1..C9`` bucket), the Tri31 (no
    strains) and the stresses are dropped."""
    src = tmp_path / "tri.ladruno"
    shutil.copy(_SIMPLEX / "tri.ladruno", src)
    with h5py.File(src, "r+") as f:
        stage = f["MODEL_STAGE[1]"]
        del stage["RESULTS/ON_ELEMENTS/stresses"]
        del stage["RESULTS/ON_ELEMENTS/strains/33016-LadrunoLST[0:0:0]"]
        del stage["MODEL/ELEMENTS/33016-LadrunoLST[0:0]"]
        del stage["MODEL/ELEMENTS/33-Tri31[100:0]"]
        for key in ("31-FourNodeQuad[201:0:0]", "209-SixNodeTri[0:0:0]"):
            cm = stage["RESULTS/ON_ELEMENTS/strains"][key]["COLUMN_MAP"]
            cm.attrs["COMP_NAMES"] = np.bytes_("\n".join(["eps_xx,eps_yy,gamma_xy"] * cm["GAUSS_ID"].shape[0]))
    return _pair(src)


def test_python_and_the_app_agree_on_the_groups_that_fall_back(mixed_tri_set, tmp_path, monkeypatch):
    """#3085, damage case "one group has no GP_PARAM": the still asks ``node``, and the app
    draws the SixNodeTri alone as its element mean and names it in ``topology_fallback``:
    exactly the group ``_app_has_natural_coords`` finds without coordinates in the same file.
    The BezierTri6 is extrapolated: its corner at the field's minimum reads its own x
    (closed form)."""
    from apeGmsh.viewers.session._results_stills import _app_has_natural_coords

    with h5py.File(mixed_tri_set, "r") as f:
        stage = f["MODEL_STAGE[1]"]
        strains = stage["RESULTS/ON_ELEMENTS/strains"]
        python = sorted(f"strains/{k}" for k in strains if not _app_has_natural_coords(stage, k))
        bezier = [int(n) for n in stage["MODEL/ELEMENTS/33000-BezierTri6[1000:1]/CONNECTIVITY"][0, 1:4]]
    assert python == ["strains/209-SixNodeTri[0:0:0]"]
    manifests = _app_manifests(monkeypatch)
    with Results.from_ladruno(mixed_tri_set) as r:
        out = r.render(tmp_path / "mixed.png", view="contour", component="strain_xx")
        x = dict(zip(np.asarray(r.fem.nodes.ids).tolist(), np.asarray(r.fem.nodes.coords, dtype=float)[:, 0].tolist()))
    assert out.is_file() and out.stat().st_size > 0
    (entry,) = manifests[0]["shots"]
    assert entry["status"] == "ok", entry.get("reason")
    contour = entry["contour"]
    assert (contour["field"], contour["component"], contour["topology"]) == ("strains", "strains/eps_xx", "gauss_node")
    fallback = contour["topology_fallback"]
    assert (fallback["asked"], fallback["drawn"], sorted(fallback["groups"])) == ("gauss_node", "cell_averaged", python)
    rec = contour["extrema"]["recorded"]
    assert rec["min"]["node"] in bezier
    assert rec["min"]["value"] == pytest.approx(x[rec["min"]["node"]], abs=1e-9)


def test_bezier_still_without_gp_param_draws_its_mean_through_the_app(bezier_set, tmp_path, monkeypatch):
    """The 2026-11-13 rulings on #2924 item 2, kept by #3085 for a group with no GP_PARAM:
    the still asks cell_averaged and the app draws the element mean averaged at the nodes.
    One element, so every node holds the mean of the Gauss values Python reads: min = max =
    that mean."""
    _drop_gp_param(bezier_set, _BEZIER[_TET10 if bezier_set.name == _TET10.name else _TRI6])
    manifests = _app_manifests(monkeypatch)
    with Results.from_ladruno(bezier_set) as r:
        out = r.render(tmp_path / f"{bezier_set.stem}.png", view="contour", component="stress_xx")
        stage = r.stage(r.stages[0].id)
        mean = float(np.mean(np.asarray(stage.elements.gauss.get(component="stress_xx", time=0).values)[0]))
    assert out.is_file() and out.stat().st_size > 0
    (entry,) = manifests[0]["shots"]
    assert entry["status"] == "ok", entry.get("reason")
    contour = entry["contour"]
    assert (contour["field"], contour["component"]) == ("stress", "stress/sigma_xx")
    assert contour["topology"] == "element_mean_averaged"
    rec = contour["extrema"]["recorded"]
    assert (rec["min"]["value"], rec["max"]["value"]) == pytest.approx((mean, mean), rel=1e-9)
