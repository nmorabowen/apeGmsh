"""ADR 0094 S2 — fem.assess() / results.assess() acceptance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.assess import CATALOG_SEVERITY, AssessmentReport, Finding
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees._internal.lineage import WARNING_PREFIX, Lineage
from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


# ---------------------------------------------------------------------------
# FEMData builders (in-memory; no live gmsh)
# ---------------------------------------------------------------------------


def _empty_sets() -> tuple[PhysicalGroupSet, LabelSet]:
    return PhysicalGroupSet({}), LabelSet({})


def _make_fem(
    *,
    node_ids: list[int],
    coords: list[tuple[float, float, float]],
    groups: list[tuple[object, list[int], list[list[int]]]] | None = None,
    pgs: dict[tuple[int, int], dict] | None = None,
) -> FEMData:
    ids = np.asarray(node_ids, dtype=np.int64)
    xyz = np.asarray(coords, dtype=np.float64)
    pgset = PhysicalGroupSet(pgs or {})
    labels = LabelSet({})
    typed: dict[int, ElementGroup] = {}
    types = []
    n_elems = 0
    for info, eids, conn in groups or []:
        typed[info.code] = ElementGroup(
            element_type=info,
            ids=np.asarray(eids, dtype=np.int64),
            connectivity=np.asarray(conn, dtype=np.int64),
        )
        types.append(info)
        n_elems += len(eids)
    nodes = NodeComposite(
        node_ids=ids, node_coords=xyz, physical=pgset, labels=labels,
    )
    elements = ElementComposite(
        groups=typed, physical=pgset, labels=labels,
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=MeshInfo(
            n_nodes=int(ids.size),
            n_elems=n_elems,
            bandwidth=0,
            types=types,
        ),
    )


def _tet4_info(*, count: int = 1):
    return make_type_info(
        code=4, gmsh_name="Tetrahedron 4", dim=3, order=1, npe=4, count=count,
    )


def _tet10_info(*, count: int = 1):
    return make_type_info(
        code=11, gmsh_name="Tetrahedron 10", dim=3, order=2, npe=10, count=count,
    )


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def _assert_catalog_severities(report: AssessmentReport) -> None:
    for finding in report.findings:
        assert finding.code in CATALOG_SEVERITY, finding
        assert finding.severity == CATALOG_SEVERITY[finding.code], finding
        if finding.code == "RES.U_VS_DIAG":
            assert finding.severity == "info"
            assert finding.severity != "error"


def _codes(report: AssessmentReport) -> list[str]:
    return [f.code for f in report.findings]


# ---------------------------------------------------------------------------
# FEMData.assess
# ---------------------------------------------------------------------------


def test_empty_model_is_model_empty() -> None:
    fem = _make_fem(node_ids=[], coords=[])
    report = fem.assess()
    assert isinstance(report, AssessmentReport)
    assert "MODEL.EMPTY" in _codes(report)
    _assert_catalog_severities(report)
    err = next(f for f in report.findings if f.code == "MODEL.EMPTY")
    assert err.severity == "error"


def test_planted_unbridged_pair_is_coincident_unbridged() -> None:
    fem = _make_fem(
        node_ids=[1, 2, 3],
        coords=[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
    )
    report = fem.assess()
    assert "MESH.COINCIDENT_UNBRIDGED" in _codes(report)
    finding = next(
        f for f in report.findings if f.code == "MESH.COINCIDENT_UNBRIDGED"
    )
    assert finding.severity == "warning"
    assert finding.detail is not None
    assert (2, 3) in finding.detail["pairs"]
    # Bridged-only pair must not appear — emptiness is the only branch.
    assert all(
        isinstance(p, tuple) and len(p) == 2
        for p in finding.detail["pairs"]
    )
    _assert_catalog_severities(report)


def test_bridged_coincident_pair_is_not_a_finding() -> None:
    info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    fem = _make_fem(
        node_ids=[1, 2],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        groups=[(info, [10], [[1, 2]])],
    )
    report = fem.assess()
    assert "MESH.COINCIDENT_UNBRIDGED" not in _codes(report)


def test_inverted_tet4_is_mesh_inverted() -> None:
    # Right-handed tet is fine; swapping the last two corners inverts it.
    info = _tet4_info()
    good = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        ],
        groups=[(info, [1], [[1, 2, 3, 4]])],
    )
    assert "MESH.INVERTED" not in _codes(good.assess())

    bad = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        ],
        groups=[(info, [7], [[1, 2, 4, 3]])],
    )
    report = bad.assess()
    assert "MESH.INVERTED" in _codes(report)
    finding = next(f for f in report.findings if f.code == "MESH.INVERTED")
    assert finding.severity == "error"
    assert 7 in finding.detail["ids"]
    _assert_catalog_severities(report)


def _hex8_info(*, count: int = 1):
    return make_type_info(
        code=5, gmsh_name="Hexahedron 8", dim=3, order=1, npe=8, count=count,
    )


def _tri3_info(*, count: int = 1):
    return make_type_info(
        code=2, gmsh_name="Triangle 3", dim=2, order=1, npe=3, count=count,
    )


_VALID_SKEWED_HEX8 = [
    ( 0.258, -0.357, -0.363),
    ( 1.087, -0.350, -0.320),
    ( 0.588,  1.381, -0.319),
    ( 0.260,  1.254,  0.224),
    (-0.369,  0.155,  0.888),
    ( 0.855, -0.028,  0.712),
    ( 1.416,  1.030,  0.805),
    (-0.392,  0.747,  0.764),
]


def test_unit_cube_hex8_is_not_inverted() -> None:
    info = _hex8_info()
    fem = _make_fem(
        node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        ],
        groups=[(info, [1], [[1, 2, 3, 4, 5, 6, 7, 8]])],
    )
    assert "MESH.INVERTED" not in _codes(fem.assess())


def test_valid_skewed_hex8_is_not_inverted() -> None:
    info = _hex8_info()
    fem = _make_fem(
        node_ids=list(range(1, 9)),
        coords=_VALID_SKEWED_HEX8,
        groups=[(info, [1], [[1, 2, 3, 4, 5, 6, 7, 8]])],
    )
    assert "MESH.INVERTED" not in _codes(fem.assess())


def test_inverted_skewed_hex8_is_mesh_inverted() -> None:
    # Swap corners 0/2 (1-based nodes 1/3). A 6/7 swap stays positive
    # under the corrected tet split; 0/2 does not.
    info = _hex8_info()
    conn = [3, 2, 1, 4, 5, 6, 7, 8]
    fem = _make_fem(
        node_ids=list(range(1, 9)),
        coords=_VALID_SKEWED_HEX8,
        groups=[(info, [3], [conn])],
    )
    report = fem.assess()
    assert "MESH.INVERTED" in _codes(report)
    finding = next(f for f in report.findings if f.code == "MESH.INVERTED")
    assert 3 in finding.detail["ids"]
    _assert_catalog_severities(report)


def test_planar_inclined_tri3_is_judged() -> None:
    info = _tri3_info()
    # Right-handed on the plane z = x (all nodes coplanar).
    fem = _make_fem(
        node_ids=[1, 2, 3],
        coords=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)],
        groups=[(info, [1], [[1, 2, 3]])],
    )
    report = fem.assess()
    assert "MESH.INVERTED" not in _codes(report)
    assert "non-planar" not in report.text


def test_tri3_in_nonplanar_mesh_is_skipped_not_judged() -> None:
    info = _tri3_info()
    fem = _make_fem(
        node_ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 2.0),
        ],
        groups=[(info, [1], [[1, 2, 3]])],
    )
    report = fem.assess()
    assert "MESH.INVERTED" not in _codes(report)
    assert "2D cells in non-planar 3D space" in report.text


def test_quadratic_cells_are_skipped_not_judged() -> None:
    info = _tet10_info()
    fem = _make_fem(
        node_ids=list(range(1, 11)),
        coords=[(float(i), 0.0, 0.0) for i in range(10)],
        groups=[(info, [1], [list(range(1, 11))])],
    )
    report = fem.assess()
    assert "MESH.INVERTED" not in _codes(report)
    assert "MESH.INVERTED" in report.text
    assert "Skipped" in report.text


def test_empty_named_pg() -> None:
    pgs = {
        (2, 1): {
            "name": "EmptyFace",
            "node_ids": np.array([], dtype=np.int64),
            "node_coords": np.zeros((0, 3), dtype=np.float64),
            "element_ids": np.array([], dtype=np.int64),
        },
    }
    fem = _make_fem(
        node_ids=[1, 2],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        pgs=pgs,
    )
    report = fem.assess()
    assert "MESH.EMPTY_PG" in _codes(report)
    finding = next(f for f in report.findings if f.code == "MESH.EMPTY_PG")
    assert finding.severity == "warning"
    assert "EmptyFace" in finding.detail["names"]


def test_figures_true_raises_on_fem() -> None:
    fem = _make_fem(node_ids=[1], coords=[(0.0, 0.0, 0.0)])
    with pytest.raises(NotImplementedError, match="S3"):
        fem.assess(figures=True)


# ---------------------------------------------------------------------------
# Results.assess
# ---------------------------------------------------------------------------


def _write_nodes(
    tmp_path: Path,
    *,
    node_ids: np.ndarray,
    components: dict[str, np.ndarray],
    kind: str = "static",
    name: str = "g",
    **stage_kw: object,
) -> Path:
    path = tmp_path / "results.h5"
    n_steps = next(iter(components.values())).shape[0]
    time = np.arange(float(n_steps))
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name=name, kind=kind, time=time, **stage_kw)
        w.write_nodes(sid, "partition_0", node_ids=node_ids, components=components)
        w.end_stage()
    return path


def test_injected_nan_is_res_nan(tmp_path: Path) -> None:
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    ux = np.array([[1.0, np.nan, 3.0]])
    path = _write_nodes(tmp_path, node_ids=node_ids, components={"displacement_x": ux})
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    assert "RES.NAN" in _codes(report)
    finding = next(f for f in report.findings if f.code == "RES.NAN")
    assert finding.severity == "error"
    # Recorded id 2 carries the NaN. Do not invent fill-NaNs for missing ids.
    assert 2 in finding.detail["ids"]
    _assert_catalog_severities(report)


def test_injected_inf_is_res_inf(tmp_path: Path) -> None:
    node_ids = np.array([1, 2], dtype=np.int64)
    ux = np.array([[1.0, np.inf]])
    path = _write_nodes(tmp_path, node_ids=node_ids, components={"displacement_x": ux})
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    assert "RES.INF" in _codes(report)
    assert next(f for f in report.findings if f.code == "RES.INF").severity == "error"


def test_dirty_lineage_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node_ids = np.array([1, 2], dtype=np.int64)
    path = _write_nodes(
        tmp_path,
        node_ids=node_ids,
        components={"displacement_x": np.zeros((1, 2))},
    )
    dirty = Lineage(warnings=(f"{WARNING_PREFIX} planted drift",))
    monkeypatch.setattr(Results, "lineage", property(lambda self: dirty))
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    assert "RES.LINEAGE" in _codes(report)
    finding = next(f for f in report.findings if f.code == "RES.LINEAGE")
    assert finding.severity == "warning"
    assert report.lineage is dirty
    _assert_catalog_severities(report)


def test_unbound_fem_and_no_stage(tmp_path: Path) -> None:
    path = tmp_path / "empty_stages.h5"
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    codes = _codes(report)
    assert "RES.UNBOUND_FEM" in codes
    assert "RES.NO_STAGE" in codes
    assert "MESH.*" in report.text
    assert "no bound FEMData" in report.text
    assert "`RES.NAN` — no stages" in report.text
    _assert_catalog_severities(report)


def test_energy_err_skip_is_not_pass(tmp_path: Path) -> None:
    node_ids = np.array([1, 2], dtype=np.int64)
    path = _write_nodes(
        tmp_path,
        node_ids=node_ids,
        components={"displacement_x": np.zeros((1, 2))},
    )
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    assert "RES.ENERGY_ERR" not in _codes(report)
    assert "RES.ENERGY_ERR" in report.text
    assert "Skipped" in report.text


def test_nonfinite_energy_err_is_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    path = _write_nodes(
        tmp_path,
        node_ids=np.array([1], dtype=np.int64),
        components={"displacement_x": np.zeros((1, 1))},
    )

    def _nan_energy(self, *, region=None, stage=None):
        return pd.DataFrame({"ERR": [np.nan]}, index=[0.0])

    monkeypatch.setattr(Results, "energy", _nan_energy)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.ENERGY_ERR")
    assert "non-finite" in finding.message
    _assert_catalog_severities(report)


def test_energy_valueerror_on_first_stage_tries_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    path = tmp_path / "two_stages.h5"
    node_ids = np.array([1], dtype=np.int64)
    ux = np.zeros((1, 1))
    time = np.array([0.0])
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        s0 = w.begin_stage(name="a", kind="static", time=time, stage_id="a")
        w.write_nodes(s0, "partition_0", node_ids=node_ids, components={"displacement_x": ux})
        w.end_stage()
        s1 = w.begin_stage(name="b", kind="static", time=time, stage_id="b")
        w.write_nodes(s1, "partition_0", node_ids=node_ids, components={"displacement_x": ux})
        w.end_stage()

    def _energy(self, *, region=None, stage=None):
        if stage == "a":
            raise ValueError("channel absent")
        return pd.DataFrame({"ERR": [0.05]}, index=[0.0])

    monkeypatch.setattr(Results, "energy", _energy)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        report = r.assess()
    finding = next(f for f in report.findings if f.code == "RES.ENERGY_ERR")
    assert finding.detail["stage"] == "b"
    assert "0.05" in finding.message


def test_u_vs_diag_is_always_info(tmp_path: Path) -> None:
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    # diag of [(0,0,0), (0,0,1)] is 1.0; ‖u‖_max = 0.25
    uz = np.array([[0.0, 0.25]])
    path = _write_nodes(
        tmp_path, node_ids=node_ids, components={"displacement_z": uz},
    )
    with Results.from_native(
        path, model=_open_model_from_h5(path), fem=fem,
    ) as r:
        report = r.assess()
    u = [f for f in report.findings if f.code == "RES.U_VS_DIAG"]
    assert u, report.text
    assert u[0].severity == "info"
    assert u[0].severity != "error"
    _assert_catalog_severities(report)


def test_u_vs_diag_skips_mode_stages(tmp_path: Path) -> None:
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    path = _write_nodes(
        tmp_path,
        node_ids=node_ids,
        components={"displacement_z": np.array([[0.0, 1.0]])},
        kind="mode",
        name="mode1",
        eigenvalue=1.0,
        frequency_hz=1.0,
        period_s=1.0,
        mode_index=1,
    )
    with Results.from_native(
        path, model=_open_model_from_h5(path), fem=fem,
    ) as r:
        report = r.assess()
    assert "RES.U_VS_DIAG" not in _codes(report)
    assert "RES.U_VS_DIAG" in report.text
    assert "mode" in report.text


def test_results_assess_sees_planted_coincident_pair(tmp_path: Path) -> None:
    fem = _make_fem(
        node_ids=[1, 2, 3],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    )
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    path = _write_nodes(
        tmp_path,
        node_ids=node_ids,
        components={"displacement_x": np.zeros((1, 3))},
    )
    with Results.from_native(
        path, model=_open_model_from_h5(path), fem=fem,
    ) as r:
        report = r.assess()
    assert "MESH.COINCIDENT_UNBRIDGED" in _codes(report)


def test_figures_true_raises_on_results(tmp_path: Path) -> None:
    path = _write_nodes(
        tmp_path,
        node_ids=np.array([1], dtype=np.int64),
        components={"displacement_x": np.zeros((1, 1))},
    )
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        with pytest.raises(NotImplementedError, match="S3"):
            r.assess(figures=True)


def test_report_is_frozen() -> None:
    fem = _make_fem(node_ids=[], coords=[])
    report = fem.assess()
    with pytest.raises(Exception):
        report.findings = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        report.text = ""  # type: ignore[misc]


def test_model_diagonal_reexport() -> None:
    from apeGmsh.results._geometry import model_diagonal as promoted
    from apeGmsh.results.plot._arrows import model_diagonal as reexport
    assert promoted is reexport
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    assert reexport(coords) == pytest.approx(1.0)


def test_assess_not_a_composite_and_not_reexported() -> None:
    from apeGmsh import apeGmsh as Session
    from apeGmsh import __all__ as root_all
    assert "assess" not in root_all
    names = [row[0] for row in Session._COMPOSITES]
    assert "assess" not in names


def test_finding_shape() -> None:
    f = Finding(code="MODEL.EMPTY", severity="error", message="x")
    assert f.detail is None
