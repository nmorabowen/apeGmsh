"""ADR 0018 C3 — :class:`ModelData` end-to-end + parity + fail-loud.

Acceptance gate from
``opensees/architecture/modeldata-enrichment-scope.md`` §3 C3:

* Round-trip — declarative inject through the schema-owning emitter
  reaches the viewer's orientation join unchanged.
* Parity — for the same FEM, ``ModelData`` and ``apeSees(fem).h5()``
  produce the same ``{fem_eid: vecxz}`` via
  ``h5_reader.element_local_axes_vecxz()`` (the testable form of
  INV-16; whole-file byte-equivalence is not achievable because the
  vanilla deck omits materials / sections / integration the bridge
  full-path emits).
* No ``ModelData`` marker — INV-16 spirit.
* Fail-loud — INV-5/6/7/9/11/12 raise paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import ModelData, apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import make_two_node_beam


# ---------------------------------------------------------------------------
# End-to-end round-trip through the orientation join
# ---------------------------------------------------------------------------

def test_oriented_elements_round_trip(tmp_path: Path) -> None:
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6, model_name="m")
    md.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    out = tmp_path / "md.h5"
    md.write(str(out))

    with h5_reader.open(str(out)) as model:
        join = model.element_local_axes_vecxz()

    # The two-node-beam fixture's "Cols" PG contains element id 1.
    assert set(join.keys()) == {1}
    np.testing.assert_allclose(join[1], (1.0, 0.0, 0.0))


def test_two_pgs_distinct_vecxz(tmp_path: Path) -> None:
    """Multiple oriented_elements calls accumulate; each produces one
    transform group + element_meta rows keyed by fem_eid."""
    # Use a 4-element frame (two PGs in one fixture) — the FEMStub
    # `make_two_node_beam` carries only one PG, so we synthesize a
    # second call for the same PG with a different vecxz (the writer
    # binds rows by tag, so re-using the PG still produces two
    # transform groups but only the SECOND call's vecxz wins for the
    # shared element id).  To get genuine multi-eid parity we use the
    # 4-element fixture below.
    from tests.opensees.fixtures.fem_stub import _ElementGroupView, _ElementsStub, _NodesStub, FEMStub

    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        node_pgs={"Base": [1, 3]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Col_A": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
            "Col_B": _ElementGroupView(ids=(2,), connectivity=((3, 4),)),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)

    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    md.oriented_elements(
        pg="Col_A", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    md.oriented_elements(
        pg="Col_B", ele_type="forceBeamColumn", vecxz=(0.0, 1.0, 0.0),
    )

    out = tmp_path / "two.h5"
    md.write(str(out))

    with h5_reader.open(str(out)) as model:
        join = model.element_local_axes_vecxz()

    assert set(join.keys()) == {1, 2}
    np.testing.assert_allclose(join[1], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(join[2], (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# Parity with apeSees(fem).h5() — INV-16 (testable form)
# ---------------------------------------------------------------------------

def test_parity_with_apesees_h5(tmp_path: Path) -> None:
    """For the same FEM, the orientation join from ``ModelData.write``
    and from ``apeSees(fem).h5()`` must agree exactly.

    Whole-file byte-equivalence is not achievable (the vanilla deck
    omits materials / sections / integration the bridge full-path
    emits); the testable INV-16 form is the join's output — what the
    viewer / future P2 actually reads."""
    # --- bridge path ---
    fem_a = make_two_node_beam()
    ops = apeSees(cast("object", fem_a))
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    bridge_path = tmp_path / "bridge.h5"
    ops.h5(str(bridge_path))

    # --- declarative path ---
    fem_b = make_two_node_beam()
    md = ModelData(cast("object", fem_b), ndm=3, ndf=6)
    md.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    md_path = tmp_path / "modeldata.h5"
    md.write(str(md_path))

    # --- compare via the reader's orientation join ---
    with h5_reader.open(str(bridge_path)) as ma, h5_reader.open(str(md_path)) as mb:
        join_a = ma.element_local_axes_vecxz()
        join_b = mb.element_local_axes_vecxz()

    assert set(join_a.keys()) == set(join_b.keys()) == {1}
    np.testing.assert_allclose(join_a[1], join_b[1])
    np.testing.assert_allclose(join_a[1], (1.0, 0.0, 0.0))


def test_no_modeldata_marker_in_file(tmp_path: Path) -> None:
    """INV-16 spirit: nothing in the file identifies the author as
    ``ModelData``. A consumer must not be able to branch on
    provenance."""
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6, model_name="m")
    md.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    out = tmp_path / "marker.h5"
    md.write(str(out))

    with h5py.File(str(out), "r") as f:
        def _walk(name: str, obj) -> None:  # noqa: ANN001
            assert "modeldata" not in name.lower(), (
                f"ModelData provenance leaked in path: {name!r}"
            )
            for k, v in obj.attrs.items():
                key = str(k).lower()
                assert "modeldata" not in key, (
                    f"ModelData provenance leaked in attr name "
                    f"{name}@{k!r}"
                )
                if isinstance(v, (bytes, str)):
                    val = (v.decode("utf-8", "replace") if isinstance(v, bytes)
                           else v).lower()
                    assert "modeldata" not in val, (
                        f"ModelData provenance leaked in attr value "
                        f"{name}@{k!r} = {v!r}"
                    )

        f.visititems(_walk)
        for k, v in f.attrs.items():
            assert "modeldata" not in str(k).lower()


# ---------------------------------------------------------------------------
# Fail-loud at construction
# ---------------------------------------------------------------------------

def test_raises_on_non_fem_input() -> None:
    with pytest.raises(TypeError, match="must be a FEMData-like object"):
        ModelData(object(), ndm=3, ndf=6)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_ndm", [0, 1, 4, -1])
def test_raises_on_bad_ndm(bad_ndm: int) -> None:
    fem = make_two_node_beam()
    with pytest.raises(ValueError, match="ndm must be 2 or 3"):
        ModelData(cast("object", fem), ndm=bad_ndm, ndf=6)


@pytest.mark.parametrize("bad_ndf", [0, -1, -6])
def test_raises_on_bad_ndf(bad_ndf: int) -> None:
    fem = make_two_node_beam()
    with pytest.raises(ValueError, match="ndf must be > 0"):
        ModelData(cast("object", fem), ndm=3, ndf=bad_ndf)


# ---------------------------------------------------------------------------
# Fail-loud at inject (delegated to H5Emitter, surfaced through ModelData)
# ---------------------------------------------------------------------------

def test_raises_on_unknown_pg() -> None:
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    with pytest.raises(ValueError, match="not found in the bound FEMData"):
        md.oriented_elements(
            pg="DoesNotExist",
            ele_type="forceBeamColumn",
            vecxz=(1.0, 0.0, 0.0),
        )


def test_raises_on_unknown_ele_type() -> None:
    """INV-7: unknown / non-beam ele_type at inject time."""
    fem = make_two_node_beam()
    md = ModelData(cast("object", fem), ndm=3, ndf=6)
    with pytest.raises(ValueError, match="no transf slot"):
        md.oriented_elements(
            pg="Cols",
            ele_type="not_a_real_element",
            vecxz=(1.0, 0.0, 0.0),
        )
