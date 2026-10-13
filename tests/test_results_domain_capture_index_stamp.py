"""#1764: DomainCapture stamps ``element_index_space``, or refuses.

A capture queries OpenSees by ops tag and labels each element row with the
record's ``fem_eid``. Under the bridge's default ``element_tags=
"sequential"`` the two differ, and each path below used to fall back to
querying the ``fem_eid`` as if it were an ops tag: the row then held
another element's response, or the element was dropped, with no warning.
On origin/main b23c182c a 2-D portal (pairing fem->tag {5:3, 4:4, 6:5})
captured fem 5's row as Fy = 6257.8, which is tag 5's response (column fem
6); fem 5's own value is -6257.8.

Each silent path now refuses before the run file is written:

- no bridge, no model and no ``tag_map=``;
- the bridge could not archive its sidecar model.h5;
- reading the pairing raised (it was swallowed);
- the pairing does not cover every recorded id (``to_ops`` passes a
  partly-covered list through untranslated).

A capture that runs is stamped ``element_index_space = "fem"``: a
node-only one, or an element one whose every id the pairing covers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.results.capture._domain import DomainCapture
from apeGmsh.results.capture.spec import (
    DomainCaptureSpec,
    ResolvedDomainCaptureRecord,
)
from apeGmsh.results.readers._native import NativeReader
from apeGmsh.results.schema import _native

from tests.test_results_domain_capture import _FakeOps, _make_spec, _MockFem

_REFUSAL = "DomainCapture: the spec records"


def _node_record() -> ResolvedDomainCaptureRecord:
    return ResolvedDomainCaptureRecord(
        category="nodes", name="disp", components=("displacement_x",),
        dt=None, n_steps=None, node_ids=np.array([1], dtype=np.int64),
    )


def _element_record(ids: list[int]) -> ResolvedDomainCaptureRecord:
    return ResolvedDomainCaptureRecord(
        category="elements", name="f",
        components=("nodal_resisting_force_x",),
        dt=None, n_steps=None, element_ids=np.asarray(ids, dtype=np.int64),
        element_class_name="ElasticBeam3d",
    )


def _run(cap: DomainCapture) -> None:
    with cap as c:
        c.begin_stage("s", kind="static")
        c.step(t=1.0)
        c.end_stage()


def _verified(path: Path) -> bool:
    with NativeReader(path) as r:
        assert r.stages()
        return r.element_index_verified


class _SidecarFails:
    """A bridge whose ``h5()`` cannot archive the model (ADR 0055 degrade)."""

    def h5(self, path: str) -> None:
        raise NotImplementedError("stage-claimed phantom nodes")


class _SidecarUnreadable:
    """A bridge whose sidecar is written but cannot be read back."""

    def h5(self, path: str) -> None:
        Path(path).write_bytes(b"not an hdf5 file")


# ---------------------------------------------------------------------------
# Stamped
# ---------------------------------------------------------------------------


def test_node_only_capture_is_stamped(tmp_path: Path) -> None:
    fem = _MockFem([1])
    spec = _make_spec(_node_record(), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    _run(DomainCapture(spec, path, fem, ops=_FakeOps()))
    with h5py.File(path, "r") as f:
        assert f.attrs[_native.ATTR_ELEMENT_INDEX_SPACE] == "fem"
    assert _verified(path)


def test_demo_results_are_stamped(tmp_path: Path) -> None:
    from apeGmsh.results.demo import make_demo_results

    results = make_demo_results(path=tmp_path)
    with NativeReader(results._path) as r:
        assert r.element_index_space == "fem"
        assert r.element_index_verified


# ---------------------------------------------------------------------------
# Refused: each formerly silent path
# ---------------------------------------------------------------------------


def test_no_bridge_no_model_refuses(tmp_path: Path) -> None:
    fem = _MockFem([1])
    spec = _make_spec(_element_record([5]), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    with pytest.raises(ValueError, match="no bridge, no model"):
        _run(DomainCapture(spec, path, fem, ops=_FakeOps()))
    assert not path.exists()


def test_sidecar_failure_refuses(tmp_path: Path) -> None:
    fem = _MockFem([1])
    spec = _make_spec(_element_record([5]), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    with pytest.warns(UserWarning, match="could not archive"):
        with pytest.raises(ValueError, match="could not archive a model.h5"):
            _run(DomainCapture(spec, path, fem, ops=_FakeOps(),
                               bridge=_SidecarFails()))
    assert not path.exists()


def test_unreadable_pairing_refuses_instead_of_being_swallowed(
    tmp_path: Path,
) -> None:
    fem = _MockFem([1])
    spec = _make_spec(_element_record([5]), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    with pytest.raises(ValueError, match="reading the sidecar's pairing failed"):
        _run(DomainCapture(spec, path, fem, ops=_FakeOps(),
                           bridge=_SidecarUnreadable()))
    assert not path.exists()
    assert not path.with_suffix(".model.h5").exists()


def test_from_h5_unreadable_pairing_refuses_element_capture(
    tmp_path: Path,
) -> None:
    # A model.h5 whose /meta reads but whose model does not load: the
    # pairing read used to be swallowed.
    from tests.fixtures.schema import OPENSEES_CURRENT

    model_path = tmp_path / "model.h5"
    with h5py.File(model_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = OPENSEES_CURRENT
        meta.attrs["opensees_schema_version"] = OPENSEES_CURRENT
        meta.attrs["ndm"] = 3
        meta.attrs["ndf"] = 6
        meta.attrs["snapshot_id"] = "stub-snapshot"
        meta.attrs["model_name"] = "stub"
    fem = _MockFem([1])
    spec = DomainCaptureSpec()
    spec.elements(ids=[5], components=["nodal_resisting_force_x"],
                  element_class_name="ElasticBeam3d")
    with pytest.raises(ValueError, match="reading the pairing from"):
        _run(DomainCapture.from_h5(
            model_path, spec=spec, fem=fem, output=tmp_path / "run.h5",
            ops=_FakeOps(),
        ))


def test_incomplete_pairing_refuses(tmp_path: Path) -> None:
    # {4: 4} covers 4 but not 7: ElementTagTranslator.to_ops would pass
    # [4, 7] through untranslated.
    fem = _MockFem([1])
    spec = _make_spec(_element_record([4, 7]), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    with pytest.raises(ValueError, match=r"fem_eids \[7\]"):
        _run(DomainCapture(spec, path, fem, ops=_FakeOps(),
                           tag_map={4: 4}))
    assert not path.exists()


# ---------------------------------------------------------------------------
# The bridge path: sequential tags, a real sidecar pairing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def portal() -> dict[str, Any]:
    H, L = 3.0, 5.0
    with apeGmsh(model_name="b5f_portal", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0, 0, 0)
        p1 = geo.add_point(0, H, 0)
        p2 = geo.add_point(L, H, 0)
        p3 = geo.add_point(L, 0, 0)
        c1 = geo.add_line(p0, p1)
        bm = geo.add_line(p1, p2)
        c2 = geo.add_line(p3, p2)
        br = geo.add_line(p0, p2)
        for c in (c1, bm, c2, br):
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        g.physical.add(1, [c1, c2], name="Cols")
        g.physical.add(1, [bm], name="Beam")
        g.physical.add(1, [br], name="Brace")
        g.mesh.generation.generate(dim=1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem, default_orientation=None)   # sequential tags
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    # Beam first, and the Brace is never built.
    ops.element.elasticBeamColumn(pg="Beam", transf=tr, A=0.02, E=2e11, Iz=3e-4)
    ops.element.elasticBeamColumn(pg="Cols", transf=tr, A=0.02, E=2e11, Iz=5e-4)
    return {
        "ops": ops,
        "beam": [int(e) for e in fem.elements.select(pg="Beam").ids],
        "brace": [int(e) for e in fem.elements.select(pg="Brace").ids],
        "node": int(fem.nodes.ids[0]),
    }


def test_bridge_capture_of_an_unbuilt_element_refuses(
    tmp_path: Path, portal: dict,
) -> None:
    ops = portal["ops"]
    spec = DomainCaptureSpec(opensees=ops)
    spec.elements(ids=portal["beam"] + portal["brace"],
                  components=["nodal_resisting_force_x"],
                  element_class_name="ElasticBeam2d")
    path = tmp_path / "run.h5"
    with pytest.raises(ValueError, match=re.escape(f"fem_eids {portal['brace']}")):
        _run(ops.domain_capture(spec, path=str(path), ops=_FakeOps()))
    assert not path.exists()
    assert not path.with_suffix(".model.h5").exists()


def test_bridge_node_capture_is_stamped(tmp_path: Path, portal: dict) -> None:
    ops = portal["ops"]
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(ids=[portal["node"]], components=["displacement_x"])
    path = tmp_path / "run.h5"
    _run(ops.domain_capture(spec, path=str(path), ops=_FakeOps()))
    assert _verified(path)


@pytest.mark.parametrize("eid, bad, match", [
    (5, {5.7: 3}, "is not an int"),
    (5, {"5": 3}, "is not an int"),
    (1, {True: 3}, "is not an int"),       # int(True) == 1 would cover eid 1
    (5, {5: 3.0}, "is not an int"),
    (5, {5.0: 3}, "is not an int"),
    (5, [(5, 3)], "must be a"),
], ids=["float_key", "str_key", "bool_key", "float_value",
        "integral_float_key", "list"])
def test_malformed_tag_map_refuses(
    tmp_path: Path, eid: int, bad: Any, match: str,
) -> None:
    # Coercing these would truncate 5.7 -> 5 (or similar), cover the
    # record, and stamp the file as if the pairing were the model's.
    fem = _MockFem([1])
    spec = _make_spec(_element_record([eid]), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    with pytest.raises((TypeError, ValueError), match=match):
        _run(DomainCapture(spec, path, fem, ops=_FakeOps(), tag_map=bad))
    assert not path.exists()


def test_numpy_int_tag_map_is_accepted(tmp_path: Path) -> None:
    fem = _MockFem([1])
    spec = _make_spec(_node_record(), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    _run(DomainCapture(spec, path, fem, ops=_FakeOps(),
                       tag_map={np.int64(5): np.int32(3)}))
    assert _verified(path)


def test_empty_mapping_refuses(tmp_path: Path) -> None:
    # An empty map is a pairing that covers nothing, not "no pairing".
    fem = _MockFem([1])
    spec = _make_spec(_element_record([4]), snapshot_id=fem.snapshot_id)
    path = tmp_path / "run.h5"
    with pytest.raises(ValueError, match=r"fem_eids \[4\]"):
        _run(DomainCapture(spec, path, fem, ops=_FakeOps(), tag_map={}))
    assert not path.exists()


@pytest.mark.parametrize("as_translator", [False, True],
                         ids=["mapping", "translator"])
@pytest.mark.parametrize("pairing, refused", [
    ({4: 4, 7: 9}, False),   # covers both ids (7 -> tag 9)
    ({4: 4, 9: 7}, True),    # 7 is an ops tag here, not a fem_eid
    ({4: 4}, True),          # 7 unknown
])
def test_tag_map_must_cover_every_id(
    tmp_path: Path, pairing: dict, refused: bool, as_translator: bool,
) -> None:
    # tag_map= takes a plain {fem_eid: ops_tag} mapping, or the
    # ElementTagTranslator it took before.
    from apeGmsh.results.readers._tag_translation import ElementTagTranslator

    tag_map: Any = pairing
    if as_translator:
        tag_map = ElementTagTranslator(
            pairing, {t: f for f, t in pairing.items()},
        )
    fem = _MockFem([1])
    spec = _make_spec(_element_record([4, 7]), snapshot_id=fem.snapshot_id)
    cap = DomainCapture(spec, tmp_path / "run.h5", fem, ops=_FakeOps(),
                        tag_map=tag_map)
    if refused:
        with pytest.raises(ValueError, match=r"fem_eids \[7\]"):
            cap.__enter__()
    else:
        cap.__enter__()
        cap.close()
        with h5py.File(tmp_path / "run.h5", "r") as f:
            assert f.attrs[_native.ATTR_ELEMENT_INDEX_SPACE] == "fem"
