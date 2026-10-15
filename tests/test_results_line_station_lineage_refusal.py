"""The elastic-beam line-stations lineage refusal fires (#1956, G3 review).

An elastic-beam line-stations record reads ``localForce`` end forces, so
``refuse_non_elastic_local_force`` (``results/spec/_emit.py``) checks every
element's type token in ``model=`` against the elastic-beam family. That
check is only as good as the pairing: a model built from another or a
re-meshed FEMData can hold the same FEM ids with other element types. So
the function first refuses a model whose ``lineage.fem_hash`` is not the
spec's ``fem_snapshot_id``. Before this file, removing that refusal passed
every test.

Each refusal case is paired with a control that differs only in the
lineage, so the raise can come from nothing but the lineage check: every
element is an ``elasticBeamColumn`` and the type check alone passes.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gmsh
import numpy as np
import pytest

from apeGmsh.results.spec._emit import refuse_non_elastic_local_force
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

_LINEAGE = "was not built from the spec's FEMData"


def _record(eids: list[int]) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name="girder",
        components=("axial_force",), dt=None, n_steps=None,
        element_ids=np.asarray(eids, dtype=np.int64),
        element_class_name="ElasticBeam2d",
    )


def _duck_model(fem_hash: str, eids: list[int]) -> Any:
    rows = [SimpleNamespace(fem_eid=e, type_token="elasticBeamColumn")
            for e in eids]
    return SimpleNamespace(lineage=SimpleNamespace(fem_hash=fem_hash),
                           elements=lambda: rows)


def test_same_lineage_passes() -> None:
    """Control: every element is an elastic beam and the lineage matches."""
    refuse_non_elastic_local_force(
        _record([3, 4]), _duck_model("fem-a", [3, 4]), fem_snapshot_id="fem-a",
    )


def test_other_lineage_is_refused() -> None:
    """The same elastic beams under another FEMData's lineage are refused,
    naming the record and both hashes."""
    with pytest.raises(ValueError, match=_LINEAGE) as exc:
        refuse_non_elastic_local_force(
            _record([3, 4]), _duck_model("fem-b", [3, 4]),
            fem_snapshot_id="fem-a",
        )
    msg = str(exc.value)
    assert "'girder'" in msg and "'fem-b'" in msg and "'fem-a'" in msg


# ---------------------------------------------------------------------------
# Through the transcoder, with a real model.h5
# ---------------------------------------------------------------------------


def _beam_fem() -> Any:
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="g17_lineage_beam", verbose=False) as g:
        geo = g.model.geometry
        c = geo.add_line(geo.add_point(0.0, 0.0, 0.0),
                         geo.add_point(4.0, 0.0, 0.0))
        gmsh.model.mesh.setTransfiniteCurve(c, 3)
        g.physical.add(1, [c], name="Beam")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    """A 2-D elastic beam's FEMData and the model.h5 the bridge writes."""
    from apeGmsh.opensees import apeSees

    fem = _beam_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(pg="Beam", transf=ops.geomTransf.Linear(),
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    path = tmp_path_factory.mktemp("lineage") / "model.h5"
    ops.h5(str(path))
    return fem, path


def _transcode(fem: Any, model_h5: Path, snapshot_id: str,
               tmp_path: Path) -> None:
    from apeGmsh.results.transcoders import RecorderTranscoder

    eids = [int(e) for e in fem.elements.select(pg="Beam").ids]
    n = len(eids)
    # One localForce row per step: time, then 6 end forces per element.
    np.savetxt(tmp_path / "girder_line_stations_local_force.out",
               np.column_stack([[1.0], np.ones((1, 6 * n))]))
    spec = ResolvedRecorderSpec(fem_snapshot_id=snapshot_id,
                                records=(_record(eids),))
    RecorderTranscoder(spec, tmp_path, tmp_path / "out.h5", fem,
                       stage_name="s", stage_kind="static",
                       model_h5_src=model_h5).run()


def test_transcoder_same_lineage_transcodes(
    built: tuple[Any, Path], tmp_path: Path,
) -> None:
    """Control: the spec resolved against the model's own FEMData."""
    fem, model_h5 = built
    _transcode(fem, model_h5, fem.snapshot_id, tmp_path)
    assert (tmp_path / "out.h5").exists()


def test_transcoder_refuses_a_model_from_another_femdata(
    built: tuple[Any, Path], tmp_path: Path,
) -> None:
    """The same files and model under a spec resolved against another
    FEMData: the transcode refuses before reading a value."""
    fem, model_h5 = built
    with pytest.raises(ValueError, match=_LINEAGE):
        _transcode(fem, model_h5, "another-femdata", tmp_path)
    assert not (tmp_path / "out.h5").exists()
