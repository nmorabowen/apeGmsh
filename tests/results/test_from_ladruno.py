"""``Results.from_ladruno`` factory + public-API tests (recorder-plan L2b).

Exercises the self-sufficient path: ``from_ladruno`` with **no** ``model_h5``
builds the broker from the ``.ladruno`` itself, so these run against the
committed fork fixtures with no sibling ``model.h5`` and no fork at test time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
TRUSS = FIXTURES / "truss2d.ladruno"
BEAM = FIXTURES / "beam3d.ladruno"


def test_self_sufficient_no_model_h5() -> None:
    r = Results.from_ladruno(TRUSS)
    assert r.fem is not None
    # A broker is bound even without a model_h5 (built from the file).
    assert r.model is not None
    assert r.model.fem is not None


def test_nodes_get_displacement_x() -> None:
    r = Results.from_ladruno(TRUSS)
    slab = r.nodes.get(component="displacement_x")
    assert slab.component == "displacement_x"
    assert slab.values.shape == (4, 3)
    assert slab.node_ids.tolist() == [1, 2, 3]
    # Node 1 fixed in x → zero; tip node 3 grows with the load ramp.
    i3 = slab.node_ids.tolist().index(3)
    assert np.all(np.diff(slab.values[:, i3]) > 0)


def test_minimal_broker_ndm_ndf() -> None:
    # The self-sufficient broker uses ndm from INFO/SPATIAL_DIM and
    # ndf=ndm (a .ladruno doesn't record ndf — see opensees_model docs).
    rt = Results.from_ladruno(TRUSS)
    assert rt.model.ndm == 2 and rt.model.ndf == 2
    rb = Results.from_ladruno(BEAM)
    assert rb.model.ndm == 3 and rb.model.ndf == 3


def test_rejects_non_ladruno(tmp_path: Path) -> None:
    import h5py

    bad = tmp_path / "x.ladruno"
    with h5py.File(bad, "w") as h:
        info = h.create_group("INFO")
        info.attrs["GENERATOR"] = "MPCO"
        info.attrs["FORMAT_VERSION"] = 1
    with pytest.raises(ValueError, match="expected 'Ladruno'"):
        Results.from_ladruno(bad)


def test_time_slice_last_step() -> None:
    r = Results.from_ladruno(TRUSS)
    slab = r.nodes.get(component="displacement_x", time=-1)
    assert slab.values.shape == (1, 3)
