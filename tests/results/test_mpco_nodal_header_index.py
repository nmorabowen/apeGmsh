"""MPCO reads nodal-force buckets at every header index (#2115 item 2).

MPCO opens a second response header (``[<rule>:<cust>:1]``) when elements
of one class write a different META. The nodal-force reader used to leave
every ``header_idx > 0`` bucket unread, named in a
``NodalForceBucketSkippedWarning``. It now reads such a bucket once its own
META spells the catalog layout (``NUM_COLUMNS``, the single META block, and
the ``META/COMPONENTS`` names at each node in catalog order), and warns by
name when it does not.

The oracle is a file with two headers for one class: each header holds its
own elements and its own end forces, and every element must read back the
values of the header that holds it.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import NODAL_FORCE_CATALOG
from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_nodal_io as _mn
from apeGmsh.results.readers._mpco_nodal_io import (
    NodalForceBucketSkippedWarning,
)
from tests.conftest import _stub_model_h5_path
from tests.test_results_mpco_nodal_mock import _create_skeleton

_T = 2
#: Header 0 holds these elements, header 1 the others.
_EIDS = {0: np.array([7, 11], dtype=np.int64),
         1: np.array([21, 33], dtype=np.int64)}

_G3 = ("Px", "Py", "Pz", "Mx", "My", "Mz")
_L3 = ("N", "Vy", "Vz", "T", "My", "Mz")


def _names(bases: tuple[str, ...], *, sep: str = "_", n_nodes: int = 2) -> str:
    return "0." + ",".join(
        f"{b}{sep}{n}" for n in range(1, n_nodes + 1) for b in bases)


#: (id, class, MPCO frame, catalog token, bracket stem, COMPONENTS body),
#: the column names each class's ``setResponse`` writes in the fork.
_CASES = [
    ("eb3d-global", "ElasticBeam3d", "globalForce", "global_force",
     "5-ElasticBeam3d[1:0:", _names(_G3)),
    ("eb3d-local", "ElasticBeam3d", "localForce", "local_force",
     "5-ElasticBeam3d[1:0:", _names(_L3)),
    ("fbc3d-global", "ForceBeamColumn3d", "globalForce", "global_force",
     "74-ForceBeamColumn3d[1000:1:", _names(_G3)),
    ("fbc3d-local", "ForceBeamColumn3d", "localForce", "local_force",
     "74-ForceBeamColumn3d[1000:1:", _names(_L3)),
    ("eb2d-global", "ElasticBeam2d", "globalForce", "global_force",
     "3-ElasticBeam2d[1:0:", _names(("Px", "Py", "Mz"))),
    ("eb2d-local", "ElasticBeam2d", "localForce", "local_force",
     "3-ElasticBeam2d[1:0:", _names(("N", "V", "M"))),
    ("dbc2d-local", "DispBeamColumn2d", "localForce", "local_force",
     "62-DispBeamColumn2d[1000:1:", _names(("N", "V", "M"), sep="")),
]


def _flat(header: int, width: int) -> np.ndarray:
    """``(T, E, width)``, every slot distinct and each header its own:
    ``header * 0.25 + t * 1000 + eid * 10 + col``."""
    eids = _EIDS[header]
    flat = np.empty((_T, eids.size, width))
    for t in range(_T):
        for i, e in enumerate(eids):
            flat[t, i] = header * 0.25 + t * 1000 + e * 10 + np.arange(width)
    return flat


def _add_bucket(
    stage: "h5py.Group", frame: str, key: str, eids: np.ndarray,
    flat: np.ndarray, components: str | None, *,
    num_columns: int | None = None, gauss_id: int = -1,
) -> None:
    """A single-block nodal-force bucket as the fork's MPCO recorder writes
    it; ``components=None`` leaves out ``META/COMPONENTS``."""
    width = flat.shape[2]
    ncol = width if num_columns is None else num_columns
    grp = stage["RESULTS/ON_ELEMENTS"].require_group(frame).create_group(key)
    grp.attrs["NUM_COLUMNS"] = np.array([ncol], dtype=np.int32)
    meta = grp.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.array([[1]], dtype=np.int32))
    meta.create_dataset("GAUSS_IDS",
                        data=np.array([[gauss_id]], dtype=np.int32))
    meta.create_dataset("NUM_COMPONENTS",
                        data=np.array([[ncol]], dtype=np.int32))
    if components is not None:
        meta.create_dataset("COMPONENTS",
                            data=np.array([components.encode("ascii")]))
    grp.create_dataset("ID", data=eids.reshape(-1, 1).astype(np.int32))
    data = grp.create_group("DATA")
    for t in range(flat.shape[0]):
        ds = data.create_dataset(f"STEP_{t}", data=flat[t])
        ds.attrs["STEP"] = t
        ds.attrs["TIME"] = float(t + 1)


def _two_header_file(
    tmp_path: Path, frame: str, stem: str, comps0: str,
    comps1: str | None, *, width: int, width1: int | None = None,
    num_columns1: int | None = None, gauss_id1: int = -1,
) -> Path:
    path = tmp_path / "two_headers.mpco"
    f, stage_name = _create_skeleton(
        path, node_ids=np.array([1, 2, 3], dtype=np.int32),
        coords=np.zeros((3, 3)), n_steps=_T, dt=1.0,
    )
    try:
        stage = f[stage_name]
        _add_bucket(stage, frame, f"{stem}0]", _EIDS[0], _flat(0, width),
                    comps0)
        _add_bucket(stage, frame, f"{stem}1]", _EIDS[1],
                    _flat(1, width if width1 is None else width1), comps1,
                    num_columns=num_columns1, gauss_id=gauss_id1)
    finally:
        f.close()
    return path


def _expected(header: int, width: int, n_comp: int, k: int) -> np.ndarray:
    """``(T, E, 2)``: component ``k`` at end i and end j."""
    return _flat(header, width)[:, :, [k, k + n_comp]]


def _by_element(slab) -> dict[int, np.ndarray]:
    return {int(e): slab.values[:, i, :]
            for i, e in enumerate(slab.element_ids)}


# =====================================================================
# Two headers, one class: each reads its own values
# =====================================================================

@pytest.mark.parametrize(
    ("cls", "frame", "token", "stem", "comps"),
    [c[1:] for c in _CASES], ids=[c[0] for c in _CASES],
)
def test_each_header_reads_its_own_values(
    tmp_path: Path, cls: str, frame: str, token: str, stem: str, comps: str,
) -> None:
    layout = NODAL_FORCE_CATALOG[(cls, token)]
    width, n_comp = layout.flat_size_per_element, layout.n_components_per_node
    path = _two_header_file(tmp_path, frame, stem, comps, comps, width=width)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NodalForceBucketSkippedWarning)
        with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
            assert set(layout.component_layout) <= set(
                r.elements.available_components())
            for k, comp in enumerate(layout.component_layout):
                got = _by_element(r.elements.get(component=comp))
                assert sorted(got) == sorted(
                    [*_EIDS[0].tolist(), *_EIDS[1].tolist()]), comp
                for h in (0, 1):
                    want = _expected(h, width, n_comp, k)
                    for i, e in enumerate(_EIDS[h]):
                        np.testing.assert_array_equal(
                            got[int(e)], want[:, i, :],
                            err_msg=f"{comp} header {h} element {e}")


def test_id_filter_reads_the_second_header_alone(tmp_path: Path) -> None:
    _, cls, frame, token, stem, comps = _CASES[0]
    layout = NODAL_FORCE_CATALOG[(cls, token)]
    width = layout.flat_size_per_element
    path = _two_header_file(tmp_path, frame, stem, comps, comps, width=width)
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        slab = r.elements.get(component="nodal_resisting_moment_y", ids=[33])
    np.testing.assert_array_equal(slab.element_ids, [33])
    np.testing.assert_array_equal(
        slab.values, _expected(1, width, 6, 4)[:, 1:2, :])


def test_scan_returns_both_headers(tmp_path: Path) -> None:
    _, cls, frame, token, stem, comps = _CASES[1]
    width = NODAL_FORCE_CATALOG[(cls, token)].flat_size_per_element
    path = _two_header_file(tmp_path, frame, stem, comps, comps, width=width)
    with h5py.File(path, "r") as f:
        _tok, buckets, skipped = _mn.scan_nodal_force_buckets(
            f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"],
            canonical_component="nodal_resisting_force_local_x",
        )
    assert skipped == []
    assert sorted(b.elem_key.header_idx for b in buckets) == [0, 1]


# =====================================================================
# A second header whose META does not spell the catalog warns by name
# =====================================================================

_G3_SWAPPED = ("Py", "Px", "Pz", "Mx", "My", "Mz")
_L3_SWAPPED = ("Vy", "N", "Vz", "T", "My", "Mz")

#: (id, frame, stem, header-1 COMPONENTS, header-1 width, NUM_COLUMNS,
#:  GAUSS_IDS, the reason the warning must name)
_MISMATCHES = [
    ("swapped-global", "globalForce", "5-ElasticBeam3d[1:0:",
     _names(_G3_SWAPPED), 12, None, -1, "do not spell"),
    ("swapped-local", "localForce", "5-ElasticBeam3d[1:0:",
     _names(_L3_SWAPPED), 12, None, -1, "do not match the catalog"),
    ("four-node-names", "globalForce", "5-ElasticBeam3d[1:0:",
     _names(("Px", "Py", "Pz"), n_nodes=4), 12, None, -1, "do not spell"),
    ("three-node-columns", "globalForce", "5-ElasticBeam3d[1:0:",
     _names(_G3, n_nodes=3), 18, None, -1, "NUM_COLUMNS=18"),
    ("2d-names-on-3d-class", "localForce", "5-ElasticBeam3d[1:0:",
     _names(("N", "V", "M"), n_nodes=4), 12, None, -1,
     "do not match the catalog"),
    ("unnamed-columns", "globalForce", "5-ElasticBeam3d[1:0:",
     "0." + ",".join(f"C{i}" for i in range(1, 13)), 12, None, -1,
     "do not decode"),
    ("no-components", "globalForce", "5-ElasticBeam3d[1:0:",
     None, 12, None, -1, "carries no COMPONENTS"),
    ("gauss-point", "globalForce", "5-ElasticBeam3d[1:0:",
     _names(_G3), 12, None, 0, "GAUSS_IDS"),
    ("column-count", "globalForce", "5-ElasticBeam3d[1:0:",
     _names(_G3).rpartition(",")[0], 12, None, -1, "names 11 columns"),
]


@pytest.mark.parametrize(
    ("frame", "stem", "comps1", "width1", "ncol1", "gid1", "reason"),
    [m[1:] for m in _MISMATCHES], ids=[m[0] for m in _MISMATCHES],
)
def test_second_header_meta_mismatch_warns_by_name(
    tmp_path: Path, frame: str, stem: str, comps1: str | None, width1: int,
    ncol1: int | None, gid1: int, reason: str,
) -> None:
    token = "global_force" if frame == "globalForce" else "local_force"
    layout = NODAL_FORCE_CATALOG[("ElasticBeam3d", token)]
    comps0 = _names(_G3 if frame == "globalForce" else _L3)
    path = _two_header_file(
        tmp_path, frame, stem, comps0, comps1, width=12, width1=width1,
        num_columns1=ncol1, gauss_id1=gid1,
    )
    comp = layout.component_layout[0]
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.warns(NodalForceBucketSkippedWarning) as rec:
            slab = r.elements.get(component=comp)
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, NodalForceBucketSkippedWarning)]
    assert len(msgs) == 1, msgs
    assert f"{stem}1]" in msgs[0]
    assert "'ElasticBeam3d'" in msgs[0]
    assert "response header 1" in msgs[0]
    assert reason in msgs[0], msgs[0]
    # Header 0 still reads, alone and with its own values.
    np.testing.assert_array_equal(slab.element_ids, _EIDS[0])
    np.testing.assert_array_equal(slab.values, _expected(0, 12, 6, 0))
