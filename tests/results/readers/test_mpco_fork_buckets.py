"""MPCO reads of the fork's bucket shapes, without the fork (#1970).

The fork's MPCO recorder writes three shapes the reader used to answer with
an empty slab and no warning. Each file here is synthesised in the exact
shape fork build 79e0623670 writes (dumped from the live-fork files of
``tests/opensees/integration_ladruno/test_element_families_ladruno_live.py``):

* ``33018-LadrunoBrick20[0:0:0]``: the bracket's integration rule is 0, with
  27 sequential GP blocks of 6 components, under ``stress`` or ``stresses``
  (the group is the keyword the deck asked for);
* ``33016-LadrunoLST[0:0:0]``: rule 0 and one element-level block
  (``GAUSS_IDS == [-1]``) of nine unnamed columns ``C1..C9``;
* an element-only file (``-E`` channels, no ``-N``): no ``ON_NODES`` at all,
  and STEP keys numbered by the process's step counter, not from 0.

The expected values are a closed form of (time, element, GP, component), laid
out GP-slowest as the recorder packs them, not decoded through the catalog.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    RESPONSE_CATALOG,
    IntRule,
    lookup,
)
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.readers._protocol import ResultLevel

STRESS_3D = ("stress_xx", "stress_yy", "stress_zz",
             "stress_xy", "stress_yz", "stress_xz")
STRESS_2D = ("stress_xx", "stress_yy", "stress_xy")
#: LadrunoLST's interior rule in the element's GP order (LadrunoLST.cpp).
LST_GP = np.array([[2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 6, 1 / 6]])
H20_KEY = "33018-LadrunoBrick20[0:0:0]"
LST_KEY = "33016-LadrunoLST[0:0:0]"


def _value(t: int, e: int, gp: int, comp: int) -> float:
    return 1000.0 * t + 100.0 * e + 3.0 * gp + 0.25 * comp + 1.0


def _flat(n_steps: int, n_elem: int, n_gp: int, n_comp: int) -> np.ndarray:
    """``(T, E, n_gp * n_comp)``, GP-slowest, components fastest."""
    out = np.empty((n_steps, n_elem, n_gp * n_comp))
    for t in range(n_steps):
        for e in range(n_elem):
            for gp in range(n_gp):
                for c in range(n_comp):
                    out[t, e, gp * n_comp + c] = _value(t, e, gp, c)
    return out


def _file(path: Path, *, nodal: bool, step0: int = 0) -> h5py.File:
    f = h5py.File(path, "w")
    stage = f.create_group("MODEL_STAGE[1]")
    nodes = stage.create_group("MODEL/NODES")
    nodes.create_dataset("ID", data=np.array([[1], [2]], dtype=np.int32))
    nodes.create_dataset("COORDINATES", data=np.zeros((2, 3)))
    results = stage.create_group("RESULTS")
    results.create_group("ON_ELEMENTS")
    if nodal:
        data = results.create_group("ON_NODES/DISPLACEMENT/DATA")
        for k in range(2):
            ds = data.create_dataset(f"STEP_{step0 + k}", data=np.zeros((2, 3)))
            ds.attrs["TIME"] = np.array([0.5 * (k + 1)])
    return f


def _bucket(
    f: h5py.File, group: str, key: str, *, flat: np.ndarray, ids: list[int],
    gauss_ids: list[int], n_comp: list[int], components: str, step0: int = 0,
) -> None:
    b = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].require_group(group).create_group(key)
    b.attrs["NUM_COLUMNS"] = np.array([flat.shape[2]], dtype=np.int32)
    meta = b.create_group("META")
    n = len(gauss_ids)
    meta.create_dataset("MULTIPLICITY", data=np.ones((n, 1), dtype=np.int32))
    meta.create_dataset("GAUSS_IDS", data=np.array(gauss_ids, np.int32).reshape(-1, 1))
    meta.create_dataset("NUM_COMPONENTS", data=np.array(n_comp, np.int32).reshape(-1, 1))
    meta.create_dataset("COMPONENTS", data=np.array([components.encode()]))
    b.create_dataset("ID", data=np.array(ids, dtype=np.int32).reshape(-1, 1))
    data = b.create_group("DATA")
    for k in range(flat.shape[0]):
        ds = data.create_dataset(f"STEP_{step0 + k}", data=flat[k])
        ds.attrs["TIME"] = np.array([0.5 * (k + 1)])


def _h20(f: h5py.File, group: str, *, step0: int = 0) -> None:
    names = "sigma11,sigma22,sigma33,sigma12,sigma23,sigma13"
    _bucket(f, group, H20_KEY, flat=_flat(2, 2, 27, 6), ids=[7, 9],
            gauss_ids=list(range(27)), n_comp=[6] * 27,
            components=";".join(f"0.1.4.{names}" for _ in range(27)),
            step0=step0)


def _lst(f: h5py.File, group: str, *, width: int = 9, named: bool = False) -> None:
    labels = [f"s{i}" if named else f"C{i}" for i in range(1, width + 1)]
    _bucket(f, group, LST_KEY, flat=_flat(2, 3, 1, width), ids=[4, 5, 6],
            gauss_ids=[-1], n_comp=[width],
            components="0." + ",".join(labels))


def _check(slab, comp: int, ids: list[int], n_gp: int) -> None:
    index = np.asarray(slab.element_index)
    values = np.asarray(slab.values)
    assert values.shape == (2, len(ids) * n_gp)
    assert index.tolist() == [i for i in ids for _ in range(n_gp)]
    for e in range(len(ids)):
        for gp in range(n_gp):
            col = e * n_gp + gp
            for t in range(2):
                assert values[t, col] == _value(t, e, gp, comp), (t, e, gp)


@pytest.mark.parametrize("group", ["stress", "stresses"])
def test_rule0_brick20_gauss_stress_reads(tmp_path: Path, group: str) -> None:
    path = tmp_path / "h20.mpco"
    with _file(path, nodal=True) as f:
        _h20(f, group)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        assert set(STRESS_3D) <= set(r.available_components(stage.id, ResultLevel.GAUSS))
        for k, comp in enumerate(STRESS_3D):
            slab = r.read_gauss(stage.id, comp)
            _check(slab, k, [7, 9], 27)
        xi = np.asarray(slab.natural_coords)
        a = np.sqrt(0.6)
        assert xi.shape == (54, 3)
        assert np.all(np.isclose(np.abs(xi), a) | (xi == 0.0))
        assert len({tuple(p) for p in np.round(xi[:27], 12)}) == 27


@pytest.mark.parametrize("group", ["stress", "stresses"])
def test_rule0_lst_unnamed_element_block_reads(tmp_path: Path, group: str) -> None:
    path = tmp_path / "lst.mpco"
    with _file(path, nodal=True) as f:
        _lst(f, group)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        for k, comp in enumerate(STRESS_2D):
            slab = r.read_gauss(stage.id, comp)
            index = np.asarray(slab.element_index)
            values = np.asarray(slab.values)
            assert index.tolist() == [4, 4, 4, 5, 5, 5, 6, 6, 6]
            # The block is one element-level "GP" of nine columns: GP g's
            # component k is column 3 * g + k of that block.
            for e in range(3):
                for gp in range(3):
                    for t in range(2):
                        assert values[t, 3 * e + gp] == _value(t, e, 0, 3 * gp + k)
        np.testing.assert_allclose(
            np.asarray(slab.natural_coords)[:3, :2], LST_GP, atol=1e-12,
        )


def test_named_element_block_still_refuses(tmp_path: Path) -> None:
    """Columns the element named itself are not renamed by the catalog."""
    path = tmp_path / "lst_named.mpco"
    with _file(path, nodal=True) as f:
        _lst(f, "stresses", named=True)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError, match=r"33016-LadrunoLST\[0:0:0\]"):
            r.read_gauss(stage.id, "stress_xx")


def test_unnamed_block_of_another_width_refuses(tmp_path: Path) -> None:
    path = tmp_path / "lst_wide.mpco"
    with _file(path, nodal=True) as f:
        _lst(f, "stresses", width=10)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError, match=r"33016-LadrunoLST\[0:0:0\]"):
            r.read_gauss(stage.id, "stress_xx")


def test_rule0_with_two_catalog_rules_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 0 picks the class's one catalog rule; with two it refuses."""
    monkeypatch.setitem(
        RESPONSE_CATALOG, ("LadrunoBrick20", IntRule.Hex_GL_2, "stress"),
        lookup("Brick", IntRule.Hex_GL_2, "stress"),
    )
    path = tmp_path / "h20.mpco"
    with _file(path, nodal=True) as f:
        _h20(f, "stresses")
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError, match=r"integration rule 0"):
            r.read_gauss(stage.id, "stress_xx")


def test_element_only_file_has_its_steps(tmp_path: Path) -> None:
    path = tmp_path / "element_only.mpco"
    with _file(path, nodal=False) as f:
        _h20(f, "stress", step0=8)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        assert stage.n_steps == 2
        np.testing.assert_array_equal(r.time_vector(stage.id), [0.5, 1.0])
        _check(r.read_gauss(stage.id, "stress_yy"), 1, [7, 9], 27)


def test_element_only_file_with_two_cadences_refuses(tmp_path: Path) -> None:
    path = tmp_path / "element_only_bad.mpco"
    with _file(path, nodal=False) as f:
        _h20(f, "stress")
        _bucket(f, "strain", H20_KEY, flat=_flat(1, 2, 27, 6), ids=[7, 9],
                gauss_ids=list(range(27)), n_comp=[6] * 27,
                components="x")
    with MPCOReader(path) as r, pytest.raises(ValueError, match="step-count"):
        r.stages()
