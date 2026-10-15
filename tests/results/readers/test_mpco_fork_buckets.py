"""MPCO reads of the fork's bucket shapes, without the fork (#1970).

The fork's MPCO recorder writes the bracket's integration rule as 0
(``[0:0:0]``) for its own classes and for the stock SixNodeTri and
EightNodeQuad, and the reader used to answer each with an empty slab and no
warning. Every file here is synthesised in the shape fork build 79e0623670
writes. The shapes were dumped from live runs of each class, with each read
checked against ``eleResponse``:

* **Gauss-tagged blocks**: one META block per GP (``GAUSS_IDS`` 0..n-1).
  BezierTri6, BezierTet10, LadrunoBrick, LadrunoBrick20, SixNodeTri,
  EightNodeQuad, and the LadrunoBrick material-state buckets.
* **One unnamed element-level block**: ``GAUSS_IDS == [-1]``, columns
  ``C1..Cn``. LadrunoQuad (``C1..C12``), LadrunoCST (``C1..C3``), LadrunoLST
  (``C1..C9``).
* **An element-only file** (``-E`` channels, no ``-N``): no ``ON_NODES`` at
  all, and STEP keys numbered by the process's step counter, not from 0.

The group is the keyword the deck asked for, so each stress case is read under
both ``stress`` and ``stresses``. Expected values are a closed form of (time,
element, GP, component), laid out GP-slowest as the recorder packs them.
Expected GP coordinates are written out from each rule, in the element's own
order: the live ``integrationPoints`` where the element answers it, and
otherwise the fork source cited at each family.
"""
from __future__ import annotations

from dataclasses import dataclass
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
LABELS_3D = "sigma11,sigma22,sigma33,sigma12,sigma23,sigma13"
LABELS_2D = "sigma11,sigma22,sigma12"

G2 = 1 / np.sqrt(3.0)                  # 2-point Gauss-Legendre abscissa
G3 = np.sqrt(0.6)                      # 3-point Gauss-Legendre abscissa
TA = (5 + 3 * np.sqrt(5.0)) / 20       # 4-point tet rule (a, b, b)
TB = (5 - np.sqrt(5.0)) / 20


def _hex27() -> np.ndarray:
    """LadrunoBrick20's 27 GPs in the element's own order (live
    ``integrationPoints``): corners, edge midpoints, faces, centroid."""
    a = G3
    corner = [(-a, -a, -a), (a, -a, -a), (a, a, -a), (-a, a, -a),
              (-a, -a, a), (a, -a, a), (a, a, a), (-a, a, a)]
    edge = [(0, -a, -a), (a, 0, -a), (0, a, -a), (-a, 0, -a),
            (0, -a, a), (a, 0, a), (0, a, a), (-a, 0, a),
            (-a, -a, 0), (a, -a, 0), (a, a, 0), (-a, a, 0)]
    face = [(a, 0, 0), (0, a, 0), (0, 0, a), (-a, 0, 0), (0, -a, 0),
            (0, 0, -a)]
    return np.array(corner + edge + face + [(0, 0, 0)], dtype=float)


@dataclass(frozen=True)
class Family:
    cls: str
    tag: int
    n_gp: int
    comps: tuple[str, ...]
    unnamed: bool                # one GAUSS_IDS=[-1] block of C1..Cn
    gp: np.ndarray               # expected natural coords, (n_gp, d), in order

    @property
    def key(self) -> str:
        return f"{self.tag}-{self.cls}[0:0:0]"


FAMILIES = [
    # BezierTri6 and BezierTet10: the live ``integrationPoints`` order.
    Family("BezierTri6", 33000, 3, STRESS_2D, False,
           np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])),
    Family("BezierTet10", 33001, 4, STRESS_3D, False,
           np.array([[TA, TB, TB], [TB, TA, TB], [TB, TB, TA],
                     [TB, TB, TB]])),
    # LadrunoBrick.cpp:649-652: i, j, k over sg = {-1/sqrt3, +1/sqrt3},
    # x slowest.
    Family("LadrunoBrick", 33002, 8, STRESS_3D, False,
           np.array([[x, y, z] for x in (-G2, G2) for y in (-G2, G2)
                     for z in (-G2, G2)])),
    # LadrunoQuad.cpp:89-92: counter-clockwise from (-, -).
    Family("LadrunoQuad", 33007, 4, STRESS_2D, True,
           np.array([[-G2, -G2], [G2, -G2], [G2, G2], [-G2, G2]])),
    Family("LadrunoCST", 33008, 1, STRESS_2D, True,
           np.array([[1 / 3, 1 / 3]])),
    # LadrunoLST answers no integrationPoints; its order is the fork
    # source's (LadrunoLST.cpp, the SixNodeTri anchors).
    Family("LadrunoLST", 33016, 3, STRESS_2D, True,
           np.array([[2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 6, 1 / 6]])),
    Family("LadrunoBrick20", 33018, 27, STRESS_3D, False, _hex27()),
    # SixNodeTri answers no integrationPoints either: SixNodeTri.cpp:127-132.
    Family("SixNodeTri", 209, 3, STRESS_2D, False,
           np.array([[2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 6, 1 / 6]])),
    # EightNodeQuad.cpp:127-144: corners counter-clockwise, then the
    # edge midpoints, then the centre.
    Family("EightNodeQuad", 208, 9, STRESS_2D, False,
           np.array([[-G3, -G3], [G3, -G3], [G3, G3], [-G3, G3],
                     [0.0, -G3], [G3, 0.0], [0.0, G3], [-G3, 0.0],
                     [0.0, 0.0]])),
]


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


def _gauss_bucket(
    f: h5py.File, group: str, key: str, *, n_gp: int, labels: str,
    ids: list[int], n_steps: int = 2, step0: int = 0,
) -> None:
    """One META block per GP, as the fork tags GP-aware responses."""
    n_comp = len(labels.split(","))
    _bucket(f, group, key, flat=_flat(n_steps, len(ids), n_gp, n_comp),
            ids=ids, gauss_ids=list(range(n_gp)), n_comp=[n_comp] * n_gp,
            components=";".join(f"0.1.4.{labels}" for _ in range(n_gp)),
            step0=step0)


def _unnamed_bucket(
    f: h5py.File, group: str, key: str, *, n_gp: int, n_comp: int,
    ids: list[int], named: bool = False,
) -> None:
    """One element-level block, as the fork writes an untagged response.

    The columns are the same GP-slowest packing; only META differs.
    """
    width = n_gp * n_comp
    labels = [f"s{i}" if named else f"C{i}" for i in range(1, width + 1)]
    _bucket(f, group, key, flat=_flat(2, len(ids), n_gp, n_comp), ids=ids,
            gauss_ids=[-1], n_comp=[width],
            components="0." + ",".join(labels))


def _write_family(f: h5py.File, fam: Family, group: str, ids: list[int]) -> None:
    if fam.unnamed:
        _unnamed_bucket(f, group, fam.key, n_gp=fam.n_gp,
                        n_comp=len(fam.comps), ids=ids)
    else:
        _gauss_bucket(f, group, fam.key, n_gp=fam.n_gp, ids=ids,
                      labels=LABELS_3D if fam.comps == STRESS_3D else LABELS_2D)


def _check(slab, comp: int, ids: list[int], n_gp: int) -> None:
    index = np.asarray(slab.element_index)
    values = np.asarray(slab.values)
    assert values.shape == (2, len(ids) * n_gp)
    assert index.tolist() == [i for i in ids for _ in range(n_gp)]
    for e in range(len(ids)):
        for gp in range(n_gp):
            for t in range(2):
                assert values[t, e * n_gp + gp] == _value(t, e, gp, comp), (t, e, gp)


def _check_coords(slab, fam: Family, n_elem: int) -> None:
    xi = np.asarray(slab.natural_coords)
    d = fam.gp.shape[1]
    assert xi.shape[0] == n_elem * fam.n_gp
    for e in range(n_elem):
        got = xi[e * fam.n_gp:(e + 1) * fam.n_gp, :d]
        np.testing.assert_allclose(got, fam.gp, atol=1e-12)


@pytest.mark.parametrize("group", ["stress", "stresses"])
@pytest.mark.parametrize("fam", FAMILIES, ids=[f.cls for f in FAMILIES])
def test_rule0_family_gauss_stress_reads(
    tmp_path: Path, fam: Family, group: str,
) -> None:
    ids = [7, 9, 11]
    path = tmp_path / "fam.mpco"
    with _file(path, nodal=True) as f:
        _write_family(f, fam, group, ids)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        assert set(fam.comps) <= set(
            r.available_components(stage.id, ResultLevel.GAUSS)
        )
        for k, comp in enumerate(fam.comps):
            slab = r.read_gauss(stage.id, comp)
            _check(slab, k, ids, fam.n_gp)
            _check_coords(slab, fam, len(ids))


@pytest.mark.parametrize("group, labels, parent", [
    ("material.equivalentPlasticStrain", "PLE+,PLE-", "equivalent_plastic_strain"),
    ("material.damage", "d+,d-", "damage"),
])
def test_rule0_material_state_reads(
    tmp_path: Path, group: str, labels: str, parent: str,
) -> None:
    """LadrunoBrick + ASDConcrete3D: the material-state path, rule 0."""
    fam = FAMILIES[2]
    assert fam.cls == "LadrunoBrick"
    ids = [7, 9]
    path = tmp_path / "mat.mpco"
    with _file(path, nodal=True) as f:
        _gauss_bucket(f, group, fam.key, n_gp=8, labels=labels, ids=ids)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        for k, suffix in enumerate(("tension", "compression")):
            slab = r.read_gauss(stage.id, f"{parent}_{suffix}")
            _check(slab, k, ids, 8)
            _check_coords(slab, fam, len(ids))


def test_named_element_block_still_refuses(tmp_path: Path) -> None:
    """Columns the element named itself are not renamed by the catalog."""
    path = tmp_path / "lst_named.mpco"
    with _file(path, nodal=True) as f:
        _unnamed_bucket(f, "stresses", "33016-LadrunoLST[0:0:0]", n_gp=3,
                        n_comp=3, ids=[4, 5], named=True)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError, match=r"33016-LadrunoLST\[0:0:0\]"):
            r.read_gauss(stage.id, "stress_xx")


def test_unnamed_block_of_another_width_refuses(tmp_path: Path) -> None:
    path = tmp_path / "lst_wide.mpco"
    with _file(path, nodal=True) as f:
        _unnamed_bucket(f, "stresses", "33016-LadrunoLST[0:0:0]", n_gp=1,
                        n_comp=10, ids=[4, 5])
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError, match=r"33016-LadrunoLST\[0:0:0\]"):
            r.read_gauss(stage.id, "stress_xx")


def test_rule0_gauss_tagged_width_refusal_names_the_remedy(tmp_path: Path) -> None:
    """LadrunoBrick20 ``formulation="uri"``: 8 GP blocks against the 27-GP row.

    The live fork writes this as ``33018-LadrunoBrick20[0:0:0]`` with
    ``NUM_COLUMNS == 48``. The fork's native ``.ladruno`` recorder reads the
    same run equal to ``eleResponse``, so the message points there. It names
    the remedy without the fork's tokens (``tests/test_fork_token_ratchet.py``).
    """
    path = tmp_path / "uri.mpco"
    with _file(path, nodal=True) as f:
        _gauss_bucket(f, "stresses", "33018-LadrunoBrick20[0:0:0]", n_gp=8,
                      labels=LABELS_3D, ids=[7])
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError) as err:
            r.read_gauss(stage.id, "stress_xx")
    msg = str(err.value)
    assert "33018-LadrunoBrick20[0:0:0]" in msg
    assert "NUM_COLUMNS=48" in msg
    assert "integration rule 0" in msg and "27-GP layout" in msg
    assert "fork's native recorder" in msg


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
        _write_family(f, FAMILIES[6], "stresses", [7])
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        with pytest.raises(ValueError, match=r"integration rule 0"):
            r.read_gauss(stage.id, "stress_xx")


def test_element_only_file_has_its_steps(tmp_path: Path) -> None:
    path = tmp_path / "element_only.mpco"
    with _file(path, nodal=False) as f:
        _gauss_bucket(f, "stress", "33018-LadrunoBrick20[0:0:0]", n_gp=27,
                      labels=LABELS_3D, ids=[7, 9], step0=8)
    with MPCOReader(path) as r:
        (stage,) = r.stages()
        assert stage.n_steps == 2
        np.testing.assert_array_equal(r.time_vector(stage.id), [0.5, 1.0])
        _check(r.read_gauss(stage.id, "stress_yy"), 1, [7, 9], 27)


def test_element_only_file_with_two_cadences_refuses(tmp_path: Path) -> None:
    path = tmp_path / "element_only_bad.mpco"
    with _file(path, nodal=False) as f:
        _gauss_bucket(f, "stress", "33018-LadrunoBrick20[0:0:0]", n_gp=27,
                      labels=LABELS_3D, ids=[7, 9])
        _gauss_bucket(f, "strain", "33018-LadrunoBrick20[0:0:0]", n_gp=27,
                      labels=LABELS_3D, ids=[7, 9], n_steps=1)
    with MPCOReader(path) as r, pytest.raises(ValueError, match="step-count"):
        r.stages()
