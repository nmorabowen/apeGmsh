"""Line stations take one ranked source per element, on both formats (#2109).

The rank is ``section.force`` (0) > ``localForce`` (1) > ``basicForce``
end moments (2, #2107). ``globalForce`` and every other element-level
token are never a source: global end forces are not local station values.
Before #2109 a ``.ladruno`` read fed any element-level token into line
stations at rank 2, so a ``globalForce``-only file read global end moments
as ``bending_moment_*`` (wrong sign and frame), and ``basicForce`` beside
``globalForce`` gave each element four stations. MPCO read no
``basicForce`` stations at all and passed ``globalForce`` over silently.

The oracle is statics on #2107's 3-D cantilever: along +x, fixed at
x = 0, L = 4, two elements (11 on [0, 2], 12 on [2, 4]), tip load
(Fx, Fy, Fz, Mx, My, Mz) = (1e3, 2e3, 3e3, 4e2, 5e2, 6e2), local y = Y and
z = Z, so global and local end forces coincide. In the section convention
``Mz(x) = Mz + Fy (L - x)`` and ``My(x) = My - Fz (L - x)``: over the beam
``bending_moment_z`` runs 8600 -> 600 and ``bending_moment_y`` -11500 -> 500.
Each synthetic bucket holds the end forces OpenSees reports for that load:
``localForce`` / ``globalForce`` end i ``-M(x_a)``, end j ``+M(x_b)``, and
``basicForce`` ``Mz_1 = -Mz(x_a)``, ``Mz_2 = +Mz(x_b)`` (#2107's identity).
Main read the ``globalForce``-only file as ``bending_moment_z`` =
[-8600, ..., 600].
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    ELE_TAG_ForceBeamColumn2d,
    ELE_TAG_ForceBeamColumn3d,
    ResponseLayout,
    flatten,
    lookup_custom_rule,
    resolve_layout_from_gp_x,
)
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_element_io import (
    LineStationBucketSkippedWarning,
)
from apeGmsh.results.readers._mpco import MPCOBucketSkippedWarning, MPCOReader
from apeGmsh.results.readers._protocol import ResultLevel
from tests.test_results_mpco_line_mock import (
    _add_line_station_bucket,
    _create_skeleton,
)

RTOL = 1e-10
L = 4.0
FX, FY, FZ, MX, MY, MZ = 1.0e3, 2.0e3, 3.0e3, 4.0e2, 5.0e2, 6.0e2
LAMBDA = np.array([0.5, 1.0])          # two load steps
#: element id -> (x_a, x_b)
SPAN = {11: (0.0, 2.0), 12: (2.0, 4.0), 13: (0.0, 2.0), 14: (2.0, 4.0),
        21: (0.0, 2.0), 22: (2.0, 4.0)}
MOMENTS = ("bending_moment_z", "bending_moment_y")
#: The card's statics at the fixed end and the tip, full load.
CARD_ENDS = {"bending_moment_z": (8600.0, 600.0),
             "bending_moment_y": (-11500.0, 500.0)}

#: The format's warning for a bucket it leaves out of a line-station read.
SKIP_WARNING: dict[str, type[Warning]] = {
    "ladruno": LineStationBucketSkippedWarning,
    "mpco": MPCOBucketSkippedWarning,
}


def _section(comp: str, x: float) -> float:
    """Statics on the free body ``x..L`` at the full load, section signs."""
    return {
        "axial_force": FX, "torsion": MX,
        "bending_moment_z": MZ + FY * (L - x),
        "bending_moment_y": MY - FZ * (L - x),
    }[comp]


# ---------------------------------------------------------------------
# Bucket rows: what OpenSees reports for one element at the full load
# ---------------------------------------------------------------------

_END_NAMES_LOCAL = "N,Vy,Vz,T,My,Mz"
_END_NAMES_GLOBAL = "Px,Py,Pz,Mx,My,Mz"


def _end_forces(eid: int) -> list[float]:
    """12 end forces, local = global frame: end i ``-s(x_a)``, end j ``+s(x_b)``."""
    xa, xb = SPAN[eid]
    row: list[float] = []
    for sign, x in ((-1.0, xa), (1.0, xb)):
        row += [sign * FX, 0.0, 0.0, sign * MX,
                sign * _section("bending_moment_y", x),
                sign * _section("bending_moment_z", x)]
    return row


def _names_12(bases: str) -> str:
    return ",".join(f"{b}_{s}" for s in (1, 2) for b in bases.split(","))


def _basic_row(eid: int) -> list[float]:
    """``N,Mz_1,Mz_2,My_1,My_2,T`` = ``q``: ``Mz_1 = -Mz(x_a)``, ``Mz_2 = Mz(x_b)``."""
    xa, xb = SPAN[eid]
    return [FX,
            -_section("bending_moment_z", xa), _section("bending_moment_z", xb),
            -_section("bending_moment_y", xa), _section("bending_moment_y", xb),
            MX]


_SECTION_XI = np.array([-1.0, 0.0, 1.0])          # Lobatto-3
_SECTION_NAMES = ("P", "Mz", "My", "T")
_SECTION_CANON = ("axial_force", "bending_moment_z", "bending_moment_y",
                  "torsion")


def _section_row(eid: int) -> list[float]:
    """``P,Mz,My,T`` at each of the three stations, statics."""
    xa, xb = SPAN[eid]
    row: list[float] = []
    for xi in _SECTION_XI:
        x = xa + (xi + 1.0) / 2.0 * (xb - xa)
        row += [_section(c, x) for c in _SECTION_CANON]
    return row


ELEMENT_BUCKETS: dict[str, tuple[str, str, Callable[[int], list[float]]]] = {
    # token: (bracket key, names, row)
    "localForce": ("5-ElasticBeam3d[1:0:0]", _names_12(_END_NAMES_LOCAL),
                   _end_forces),
    "globalForce": ("5-ElasticBeam3d[1:0:0]", _names_12(_END_NAMES_GLOBAL),
                    _end_forces),
    "basicForce": ("5-ElasticBeam3d[1:0:0]", "N,Mz_1,Mz_2,My_1,My_2,T",
                   _basic_row),
}


# ---------------------------------------------------------------------
# Writers, one per format
# ---------------------------------------------------------------------

class _File:
    """A minimal file of one format; ``add`` buckets, then ``reader()``."""

    def __init__(self, fmt: str, path: Path, *, ndm: int = 3) -> None:
        self.fmt = fmt
        self.path = path
        if fmt == "ladruno":
            with h5py.File(path, "w") as f:
                info = f.create_group("INFO")
                info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
                info.attrs["FORMAT_VERSION"] = np.array([1], dtype=np.int32)
                info.attrs["SPATIAL_DIM"] = np.array([ndm], dtype=np.int32)
                f.create_group("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS")
        else:
            f, _ = _create_skeleton(
                path, node_ids=np.array([1, 2, 3], dtype=np.int32),
                coords=np.array([[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]]),
                n_steps=LAMBDA.size, dt=1.0,
            )
            del f["INFO/SPATIAL_DIM"]
            f["INFO"].create_dataset("SPATIAL_DIM", data=ndm)
            f.close()

    def add(self, token: str, ids: list[int], *, key: str | None = None,
            names: str | None = None, row: Any = None,
            data_width: int | None = None) -> None:
        """An element-level bucket: ``row(eid)`` scaled by each step."""
        k0, n0, r0 = ELEMENT_BUCKETS.get(token, ("", "", None))
        key, names, row = key or k0, names or n0, row or r0
        data = np.array([[row(e) for e in ids] for _ in LAMBDA])
        data *= LAMBDA[:, None, None]
        if data_width is not None:          # a layout that misses its DATA
            data = np.concatenate([data] * (data_width // data.shape[-1]),
                                  axis=-1)
        with h5py.File(self.path, "r+") as f:
            if self.fmt == "ladruno":
                on_e = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
                grp = on_e.require_group(token).create_group(key)
                self._ladruno_column_map(grp, [names], level=0)
                self._ladruno_data(grp, ids, data)
            else:
                on_e = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
                grp = on_e.require_group(token).create_group(key)
                meta = grp.create_group("META")
                meta.create_dataset(
                    "COMPONENTS", data=np.array([f"0.{names}".encode()]))
                grp.create_dataset(
                    "ID", data=np.array(ids, dtype=np.int32)[:, None])
                dg = grp.create_group("DATA")
                for t in range(LAMBDA.size):
                    ds = dg.create_dataset(f"STEP_{t}", data=data[t])
                    ds.attrs["STEP"] = t
                    ds.attrs["TIME"] = float(t + 1)

    def add_section(self, ids: list[int], *, ndm: int = 3) -> None:
        """A ``section.force`` bucket of a force beam, three stations."""
        names = _SECTION_NAMES if ndm == 3 else ("P", "Mz")
        canon = _SECTION_CANON if ndm == 3 else _SECTION_CANON[:2]
        per_station = [
            [_section_row(e)[s * 4:s * 4 + len(canon)] for e in ids]
            for s in range(_SECTION_XI.size)
        ]                                                   # (S, E, C)
        if self.fmt == "ladruno":
            cls = "ForceBeamColumn3d" if ndm == 3 else "ForceBeamColumn2d"
            flat = np.array(per_station).transpose(1, 0, 2).reshape(
                len(ids), -1)
            data = LAMBDA[:, None, None] * flat[None]
            with h5py.File(self.path, "r+") as f:
                grp = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].require_group(
                    "section.force").create_group(f"73-{cls}[1000:1:0]")
                self._ladruno_column_map(
                    grp, [",".join(names)] * _SECTION_XI.size, level=2)
                self._ladruno_data(grp, ids, data)
            return
        per_comp = {
            c: np.array([[[lam * per_station[s][e][k]
                           for s in range(_SECTION_XI.size)]
                          for e in range(len(ids))] for lam in LAMBDA])
            for k, c in enumerate(canon)
        }
        cls = "ForceBeamColumn3d" if ndm == 3 else "ForceBeamColumn2d"
        tag = ELE_TAG_ForceBeamColumn3d if ndm == 3 else ELE_TAG_ForceBeamColumn2d
        shape = resolve_layout_from_gp_x(
            lookup_custom_rule(cls, "section_force"), _SECTION_XI, (2, 1))
        layout = ResponseLayout(
            n_gauss_points=_SECTION_XI.size,
            natural_coords=shape.natural_coords,
            coord_system=shape.coord_system,
            n_components_per_gp=len(canon), component_layout=canon,
            class_tag=shape.class_tag,
        )
        with h5py.File(self.path, "r+") as f:
            _add_line_station_bucket(
                f["MODEL_STAGE[1]"], class_tag=tag, class_name=cls,
                custom_rule_idx=1, header_idx=0,
                element_ids=np.array(ids, dtype=np.int32), gp_x=_SECTION_XI,
                section_names=names, flat_data=flatten(per_comp, layout),
            )

    @staticmethod
    def _ladruno_column_map(grp: Any, lines: list[str], *, level: int) -> None:
        cm = grp.create_group("COLUMN_MAP")
        n = len(lines)
        gauss = np.arange(n) if level == 2 else np.full(n, -1)
        for name, col in (("LEVELS", np.full(n, level)), ("GAUSS_ID", gauss),
                          ("MULTIPLICITY", np.ones(n)),
                          ("SECTION_TAG", np.full(n, -1)),
                          ("FIBER_ID", np.full(n, -1)),
                          ("NUM_COMP", [len(s.split(",")) for s in lines])):
            cm.create_dataset(name, data=np.asarray(col, dtype=np.int32)[:, None])
        cm.attrs["COMP_NAMES"] = np.bytes_("\n".join(lines).encode())

    @staticmethod
    def _ladruno_data(grp: Any, ids: list[int], data: np.ndarray) -> None:
        grp.create_dataset("DATA", data=data)
        grp.create_dataset("ID", data=np.array(ids, dtype=np.int32)[:, None])
        grp.create_dataset("STEP", data=np.arange(LAMBDA.size, dtype=np.int32))
        grp.create_dataset("TIME", data=np.arange(1.0, LAMBDA.size + 1.0))

    def reader(self) -> Any:
        return (LadrunoReader(self.path) if self.fmt == "ladruno"
                else MPCOReader(self.path))


def _read(r: Any, comp: str, ids: list[int] | None) -> Any:
    stage = r.stages()[0].id
    return r.read_line_stations(
        stage, comp, element_ids=None if ids is None else np.array(ids))


def _listed(r: Any) -> set[str]:
    return set(r.available_components(r.stages()[0].id,
                                      ResultLevel.LINE_STATIONS))


def _rows(slab: Any, eid: int) -> list[int]:
    return [i for i, e in enumerate(slab.element_index) if int(e) == eid]


def _assert_statics(slab: Any, comp: str, eid: int, xi: np.ndarray) -> None:
    rows = _rows(slab, eid)
    assert len(rows) == xi.size, (comp, eid, slab.element_index.tolist())
    np.testing.assert_allclose(slab.station_natural_coord[rows], xi,
                               rtol=0, atol=1e-12)
    xa, xb = SPAN[eid]
    x = xa + (xi + 1.0) / 2.0 * (xb - xa)
    want = np.outer(LAMBDA, [_section(comp, v) for v in x])
    np.testing.assert_allclose(slab.values[:, rows], want, rtol=RTOL,
                               err_msg=f"{comp} of element {eid}")


FORMATS = ["ladruno", "mpco"]


# ---------------------------------------------------------------------
# globalForce is never a station source
# ---------------------------------------------------------------------

@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_global_force_only_reads_no_stations_and_names_it(
    tmp_path: Path, fmt: str, comp: str,
) -> None:
    """The #2109 repro: main read ``bending_moment_z`` = [-8600, .., 600]
    from global end forces. Now nothing lists or reads, and the read names
    the ``globalForce`` bucket and its elements."""
    f = _File(fmt, tmp_path / f"cantilever.{fmt}")
    f.add("globalForce", [11, 12])
    with f.reader() as r:
        assert comp not in _listed(r)
        for ids in ([11, 12], None):
            with pytest.warns(SKIP_WARNING[fmt],
                              match=r"globalForce.*\[11, 12\].*never a "
                                    r"line-station source"):
                slab = _read(r, comp, ids)
            assert slab.element_index.size == 0
            assert slab.values.shape == (LAMBDA.size, 0)


@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_basic_force_beside_global_force_is_the_one_source(
    tmp_path: Path, fmt: str, comp: str,
) -> None:
    """Main gave each element four stations (basicForce and globalForce at
    rank 2, reading 8600 and -8600 at one station). Now basicForce alone
    reads, ``[-M_1, +M_2]``, equal to statics, and nothing warns."""
    f = _File(fmt, tmp_path / f"cantilever.{fmt}")
    f.add("basicForce", [11, 12])
    f.add("globalForce", [11, 12])
    with f.reader() as r:
        assert comp in _listed(r)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SKIP_WARNING[fmt])
            slab = _read(r, comp, [11, 12])
    for eid in (11, 12):
        _assert_statics(slab, comp, eid, np.array([-1.0, 1.0]))
    full = slab.values[-1]
    np.testing.assert_allclose([full[0], full[-1]], CARD_ENDS[comp],
                               rtol=RTOL)


@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_each_element_reads_its_highest_ranked_source_only(
    tmp_path: Path, fmt: str, comp: str,
) -> None:
    """Four elements, one per rank:

    * 11 records section.force, localForce, basicForce and globalForce:
      three section stations;
    * 12 records localForce, basicForce and globalForce: two localForce
      stations;
    * 13 records basicForce and globalForce: two basicForce stations;
    * 14 records globalForce only: no stations, named in the warning.
    """
    f = _File(fmt, tmp_path / f"frame.{fmt}")
    f.add_section([11])
    f.add("localForce", [11, 12])
    f.add("basicForce", [11, 12, 13])
    f.add("globalForce", [11, 12, 13, 14])
    with f.reader() as r:
        for ids in ([11, 12, 13, 14], None):
            with pytest.warns(SKIP_WARNING[fmt],
                              match=r"globalForce.*element\(s\) \[14\]"):
                slab = _read(r, comp, ids)
            assert sorted(set(slab.element_index.tolist())) == [11, 12, 13]
            _assert_statics(slab, comp, 11, _SECTION_XI)
            _assert_statics(slab, comp, 12, np.array([-1.0, 1.0]))
            _assert_statics(slab, comp, 13, np.array([-1.0, 1.0]))


# ---------------------------------------------------------------------
# MPCO's 2-D localForce refusal spares section-served elements (#2098 1)
# ---------------------------------------------------------------------

def test_mpco_refused_2d_local_force_spares_section_stations(
    tmp_path: Path,
) -> None:
    """A force beam (21) records ``section.force`` and a ``localForce``
    whose META names 6 columns over 12-wide DATA (refused, #2010). Its
    section stations read; before #2109 the refusal raised for it and for
    every ``element_ids=None`` read. Element 22, in the refused bucket with
    no section stations, still raises naming the class. (``.ladruno`` has
    held this since #2095: ``test_results_ladruno_local_force_2d_gate``.)
    """
    f = _File("mpco", tmp_path / "frame2d.mpco", ndm=2)
    f.add_section([21], ndm=2)
    f.add("localForce", [21], key="73-ForceBeamColumn2d[1000:1:0]",
          names="N_1,V_1,M_1,N_2,V_2,M_2", row=lambda e: [1.0] * 6,
          data_width=12)
    with f.reader() as r:
        for ids in ([21], None):
            slab = _read(r, "bending_moment_z", ids)
            _assert_statics(slab, "bending_moment_z", 21, _SECTION_XI)
    f.add("localForce", [22], key="5-ElasticBeam2d[1:0:0]",
          names="N_1,V_1,M_1,N_2,V_2,M_2", row=lambda e: [1.0] * 6,
          data_width=12)
    with f.reader() as r:
        with pytest.raises(ValueError, match=r"ElasticBeam2d.*\[22\]"):
            _read(r, "bending_moment_z", [21, 22])


# ---------------------------------------------------------------------
# .ladruno names a bucket whose COLUMN_MAP misses its DATA (#2098 item 3)
# ---------------------------------------------------------------------

def test_ladruno_read_names_an_unreadable_local_force_bucket(
    tmp_path: Path,
) -> None:
    """Six names over 12 DATA columns at multiplicity 1: ``parse_blocks``
    fails. The listing has named it since #2095; the read dropped it with
    no word. Now the read names it too, and the other buckets still read."""
    f = _File("ladruno", tmp_path / "cantilever.ladruno")
    f.add("basicForce", [11])
    f.add("localForce", [12], data_width=24)
    with f.reader() as r:
        with pytest.warns(LineStationBucketSkippedWarning,
                          match=r"localForce bucket '5-ElasticBeam3d"
                                r"\[1:0:0\]'.*unreadable"):
            slab = _read(r, "bending_moment_z", [11, 12])
    assert sorted(set(slab.element_index.tolist())) == [11]
    _assert_statics(slab, "bending_moment_z", 11, np.array([-1.0, 1.0]))


# ---------------------------------------------------------------------
# MPCO basicForce: the end moments only, as on .ladruno (#2107)
# ---------------------------------------------------------------------

def test_mpco_basic_force_non_moment_pair_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """``Vy_1,Vy_2`` under ``basicForce`` has no basic-to-section identity:
    no stations, and the read names the bucket and the component, as the
    ``.ladruno`` read has since #2107. The end-moment pair beside it reads."""
    f = _File("mpco", tmp_path / "cantilever.mpco")
    f.add("basicForce", [11, 12], names="N,Vy_1,Vy_2,My_1,My_2,T")
    with f.reader() as r:
        assert "shear_y" not in _listed(r)
        assert "bending_moment_y" in _listed(r)
        with pytest.warns(MPCOBucketSkippedWarning,
                          match=r"basicForce.*'shear_y' at stations \[1, 2\]"):
            slab = _read(r, "shear_y", [11, 12])
        assert slab.element_index.size == 0
        slab = _read(r, "bending_moment_y", [11, 12])
    for eid in (11, 12):
        _assert_statics(slab, "bending_moment_y", eid, np.array([-1.0, 1.0]))


# =====================================================================
# Live, on the fork: #2107's cantilever, each token combination, both
# formats (CI ``live-fork`` at FORK_PIN)
# =====================================================================

#: case -> (beam family, recorded tokens, the source that must win or None,
#: stations per element)
LIVE_CASES: dict[str, tuple[str, tuple[str, ...], "str | None", int]] = {
    "global_only": ("elasticBeamColumn", ("globalForce",), None, 0),
    "basic_global": ("elasticBeamColumn", ("basicForce", "globalForce"),
                     "basicForce", 2),
    "local_basic_global": ("elasticBeamColumn",
                           ("localForce", "basicForce", "globalForce"),
                           "localForce", 2),
    "section_all": ("forceBeamColumn",
                    ("section.force", "localForce", "basicForce",
                     "globalForce"), "section.force", 4),
}
LIVE_KINDS = {"ladruno": "Ladruno", "mpco": "MPCO"}


@pytest.fixture(scope="module", params=sorted(LIVE_CASES))
def live_cantilever(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Solve #2107's two-element cantilever in two load steps, recording
    the case's tokens with both recorders."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter
    from apeGmsh.results import Results
    from tests.opensees.integration_ladruno.test_ladruno_line_stations_live import (  # noqa: E501
        TIP,
        _beam_3d,
        _cantilever_fem,
        _fem_to_ops,
    )

    case = request.param
    family, tokens, _, _ = LIVE_CASES[case]
    tmp = tmp_path_factory.mktemp(f"source_rank_{case}")
    fem = _cantilever_fem(f"rank_{case}")
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))  # local y = Y, z = Z
    _beam_3d(ops, family, tr)
    ops.fix(label="a", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="b"), forces=TIP)
    paths = {fmt: str(tmp / f"cantilever.{fmt}") for fmt in LIVE_KINDS}
    for fmt, kind in LIVE_KINDS.items():
        # A node channel too: an element-only MPCO file reads 0 steps (#1970).
        getattr(ops.recorder, kind)(file=paths[fmt],
                                    nodal_responses=("displacement",),
                                    elem_responses=tokens)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Beam").ids]
    assert any(fem_to_ops[e] != e for e in eids)       # not vacuous
    coords = {int(n): np.asarray(c, dtype=float)
              for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    ends = {int(e): (float(coords[int(c[0])][0]), float(coords[int(c[-1])][0]))
            for grp in fem.elements
            for e, c in zip(grp.ids, grp.connectivity)}
    for _ in range(2):
        assert emitter.analyze(steps=1) == 0
    emitter.ops.remove("recorders")
    return {
        "case": case, "eids": eids, "ends": ends,
        "results": {
            fmt: (Results.from_ladruno if fmt == "ladruno"
                  else Results.from_mpco)(paths[fmt], fem=fem,
                                          model_h5=model_h5)
            for fmt in LIVE_KINDS
        },
    }


@pytest.mark.ladruno_fork
@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_live_cantilever_reads_one_ranked_source(
    live_cantilever: dict[str, Any], fmt: str, comp: str,
) -> None:
    """Each element reads its highest-ranked recorded source, equal to
    statics at ``rtol=1e-10`` (``[8600, 600]`` / ``[-11500, 500]`` over the
    beam at the full load); a ``globalForce``-only file reads nothing and
    names the bucket."""
    sol = live_cantilever
    _, _, winner, n_st = LIVE_CASES[sol["case"]]
    ls = sol["results"][fmt].elements.line_stations
    if winner is None:
        assert comp not in ls.available_components()
        with pytest.warns(SKIP_WARNING[fmt],
                          match=r"globalForce.*never a line-station source"):
            slab = ls.get(component=comp, ids=sol["eids"])
        assert np.asarray(slab.element_index).size == 0
        return
    assert comp in ls.available_components()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SKIP_WARNING[fmt])
        slab = ls.get(component=comp, ids=sol["eids"])
    values = np.asarray(slab.values)
    xi_all = np.asarray(slab.station_natural_coord)
    index = np.asarray(slab.element_index)
    for eid in sol["eids"]:
        rows = [i for i, e in enumerate(index) if int(e) == eid]
        assert len(rows) == n_st, (fmt, sol["case"], eid, index.tolist())
        xa, xb = sol["ends"][eid]
        x = xa + (xi_all[rows] + 1.0) / 2.0 * (xb - xa)
        want = np.outer(LAMBDA, [_section(comp, v) for v in x])
        np.testing.assert_allclose(
            values[:, rows], want, rtol=RTOL, atol=1e-9,
            err_msg=f"{fmt} {sol['case']} {comp} of element {eid}")
    # The card's numbers: the fixed end and the tip at the full load.
    x_all = np.array([
        sol["ends"][int(e)][0]
        + (xi + 1.0) / 2.0 * (sol["ends"][int(e)][1] - sol["ends"][int(e)][0])
        for e, xi in zip(index, xi_all)
    ])
    root_to_tip = values[-1, np.argsort(x_all, kind="stable")]
    np.testing.assert_allclose([root_to_tip[0], root_to_tip[-1]],
                               CARD_ENDS[comp], rtol=RTOL)
