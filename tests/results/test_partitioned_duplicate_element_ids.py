"""#2221 items 2 and 3: an element id a stitch would return twice refuses.

2. **Across partitions.** Each element lives on exactly one rank, so the
   MPCO and ``.ladruno`` partition stitches concatenate element rows
   without dedupe. Before the fix, part files holding ``[7, 11]`` and
   ``[11, 33]`` read back ``[7, 11, 11, 33]`` with no word. The stitch now
   raises :class:`MPCOElementInTwoBuckets`, naming the id and both part
   files. Nodes are not refused: a boundary node shared by two parts is
   the design, and the node merge collapses it.
3. **Inside one ``.ladruno`` element read.** ``read_element_slab``
   stitches every bucket of a token; an id in two buckets (or twice in
   one) read twice. It now refuses the same way.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
from apeGmsh.results.readers._mpco_refusals import MPCOElementInTwoBuckets
from tests.results.test_mpco_mixed_dim_gauss import (
    _TET,
    _fiber_file,
    _gauss,
    _section_force_file,
    _stress_file,
)
from tests.test_results_mpco_layer_mock import (  # noqa: F401 (fixture)
    simple_layer_mpco,
)
from tests.test_results_mpco_spring_mock import _make_spring_mpco

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
_STAGE = "MODEL_STAGE[1]"


# =====================================================================
# MPCO partitioned
# =====================================================================

def _mpco_parts(tmp_path: Path, ids_0: list[int], ids_1: list[int]) -> list[Path]:
    """Two tet parts; both carry nodes 1..4 (shared boundary nodes)."""
    return [
        _stress_file(tmp_path / "Rec.part-0.mpco", [(_TET, ids_0)]),
        _stress_file(tmp_path / "Rec.part-1.mpco", [(_TET, ids_1)]),
    ]


class TestMPCOPartitioned:
    def test_element_in_two_parts_refuses(self, tmp_path: Path) -> None:
        p0, _ = _mpco_parts(tmp_path, [7, 11], [11, 33])
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _gauss(p0, "stress_xx")
        msg = str(info.value)
        assert "element 11 comes back from two partition files" in msg
        assert "Rec.part-0.mpco and Rec.part-1.mpco" in msg

    def test_disjoint_parts_read_unchanged(self, tmp_path: Path) -> None:
        p0, _ = _mpco_parts(tmp_path, [7, 11], [33])
        slab = _gauss(p0, "stress_xx")
        np.testing.assert_array_equal(slab.element_index, [7, 11, 33])
        np.testing.assert_array_equal(slab.values, [[7.0, 11.0, 33.0]])

    def test_a_read_that_skips_the_shared_id_answers(
        self, tmp_path: Path,
    ) -> None:
        """The refusal is about what the read returns."""
        p0, _ = _mpco_parts(tmp_path, [7, 11], [11, 33])
        slab = _gauss(p0, "stress_xx", ids=[7, 33])
        np.testing.assert_array_equal(slab.element_index, [7, 33])

    def test_shared_node_still_reads(self, tmp_path: Path) -> None:
        """Both parts hold nodes 1..4; the node read merges them."""
        with MPCOMultiPartitionReader(
            _mpco_parts(tmp_path, [7, 11], [11, 33]),
        ) as r:
            slab = r.read_nodes(r.stages()[0].id, "displacement_x")
        np.testing.assert_array_equal(slab.node_ids, [1, 2, 3, 4])


# =====================================================================
# .ladruno partitioned (truss2d.part-0 holds element 1, part-1 element 2,
# both hold node 2)
# =====================================================================

def _truss_parts(tmp_path: Path, part1_element: int) -> list[Path]:
    out = []
    for i in (0, 1):
        dst = tmp_path / f"truss2d.part-{i}.ladruno"
        shutil.copy(_FIX / f"truss2d.part-{i}.ladruno", dst)
        out.append(dst)
    with h5py.File(out[1], "r+") as f:
        f[f"{_STAGE}/RESULTS/ON_ELEMENTS/basicForce/12-Truss[1:0:0]/ID"][...] = (
            np.array([[part1_element]])
        )
    return out


class TestLadrunoPartitioned:
    def test_element_in_two_parts_refuses(self, tmp_path: Path) -> None:
        with LadrunoMultiPartitionReader(_truss_parts(tmp_path, 1)) as r:
            with pytest.raises(MPCOElementInTwoBuckets) as info:
                r.read_elements(r.stages()[0].id, "basicForce")
        msg = str(info.value)
        assert "element 1 comes back from two partition files" in msg
        assert "truss2d.part-0.ladruno and truss2d.part-1.ladruno" in msg

    def test_disjoint_parts_read_unchanged(self, tmp_path: Path) -> None:
        with LadrunoMultiPartitionReader(_truss_parts(tmp_path, 2)) as r:
            slab = r.read_elements(r.stages()[0].id, "basicForce")
        np.testing.assert_array_equal(slab.element_ids, [1, 2])

    def test_shared_node_still_reads(self, tmp_path: Path) -> None:
        """Node 2 sits in both parts; it reads once, beside the refusal."""
        with LadrunoMultiPartitionReader(_truss_parts(tmp_path, 1)) as r:
            slab = r.read_nodes(r.stages()[0].id, "displacement_x")
        np.testing.assert_array_equal(slab.node_ids, [1, 2, 3])


# =====================================================================
# .ladruno single file: read_element_slab's bucket stitch
# =====================================================================

_FORCE = f"{_STAGE}/RESULTS/ON_ELEMENTS/force"
_QUAD_KEY = "31-FourNodeQuad[201:0:0]"
_QUAD_KEY_1 = "31-FourNodeQuad[201:0:1]"


def _quad_force(path: Path, *, copy_ids: "list[int] | None" = None,
                ids_in_one: "list[int] | None" = None) -> Path:
    """``quad2d.ladruno`` (quad element 1). ``copy_ids`` adds a second
    ``force`` bucket with those ids; ``ids_in_one`` rewrites the one
    bucket to hold those ids (its row repeated)."""
    shutil.copy(_FIX / "quad2d.ladruno", path)
    with h5py.File(path, "r+") as f:
        token = f[_FORCE]
        if copy_ids is not None:
            token.copy(_QUAD_KEY, _QUAD_KEY_1)
            g = token[_QUAD_KEY_1]
            d = g["DATA"][...]
            del g["DATA"], g["ID"]
            g.create_dataset("DATA", data=np.repeat(d, len(copy_ids), axis=1))
            g.create_dataset("ID", data=np.array(copy_ids)[:, None])
        if ids_in_one is not None:
            g = token[_QUAD_KEY]
            d = g["DATA"][...]
            del g["DATA"], g["ID"]
            g.create_dataset("DATA", data=np.repeat(d, len(ids_in_one), axis=1))
            g.create_dataset("ID", data=np.array(ids_in_one)[:, None])
    return path


def _force(path: Path):
    with LadrunoReader(path) as r:
        return r.read_elements(r.stages()[0].id, "force")


class TestLadrunoElementStitch:
    def test_element_in_two_buckets_refuses(self, tmp_path: Path) -> None:
        path = _quad_force(tmp_path / "dup.ladruno", copy_ids=[1])
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _force(path)
        msg = str(info.value)
        assert msg.startswith("dup.ladruno: element read of token 'force'")
        assert "element 1 appears in two .ladruno buckets" in msg
        assert f"{_FORCE}/{_QUAD_KEY}" in msg and f"{_FORCE}/{_QUAD_KEY_1}" in msg

    def test_element_twice_in_one_bucket_refuses(self, tmp_path: Path) -> None:
        path = _quad_force(tmp_path / "twice.ladruno", ids_in_one=[1, 1])
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _force(path)
        assert "element 1 appears twice in the .ladruno bucket" in str(info.value)

    def test_disjoint_buckets_read_unchanged(self, tmp_path: Path) -> None:
        path = _quad_force(tmp_path / "ok.ladruno", copy_ids=[9])
        slab = _force(path)
        np.testing.assert_array_equal(slab.element_ids, [1, 9])
        assert slab.values.shape == (2, 2, 8)

    def test_a_read_that_skips_the_repeat_answers(self, tmp_path: Path) -> None:
        path = _quad_force(tmp_path / "dup.ladruno", copy_ids=[1, 9])
        with LadrunoReader(path) as r:
            slab = r.read_elements(
                r.stages()[0].id, "force", element_ids=np.array([9]),
            )
        np.testing.assert_array_equal(slab.element_ids, [9])


# =====================================================================
# .ladruno single file: the nodal-force decode stitches buckets too
# =====================================================================

_LOCAL_FORCE = f"{_STAGE}/RESULTS/ON_ELEMENTS/localForce"
_BEAM_KEY = "5-ElasticBeam3d[1:0:0]"
_BEAM_KEY_9 = "5-ElasticBeam3d[1:0:9]"


def _beam_dup(path: Path) -> Path:
    """``beam3d.ladruno`` with its localForce bucket (element 1) copied
    to a second key: element 1 sits in two buckets of the token."""
    shutil.copy(_FIX / "beam3d.ladruno", path)
    with h5py.File(path, "r+") as f:
        f[_LOCAL_FORCE].copy(_BEAM_KEY, _BEAM_KEY_9)
    return path


class TestLadrunoNodalForceStitch:
    @pytest.mark.parametrize("component", [
        "nodal_resisting_force_local_x", "nodal_resisting_moment_local_z",
    ])
    def test_decoded_read_refuses(self, tmp_path: Path, component: str) -> None:
        """Before the fix these read element_ids [1, 1] with no word,
        while the raw ``localForce`` read refused."""
        path = _beam_dup(tmp_path / "dup_beam.ladruno")
        with LadrunoReader(path) as r:
            with pytest.raises(MPCOElementInTwoBuckets) as info:
                r.read_elements(r.stages()[0].id, component)
        msg = str(info.value)
        assert msg.startswith(
            f"dup_beam.ladruno: element read of {component!r} "
            f"(token 'localForce')",
        )
        assert f"{_LOCAL_FORCE}/{_BEAM_KEY}" in msg
        assert f"{_LOCAL_FORCE}/{_BEAM_KEY_9}" in msg

    def test_raw_token_read_refuses(self, tmp_path: Path) -> None:
        path = _beam_dup(tmp_path / "dup_beam.ladruno")
        with LadrunoReader(path) as r:
            with pytest.raises(MPCOElementInTwoBuckets, match="element 1 "):
                r.read_elements(r.stages()[0].id, "localForce")

    def test_undoubled_file_reads(self, tmp_path: Path) -> None:
        shutil.copy(_FIX / "beam3d.ladruno", tmp_path / "ok.ladruno")
        with LadrunoReader(tmp_path / "ok.ladruno") as r:
            slab = r.read_elements(
                r.stages()[0].id, "nodal_resisting_force_local_x",
            )
        np.testing.assert_array_equal(slab.element_ids, [1])


# =====================================================================
# MPCO partitioned: the line-station, fiber, layer and spring stitches
# =====================================================================

def _multi_read(paths: list[Path], method: str, component: str):
    with MPCOMultiPartitionReader(paths) as r:
        return getattr(r, method)(r.stages()[0].id, component)


def _assert_two_parts(info, eid: int) -> None:
    msg = str(info.value)
    assert f"element {eid} comes back from two partition files" in msg
    assert "part-0.mpco and " in msg and "part-1.mpco" in msg


class TestMPCOPartitionedStitches:
    def test_line_stations(self, tmp_path: Path) -> None:
        paths = [
            _section_force_file(tmp_path / "L.part-0.mpco", [1, 2], []),
            _section_force_file(tmp_path / "L.part-1.mpco", [2, 3], []),
        ]
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _multi_read(paths, "read_line_stations", "axial_force")
        _assert_two_parts(info, 2)

    def test_line_stations_disjoint(self, tmp_path: Path) -> None:
        paths = [
            _section_force_file(tmp_path / "L.part-0.mpco", [1, 2], []),
            _section_force_file(tmp_path / "L.part-1.mpco", [3], []),
        ]
        slab = _multi_read(paths, "read_line_stations", "axial_force")
        assert sorted(set(slab.element_index.tolist())) == [1, 2, 3]

    def test_fibers(self, tmp_path: Path) -> None:
        paths = [
            _fiber_file(tmp_path / "F.part-0.mpco", [10, 11], [12]),
            _fiber_file(tmp_path / "F.part-1.mpco", [11], [13]),
        ]
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _multi_read(paths, "read_fibers", "fiber_stress")
        _assert_two_parts(info, 11)

    def test_fibers_disjoint(self, tmp_path: Path) -> None:
        paths = [
            _fiber_file(tmp_path / "F.part-0.mpco", [10, 11], [12]),
            _fiber_file(tmp_path / "F.part-1.mpco", [14], [13]),
        ]
        slab = _multi_read(paths, "read_fibers", "fiber_stress")
        assert sorted(set(slab.element_index.tolist())) == [10, 11, 12, 13, 14]

    def test_springs(self, tmp_path: Path) -> None:
        paths = [
            _make_spring_mpco(
                tmp_path / f"S.part-{i}.mpco", element_ids=ids, n_springs=1,
                force_values=np.full((2, 2, 1), 3.0),
            )
            for i, ids in enumerate(([1, 2], [2, 3]))
        ]
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _multi_read(paths, "read_springs", "spring_force")
        _assert_two_parts(info, 2)

    def test_layers(self, tmp_path: Path, simple_layer_mpco) -> None:  # noqa: F811
        """Both parts are a copy of the ASDShellQ4 file (elements 20, 21)."""
        paths = []
        for i in (0, 1):
            dst = tmp_path / f"Y.part-{i}.mpco"
            shutil.copy(simple_layer_mpco["path"], dst)
            paths.append(dst)
        with pytest.raises(MPCOElementInTwoBuckets) as info:
            _multi_read(paths, "read_layers", "fiber_stress")
        _assert_two_parts(info, 20)
