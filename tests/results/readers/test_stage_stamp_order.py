"""MODEL_STAGE enumeration order — numeric stamp, not lexicographic.

Both single-file readers enumerate ``MODEL_STAGE[<stamp>]`` groups.
Lexicographic sorting puts ``MODEL_STAGE[10]`` before ``MODEL_STAGE[2]``
once a run reaches ten stages, scrambling stage ids and the viewer's
stage timeline. The readers must sort by the numeric stamp. (The
multi-file readers delegate stage enumeration to reader 0, so covering
the single-file readers covers them too.)
"""
from __future__ import annotations

from pathlib import Path

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.schema._versions import LADRUNO_SUPPORTED_FORMAT_VERSIONS

# Stamps deliberately non-contiguous and crossing the 1→2-digit
# boundary: lexicographic order would be [10], [11], [2], [9].
STAMPS = (2, 9, 10, 11)


def _write_stage_groups(h5file) -> None:
    for stamp in STAMPS:
        h5file.create_group(f"MODEL_STAGE[{stamp}]")


def test_ladruno_stages_numeric_stamp_order(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "many_stages.ladruno"
    with h5py.File(path, "w") as h:
        info = h.create_group("INFO")
        info.attrs["GENERATOR"] = "Ladruno"
        info.attrs["FORMAT_VERSION"] = LADRUNO_SUPPORTED_FORMAT_VERSIONS[0]
        _write_stage_groups(h)

    with LadrunoReader(path) as r:
        stages = r.stages()
        assert [s.name for s in stages] == [
            f"MODEL_STAGE[{k}]" for k in STAMPS
        ]
        assert [s.id for s in stages] == [
            f"stage_{i}" for i in range(len(STAMPS))
        ]


def test_mpco_stages_numeric_stamp_order(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "many_stages.mpco"
    with h5py.File(path, "w") as h:
        _write_stage_groups(h)

    with MPCOReader(path) as r:
        stages = r.stages()
        assert [s.name for s in stages] == [
            f"MODEL_STAGE[{k}]" for k in STAMPS
        ]
        assert [s.id for s in stages] == [
            f"stage_{i}" for i in range(len(STAMPS))
        ]
