"""Build the B-spline grid fixtures of #3013 from the fork's fixture.

Fork release ``ladruno-v20261118`` (WP-232i, OpenSees#1099) amends 232a §2.4:
in Contract 2, ``MODEL/PARTICLES/GRID/BASIS`` is exactly one of
``gimp | linear | bspline2 | bspline3``, with no ``FORMAT_MINOR_VERSION``
bump. The fork commits no ``.ladruno`` file with a B-spline grid, so these
are ``particles_v1m2.ladruno`` (the fork's fixture, vendored here) with only
that attribute rewritten, in the fork's encoding of it: a 1-element array of
a fixed-length, null-terminated ASCII string as long as the value (the
vendored file stores ``gimp`` as ``|S4``). Nothing else changes: a B-spline
grid has the same four attributes (§2.4), and the outputs stay the GIMP
run's, so these files pin what a reader *accepts*, not what a B-spline run
computes (the live test, ``test_mpm_bspline_readers_live.py``, runs one).

- ``particles_bspline2_v1m2.ladruno``: ``BASIS = "bspline2"``.
- ``particles_bspline3_v1m2.ladruno``: ``BASIS = "bspline3"``.

The refused values (``bspline4`` and others in Contract 2, ``bspline2`` in
Contract 1) are edited copies each test suite makes in a temp folder, the same
values in Python (:func:`set_basis`) and in the viewer (a variable-length
string), never committed.

Run from the repository root: ``python tests/fixtures/mpm/make_basis_fixtures.py``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "particles_v1m2.ladruno"
GRID = "MODEL_STAGE[1]/MODEL/PARTICLES/GRID"


def set_basis(path: Path, value: str) -> None:
    """Rewrite ``GRID/BASIS`` of the copy at ``path`` in the fork's encoding."""
    raw = value.encode("ascii")
    with h5py.File(path, "r+") as f:
        grid = f[GRID]
        old = grid.attrs.get_id("BASIS").get_type()
        if old.is_variable_str() or old.get_strpad() != h5py.h5t.STR_NULLTERM:
            raise ValueError(f"{path}: BASIS is not the fork's fixed NULLTERM string")
        del grid.attrs["BASIS"]
        tid = h5py.h5t.C_S1.copy()
        tid.set_size(len(raw))
        tid.set_strpad(h5py.h5t.STR_NULLTERM)
        tid.set_cset(h5py.h5t.CSET_ASCII)
        space = h5py.h5s.create_simple((1,))
        attr = h5py.h5a.create(grid.id, b"BASIS", tid, space)
        attr.write(np.array([raw], dtype=f"S{len(raw)}"), mtype=tid)


def build(path: Path, value: str) -> None:
    shutil.copyfile(SOURCE, path)
    set_basis(path, value)


if __name__ == "__main__":
    for value in ("bspline2", "bspline3"):
        name = f"particles_{value}_v1m2.ladruno"
        build(HERE / name, value)
        print(name)
