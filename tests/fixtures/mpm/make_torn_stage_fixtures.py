"""Build the torn-stage fixtures of M2-S7 (#2991) from the fork's fixture.

A kill while a stage is laid out (the first stage, or an ``mpm reset``) can
leave that stage torn: 232a §2.5 at fork ``a1d4c7260``, "The reader rule:
opening a particle file". These files model the torn stage at the layout
level, from ``particles_v1m2.ladruno`` (the fork's fixture, vendored here):
two particle stages in a version-3 superblock file (``libver="latest"``, the
superblock that carries the consistency flags the reader rule reads), closed
cleanly, so their flags read 0x00. A test copies one and sets the flags to
0x05 or 0x01 to reach steps 2 and 3 of the open.

- ``particles_torn_last_stage_v1m2.ladruno``: ``MODEL_STAGE[2]`` (epoch 1) was
  torn while it opened: its registry holds ``ID`` and ``BODY_ID`` but no
  ``MATERIAL_TAG`` or ``MASS`` row yet (the crash rule's step 0), and no
  output. ``MODEL_STAGE[1]`` is the fork fixture's stage, 3 outputs.
- ``particles_torn_early_stage_v1m2.ladruno``: the same torn stage first
  (``MODEL_STAGE[1]``, epoch 0), then a whole stage (``MODEL_STAGE[2]``,
  epoch 1, 3 outputs): an earlier stage that does not read is corruption.
- ``build(path, "half")`` (not committed): ``MODEL_STAGE[2]`` was killed
  after ``MODEL/PARTICLES`` was written and before ``RESULTS/ON_PARTICLES``
  was created (it does not open as a particle stage, C2-PAIR). The tests
  build it in a temp folder: ``LadrunoReader`` refuses it at open (C2-PAIR),
  and every committed ``.ladruno`` must open
  (``tests/results/test_ladruno_duplicate_node_ids.py``).

Run from the repository root: ``python tests/fixtures/mpm/make_torn_stage_fixtures.py``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "particles_v1m2.ladruno"
STAGE = "MODEL_STAGE[1]"


def _tear_registry(stage: h5py.Group) -> None:
    """Killed inside the registry write: ID and BODY_ID hold their rows,
    MATERIAL_TAG and MASS none yet, and no output was written."""
    model = stage["MODEL/PARTICLES"]
    for name in ("MATERIAL_TAG", "MASS"):
        model[name].resize(0, axis=0)
    on = stage["RESULTS/ON_PARTICLES"]
    for name in ("STEP", "TIME", "OFFSETS", "ID"):
        on[name].resize(0, axis=0)
    for name in on:
        if isinstance(on[name], h5py.Group):
            on[name]["DATA"].resize(0, axis=0)


def _stage(dst: h5py.File, src: h5py.File, name: str, epoch: int) -> h5py.Group:
    src.copy(src[STAGE], dst, name=name)
    g = dst[name]
    g.attrs["PARTICLE_EPOCH"] = np.array([epoch], dtype=np.int32)
    return g


def build(path: Path, kind: str) -> None:
    with h5py.File(SOURCE, "r") as src, \
            h5py.File(path, "w", libver="latest") as dst:
        src.copy(src["INFO"], dst, name="INFO")
        first = _stage(dst, src, "MODEL_STAGE[1]", 0)
        second = _stage(dst, src, "MODEL_STAGE[2]", 1)
        if kind == "last":
            _tear_registry(second)
        elif kind == "early":
            _tear_registry(first)
        elif kind == "half":
            del second["RESULTS/ON_PARTICLES"]
        else:
            raise ValueError(kind)


if __name__ == "__main__":
    for kind, name in (("last", "particles_torn_last_stage_v1m2.ladruno"),
                       ("early", "particles_torn_early_stage_v1m2.ladruno")):
        build(HERE / name, kind)
        print(name)
