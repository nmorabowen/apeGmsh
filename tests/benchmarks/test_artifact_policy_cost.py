"""The cost of the D1 overwrite policy's hash (#1307, P4).

``artifact_verdict`` compares the target's neutral content with what the
write would produce: ``content_hash`` writes the neutral zone to an
in-memory HDF5 file and walks it, ``artifact_content_hash`` walks the
target.  Both are paid only when the target is this run's fuller file
(the bridge wrote first).  This case measures each half on a mesh of a
few tens of thousands of tetrahedra and reports them against the disk
write they gate.  The gate is a per-element budget, 10 us for the two
halves together (measured: 3.6 us on the maintainer's machine, where
the disk write is 1.4 us per element), so a return to per-element
hashing of the string columns (the lineage walk's ``repr`` route, 7 us
and non-deterministic) fails it.

Run with ``pytest -m bench tests/benchmarks/test_artifact_policy_cost.py -s``.
"""
from __future__ import annotations

import copy
import statistics
import time
from pathlib import Path

import h5py
import pytest

from apeGmsh import apeGmsh
from apeGmsh._artifact_policy import (
    artifact_content_hash,
    artifact_verdict,
    content_hash,
    provenance_scripts,
)
from apeGmsh.opensees._internal.schema_version import NEUTRAL, PROVENANCE

REPS = 3


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


@pytest.mark.bench
def test_policy_hash_costs_no_more_than_the_write_it_gates(tmp_path: Path) -> None:
    with apeGmsh(model_name="bench", _artifacts=False) as g:
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="body")
        g.physical.add_volume("body", name="body")
        g.loads.point.force(pg="body", force=(0.0, 0.0, -1.0))
        g.masses.point(pg="body", mass=1.0)
        g.mesh.sizing.set_global_size(0.08)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data()
    n_elems = fem.info.n_elems
    target = tmp_path / "bench.h5"
    fem.to_h5(str(target))

    def fresh():
        f = copy.copy(fem)
        del f._snapshot_id_cache
        return f

    hashes = [_timed(lambda: content_hash(fresh())) for _ in range(REPS)]
    reads = [_timed(lambda: artifact_content_hash(target)) for _ in range(REPS)]
    writes = [_timed(lambda: fresh().to_h5(str(target))) for _ in range(REPS)]
    # the full verdict on this run's fuller file: the only path that hashes
    with h5py.File(target, "a") as f:
        f.create_group("opensees")
    checks = [
        _timed(lambda: artifact_verdict(
            target, writes=frozenset({NEUTRAL, PROVENANCE}), overwrite=True,
            session_id=fem.session_id, content=lambda: content_hash(fem),
            scripts=provenance_scripts(fem.provenance), explicit=False,
        ))
        for _ in range(REPS)
    ]
    h, r, w, c = (statistics.median(x) for x in (hashes, reads, writes, checks))
    per_elem_us = (h + r) / n_elems * 1e6
    print(
        f"\n[artifact-policy] {n_elems} elements: content_hash {h * 1e3:.1f} ms, "
        f"target hash {r * 1e3:.1f} ms, verdict {c * 1e3:.1f} ms, "
        f"disk write {w * 1e3:.1f} ms ((hash + target) / write = {(h + r) / w:.2f}, "
        f"{per_elem_us:.2f} us per element)"
    )
    assert n_elems > 20_000, n_elems
    assert per_elem_us < 10.0, (h, r, n_elems)
