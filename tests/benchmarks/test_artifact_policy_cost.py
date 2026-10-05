"""The cost of the D1 overwrite policy's hash (#1307, P4).

``artifact_target_is_ours`` compares the target's ``/meta/snapshot_id``
with the snapshot's ``fem_hash`` (``FEMData.snapshot_id``, lineage INV-1).
The hash is cached on the snapshot and the write that follows needs it
for ``/meta/snapshot_id`` anyway, so the policy's own cost is one
``h5py`` open of the target.  This case measures both halves on a
mesh of a few tens of thousands of tetrahedra, against the write they
gate: the hash plus the identity read must cost less than the write.

Run with ``pytest -m bench tests/benchmarks/test_artifact_policy_cost.py -s``.
"""
from __future__ import annotations

import copy
import statistics
import time
from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh._artifact_policy import artifact_target_is_ours, provenance_scripts
from apeGmsh.opensees._internal.schema_version import NEUTRAL, PROVENANCE

REPS = 3


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


@pytest.mark.bench
def test_policy_hash_costs_less_than_the_write_it_gates(tmp_path: Path) -> None:
    with apeGmsh(model_name="bench", _artifacts=False) as g:
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="body")
        g.physical.add_volume("body", name="body")
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

    hashes = [_timed(lambda: fresh().snapshot_id) for _ in range(REPS)]
    writes = [_timed(lambda: fresh().to_h5(str(target))) for _ in range(REPS)]
    checks = [
        _timed(lambda: artifact_target_is_ours(
            target, writes=frozenset({NEUTRAL, PROVENANCE}), overwrite=True,
            session_id=fem.session_id, fem_hash=fem.snapshot_id,
            scripts=provenance_scripts(fem.provenance), explicit=False,
        ))
        for _ in range(REPS)
    ]
    h, w, c = (statistics.median(x) for x in (hashes, writes, checks))
    print(
        f"\n[artifact-policy] {n_elems} elements: fem_hash {h * 1e3:.1f} ms, "
        f"identity read + rule {c * 1e3:.1f} ms, write {w * 1e3:.1f} ms "
        f"(hash / write = {h / w:.2f})"
    )
    assert n_elems > 20_000, n_elems
    assert h + c < w, (h, c, w)
