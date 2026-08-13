"""ADR 0092 S6 — the ``Results`` ownership contract (INV-6), measured.

S4 proved the partitioned contact deck's SHAPE, S5 proved its NUMBERS.
S6 closes the loop on the READ side: a 2-rank contact run writes one
``.ladruno`` per rank, and ``Results`` must stitch them back into the
answer the serial twin gives.

INV-6 says the read side "already tolerates single-rank output" and that
the work reduces to a regression test. Reading the stitch showed the
contract actually rests on **three** things, only one of which was
pinned anywhere:

1. **The filename grammar is a CROSS-LIBRARY contract, and nothing
   executed it.** apeGmsh emits the recorder line ONCE, globally, with
   the filename verbatim (``apesees.py`` step 3 — "one recorder is
   sufficient even under MP"); the ``<stem>.part-<K>.ladruno`` suffix
   that :func:`discover_partition_files` matches is added by the FORK
   at ``SRC/recorder/LadrunoRecorder.cpp`` (it detects partitioning via
   ``send_self_count`` or an MPI launcher's ``PMI_SIZE``/``OMPI_*``/
   ``SLURM_NTASKS`` pair, then rewrites the stem). Both sides were
   tested against their own assumption; the two had never met. If the
   fork's spelling drifted, every partitioned read would silently fall
   back to reading ONE rank — a wrong answer that looks like a small
   model.

2. **The node stitch dedupes first-write-wins** (``_merge_node_slabs``
   / ``_merge_partition_fems``) — and a contact GHOST is exactly the
   duplicate that exercises it. The owner rank declares ghosts of the
   interface nodes its neighbour natively owns (INV-2), so those nodes
   are recorded TWICE, once per rank. Collapsing them is only correct
   because the two copies agree, which fork ADR-78 P0 and apeGmsh S5
   measured as bit-identical. Nothing in the reader checks it: if the
   ghost's value ever diverged, first-write-wins would silently pick
   whichever rank sorted first. This suite asserts the duplication is
   real (both files carry the node) AND that the stitched answer is the
   serial one.

3. **The element stitch does NOT dedupe** (``_concat_element_slabs``
   concatenates, assuming partitions hold disjoint elements). That is
   safe only because INV-7 forbids a ghost from carrying elements. The
   read side has no defence of its own, so this suite pins the
   disjointness in the produced files.

"Contact families" resolves to (b), ordinary node/element results on a
model that HAS contact: the fork's contact subsystem exposes no
recordable response at all (``LadrunoContactHandler`` is a
``ConstraintHandler``, ``LadrunoContactFE`` an analysis-layer
``FE_Element``, ``LadrunoContactDomain`` not even a ``TaggedObject`` —
no ``setResponse``/``getResponse`` anywhere, and ``element/contact.py``
defines no ``Element`` class, so there is no tag a recorder could
target). What contact changes is not WHAT is recorded but WHO records
it — which is precisely the ownership contract under test.

Same gating as the S5 twin (``APEGMSH_OPENSEES_BIN`` + Intel MPI); CI
excludes ``subprocess`` by construction, so CI skips this file.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.results import Results
from apeGmsh.results.readers._protocol import ResultLevel

# The S5 harness owns the model, the probes and the MPI launch recipe;
# S6 is the same model read back, so it imports rather than re-states.
from tests.opensees.subprocess.test_contact_partitioned_numeric_twin import (
    _build_fem,
    _dist_bin,
    _mpiexec,
    _probes,
    _run_env,
    RUN_TIMEOUT_S,
)

REL_TOL = 1.0e-10

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason=(
            "APEGMSH_OPENSEES_BIN unset or does not hold OpenSees.exe + "
            "OpenSeesMP.exe — point it at a Ladruno-fork dist\\bin to run "
            "the ADR 0092 S6 results-stitch suite"
        ),
    ),
    pytest.mark.skipif(
        _mpiexec() is None,
        reason=(
            "Intel MPI mpiexec.exe not found (set I_MPI_ROOT, or install "
            "oneAPI at the default location) — needed for the 2-rank lane"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Model — the S5 twin plus a whole-model .ladruno recorder
# ---------------------------------------------------------------------------


def _build_ops_recorded(fem, *, parallel_chain: bool, out_name: str) -> apeSees:
    """The S5 model with a recorder attached.

    The constraint handler is deliberately NOT declared: it rides the
    auto-emit, which (since the ADR 0092 S5 open-item fix) is hoisted
    above the user's ``analysis`` line and is therefore effective. So
    this suite is also a live consumer of that fix — if the hoist
    regressed, these decks would run PlainHandler and the numbers would
    stop matching the serial twin.

    The recorder carries NO filter, so every node the rank holds is
    recorded — including the owner's contact ghosts, which is the point.
    """
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e7, nu=0.0)
    ops.element.stdBrick(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    for pg in ("master", "slave", "top"):
        ops.fix(pg=pg, dofs=(1, 1, 0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="top", forces=(0.0, 0.0, -2500.0))
    if parallel_chain:
        ops.numberer.ParallelPlain()
        ops.system.Mumps()
    else:
        ops.numberer.RCM()
        ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    # Bare filename → lands beside the deck (the runners use cwd=deck.parent).
    ops.recorder.Ladruno(
        file=out_name,
        nodal_responses=("displacement",),
        elem_responses=("stresses",),
    )
    return ops


_RUN_FRAGMENT = """
set ok [analyze 1]
puts [format "S6 ok=%d" $ok]
wipe
"""

_OK_RE = re.compile(r"S6 ok=(-?\d+)")


def _append_run(deck: Path) -> None:
    with open(deck, "a", encoding="utf-8") as f:
        f.write(_RUN_FRAGMENT)


def _assert_converged(out: str, lane: str) -> None:
    oks = [int(m) for m in _OK_RE.findall(out)]
    assert oks, f"{lane}: deck printed no S6 line:\n{out[-2000:]}"
    assert all(o == 0 for o in oks), f"{lane}: analyze failed {oks}\n{out[-2000:]}"


def _run(cmd: list[str], deck: Path, env: "dict[str, str]") -> str:
    r = subprocess.run(
        cmd, cwd=deck.parent, env=env, capture_output=True, text=True,
        timeout=RUN_TIMEOUT_S,
    )
    return r.stdout + r.stderr


# ---------------------------------------------------------------------------
# The two recorded runs — built once per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, object]":
    dist, mpi = _dist_bin(), _mpiexec()
    assert dist is not None and mpi is not None
    env = _run_env(dist)

    fem = _build_fem("s6_results_stitch")
    probes = _probes(fem)

    d = tmp_path_factory.mktemp("s6")
    serial_deck, mp_deck = d / "serial.tcl", d / "mp.tcl"
    _build_ops_recorded(
        fem, parallel_chain=False, out_name="s6_serial.ladruno",
    ).tcl(str(serial_deck), flat=True)
    _build_ops_recorded(
        fem, parallel_chain=True, out_name="s6_mp.ladruno",
    ).tcl(str(mp_deck))
    _append_run(serial_deck)
    _append_run(mp_deck)

    serial_out = _run([str(dist / "OpenSees.exe"), str(serial_deck)],
                      serial_deck, env)
    _assert_converged(serial_out, "serial")
    mp_out = _run(
        [str(mpi), "-n", "2", str(dist / "OpenSeesMP.exe"), str(mp_deck)],
        mp_deck, env,
    )
    _assert_converged(mp_out, "mp")

    return {
        "dir": d, "fem": fem, "probes": probes,
        "serial_out": serial_out, "mp_out": mp_out,
    }


# ---------------------------------------------------------------------------
# (1) the cross-library filename contract
# ---------------------------------------------------------------------------


def test_fork_partition_filenames_match_the_reader_grammar(recorded) -> None:
    """The fork's per-rank suffix is what ``discover_partition_files``
    matches — the contract neither library could test alone."""
    from apeGmsh.results.readers._ladruno_multi import (
        discover_partition_files,
    )

    d: Path = recorded["dir"]           # type: ignore[assignment]
    produced = sorted(p.name for p in d.glob("s6_mp*.ladruno"))
    assert produced == ["s6_mp.part-0.ladruno", "s6_mp.part-1.ladruno"], (
        "the fork did not write the per-rank filenames apeGmsh's reader "
        f"grammar expects (found {produced}). apeGmsh emits the recorder "
        "line ONCE with the name verbatim; the '.part-<K>' suffix comes "
        "from SRC/recorder/LadrunoRecorder.cpp. A drift here makes every "
        "partitioned read silently return ONE rank's slice."
    )
    # The serial run must NOT be suffixed (single process → verbatim).
    assert (d / "s6_serial.ladruno").is_file()

    found = discover_partition_files(d / "s6_mp.part-0.ladruno")
    assert [p.name for p in found] == produced


# ---------------------------------------------------------------------------
# (2) the ghost is a real duplicate, and the stitch collapses it correctly
# ---------------------------------------------------------------------------


def _node_slab_ids_values(path, component: str):
    r = Results.from_ladruno(path)
    slab = r.nodes.get(component=component)
    return slab.node_ids, slab.values


def test_contact_ghost_is_recorded_on_both_ranks_with_equal_values(
    recorded,
) -> None:
    """The duplicate is real, and the two copies agree.

    Two jobs. First, non-vacuity for the dedupe test below: the
    interface nodes really are recorded on BOTH ranks (owner ghost +
    native owner) — measured here as the 4 slave-face nodes, rank 0
    recording 12 nodes for the 8 it owns.

    Second, and the sharper pin: ``_merge_node_slabs`` keeps whichever
    copy it meets FIRST and silently drops the rest, so the stitch is
    only correct if ghost == native. That equality is a fork property
    (ADR-78 P0 / apeGmsh S5 measured it bit-identical at the solver
    level); this asserts it survives all the way to the recorder, which
    is what the reader actually consumes.
    """
    from apeGmsh.results.readers._ladruno import LadrunoReader

    d: Path = recorded["dir"]           # type: ignore[assignment]
    probes = recorded["probes"]
    slave_nodes = set(probes["slave"])  # type: ignore[index]

    per_rank: list[dict[int, np.ndarray]] = []
    for name in ("s6_mp.part-0.ladruno", "s6_mp.part-1.ladruno"):
        with LadrunoReader(d / name) as r:
            slab = r.read_nodes("stage_0", "displacement_z")
            per_rank.append({
                int(n): slab.values[:, i]
                for i, n in enumerate(slab.node_ids)
            })

    both = set(per_rank[0]) & set(per_rank[1]) & slave_nodes
    assert both, (
        "no slave-interface node is recorded on both ranks — the contact "
        "ghost machinery (INV-2) did not produce the duplicate this "
        f"suite exists to test. rank0={sorted(per_rank[0])} "
        f"rank1={sorted(per_rank[1])} slave={sorted(slave_nodes)}"
    )
    for nid in sorted(both):
        np.testing.assert_allclose(
            per_rank[0][nid], per_rank[1][nid], rtol=1e-12, atol=1e-14,
            err_msg=(
                f"ghost node {nid}: the owner rank's recorded copy differs "
                "from its native rank's. _merge_node_slabs keeps the first "
                "and drops the second WITHOUT comparing them, so this "
                "divergence would reach users as a silently-wrong stitched "
                "value (ADR 0092 INV-6 / INV-2)."
            ),
        )


def test_stitched_nodes_dedupe_ghosts_and_match_serial(recorded) -> None:
    """One row per node id, and the values are the serial answer.

    First-write-wins in ``_merge_node_slabs`` is only correct because a
    ghost and its native copy agree (fork ADR-78 P0 / apeGmsh S5
    measured bit-identical). This asserts the consequence end-to-end.
    """
    d: Path = recorded["dir"]           # type: ignore[assignment]

    for component in ("displacement_x", "displacement_z"):
        s_ids, s_vals = _node_slab_ids_values(
            d / "s6_serial.ladruno", component)
        m_ids, m_vals = _node_slab_ids_values(
            d / "s6_mp.part-0.ladruno", component)

        assert len(set(m_ids.tolist())) == m_ids.size, (
            f"{component}: stitched node ids contain duplicates — the "
            "ghost dedupe in _merge_node_slabs did not collapse the "
            "owner's contact ghosts"
        )
        assert m_ids.tolist() == s_ids.tolist(), (
            f"{component}: stitched node set != serial node set\n"
            f"mp={m_ids.tolist()}\nserial={s_ids.tolist()}"
        )
        assert not np.isnan(m_vals).any(), (
            f"{component}: stitched values carry NaN — a node id survived "
            "the union with no partition filling it"
        )
        np.testing.assert_allclose(
            m_vals, s_vals, rtol=REL_TOL, atol=1e-14,
            err_msg=(
                f"{component}: the 2-rank stitch disagrees with the serial "
                "run. If only the interface nodes differ, the ghost and its "
                "native copy diverged and first-write-wins silently picked "
                "one (ADR 0092 INV-6 / INV-2)."
            ),
        )


# ---------------------------------------------------------------------------
# (3) element stitch — disjoint by rank, so concat cannot double-count
# ---------------------------------------------------------------------------


def test_element_results_are_disjoint_across_ranks(recorded) -> None:
    """``_concat_element_slabs`` concatenates without dedupe, so a
    ghost that carried elements (INV-7 violation) would double-count
    silently. Pin the disjointness the read side depends on."""
    from apeGmsh.results.readers._ladruno import LadrunoReader

    d: Path = recorded["dir"]           # type: ignore[assignment]

    per_rank: list[set[int]] = []
    component = None
    for name in ("s6_mp.part-0.ladruno", "s6_mp.part-1.ladruno"):
        with LadrunoReader(d / name) as r:
            if component is None:
                comps = r.available_components("stage_0", ResultLevel.GAUSS)
                assert comps, f"{name}: no gauss components recorded"
                component = comps[0]
            slab = r.read_gauss("stage_0", component)
            # GaussSlab keys rows by ``element_index`` (one row per GP).
            per_rank.append({int(e) for e in slab.element_index})

    assert per_rank[0] and per_rank[1], (
        f"a rank recorded no elements at all: {per_rank}")
    overlap = per_rank[0] & per_rank[1]
    assert not overlap, (
        f"element(s) {sorted(overlap)} are recorded on BOTH ranks — the "
        "element stitch concatenates without dedupe, so this would "
        "double-count in every partitioned read (ADR 0092 INV-7: a ghost "
        "carries geometry and SP state only, never elements)"
    )

    with LadrunoReader(d / "s6_serial.ladruno") as r:
        serial_ids = {
            int(e) for e in r.read_gauss("stage_0", component).element_index
        }
    assert per_rank[0] | per_rank[1] == serial_ids, (
        "the union of the ranks' recorded elements != the serial run's")


# ---------------------------------------------------------------------------
# (4) INV-6 proper — a rank with no element results drops out of the
#     stitch; loud only when EVERY rank is missing
# ---------------------------------------------------------------------------


def _strip_element_results(src: Path, dst: Path) -> Path:
    import h5py

    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        for k in (k for k in f if k.startswith("MODEL_STAGE[")):
            results = f[k]["RESULTS"]
            if "ON_ELEMENTS" in results:
                del results["ON_ELEMENTS"]
    return dst


def test_stitch_survives_a_rank_with_no_element_results(
    recorded, tmp_path: Path,
) -> None:
    """The INV-6 shape on REAL contact output rather than a synthesized
    manifest: one rank keeps its element results, the other has none."""
    from apeGmsh.results.readers._ladruno_element_io import (
        MissingElementResults,
    )
    from apeGmsh.results.readers._ladruno_multi import (
        LadrunoMultiPartitionReader,
    )

    d: Path = recorded["dir"]           # type: ignore[assignment]
    p0 = tmp_path / "s6_mp.part-0.ladruno"
    p1 = tmp_path / "s6_mp.part-1.ladruno"
    shutil.copy(d / "s6_mp.part-0.ladruno", p0)
    _strip_element_results(d / "s6_mp.part-1.ladruno", p1)

    with LadrunoMultiPartitionReader([p0, p1]) as r:
        comps = r.available_components("stage_0", ResultLevel.GAUSS)
        assert comps, "rank 0 still carries element results"
        slab = r.read_gauss("stage_0", comps[0])
        assert slab.element_index.size > 0

    # Strip BOTH → nothing survives, so the read is loud again.
    _strip_element_results(d / "s6_mp.part-0.ladruno", p0)
    with LadrunoMultiPartitionReader([p0, p1]) as r:
        with pytest.raises(MissingElementResults):
            r.read_gauss("stage_0", comps[0])
