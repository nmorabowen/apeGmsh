"""Partitioned 3-D space frame — OpenSeesMP-ready Tcl deck from apeGmsh.

Builds a 2-storey x 2x2-bay rigid-jointed space frame, partitions the
mesh into 4 ranks via Gmsh-native METIS (Flavor A), declares one
rigid-floor diaphragm per storey so cross-partition replication per
ADR 0027 INV-1/INV-2 fires, and emits ``partitioned_frame.tcl``.

Run:
    python examples/partition_frame.py            # emit only
    python examples/partition_frame.py --run      # try mpiexec too

Units: SI (m, N, Pa).  See docs/guide_partitioning.md for the full
walk-through this script anchors.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NodeGroupRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees

N_STORIES, N_BAYS, BAY, H, LC = 2, 2, 5.0, 3.0, 1.5
N_PARTS = 4
OUT = Path("partitioned_frame.tcl")


def build_partitioned_fem():
    """Geometry + mesh + partition + cross-partition diaphragm records."""
    sess = apeGmsh(model_name="partition_frame", verbose=False)
    sess.begin()
    try:
        gm = sess.model.geometry
        pts: dict[tuple[int, int, int], int] = {
            (s, i, j): gm.add_point(i * BAY, j * BAY, s * H)
            for s in range(N_STORIES + 1)
            for i in range(N_BAYS + 1) for j in range(N_BAYS + 1)
        }
        masters = {
            s: gm.add_point(N_BAYS * BAY / 2, N_BAYS * BAY / 2, s * H)
            for s in range(1, N_STORIES + 1)
        }

        cols = [
            gm.add_line(pts[(s - 1, i, j)], pts[(s, i, j)])
            for s in range(1, N_STORIES + 1)
            for i in range(N_BAYS + 1) for j in range(N_BAYS + 1)
        ]
        beams = [
            gm.add_line(pts[(s, i, j)], pts[(s, i + dx, j + dy)])
            for s in range(1, N_STORIES + 1)
            for i in range(N_BAYS + 1) for j in range(N_BAYS + 1)
            for dx, dy in ((1, 0), (0, 1))
            if i + dx <= N_BAYS and j + dy <= N_BAYS
        ]
        sess.physical.add_curve(cols, name="Columns")
        sess.physical.add_curve(beams, name="Beams")
        sess.physical.add_point(
            [pts[(0, i, j)] for i in range(N_BAYS + 1)
             for j in range(N_BAYS + 1)], name="Base")
        for s in range(1, N_STORIES + 1):
            sess.physical.add_point(
                [pts[(s, i, j)] for i in range(N_BAYS + 1)
                 for j in range(N_BAYS + 1)], name=f"Floor{s}")
            sess.physical.add_point([masters[s]], name=f"Master{s}")

        sess.mesh.sizing.set_global_size(LC)
        sess.mesh.generation.generate(dim=1)
        sess.mesh.partitioning.renumber(dim=1, method="simple", base=1)
        info = sess.mesh.partitioning.partition(n_parts=N_PARTS)
        fem = sess.mesh.queries.get_fem_data(dim=1)
    finally:
        sess.end()

    # Inject one rigid_diaphragm per storey directly into the broker.
    # The public g.constraints.rigid_diaphragm requires Part labels;
    # this example uses physical groups, so we resolve PG nodes
    # against the FEM snapshot and append records directly — same
    # shape the ADR 0027 cross-partition replication path consumes.
    for s in range(1, N_STORIES + 1):
        master = list(fem.nodes.select(pg=f"Master{s}").ids)
        slaves = list(fem.nodes.select(pg=f"Floor{s}").ids)
        fem.nodes.constraints._records.append(NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            master_node=int(master[0]),
            slave_nodes=[int(n) for n in slaves],
            plane_normal=np.array([0.0, 0.0, 1.0]),
            dofs=[1, 2, 6], name=f"floor_{s}",
        ))
    return fem, info


def emit_deck(fem, out_path: Path) -> None:
    """Wire apeSees and emit the partitioned Tcl deck."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    t_col = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    t_beam = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    common = dict(A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.element.elasticBeamColumn(pg="Columns", transf=t_col, **common)
    ops.element.elasticBeamColumn(pg="Beams", transf=t_beam, **common)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg=f"Master{N_STORIES}", values=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg=f"Master{N_STORIES}",
               forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    # No explicit analysis chain — apeSees auto-emits
    # numberer ParallelPlain + system Mumps per ADR 0027 INV-5
    # when len(fem.partitions) > 1.
    ops.tcl(str(out_path))


def maybe_run_openseesmp(tcl_path: Path) -> None:
    """Try ``mpiexec -np N OpenSeesMP <tcl>`` — non-fatal if missing."""
    mpi = shutil.which("mpiexec") or shutil.which("mpirun")
    osmp = shutil.which("OpenSeesMP")
    if not (mpi and osmp):
        print(f"[runtime] mpiexec / OpenSeesMP not on PATH; skipping.")
        return
    cmd = [mpi, "-np", str(N_PARTS), osmp, str(tcl_path)]
    print(f"[runtime] {' '.join(cmd)}")
    rc = subprocess.run(cmd, timeout=60).returncode
    print(f"[runtime] OpenSeesMP exited {rc}")


def main(run: bool = False) -> None:
    fem, info = build_partitioned_fem()
    emit_deck(fem, OUT)
    print(f"\n=== partition_frame.py ===")
    print(f"  frame:         {N_STORIES} storey x {N_BAYS} x {N_BAYS} bays")
    print(f"  nodes:         {fem.info.n_nodes}")
    print(f"  elements:      {fem.info.n_elems}")
    print(f"  partitions:    {len(fem.partitions)} (requested {N_PARTS})")
    for rec in fem.partitions:
        print(f"    rank {rec.id}: {rec.n_elements:3d} elements, "
              f"{rec.n_nodes:3d} nodes")
    if info.weights_per_partition is not None:
        print(f"  weighted bal:  {info.weights_per_partition}")
    print(f"  diaphragms:    {N_STORIES} "
          f"(cross-partition replication likely)")
    print(f"  Tcl:           {OUT.resolve()}")
    print(f"  MP run:        mpiexec -np {N_PARTS} OpenSeesMP {OUT}")
    if run:
        maybe_run_openseesmp(OUT)


if __name__ == "__main__":
    main(run="--run" in sys.argv[1:])
