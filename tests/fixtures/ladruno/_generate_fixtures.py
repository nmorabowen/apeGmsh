"""Regenerate the committed ``.ladruno`` test fixtures.

These fixtures ground the ``Results.from_ladruno`` reader work (recorder-plan
L2-L4) so apeGmsh's reader tests run **fork-free** — CI on stock ``openseespy``
reads the committed HDF5; only *regenerating* them needs the Ladruno fork build.

Run with the fork venv::

    C:/Users/nmora/venv/opensees_venv/Scripts/python.exe \
        tests/fixtures/ladruno/_generate_fixtures.py

Requires a Ladruno fork build of OpenSees (the banner lists
"Ladruno — modular HDF5 .ladruno recorder"). Verified against build
``605affeb`` (FORMAT_VERSION 1). Each model is intentionally tiny and
deterministic. Fixtures:

  * ``truss2d.ladruno``  — 2D, nodal displacement + element basicForce
                           (value channels; the L2 baseline).
  * ``beam3d.ladruno``   — 3D ElasticBeam3d on a skew axis → a non-identity
                           ``MODEL/LOCAL_AXES`` quaternion FRAME (L3 orientation).
  * ``energy.ladruno``   — transient with ``-G energy`` → ``ON_DOMAIN`` +
                           ``ON_REGIONS`` ``energyBalance`` (L4).
  * ``quad2d.ladruno``   — 2D FourNodeQuad recording ``stress``/``strain``
                           (Gauss-level continuum, COLUMN_MAP LEVELS=4 with
                           per-GP blocks) **and** ``globalForce``
                           (element-node forces, ``P<dof>_<node>``) — the
                           L2b-2 read_gauss + read_elements baseline.
  * ``bezier_tri6.ladruno`` / ``bezier_tet10.ladruno`` — single fork Bézier
                           (Bernstein) elements: the self-describing
                           ``FAMILY="bernstein"`` GP world-coord path. The tet
                           additionally writes ``GLOBAL_GP_COORDS`` (the tri6
                           does not), an in-file oracle for ``B(ξ)·X``.
  * ``node_envelope.ladruno`` — static cyclic pushover recorded with the
                           ``-envelope`` flag → ``RESULTS/ENVELOPES/ON_NODES``
                           per-node ``MIN``/``MAX``/``ABSMAX``/``ARG_STEP``
                           (no time series; ``ON_NODES`` is empty). The
                           Finding-B envelope-reader baseline.
  * ``monitor.h5``       — a live-telemetry **Monitor** sink (``recorder
                           Monitor``): a lightweight SWMR-HDF5 file
                           (``FORMAT="ladruno-monitor"``) with ``COLUMNS``/
                           ``STEP``/``TIME``/``FRAMES``, NOT a ``.ladruno``.
                           The ``read_monitor`` / ``tail_monitor`` baseline.
  * ``truss2d.part-0.ladruno`` / ``.part-1.ladruno`` — the single-partition
                           ``truss2d`` hand-split into a 2-partition manifest
                           (real MPI / ``mpiexec`` is unavailable in this env)
                           so the ``LadrunoMultiPartitionReader`` stitch path
                           (node-union + element-concat) is exercised
                           fork-free. Synthesized by :func:`_synthesize_partitions`
                           with pure ``h5py`` — no fork needed for this step.
"""
from __future__ import annotations

import os

import openseespy.opensees as ops

HERE = os.path.dirname(os.path.abspath(__file__))


def _truss2d(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 2.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.fix(3, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.element("truss", 2, 2, 3, 1.0, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 10.0, 0.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "basicForce")
    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 0.25)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(4)
    ops.wipe()  # flush + close the recorder


def _beam3d(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, 3.0, 1.0, 2.0)  # skew axis → non-identity local frame
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.element(
        "elasticBeamColumn", 1, 1, 2, 1.0, 2e8, 8e7, 0.1, 0.1, 0.1, 1
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "localForce")
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    ops.wipe()


def _node_envelope(path: str) -> None:
    # Static cyclic pushover recorded with ``-envelope`` → per-node
    # time-reduced MIN/MAX/ABSMAX/ARG_STEP under RESULTS/ENVELOPES/ON_NODES
    # (no time series; ON_NODES is empty). The load path drives node 3's Ux
    # +0.02 → -0.03 → +0.01 so MIN<0<MAX, ABSMAX=0.03 at step 8 (verifiable).
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 2.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.fix(3, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.element("truss", 2, 2, 3, 1.0, 1)
    # -envelope → EnvelopeSink; tiny -T dt records every step.
    ops.recorder(
        "ladruno", path, "-N", "displacement", "-T", "dt", 1e-6, "-envelope",
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 10.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1e-10, 10)
    ops.algorithm("Newton")
    ops.analysis("Static")
    for di in (0.25, 0.25, 0.25, 0.25,
               -0.5, -0.5, -0.5, -0.5, -0.5,
               0.4, 0.4, 0.4, 0.4, 0.4):
        ops.integrator("LoadControl", di)
        ops.analyze(1)
    ops.remove("recorders")
    ops.wipe()


def _energy(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1)
    ops.mass(2, 1.0, 1.0)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.region(1, "-node", 2)
    ops.timeSeries("Linear", 1, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 10.0, 0.0)
    # -G energy <regionTag> → whole-domain ON_DOMAIN + per-region ON_REGIONS
    ops.recorder("ladruno", path, "-N", "displacement", "-G", "energy", 1)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGen")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Linear")
    ops.analysis("Transient")
    ops.analyze(5, 0.01)
    ops.wipe()


def _quad2d(path: str) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    # Unit-square FourNodeQuad (4 Gauss points, 2x2 Gauss-Legendre).
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 1.0, 1.0)
    ops.node(4, 0.0, 1.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    ops.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStress", 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 5.0, 0.0)
    ops.load(4, 5.0, 0.0)
    # stress/strain → Gauss-level (LEVELS=4 NdMaterial, per-GP blocks);
    # force → element-node global forces (P<dof>_<node>, LEVELS=0). The
    # quad uses the ``force`` token (FourNodeQuad has no ``globalForce``)
    # and lays its columns out node-major — the reader reads positions
    # from the P<dof>_<node> tokens rather than assuming an order.
    ops.recorder(
        "ladruno", path, "-N", "displacement",
        "-E", "stress", "strain", "force",
    )
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 0.5)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(2)
    ops.wipe()


def _monitor(path: str) -> None:
    # Live-telemetry Monitor sink (recorder Monitor) — NOT a .ladruno: a
    # lightweight SWMR-HDF5 file (FORMAT="ladruno-monitor") with COLUMNS/
    # STEP/TIME/FRAMES. 2 nodes x 2 dofs disp over a 12-step transient; the
    # node-major channel order (node2.dof1, node2.dof2, node3.dof1, ...) and
    # the read/tail baseline for tests/results/test_monitor.py.
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 2.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.fix(3, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("Truss", 1, 1, 2, 1.0, 1)
    ops.element("Truss", 2, 2, 3, 1.0, 1)
    ops.mass(2, 1.0, 1.0)
    ops.mass(3, 1.0, 1.0)
    ops.recorder("Monitor", "-node", 2, 3, "-dof", 1, 2, "-resp", "disp",
                 "-sink", path, "-every", 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 10.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1e-10, 25)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    for _ in range(12):
        ops.analyze(1, 0.02)
    ops.wipe()


def _fiberbeam(path: str) -> None:
    """A ``forceBeamColumn`` with a fiber section — the L2b-3 section /
    fiber read baseline.

    A 2-node cantilever along x (3-point Lobatto) with a unit-square
    fiber section (2x2 patch → 4 fibers). A tip axial + transverse load
    gives a constant section ``P`` and a moment ``Mz`` that varies per
    integration station, so the per-GP station coordinate (read from
    ``QUADRATURE/GP_PARAM`` keyed by ``GAUSS_ID``) is exercised. Records:

      * ``section.force`` (``LEVELS=2``: ``P,Mz`` per station),
      * ``section.deformation`` (``eps,kappaZ`` per station),
      * ``section.fiber.stress`` / ``section.fiber.strain`` (``LEVELS=4``,
        ``MULTIPLICITY=4`` fibers, ``sigma11``/``eps11``) — the fiber
        geometry lands in ``MODEL/SECTION_ASSIGNMENTS``.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, 200000.0)
    ops.section("Fiber", 1)
    ops.patch("rect", 1, 2, 2, -0.05, -0.05, 0.05, 0.05)
    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 3.0, 2.0, 0.0)  # axial + transverse → P, varying Mz
    ops.recorder(
        "ladruno", path, "-N", "displacement",
        "-E", "section.force", "section.deformation",
        "section.fiber.stress", "section.fiber.strain",
    )
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 0.5)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(2)
    ops.wipe()


def _bezier_tri6(path: str) -> None:
    """A single fork ``BezierTri6`` — the self-describing HO/bernstein path.

    Exercises the seam the basis lib (#3/#4) rides: ``FAMILY="bernstein"``,
    ``TOPOLOGY="tri"``, ``PARAM_DOMAIN="bary"``, ``GP_PARAM`` as 3×2 **free**
    area coords, **no** ``GLOBAL_GP_COORDS`` (the reader reconstructs world
    GP coords via ``B(ξ)``), and Gauss stress/strain under the
    ``sigma_xx``/``eps_xx``/``gamma_xy`` axis-form token naming. Requires the
    Ladruno fork build with ``BezierTri6`` registered (tag 33000).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    # Corners 1-3, then mid-edge control points 4=(1-2), 5=(2-3), 6=(3-1).
    ops.node(1, 0.0, 0.0)
    ops.node(2, 2.0, 0.0)
    ops.node(3, 0.0, 2.0)
    ops.node(4, 1.0, 0.0)
    ops.node(5, 1.0, 1.0)
    ops.node(6, 0.0, 1.0)
    ops.fix(1, 1, 1)
    ops.fix(3, 1, 1)
    ops.fix(6, 1, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    ops.element("BezierTri6", 1, 1, 2, 3, 4, 5, 6, 1.0, "PlaneStress", 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 5.0, 0.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "stress", "strain")
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    ops.wipe()


def _bezier_tet10(path: str) -> None:
    """A single fork ``BezierTet10`` — the 3-D self-describing bernstein path.

    The tetrahedral sibling of ``_bezier_tri6``: ``FAMILY="bernstein"``,
    ``TOPOLOGY="tet"``, ``GP_PARAM`` as 4×3 free barycentric coords. Nodes
    are 4 corners + 6 mid-edge control points in ``TenNodeTetrahedron``
    order ``(1-2, 2-3, 1-3, 1-4, 3-4, 2-4)``; on this straight-sided
    reference tet the control points coincide with the nodes. Unlike the
    Tri6 fixture this element **does** write ``GLOBAL_GP_COORDS``, so the
    reader's ``B(ξ)·X`` reconstruction can be checked against the file's
    own GP world coords. Requires the Ladruno fork build (tag 33001).
    """
    import numpy as np

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    corners = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
               (0.0, 0.0, 1.0)]
    for i, (x, y, z) in enumerate(corners, start=1):
        ops.node(i, x, y, z)
    edges = [(1, 2), (2, 3), (1, 3), (1, 4), (3, 4), (2, 4)]
    for j, (a, b) in enumerate(edges, start=5):
        mid = (np.array(corners[a - 1]) + np.array(corners[b - 1])) / 2.0
        ops.node(j, float(mid[0]), float(mid[1]), float(mid[2]))
    for n in (1, 2, 3):
        ops.fix(n, 1, 1, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    ops.element("BezierTet10", 1, *range(1, 11), 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, 0.0, 0.0, -1.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "stress", "strain")
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    ops.wipe()


def _synthesize_partitions(source_name: str, stem: str) -> list[str]:
    """Hand-split a single-partition ``.ladruno`` into a 2-partition set.

    Real OpenSeesMP output (``<stem>.part-<N>.ladruno``) needs an MPI
    launcher that isn't available in this environment, so we synthesize a
    faithful 2-partition manifest from the committed single-partition
    ``truss2d`` with pure ``h5py``: partition 0 owns element 1 (nodes 1,2),
    partition 1 owns element 2 (nodes 2,3); node 2 is the shared boundary
    node (replicated, as a real partitioner would). This exercises the
    reader's node-union + element-concat stitch path; it does **not** claim
    to reproduce a specific MPI partitioner's byte layout.
    """
    import h5py
    import numpy as np

    src = os.path.join(HERE, source_name)
    # (partition_id, kept node ids, kept element row indices)
    layout = [
        (0, [1, 2], [0]),   # element 1 (Truss[1]) on nodes 1,2
        (1, [2, 3], [1]),   # element 2 (Truss[2]) on nodes 2,3
    ]
    written: list[str] = []
    for pid, keep_nodes, keep_elems in layout:
        out = os.path.join(HERE, f"{stem}.part-{pid}.ladruno")
        if os.path.exists(out):
            os.remove(out)
        with h5py.File(src, "r") as fin, h5py.File(out, "w") as fout:
            fin.copy("MODEL_STAGE[1]", fout)
            # INFO with a 2-partition manifest.
            info = fout.create_group("INFO")
            for k, v in fin["INFO"].attrs.items():
                info.attrs[k] = v
            info.attrs["PARTITIONED"] = np.int32(1)
            info.attrs["PARTITION_ID"] = np.int32(pid)
            info.attrs["NUM_PARTITIONS"] = np.int32(2)

            stage = fout["MODEL_STAGE[1]"]
            keep_nodes_arr = np.asarray(keep_nodes, dtype=np.int64)

            # --- MODEL/NODES subset ---
            nodes = stage["MODEL/NODES"]
            all_nids = np.asarray(nodes["ID"][...], dtype=np.int64).flatten()
            nmask = np.isin(all_nids, keep_nodes_arr)
            coords = np.asarray(nodes["COORDINATES"][...])
            del nodes["ID"]; del nodes["COORDINATES"]
            nodes.create_dataset("ID", data=all_nids[nmask].astype(np.int32))
            nodes.create_dataset("COORDINATES", data=coords[nmask])

            # --- MODEL/ELEMENTS subset (one Truss group) ---
            elems = stage["MODEL/ELEMENTS"]
            for cls in list(elems.keys()):
                conn = np.asarray(elems[cls]["CONNECTIVITY"][...])
                kept = conn[keep_elems]
                del elems[cls]["CONNECTIVITY"]
                elems[cls].create_dataset("CONNECTIVITY", data=kept)

            # --- RESULTS/ON_NODES subset (by node ID) ---
            for res in stage["RESULTS/ON_NODES"].values():
                rids = np.asarray(res["ID"][...], dtype=np.int64).flatten()
                rmask = np.isin(rids, keep_nodes_arr)
                data = np.asarray(res["DATA"][...])
                del res["ID"]; del res["DATA"]
                res.create_dataset(
                    "ID", data=rids[rmask].astype(np.int32).reshape(-1, 1),
                )
                res.create_dataset("DATA", data=data[:, rmask, :])

            # --- RESULTS/ON_ELEMENTS subset (by element row) ---
            on_e = stage["RESULTS/ON_ELEMENTS"]
            for token in on_e.values():
                for bucket in token.values():
                    eids = np.asarray(
                        bucket["ID"][...], dtype=np.int64,
                    ).flatten()
                    data = np.asarray(bucket["DATA"][...])
                    del bucket["ID"]; del bucket["DATA"]
                    bucket.create_dataset(
                        "ID",
                        data=eids[keep_elems].astype(np.int32).reshape(-1, 1),
                    )
                    bucket.create_dataset("DATA", data=data[:, keep_elems, :])
        written.append(out)
        print(f"{os.path.basename(out)}: {os.path.getsize(out)} bytes")
    return written


def main() -> None:
    for name, fn in (
        ("truss2d.ladruno", _truss2d),
        ("beam3d.ladruno", _beam3d),
        ("energy.ladruno", _energy),
        ("node_envelope.ladruno", _node_envelope),
        ("monitor.h5", _monitor),
        ("quad2d.ladruno", _quad2d),
        ("bezier_tri6.ladruno", _bezier_tri6),
        ("bezier_tet10.ladruno", _bezier_tet10),
        ("fiberbeam.ladruno", _fiberbeam),
    ):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            os.remove(path)
        fn(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"{name}: {size} bytes")
    # Partition fixtures are synthesized from the freshly written truss2d.
    _synthesize_partitions("truss2d.ladruno", "truss2d")


if __name__ == "__main__":
    main()
