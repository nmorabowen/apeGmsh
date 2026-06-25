# ============================================================================
# test_h5drm_drm_loadpattern.py
# ----------------------------------------------------------------------------
# Regression tests for the Ladruno DRM H5DRM load-pattern fixes (drm_study):
#   1. openseespy exposes H5DRM with crd_scale (P4 + the CMake _H5DRM-for-
#      OPS_INTERPRETER scoping fix): ops.pattern('H5DRM',...) must NOT raise
#      "unknown pattern type".
#   2. Coordinate transform / node-matching works (the garbage-T fix): the DRM
#      effective forces must actually drive the model (a non-fixed node moves).
#   3. Hold-final-displacement past tend (jaabell 99b7c2c11) + i2 clamp: running
#      PAST the DRM record must NOT exit(-1) / crash (the analyze() returns 0).
#
# Self-contained: synthesizes a tiny .h5drm (no ShakerMaker dependency). Run
# under the CPython 3.12 interp that loads the WORKTREE dist/bin/opensees.pyd.
# ============================================================================
import os
import sys
import numpy as np
import pytest

# --- bootstrap the worktree-built fork pyd (BUILD_GOTCHAS.md) ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIST = os.path.normpath(os.path.join(_HERE, "..", "dist", "bin"))

h5py = pytest.importorskip("h5py")
if not os.path.exists(os.path.join(_DIST, "opensees.pyd")):
    pytest.skip("worktree dist/bin/opensees.pyd not built", allow_module_level=True)
os.add_dll_directory(_DIST)
sys.path.insert(0, _DIST)
os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
import opensees as ops  # noqa: E402

CRD = 1000.0   # km -> m
H = 0.05       # km grid spacing (50 m)
DT = 0.025
NT = 60        # tend = 60*0.025 = 1.5 s
TEND = NT * DT


def _write_synthetic_h5drm(path):
    """3x3x3 station grid (km, centred at origin); analytic time-varying field
    with a spatial gradient so the DRM effective forces are nonzero. Final
    timestep is NON-zero (so hold-past-end holds a real static state)."""
    ax = np.array([-H, 0.0, H])
    xyz = np.array([(x, y, z) for x in ax for y in ax for z in ax], dtype=float)
    n = xyz.shape[0]                              # 27
    # boundary = outer shell, internal = the single centre node (0,0,0)
    internal = np.all(np.abs(xyz) < 1e-9, axis=1)   # only the centre
    t = np.arange(NT) * DT
    env = np.sin(2 * np.pi * t / (NT * DT))          # 0 at t0, sweeps, ~ -0 at end but nonzero mid
    env = env + 0.3                                  # bias so the final sample is clearly nonzero
    dis = np.zeros((3 * n, NT))
    for i in range(n):
        g = 1.0 + 0.2 * (xyz[i, 2] / H)              # vertical gradient -> nonzero K_be*u
        dis[3 * i + 0] = 1e-3 * env * g              # N
        dis[3 * i + 1] = 0.7e-3 * env * g            # E
        dis[3 * i + 2] = 0.5e-3 * env * g            # Dn (down)
    acc = np.gradient(np.gradient(dis, DT, axis=1), DT, axis=1)
    vel = np.gradient(dis, DT, axis=1)
    with h5py.File(path, "w") as f:
        d = f.create_group("DRM_Data")
        d["xyz"] = xyz
        d["internal"] = internal
        d["data_location"] = (np.arange(n) * 3).astype(np.int32)
        d["displacement"] = dis
        d["velocity"] = vel
        d["acceleration"] = acc
        m = f.create_group("DRM_Metadata")
        m["dt"] = DT
        m["tstart"] = 0.0
        m["tend"] = TEND
        m["drmbox_x0"] = np.array([0.0, 0.0, 0.0])   # centre
        for k, v in (("drmbox_xmin", -H), ("drmbox_xmax", H), ("drmbox_ymin", -H),
                     ("drmbox_ymax", H), ("drmbox_zmin", -H), ("drmbox_zmax", H)):
            m[k] = float(v)
        m["h"] = np.array([H, H, H])


def _build_box():
    """3x3x3 node box in METRES (= station_km * CRD), centred at origin, stdBrick."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ax = np.array([-H, 0.0, H]) * CRD            # [-50, 0, 50] m
    coord2tag = {}
    tag = 1
    for x in ax:
        for y in ax:
            for z in ax:
                ops.node(tag, float(x), float(y), float(z))
                coord2tag[(round(x, 3), round(y, 3), round(z, 3))] = tag
                tag += 1
    ops.nDMaterial("ElasticIsotropic", 1, 2.77e10, 0.3333, 2600.0)
    # 2x2x2 = 8 stdBricks
    etag = 1
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                xs, ys, zs = ax[ix:ix + 2], ax[iy:iy + 2], ax[iz:iz + 2]
                def T(a, b, c):
                    return coord2tag[(round(xs[a], 3), round(ys[b], 3), round(zs[c], 3))]
                # stdBrick node order (bottom face then top face, CCW)
                ns = [T(0, 0, 0), T(1, 0, 0), T(1, 1, 0), T(0, 1, 0),
                      T(0, 0, 1), T(1, 0, 1), T(1, 1, 1), T(0, 1, 1)]
                ops.element("stdBrick", etag, *ns, 1)
                etag += 1
    # fix the bottom face (z = -50) to give a static reference
    for (x, y, z), nt in coord2tag.items():
        if abs(z - ax[0]) < 1e-6:
            ops.fix(nt, 1, 1, 1)
    return coord2tag


def test_openseespy_h5drm_registered_and_drives_model(tmp_path):
    f = str(tmp_path / "syn.h5drm")
    _write_synthetic_h5drm(f)
    coord2tag = _build_box()

    # (1) H5DRM must be a recognized openseespy pattern with crd_scale support.
    #     A failure here is "unknown pattern type" (the P4 / CMake-scope regression).
    ops.pattern("H5DRM", 1, f, 1.0, CRD, 1.0, 1,
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-8, 10)
    ops.algorithm("Linear")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # (2) node-matching / transform: step within tend, a non-fixed node must move.
    top_centre = coord2tag[(0.0, 0.0, round(0.05 * CRD, 3))]
    for _ in range(20):
        assert ops.analyze(1, DT) == 0
    u = ops.nodeDisp(top_centre)
    assert max(abs(c) for c in u) > 1e-9, (
        "DRM applied no motion -> node-matching/transform regressed (garbage-T)")

    # (3) hold-past-end: continue PAST tend; analyze must keep returning 0 (no exit).
    n_to_tend = int(np.ceil(TEND / DT))
    extra = n_to_tend + 20                      # ~0.5 s past tend
    for _ in range(extra):
        assert ops.analyze(1, DT) == 0, "analyze failed past tend (hold-past-end regressed)"
    assert ops.getTime() > TEND
    # bounded (no blow-up): the fixed base keeps it finite
    u2 = ops.nodeDisp(top_centre)
    assert all(abs(c) < 1.0 for c in u2), "response diverged past tend"
    ops.wipe()


if __name__ == "__main__":
    import tempfile
    class _TP:
        def __truediv__(self, name):
            return os.path.join(tempfile.mkdtemp(), name)
    test_openseespy_h5drm_registered_and_drives_model(_TP())
    print("PASS")
