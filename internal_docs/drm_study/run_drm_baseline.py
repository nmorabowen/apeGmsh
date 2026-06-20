# ============================================================================
# run_drm_baseline.py  --  P1/P2 baseline DRM run against the CURRENT fork binary
# ----------------------------------------------------------------------------
# Loads the apeGmsh-built soil box, applies the H5DRM load pattern from
# drm_small.h5drm, runs a Newmark transient PAST tend, records interior-node
# displacement, and compares it to (a) the prescribed .h5drm field at that node
# and (b) the free-field oracle. This characterizes the CURRENT implementation:
#   - does the interior reproduce the free-field?  (mechanics work?)
#   - is the vertical (Z) component flipped?        (the Z-flip question)
#   - what happens after t > tend?                  (current = forces->0)
#
# NO C++ rebuild. Run with the python312 env that loads dist/bin/opensees.pyd.
#   <py312> Ladruno_implementation/drm_study/run_drm_baseline.py
# ============================================================================
import os
import sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
# Worktree dist (the freshly-built pyd with the DRM fixes + openseespy H5DRM),
# NOT the main-repo dist/bin (which is the stale shipped binary).
DIST = r"C:\Users\nmora\Github\OpenSees_Compile\OpenSees\.claude\worktrees\flamboyant-ptolemy-c4e013\dist\bin"
H5DRM = os.path.join(HERE, "drm_small.h5drm")
MODEL_PY = os.path.join(HERE, "drm_model.py")
DISP_OUT = os.path.join(HERE, "drm_disp.out")

CRD_SCALE = 1000.0
DT = 0.025
TEND = 30.0
N_STEPS = 1400          # 1400*0.025 = 35 s  -> 5 s past tend
CENTER_KM = np.array([6.0, 8.0, 0.0])

# --- bootstrap the FORK opensees.pyd (not pip openseespy) -------------------
os.add_dll_directory(DIST)
sys.path.insert(0, DIST)
import opensees as ops  # noqa: E402

# --- build the model from the emitted deck (swap its import for our fork) ---
src = open(MODEL_PY).read().replace("import openseespy.opensees as ops", "")
exec(compile(src, MODEL_PY, "exec"), {"ops": ops})

# --- locate the interior node at the box center (model coords (0,0,0)) ------
# find the .h5drm station nearest the center and its data row + the model node
with h5py.File(H5DRM, "r") as f:
    sxyz = f["DRM_Data/xyz"][:]
    internal = f["DRM_Data/internal"][:]
    dloc = f["DRM_Data/data_location"][:]
    dis = f["DRM_Data/displacement"][:]
i_center = int(np.argmin(np.linalg.norm(sxyz - CENTER_KM, axis=1)))
row0 = int(dloc[i_center])          # first of the 3 rows for this station
model_xyz = (sxyz[i_center] - CENTER_KM) * CRD_SCALE
print(f"center station #{i_center} xyz_km={sxyz[i_center]} internal={bool(internal[i_center])} "
      f"-> model node coords {model_xyz}")

# match to the model node tag
all_tags = ops.getNodeTags()
node_tag = min(all_tags, key=lambda t: float(np.linalg.norm(np.array(ops.nodeCoord(t)) - model_xyz)))
print(f"matched model node tag = {node_tag} at {ops.nodeCoord(node_tag)}")

# --- recorder on that interior node -----------------------------------------
ops.recorder("Node", "-file", DISP_OUT, "-time", "-node", node_tag, "-dof", 1, 2, 3, "disp")

# --- H5DRM load pattern (openseespy exposes crd_scale/transform) ------------
# pattern('H5DRM', tag, file, factor, crd_scale, dist_tol, do_transform, T..., x0...)
ops.pattern("H5DRM", 1, H5DRM, 1.0,
            CRD_SCALE,          # crd_scale: km -> m
            1.0,                # distance_tolerance (m); nodes match exactly
            1,                  # do_coordinate_transformation
            1.0, 0.0, 0.0,      # T (identity)
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0)      # x0

# --- transient analysis (linear elastic) ------------------------------------
ops.constraints("Plain")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1e-8, 10)
ops.algorithm("Linear")
ops.integrator("Newmark", 0.5, 0.25)
ops.analysis("Transient")

print(f"running {N_STEPS} steps @ dt={DT} (to t={N_STEPS*DT:.2f}s, tend={TEND}) ...")
ok = 0
for k in range(N_STEPS):
    ok = ops.analyze(1, DT)
    if ok != 0:
        print(f"  analyze failed at step {k} (t={ (k+1)*DT:.3f})")
        break
ops.wipe()  # flush recorder
print("analysis return:", ok)

# --- post-process: compare to prescribed field + oracle ---------------------
rec = np.loadtxt(DISP_OUT)            # cols: time ux uy uz
t_m, ux, uy, uz = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]

t_drm = np.arange(dis.shape[1]) * DT
d_presc = dis[row0:row0 + 3, :]       # prescribed (x=N, y=E, z=Down) at center
oracle = np.load(os.path.join(HERE, "drm_qa_direct.npz"))
# oracle: z(down), e(East), n(North), t  -> map to model (ux=N, uy=E, uz=Down)

def mx(a):
    return float(np.max(np.abs(a)))

print("\n=== interior-node displacement: model vs prescribed vs oracle ===")
print(f"model    max|ux,uy,uz| = {mx(ux):.4e} {mx(uy):.4e} {mx(uz):.4e}")
print(f"prescr.  max|N,E,Dn|   = {mx(d_presc[0]):.4e} {mx(d_presc[1]):.4e} {mx(d_presc[2]):.4e}")
print(f"oracle   max|N,E,Dn|   = {mx(oracle['n']):.4e} {mx(oracle['e']):.4e} {mx(oracle['z']):.4e}")

# sign check on Z: correlate model uz with prescribed Down over the record window
nlap = min(len(uz), d_presc.shape[1])
if nlap > 10 and mx(uz) > 0:
    cz = float(np.corrcoef(uz[:nlap], d_presc[2, :nlap])[0, 1])
    cx = float(np.corrcoef(ux[:nlap], d_presc[0, :nlap])[0, 1])
    print(f"corr(model_ux, prescribed_N)  = {cx:+.3f}")
    print(f"corr(model_uz, prescribed_Dn) = {cz:+.3f}   (NEGATIVE => Z is flipped)")
print("\nDONE")
