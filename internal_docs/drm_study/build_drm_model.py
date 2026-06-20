# ============================================================================
# build_drm_model.py  --  apeGmsh soil box matched to a ShakerMaker .h5drm grid
# ----------------------------------------------------------------------------
# Builds a structured hex soil box whose NODES coincide exactly with the
# .h5drm station grid, so OpenSees' H5DRM node-matching is trivial. Emits an
# OpenSees deck (Tcl + openseespy). The H5DRM load pattern + transient analysis
# are appended by the driver (run_drm_baseline.py) for the baseline study.
#
# Frame choice (deliberate, see SCOPE.md): the model is built in ShakerMaker's
# z-DOWN frame, in METERS (= station_km * 1000). Then H5DRM uses crd_scale=1000
# and an IDENTITY transform -> node-matching is exact AND the Z-flip question is
# isolated purely to the C++ d[2]=-d[2] negation.
#
# Run with the apeGmsh python (the one that has apeGmsh + gmsh importable).
# ============================================================================
import os
import numpy as np
import h5py

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

HERE = os.path.dirname(os.path.abspath(__file__))
H5DRM = os.path.join(HERE, "drm_small.h5drm")
OUT_TCL = os.path.join(HERE, "drm_model.tcl")
OUT_PY = os.path.join(HERE, "drm_model.py")
CRD_SCALE = 1000.0  # km -> m

# --- derive the grid from the .h5drm (origin / spacing / counts) ------------
with h5py.File(H5DRM, "r") as f:
    xyz_km = f["DRM_Data/xyz"][:]
    internal = f["DRM_Data/internal"][:]


def axis(vals):
    u = np.unique(np.round(vals, 6))
    return u.min(), u.max(), len(u), (np.diff(u).mean() if len(u) > 1 else 0.0)


x0, x1, nx, hx = axis(xyz_km[:, 0])
y0, y1, ny, hy = axis(xyz_km[:, 1])
z0, z1, nz, hz = axis(xyz_km[:, 2])
print(f"grid (km): X[{x0},{x1}] n={nx} h={hx:.4f}  "
      f"Y[{y0},{y1}] n={ny} h={hy:.4f}  Z[{z0},{z1}] n={nz} h={hz:.4f}")

# Model frame (see SCOPE.md): the C++ H5DRM transform maps
#   xyz_model = T * ((xyz_station - drmbox_x0) * crd_scale) + x0
# with drmbox_x0 = box CENTER read from the file ([6,8,0] km). So if we build the
# model CENTERED at the lateral origin (node = (station-center)*crd_scale), then
# crd_scale=CRD_SCALE, T=identity, x0=0 reproduces our node coords exactly.
# Result: meters, z-down, box centered laterally at (0,0), surface at z=0.
cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, z0  # box center used by H5DRM (z origin = surface)
ox, oy, oz = (x0 - cx) * CRD_SCALE, (y0 - cy) * CRD_SCALE, (z0 - cz) * CRD_SCALE
dx, dy, dz = (x1 - x0) * CRD_SCALE, (y1 - y0) * CRD_SCALE, (z1 - z0) * CRD_SCALE
h_m = hx * CRD_SCALE  # uniform 50 m
print(f"box (m, centered): origin=({ox},{oy},{oz}) dims=({dx},{dy},{dz}) h={h_m}")

# --- soil = SCEC_LOH_1 surface layer (box is shallower than the 1 km layer) -
# Vp=4000 m/s, Vs=2000 m/s, rho=2600 kg/m^3
Vp, Vs, rho = 4000.0, 2000.0, 2600.0
G = rho * Vs**2
nu = (Vp**2 - 2 * Vs**2) / (2.0 * (Vp**2 - Vs**2))
E = 2 * G * (1 + nu)
print(f"soil: E={E:.4e} nu={nu:.4f} rho={rho}")

# --- geometry + structured hex mesh on the exact grid -----------------------
with apeGmsh(model_name="drm_soil", save_to=os.path.join(HERE, "drm_model.h5")) as g:
    g.model.geometry.add_box(ox, oy, oz, dx, dy, dz, label="soil")
    g.physical.add_volume("soil", name="Soil")
    # base = deepest face (z-down -> max z). add_surface by picking the face at z=oz+dz.
    g.mesh.structured.set_transfinite_box("soil", size=h_m, recombine=True)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info)

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
soil = ops.nDMaterial.ElasticIsotropic(E=E, nu=nu, rho=rho)
ops.element.stdBrick(pg="Soil", material=soil)
ops.tcl(OUT_TCL)
ops.py(OUT_PY)
print("wrote", OUT_TCL)
print("wrote", OUT_PY)
print("PASS")
