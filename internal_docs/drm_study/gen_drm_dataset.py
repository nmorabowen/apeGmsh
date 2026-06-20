# ============================================================================
# gen_drm_dataset.py  --  small .h5drm generator for H5DRM load-pattern study
# ----------------------------------------------------------------------------
# Produces a tiny, fast DRM dataset (ShakerMaker FK synthetics) that we use to
# study / validate / improve OpenSees' H5DRMLoadPattern (jaabell DRM merge).
#
# Run with the ShakerMaker python (system Python 3.11):
#     python Ladruno_implementation/drm_study/gen_drm_dataset.py
#
# Design choices (deliberate, for the study):
#   * DRMBox with a real boundary layer (nelems=[2,2,2]) so node-matching and
#     the boundary/interior split are actually exercised -- not the degenerate
#     1x1x1 box.
#   * SHORT record (tmax small) so an OpenSees analysis can easily run PAST the
#     DRM record end (t > tend) and exercise the new "hold final displacement,
#     zero acceleration" branch from jaabell commit 99b7c2c11.
#   * SCEC_LOH_1 crust + single point source (known-good smoke config).
#
# ShakerMaker conventions (IMPORTANT for the Z-flip question):
#   coords in km, x=North, y=East, z=Down (+down). Units SI in the .h5drm.
# ============================================================================
import os
import numpy as np

# ----------------------------------------------------------------------------
# The shakermaker_venv FK core (core.cp310-win_amd64.pyd) needs its runtime
# DLLs (gfortran/Intel) on the DLL search path or it fails to load and the
# engine silently returns zeros. Add the candidate dirs before importing.
# ----------------------------------------------------------------------------
_SP = r"C:\Users\nmora\shakermaker_venv\Lib\site-packages\shakermaker"
for _d in [
    os.path.join(_SP, "ffsp"),
    os.path.join(_SP, "core"),
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin",
    r"C:\Program Files (x86)\Intel\oneAPI\2025.1\bin",
]:
    if os.path.isdir(_d):
        try:
            os.add_dll_directory(_d)
        except OSError:
            pass
        # PATH prepend too: transitive DLL->DLL loads don't honor add_dll_directory alone
        os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]

from shakermaker.shakermaker import ShakerMaker

# ----------------------------------------------------------------------------
# BUGFIX (stale venv ShakerMaker v1.x): Station.add_to_response computes
#   nskip = int(t[0]/dt)
# which goes NEGATIVE whenever the FK time offset t0 < 0 (the trace starts
# before t=0, the normal case). The negative slice then raises inside run()'s
# try/except and is silently swallowed -> every station's response stays zero
# ("0 of N done", max|z|=0). The repo (v2.0.0) fixes this by clipping the
# leading samples. We monkeypatch the fixed version here so the working venv
# Windows FK core is preserved without touching the user's install.
# ----------------------------------------------------------------------------
from shakermaker.station import Station as _Station


def _add_to_response_fixed(self, z, e, n, t, tmin=0, tmax=100.):
    dt = t[1] - t[0]
    if not self._initialized:
        self._t = np.arange(tmin, tmax, dt)
        self._z = 0 * self._t
        self._e = 0 * self._t
        self._n = 0 * self._t
        self._dt = dt
        self._tmin = t.min()
        self._tmax = t.max()
        self._initialized = True
        accumulate = False
    else:
        accumulate = True

    if t[0] >= 0:
        nskip_buf, nskip_sig = int(t[0] / dt), 0
    else:
        nskip_buf, nskip_sig = 0, int(-t[0] / dt)
    nwrite = min(len(z) - nskip_sig, len(self._t) - nskip_buf)
    if nwrite > 0:
        sb, ss = slice(nskip_buf, nskip_buf + nwrite), slice(nskip_sig, nskip_sig + nwrite)
        if accumulate:
            self._z[sb] += z[ss]; self._e[sb] += e[ss]; self._n[sb] += n[ss]
        else:
            self._z[sb] = z[ss]; self._e[sb] = e[ss]; self._n[sb] = n[ss]
    self._notify(t)


_Station.add_to_response = _add_to_response_fixed

from shakermaker.cm_library.LOH import SCEC_LOH_1
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stf_extensions.gaussian import Gaussian
from shakermaker.sl_extensions import DRMBox
from shakermaker.slw_extensions import DRMHDF5StationListWriter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_H5DRM = os.path.join(HERE, "drm_small.h5drm")
OUT_QA_NPZ = os.path.join(HERE, "drm_qa_direct.npz")  # free-field oracle at box center

# --- FK numerical parameters (example-07 known-good values) ---
dt, nfft, tb, dk, tmax = 0.025, 2048, 1000, 0.1, 30

crust = SCEC_LOH_1()

# --- source ---
sigma = 0.06
t0 = 6 * sigma
M0 = 1e18 / 5e14 / 2


def make_fault():
    stf = Gaussian(t0=t0, freq=1 / sigma, M0=M0, derivative=False)
    src = PointSource([0, 0, 2], [0, 90, 0], stf=stf)  # [x,y,z] km ; [strike,dip,rake] deg
    return FaultSource([src], metadata={"name": "src"})


# --- DRM box: center at (6,8,0) km, 2x2x2 elements, 50 m spacing ---
center = [6.0, 8.0, 0.0]
nelems = [2, 2, 2]
h_km = [0.05, 0.05, 0.05]  # 50 m

drm = DRMBox(center, nelems, h_km, metadata={"name": "drm_small"})
print("DRMBox stations:", drm.nstations)
print("box center (drmbox_x0):", drm.metadata.get("drmbox_x0"))

model = ShakerMaker(crust, make_fault(), drm)
if hasattr(model, "check_parameters"):  # older installed ShakerMaker lacks this preflight
    model.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)

if os.path.exists(OUT_H5DRM):
    os.remove(OUT_H5DRM)
writer = DRMHDF5StationListWriter(OUT_H5DRM)
model.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax,
          writer=writer, writer_mode="progressive")
assert os.path.exists(OUT_H5DRM), "writer did not produce .h5drm"

# --- free-field oracle: a DIRECT station at the box center (= QA point) ---
s = Station(center, metadata={"name": "direct_center"})
model_direct = ShakerMaker(crust, make_fault(), StationList([s], {}))
if hasattr(model_direct, "check_parameters"):
    model_direct.check_parameters(dt=dt, nfft=nfft, dk=dk, tb=tb, tmax=tmax)
model_direct.run(dt=dt, nfft=nfft, tb=tb, dk=dk, tmax=tmax)
z, e, n, t = s.get_response()
np.savez(OUT_QA_NPZ, z=z, e=e, n=n, t=t, center_km=np.array(center))

print("wrote:", OUT_H5DRM)
print("wrote:", OUT_QA_NPZ)
print("direct center max|z|,|e|,|n| =",
      float(np.max(np.abs(z))), float(np.max(np.abs(e))), float(np.max(np.abs(n))))
print("record duration t[-1] =", float(t[-1]), "s")
print("PASS")
