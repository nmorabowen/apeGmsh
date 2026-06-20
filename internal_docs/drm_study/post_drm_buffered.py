# ============================================================================
# post_drm_buffered.py  --  compare stabilized DRM interior-node disp vs the
#                           prescribed free-field + the oracle, over the FULL
#                           window (the box should now be bounded, not drifting).
# ----------------------------------------------------------------------------
#   python post_drm_buffered.py <disp_out_file>
# Component map: model ux<->prescribed N (x), uy<->E (y), uz<->Dn (z, z-down).
# ============================================================================
import os, sys, numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
fn = sys.argv[1] if len(sys.argv) > 1 else "drm_disp_buf_fixed.out"
rec = np.loadtxt(os.path.join(HERE, fn))      # time ux uy uz
t_m, ux, uy, uz = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]

meta = np.load(os.path.join(HERE, "_cmp_meta.npz"))
row0 = int(meta["row0"]); station_km = meta["station_km"]
with h5py.File(os.path.join(HERE, "drm_small.h5drm"), "r") as f:
    dis = f["DRM_Data/displacement"][:]
    dt = float(f["DRM_Metadata/dt"][()])
d_presc = dis[row0:row0 + 3, :]               # prescribed N,E,Dn at cmp node
t_drm = np.arange(d_presc.shape[1]) * dt

# oracle (free-field at box center)
ora = np.load(os.path.join(HERE, "drm_qa_direct.npz"))


def at(tt, vv, tq):
    return np.interp(tq, tt, vv)


def corr(a, b):
    if np.std(a) < 1e-30 or np.std(b) < 1e-30:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def ratio(m, p):
    sel = np.abs(p) > 0.1 * np.max(np.abs(p))
    return float(np.median(np.abs(m[sel]) / np.abs(p[sel]))) if sel.any() else float('nan')


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


print(f"=== {fn} ===")
print(f"comparison node = station {station_km} km (interior), prescribed row {row0}")
print(f"record dt={dt}, model steps={len(t_m)}, t range [{t_m[0]:.3f},{t_m[-1]:.3f}]")

pmax = np.max(np.abs(d_presc))
mmax = max(np.max(np.abs(ux)), np.max(np.abs(uy)), np.max(np.abs(uz)))
print("\n--- magnitudes (full window) ---")
print(f"prescribed max|N,E,Dn| = {np.max(np.abs(d_presc[0])):.4e} {np.max(np.abs(d_presc[1])):.4e} {np.max(np.abs(d_presc[2])):.4e}")
print(f"model      max|ux,uy,uz| = {np.max(np.abs(ux)):.4e} {np.max(np.abs(uy)):.4e} {np.max(np.abs(uz)):.4e}")
print(f"overall model/prescribed peak ratio = {mmax/pmax:.2f}   (BOUNDED if ~O(1), DIVERGED if >>10)")

# interpolate prescribed onto model time samples (full window)
pN = at(t_drm, d_presc[0], t_m)
pE = at(t_drm, d_presc[1], t_m)
pD = at(t_drm, d_presc[2], t_m)

print("\n--- FULL-window correlation model vs prescribed ---")
print(f"corr(ux, N)  = {corr(ux, pN):+.3f}")
print(f"corr(uy, E)  = {corr(uy, pE):+.3f}")
print(f"corr(uz, Dn) = {corr(uz, pD):+.3f}   (NEGATIVE => vertical/Z FLIPPED; POSITIVE+ratio~1 => flip-OFF correct)")

print("\n--- FULL-window amplitude ratio |model|/|prescribed| ---")
print(f"N: {ratio(ux, pN):.3f}   E: {ratio(uy, pE):.3f}   Dn: {ratio(uz, pD):.3f}   (~1 => tracks field)")

# post-arrival window (wave arrives ~2s; evaluate where the field is active)
mask = (t_m >= 2.0) & (t_m <= 25.0)
print("\n--- post-arrival window (2-25 s) correlation ---")
print(f"corr(ux, N)  = {corr(ux[mask], pN[mask]):+.3f}")
print(f"corr(uy, E)  = {corr(uy[mask], pE[mask]):+.3f}")
print(f"corr(uz, Dn) = {corr(uz[mask], pD[mask]):+.3f}")
print(f"ratio N: {ratio(ux[mask], pN[mask]):.3f}  E: {ratio(uy[mask], pE[mask]):.3f}  Dn: {ratio(uz[mask], pD[mask]):.3f}")

# quiet pre-arrival check (t<1.5s the interior should be ~still)
pre = t_m < 1.5
print(f"\n--- pre-arrival (t<1.5s) model RMS (should be small): ux={rms(ux[pre]):.2e} uy={rms(uy[pre]):.2e} uz={rms(uz[pre]):.2e}")
print(f"    prescribed RMS pre-arrival:                        N ={rms(pN[pre]):.2e} E ={rms(pE[pre]):.2e} Dn={rms(pD[pre]):.2e}")

# divergence onset
thr = 5 * pmax
allm = np.maximum.reduce([np.abs(ux), np.abs(uy), np.abs(uz)])
div = np.where(allm > thr)[0]
print(f"\ndivergence (|model|>5x presc_max={thr:.3f}): " +
      (f"onset t={t_m[div[0]]:.3f}s" if div.size else "NONE -- bounded"))
print("DONE")
