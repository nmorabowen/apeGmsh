# Post-process the Tcl DRM baseline: model interior-node disp vs prescribed field
# vs free-field oracle. Extracts Z-flip + tracking even with rigid drift present.
import os, numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
rec = np.loadtxt(os.path.join(HERE, "drm_disp_tcl.out"))   # time ux uy uz
t_m, ux, uy, uz = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]

meta = np.load(os.path.join(HERE, "_cmp_meta.npz"))
row0 = int(meta["row0"]); station_km = meta["station_km"]
with h5py.File(os.path.join(HERE, "drm_small.h5drm"), "r") as f:
    dis = f["DRM_Data/displacement"][:]
    dt = float(f["DRM_Metadata/dt"][()])
d_presc = dis[row0:row0 + 3, :]                  # prescribed (x=N, y=E, z=Down)
t_drm = np.arange(d_presc.shape[1]) * dt

# align model samples to prescribed times (model starts at t=dt)
def at(series_t, series_v, tq):
    return np.interp(tq, series_t, series_v)

# early window where drift is small (first 2 s)
mask = t_m <= 2.0
print(f"comparison node = station {station_km} km (interior), prescribed row {row0}")
print(f"record dt={dt}, model steps={len(t_m)}, t range [{t_m[0]:.3f},{t_m[-1]:.3f}]")
print("\n--- magnitudes ---")
print(f"prescribed max|N,E,Dn| = {np.max(np.abs(d_presc[0])):.4e} {np.max(np.abs(d_presc[1])):.4e} {np.max(np.abs(d_presc[2])):.4e}")
print(f"model      max|ux,uy,uz| (whole) = {np.max(np.abs(ux)):.4e} {np.max(np.abs(uy)):.4e} {np.max(np.abs(uz)):.4e}")
print(f"model      max|ux,uy,uz| (t<=2s) = {np.max(np.abs(ux[mask])):.4e} {np.max(np.abs(uy[mask])):.4e} {np.max(np.abs(uz[mask])):.4e}")

# correlations model-vs-prescribed on early window (component map: ux<->N, uy<->E, uz<->Dn)
pN = at(t_drm, d_presc[0], t_m[mask])
pE = at(t_drm, d_presc[1], t_m[mask])
pD = at(t_drm, d_presc[2], t_m[mask])
def corr(a, b):
    if np.std(a) < 1e-30 or np.std(b) < 1e-30: return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
print("\n--- early-window (t<=2s) correlation model vs prescribed ---")
print(f"corr(ux, N)  = {corr(ux[mask], pN):+.3f}")
print(f"corr(uy, E)  = {corr(uy[mask], pE):+.3f}")
print(f"corr(uz, Dn) = {corr(uz[mask], pD):+.3f}   (NEGATIVE => vertical/Z is FLIPPED)")

# amplitude ratio early (median of |model|/|presc| where presc is meaningful)
def ratio(m, p):
    sel = np.abs(p) > 0.05 * np.max(np.abs(p))
    return float(np.median(np.abs(m[sel]) / np.abs(p[sel]))) if sel.any() else float('nan')
print("\n--- early-window amplitude ratio |model|/|prescribed| ---")
print(f"N: {ratio(ux[mask], pN):.3f}   E: {ratio(uy[mask], pE):.3f}   Dn: {ratio(uz[mask], pD):.3f}  (~1 means tracks field)")

# divergence onset: first time |model| exceeds 5x prescribed max
thr = 5 * np.max(np.abs(d_presc))
div = np.where(np.maximum.reduce([np.abs(ux), np.abs(uy), np.abs(uz)]) > thr)[0]
print(f"\ndivergence (|model|>5x presc_max={thr:.3f}) onset: t = {t_m[div[0]]:.3f}s" if div.size else "\nno divergence")
print("DONE")
