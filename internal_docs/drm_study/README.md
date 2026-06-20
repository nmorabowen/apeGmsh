# `drm_study/` — Ladruno fork DRM reference study (copied into apeGmsh)

**Provenance:** copied 2026-06-19 from the OpenSees-fork worktree
`OpenSees/Ladruno_implementation/drm_study/` (branch `ladruno`, fork PR
`nmorabowen/OpenSees#296`). This is the **validated reference implementation** that
[ADR 0066](../../src/apeGmsh/opensees/architecture/decisions/0066-h5drm-drm-authoring.md)
("H5DRM DRM-load authoring") calls *"the spec by example"*. The apeGmsh
implementation must reproduce its result (98/98 nodes matched; interior
reproduces the free-field, post-arrival corr 0.91–0.98, ratio ~1.0; bounded past
`tend`).

It lives here so the consuming code (apeGmsh) carries its own grounding — the
fork tree is a separate repo on a different branch.

## What each file is (and which ADR requirement it grounds)

| File | Role | ADR req |
|------|------|---------|
| `SCOPE.md` | The **guide**: the full study log — C++ delta, the garbage-T / Z-flip / hold-past-end findings, the buffer-box discovery, P0–P4 plan. Read this first. | all |
| `build_drm_model.py` | apeGmsh box matched to the `.h5drm` station grid (inner DRM region). Uses `add_box` + `set_transfinite_box`. | **R2** |
| `build_drm_buffered.py` | Stabilized box: inner grid + exterior buffer (4 sides + bottom) + fixed/lysmer boundary on **non-dataset** nodes. Hand-emits Tcl in numpy (apeGmsh lacked the buffer builder). | **R3** |
| `setup_tcl_run.py` | Emits the `pattern H5DRM …` line + transient driver (the frame handshake). | **R1** |
| `add_lysmer.py` | LysmerTriangle on outer faces (the "wrong on b-nodes" cautionary tale). | R3 |
| `run_drm_baseline.py` | openseespy driver (the registered `ops.pattern('H5DRM', …)` invocation). | R1 |
| `post_drm.py`, `post_drm_buffered.py` | Validation: interior vs free-field oracle + Z-flip / divergence diagnostics. | acceptance |
| `gen_drm_dataset.py` | Synthetic `.h5drm` fixture generator (ShakerMaker DRMBox, SCEC_LOH_1). | fixture |
| `test_h5drm_drm_loadpattern.py` | The **fork regression test** to port — self-contained synthetic `.h5drm`, guards the 3 core fixes. The apeGmsh acceptance test is a port of this. | acceptance |

**Not copied** (regenerable run artifacts): `*.h5drm` (8.6 MB ea.), `*.out`,
`*.txt`, `*.npz`, `drm_model.h5`, generated `*.tcl`. Regenerate via
`gen_drm_dataset.py` + the build scripts if needed.

## Canonical `ops.pattern('H5DRM', …)` signature (fork-verified, from the test)

```python
ops.pattern("H5DRM", tag, file, factor, crd_scale, dist_tol, do_transform,
            T00,T01,T02, T10,T11,T12, T20,T21,T22,   # 3x3 row-major
            x00,x01,x02)                              # x0 offset
# identity + km->m: factor=1.0, crd_scale=1000.0, dist_tol=1.0, do_transform=1,
#                   T=I, x0=(0,0,0)
```
