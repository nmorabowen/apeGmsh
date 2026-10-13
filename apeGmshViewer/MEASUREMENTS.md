# apeGmshViewer measurements

The ledger ADR 0112 D5 asks for: the same numbers, on the maintainer's models,
one row per measured model and commit. The data-server decision and the kill
criteria are read from here, not from estimates.

Add a row with `npm run measure -- <model.h5>` (it prints the row and a
`detail:` line). Never edit an old row; a re-measure is a new row.

## Machine

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| GPU | AMD Radeon 860M (integrated), through ANGLE on Direct3D 11 |
| Display | 60 Hz; `requestAnimationFrame` is vsync-locked, so **60 fps is the ceiling** of the orbit column |
| Window | 1600 x 1000 (canvas 1584 x 961 at device pixel ratio 1) |
| Stack | Electron 44.5.1, three.js r186, h5wasm 0.10.3, Node 24.19 for the scripts |

## Method

- **first frame**: from `npm run measure` starting the Electron binary to the
  first frame painted with the mesh. It includes Electron start-up (about
  0.4 to 1.4 s of it; the `detail:` line splits it out), the HDF5 read in the
  main process, the copy to the renderer and the mesh build.
- **median fps (orbit)**: one full turn about the vertical (Z) axis through the
  fitted centre over 6 s, the camera moved and the scene rendered every
  animation frame; the median of
  1000 / frame interval. It is capped by vsync, so it answers "is it
  interactive", not "how much headroom". **render CPU** is the median CPU time
  of one `renderer.render` call (CPU only, no GPU time).
- **uncapped orbit** (`npm run measure -- <model.h5> --uncapped`): the same
  orbit with Chromium's `disable-frame-rate-limit` and `disable-gpu-vsync`
  switches, so frames are paced by the CPU + GPU pipeline, not the display.
  The fps is the mean over the 6 s orbit (frames / duration); uncapped frame
  intervals are below the timer resolution, so a median would be quantised.
  The other columns are not comparable in this mode, so it prints its own row.
- **memory**: working set of the main and renderer processes after the orbit
  and the click (`app.getAppMetrics()`), in MB. The GPU process, about 85 to
  215 MB, is reported in the `detail:` line and not in the column.
- **inspector fill**: a scripted click. The script projects the middle of a
  chosen element (a beam-column if the model has one, else an element whose
  chain resolves) to the screen and goes through the same pick path as a mouse
  click, then times until the inspector is painted. "(picked a neighbour)"
  means the ray hit a different element in front of the chosen one; the time
  is still click → chain painted. Of each fill, pick plus state plus DOM take
  2 to 11 ms; the rest is waiting for the next painted frame.

## Rows

| date | commit | model | size (MB) | nodes | elements | first frame (ms) | median fps (orbit) | render CPU (ms) | memory MB main+renderer | inspector fill (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | 14ac06c0 | `fixtures/shoebuckle.h5` | 0.18 | 124 | 123 | 2121 | 59.9 | 0.20 | 223 (128 + 94) | 71.3 |
| 2026-10-03 | 14ac06c0 | maintainer: `ladruno_4D6-24_coarse.model.h5` | 3.46 | 4254 | 5076 (+2028 OpenSees-only) | 2323 | 59.9 | 0.50 | 231 (135 + 96) | 56.1 (picked a neighbour) |
| 2026-10-03 | 14ac06c0 | maintainer: `footing_analysis_composed.h5` | 0.50 | 1218 | 575 | 2106 | 59.9 | 0.30 | 213 (129 + 84) | 38.3 |
| 2026-10-03 | 14ac06c0 | scale probe: `stress_box.h5` | 26.17 | 40093 | 220889 | 3298 | 59.9 | 0.30 | 412 (213 + 199) | 51.1 (picked a neighbour) |
| 2026-10-03 | f030ba33 | maintainer: `sanramon_1A` `model.h5` | 26.00 | 13893 | 16840 | 3821 | 59.9 | 0.30 | 276 (152 + 124) | 47.2 (picked a neighbour) |
| 2026-10-03 | be94cc88 | maintainer: `sanramon_1A` `model.h5`, from `sanramon_1A_h5.py` | 26.00 | 13893 | 16840 | 15189 (machine loaded) | 59.9 | 0.30 | 223 (109 + 115) | 204.1 (picked a neighbour) |

Notes on the rows:

- `ladruno_4D6-24_coarse.model.h5` is the largest maintainer model the
  program's auditor found
  (`C:\Users\nmora\Documents\gitAPE\ladruno-concrete-validation\runs\01_3d_embedded_rebar\sheikh_uzumeri\`):
  1600 `LadrunoBrick` hexes, 928 quads, 2192 rebar line cells and 2028
  `CorotTruss` rebar elements that have no mesh cell. Schema 2.33.0 / 2.21.0.
  The clicked element was a `CorotTruss`, whose chain is element →
  `uniaxialMaterial Steel02` (named `rebar_long` in `/opensees/names`).
- `footing_analysis_composed.h5` (`C:\Users\nmora\Github\apeGmsh\`) is schema
  2.26.0 / 2.19.0, older than that day's reader window (ADR 0113 has since replaced the window with a floor): it opened with a warning banner.
  Its elements are `BezierTri6`, which the syntax table does not know, so the
  inspector reports the chain as not decoded, by type name.
- `sanramon_1A` `model.h5` is the maintainer's San Ramon building. Source:
  `sanramon_1A_h5.py`, which builds it from the maintainer's
  `Documents\gitAPE\Epistemic Uncertanty` project at apeGmsh `32d6a85b`; the
  file is read in place from a session scratchpad and is not committed. 3510 `line2`
  cells, of which 736 are `elasticBeamColumn` columns (`Columns_Set`), 12 841
  `quad4` `ASDShellQ4` shells on 6 `ElasticMembranePlateSection`s named in
  `/opensees/names`, and 489 `point1`. The measured click hit one of the 2774
  `line2` cells that are in no physical group and have no OpenSees element, so
  its inspector says "no row in element_meta"; the stills select a column and
  a shell directly.
- `stress_box.h5` is not a maintainer model. It is a scale probe: two
  tet-meshed blocks, 220 889 tets written by `fem.to_h5` (no `/opensees`
  zone), 7.6 times the largest maintainer file. It is not committed.
- The capped orbit is pinned at the 60 Hz vsync cap on every model; the
  uncapped rows below give the throughput behind it.
- `be94cc88` is the Z-up turntable navigation (#1289); the scripted orbit
  now goes through the turntable code a right-drag uses. Its rows were taken
  while the machine was loaded by other work: Electron took about 10 s to
  start (about 1 s before), which inflates the first frame and the paint
  share of the inspector fill (pick plus state plus DOM stayed at 11 ms). The
  orbit stayed at the 60 fps cap.

### Uncapped orbit (GPU throughput)

| date | commit | model | size (MB) | elements | orbit frames | orbit ms | mean fps (uncapped) |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | efa99699 | `fixtures/shoebuckle.h5` | 0.18 | 123 | 10346 | 6000 | 1724 |
| 2026-10-03 | efa99699 | maintainer: `ladruno_4D6-24_coarse.model.h5` | 3.46 | 5076 (+2028 OpenSees-only) | 4383 | 6009 | 729 |
| 2026-10-03 | efa99699 | maintainer: `footing_analysis_composed.h5` | 0.50 | 575 | 7921 | 6002 | 1320 |
| 2026-10-03 | efa99699 | scale probe: `stress_box.h5` | 26.17 | 220889 | 3866 | 6017 | 643 |
| 2026-10-03 | f030ba33 | maintainer: `sanramon_1A` `model.h5` | 26.00 | 16840 | 4736 | 6004 | 789 |
| 2026-10-03 | be94cc88 | maintainer: `sanramon_1A` `model.h5`, from `sanramon_1A_h5.py` | 26.00 | 16840 | 4179 | 6005 | 696 |

The `be94cc88` row is the median of five runs on the loaded machine (618 to
770 fps). Interleaved with the code before the navigation change on the same
load, the uncapped orbit read 755 and 868 fps at `c69723af` against 744 and
696 at `be94cc88`: the runs overlap, and the change adds no draw work (the
same buffers; one quaternion product per frame).

The scale probe draws 18 718 boundary triangles and still renders 643 frames
a second end to end on the integrated GPU, about 21 times the 30 fps
threshold. (`efa99699` is a merge of `origin/main`; the app code is the same as
at `14ac06c0` except for this measure mode and the decoder fixes, which do not
touch rendering.)

## Kill-criteria verdict (P0)

**(1) Interactive frame rate on a model of the size the maintainer uses.**
Threshold: a median orbit of at least 30 fps, **maintainer-confirmed
(2026-10-03)**. **Pass.** Every measured model orbits at the 60 fps cap,
including the largest maintainer model (3.46 MB, 4254 nodes, about 7100
elements) and a scale probe 7.6 times larger (26 MB, 220 889 tets). Without
the vsync cap the same orbits run at 643 to 1724 fps (mean), so the margin is
measured, not inferred from CPU time. The read
in the main process took 0.1 to 0.4 s on all of them, so nothing here asks for
a data server (D5).

**(2) The inspector is filled from the file alone.** **Pass, with the
interpretation listed below.** No Python and no apeGmsh code ran: the app
reads the file with h5wasm and resolves the chain in TypeScript. Of the four
links of the beam chain, the last one (section → material) is read directly
from an HDF5 path. The other three (element → geomTransf, element →
beamIntegration, beamIntegration → section) are bare tags inside positional
OpenSees arguments. The app finds them with a table of OpenSees command syntax
(`src/chain/signatures.ts`: 16 element types, 6 integration rules), which is
OpenSees knowledge, not apeGmsh semantics, and never decodes apeGmsh replay
tokens. The table is the price; the gaps section says what in the file would
remove it.

**A beam chain on a maintainer model: `sanramon_1A` model.h5.** The archived
maintainer files the auditor found had no beam-columns, so the maintainer's
San Ramon building was rebuilt on `main` (`32d6a85b`). Its columns are
`elasticBeamColumn` with inline properties (no section), so the beam chain is
element → geomTransf, with A, E, G, J, Iy, Iz shown as positional args; its
`ASDShellQ4` shells resolve element → `ElasticMembranePlateSection`, a section
with no material (E, nu, h, rho are its own params). Both resolve with no
unresolved link (`screenshots/sanramon_1A.beam.png`, `sanramon_1A.shell.png`).
The full beam → integration → section → material chain is shown on the fixture
(`examples/shoebuckle_arch.py` on `32d6a85b`).

## Inspector fields and their HDF5 sources

For the V0 architects (#1287). **read** means the value is copied from the
path. **interpreted** means the app needed OpenSees syntax to know which value
to follow. `{alias}` is a gmsh alias (`line2`), `{type}` an OpenSees type
token, `i` a neutral-zone row and `r` an element_meta row.

### Element

| Field | HDF5 source | How |
|---|---|---|
| FEM id | `/elements/{alias}/ids[i]` | read |
| cell type | `/elements/{alias}` (group name); the drawn shape comes from its `@code` (gmsh's type numbering) | read |
| nodes | `/elements/{alias}/connectivity[i]`; for an OpenSees-only element, `/opensees/element_meta/{type}/inline_connectivity[r]` | read |
| physical groups | `/physical_groups/element_side/*/element_ids` (membership) | read |
| row in element_meta | `/opensees/element_meta/*/fem_eids[r]` equal to the FEM id | read (a join on the id) |
| OpenSees type | `/opensees/element_meta/{type}` (group name) | read |
| OpenSees tag | `/opensees/element_meta/{type}/ids[r]` | read |
| args | `/opensees/element_meta/{type}/args[r]`, with `args_str[r]` where a slot is a string; trailing NaN padding to the type's widest row is dropped | read |
| link: transfTag → geomTransf | `args[r][k]` matched to `/opensees/transforms/*@tag`; `k` from the element's OpenSees syntax (`dispBeamColumn`/`forceBeamColumn`: slot 0, new-style 2-tag form only, the old `numIntgrPts secTag transfTag` form is refused; `elasticBeamColumn`: slot 1, 3 or 6) | **interpreted** |
| link: integrationTag → beamIntegration | `args[r][1]` matched to `/opensees/beam_integration/*@tag` (`dispBeamColumn`, `forceBeamColumn`) | **interpreted** |
| inline section properties (`elasticBeamColumn` without a section) | `args[r][0..2]` (2-D: A E Iz) or `args[r][0..5]` (3-D: A E G J Iy Iz) | read, but positional: the inspector shows them in `args` unnamed; naming them would need the same syntax table |
| link: matTag / secTag (other elements) | `args[r][k]` matched to `/opensees/materials/{uniaxial,nd}/*@tag` or `/opensees/sections/*@tag` (`Truss`, `CorotTruss`: slot 1; bricks and tets: slot 0; shells (`ASDShellQ4`, `ShellMITC4`, `ShellDKGQ`, `ShellNLDKGQ`): slot 0; `quad`: slot 2; `SSPquad`: slot 0; `elasticBeamColumn` section form: slot 0) | **interpreted** |

### geomTransf, beamIntegration, section, material (each object)

| Field | HDF5 source | How |
|---|---|---|
| name | `/opensees/names` row whose `kind` is the object's family (`geomTransf`, `beamIntegration`, `section`, `uniaxialMaterial`, `nDMaterial`) and whose `tag` matches; else the object's group name, which the writer sets to `{type}_{tag}` | read |
| type | `{object}@type` | read |
| tag | `{object}@tag` | read |
| params | `{object}@params[j]`, with `{object}@params_str[j]` where a slot is a string | read, but positional: the file has no parameter names, so the inspector shows `[0] 2e11 · [1] 3.75e8 ...`, not `E`, `fy` |
| other attributes | `{object}@{attr}` (for example a transform's orientation attributes and `__deviation__`) | read |
| tables | every dataset under the object (`patches`, `fibers`, `layers`, `per_element_vecxz`, `per_element_emitted_tag`): a plain array as its values (`[1, 0, 0]`; a zero-width one, such as a 2-D transform's `per_element_vecxz` of shape (1, 0), as "empty"), a compound table as row count and columns. Rows are shown in file order and joined to no element: elements reach a transform through the transfTag slot (#1295) | read |
| link: secTag → section (beamIntegration) | `{integration}@params[0]` matched to `/opensees/sections/*@tag` (`Legendre`, `Lobatto`, `Radau`, `NewtonCotes`, `Trapezoidal`, `CompositeSimpson`) | **interpreted** |
| link: section → material (Fiber) | `/opensees/sections/{name}/{patches,fibers,layers}[j].material_ref`, an HDF5 path | read |
| section with no material (`ElasticMembranePlateSection`, `Elastic`, `ElasticShear`) | none: the constants (for `ElasticMembranePlateSection`: E, nu, h, rho) are the section's own `@params`; the chain ends at the section. Which section types reference no material is part of the syntax table | **interpreted** (that the chain ends here) |

## Gaps found (evidence for V0 and V2)

1. **Cross-references are bare tags in positional arguments.** Element →
   transform, element → integration and integration → section need the
   syntax table above. The Fiber section's `material_ref` shows the read-only
   alternative already in the schema: an HDF5 path. Path attributes such as
   `transf_ref`, `integration_ref` and `section_ref` would make every link of
   the beam chain a read.
2. **Parameters have no names in the file.** Materials, sections,
   transforms and integrations store `params` / `params_str` arrays. The
   inspector can show `[1] 375000000` but not `sy = 375 MPa`.
   `architecture/h5-schema.md` (`/opensees/materials`) still documents named
   attributes (`fy=420e6, E=200e9, ...`); the files written today do not
   carry them, so that section lags the writer.
3. **Embedded rebar cells are not linked to their OpenSees elements.** In
   `ladruno_4D6-24_coarse.model.h5` the 2192 rebar `line2` cells have no row in
   `element_meta` (`fem_eids` has no match), and the 2028 `CorotTruss` rows
   carry `fem_eids = -1` with `inline_connectivity`. Clicking a rebar cell
   says "no row in element_meta"; the truss and its material are reachable
   only through the OpenSees-only overlay, which the picker prefers when the
   two coincide.
4. **`/meta/ndm` is the mesh dimension, not the OpenSees ndm.** The 2-D
   fixture (`ops.model(ndm=2, ndf=3)`) has `/meta/ndm = 1`, and the 3-D
   `sanramon_1A` building (ndf 6) has `/meta/ndm = 2`.
5. **`/opensees/transforms/{name}` differs from the schema document.** The
   fixture's `Linear_1` has `per_element_vecxz` of shape (1, 0) and
   `per_element_emitted_tag` of shape (1,), not one row per element as the
   document describes, plus an undocumented `__deviation__` attribute.
6. **Names exist only when the user registered one.** Without `name=`, every
   object is shown by its writer group name (`Fiber_1`, `ASDSteel1D_1`).

## Headless capture (ADR 0112, "Open questions")

**Works on this machine.** `npm run capture` opens a hidden `BrowserWindow`
(`show: false`), selects the scripted target and calls
`webContents.capturePage()`; the stills in `screenshots/` were made that way
(`--pick=ASDShellQ4` for `sanramon_1A.shell.png`; each still is under 0.6 MB).
Two limits: the hidden window is clamped to the screen height, so a tall chain
needs the second still (`<out>.chain-end.png`, inspector scrolled to the end);
and the run needs a desktop session, since Electron still creates a GPU
context.

## Geometry sibling size (ADR 0112 D2a, V2b #1305, V0 amendment 4)

Every session run now writes `<stem>.geometry.h5` beside `model.h5`. The
amendment asks for its size on the measured models; if the sibling were
larger than `model.h5` on the largest model, Q2 (when geometry is captured)
would be reopened. Rows were produced on 2026-10-04 with the V2b branch
(the PR for #1305) by rebuilding each model with `APEGMSH_ARTIFACT_DIR` set,
so the session's unconditional `end()` wrote the neutral `<name>.h5` and the
sibling; `model.h5` is the bridge's composed file (`ops.h5`, neutral +
`/opensees`), the one the rows above measure. Sizes in MB.

| model | `model.h5` (composed) | neutral `<name>.h5` | `<name>.geometry.h5` | sibling / model.h5 | entities (0/1/2/3-D) | triangles | curve points | memberships | `@source` |
|---|---|---|---|---|---|---|---|---|---|
| `fixtures/shoebuckle.h5` (`examples/shoebuckle_arch.py`) | 0.19 | 0.15 | 0.02 | 0.11 | 8 (5/3/0/0) | 0 | 96 | 15 | `temp_mesh` (the example meshes with `dim=1`; no surfaces) |
| maintainer: `sanramon_1A` | 27.27 | 17.68 | 1.47 | 0.054 | 1430 (489/776/165/0) | 25 682 | 24 832 | 719 | `mesh` |
| maintainer: `ladruno_4D6-24_coarse` (`su_4D6-24_coarse`, `build_fem` only) | 3.46 (row above) | 3.01 | 0.32 | 0.09 | 603 | 1 856 | 7 680 | 230 | `mesh` |

- The sibling is 5 to 11 % of `model.h5` on every model, and 5.4 % on the
  largest (San Ramon 1A). **Q2 stays closed.**
- The San Ramon file is 25 682 triangles for 12 841 shell quads (two per
  quad), 24 832 curve points (776 curves x 32 samples) and 719 label /
  physical-group memberships; `@status = ok`. A first capture lost 13 slabs
  (`partial`, 4 008 triangles): their elements use the nodes of the column
  lines *embedded* in them, which a per-surface
  `getNodes(includeBoundary=True)` does not return. The writer reads the
  model's node table once instead; `tests/test_geometry_artifact.py` holds
  the case.
- Capture cost at `generate()` exit is below the build noise: San Ramon
  1A builds in 6.7 s with the capture and both writes included.

## Results sizes (V4)

ADR 0112 D5 leaves the results-size strategy (read lazily in-process, or a
thin local data server that imports only a reader) to measured results sizes.
These rows are that input (V4a, #1707). They time the smallest real read
path, `src/reader/results.ts`; no UI and no store are built on it. The
layout read is `architecture/h5-schema.md`, "`results.h5`: the native results
file" (V4s, #1709), results zone 1.1.0, floor 1.0.0.

### Machine (not the P0 machine above)

| Item | Value |
|---|---|
| OS | Windows 11 IoT Enterprise LTSC 10.0.26100 |
| CPU, RAM | Intel Core i7-13700KF, 64 GB |
| GPU | NVIDIA GeForce RTX 3070, through ANGLE on Direct3D 11 |
| Disk | the files sit on a Samsung 980 PRO NVMe SSD |
| Stack | Electron 44.5.1, h5wasm 0.10.3 (HDF5 2.0.0), Node 24 |

### Method

`npm run build`, then one fresh Electron process per run:

```
node_modules/.bin/electron . --mode=measure-results --file=<x.results.h5> --out=<json> --commit=<sha>
```

It prints the row and a `detail:` line (`src/main/main.ts`, the
`measure-results` block). All reads are in the main process, by hyperslab
(`Dataset.slice`); the datasets are contiguous, as `NativeWriter` writes them.

- **open+index**: `openResults`: the version gate, every stage's `time`,
  every `_ids` and `_element_index`, and the shape and dtype of every
  component. No component data is read.
- **nodal step**: `displacement_x/y/z` at the middle step, 3 hyperslabs of
  `(1, N)`. **cold** is the first read of that step in the process; **warm**
  is the median of 21 repeats. The OS file cache was not flushed (the files
  had just been written or read), so cold means a cold HDF5 and h5wasm, not a
  cold disk.
- **element step**: the first stress component of the gauss group at the
  middle step, one `(1, E, G)` hyperslab.
- **node history**: `displacement_x` of the middle node over all steps, one
  `(T, 1)` hyperslab: a strided read, one value per row of a contiguous
  `(T, N)` dataset.
- **transfer main→renderer**: bytes of one nodal step (3 float64 arrays) and
  the mean, over the scrub, of the `ipcRenderer.invoke` round trip minus the
  read time in main (structured clone both ways). The `detail:` line has the
  same for one gauss step.
- **apply**: per step in the renderer, mean over the scrub: the displacement
  magnitude mapped through jet into a `Float32Array(3N)` colour buffer
  (colour), then `bufferSubData` of the whole buffer, a points draw of N
  vertices and `gl.finish()` (upload/draw). That is the WebGL work three.js
  r186 does for a `needsUpdate` colour attribute, as raw WebGL2 calls: the
  renderer is a blank hidden page, because the app page would start its own
  flow on the file, and three.js is bundled inside that page.
- **scrub**: steps 0 to 99 end to end (invoke, read, transfer, colour, upload,
  draw, finish), one after the other; steps per second.
- **peak memory**: the largest working set of the main and renderer
  processes (`app.getAppMetrics()`), sampled every 20 ms over the whole run.
- **eager whole-history**: the float64 bytes every component of the stage
  would take in memory if it were loaded at once.

### Measured files

None of the ledger's model files is on this machine (`ladruno_4D6-24_coarse`
and `stress_box.h5` were on the maintainer's other machine), so the meshes are
rebuilt to the same size. Every file is written by apeGmsh's own
`NativeWriter` (`src/apeGmsh/results/writers/_native.py`) with
`source_type = "domain_capture"`, `/model` embedded and one partition, by
the repository's `scripts/make_viewer_results_fixtures.py` (then at
`apeGmshViewer/fixtures/make_results_fixtures.py.txt`; same code) on `9a6d3239`. **The values are
synthetic: no solve ran.** They are closed forms of the ids and the step
(the generator's docstring), fed to the writer; `domain_capture` here does
not mean a capture ran. The read path does not look at the values, and a
solve writes the same dtypes and shapes. The files are not committed
(`out/results/`, git-ignored).

- **(a)** `a_maintainer_100`: the maintainer-sized model. A unit square of
  13 x 13 structured triangles extruded in 21 layers: 4 312 nodes and 7 098
  6-node wedges (the ledger's largest maintainer model is 4 254 nodes and
  7 104 elements). 100 steps; `displacement_x/y/z` and 6 stress components at
  6 Gauss points.
- **(b)** `b_stress_box_100`: the scale probe. Two touching tet-meshed blocks,
  as `stress_box.h5` was; the unstructured mesher gave 217 820 tets and
  40 843 nodes (the ledger's probe had 220 889 tets and 40 093 nodes, 1.4 %
  more tets). 100 steps; 3 nodal and 6 stress components at 1 Gauss point.
- **(c)** `c_maintainer_5000`: the (a) mesh over 5 000 steps (a ground-motion
  history at dt = 0.01 s); 3 nodal components and `stress_xx` only, at 6
  Gauss points (6 stress components would have been a 10 GB file).

### Rows

Two cold runs of each file, one after the other, at `9a6d3239`.

| date | commit | file | size (MB) | nodes | elements | steps | fields | open+index (ms) | nodal step cold / warm (ms) | element step cold / warm (ms) | node history (ms) | transfer main→renderer | apply colour + upload/draw (ms) | scrub (steps/s) | peak memory MB main+renderer | eager whole-history (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-12 | 9a6d3239 | (a) `a_maintainer_100` (synthetic, real NativeWriter), run 1 | 206.1 | 4312 | 7098 | 100 | 3 nodal + 6 x 6-pt gauss | 25.1 | 2.87 / 0.96 | 0.77 / 0.37 | 1.3 | 101 KB in 0.77 ms | 0.08 + 0.03 | 609.0 | 234 (134 + 102) | 204.8 |
| 2026-10-12 | 9a6d3239 | (a) `a_maintainer_100` (synthetic, real NativeWriter), run 2 | 206.1 | 4312 | 7098 | 100 | 3 nodal + 6 x 6-pt gauss | 26.9 | 2.92 / 0.93 | 0.61 / 0.42 | 1.8 | 101 KB in 0.72 ms | 0.09 + 0.03 | 584.1 | 234 (134 + 102) | 204.8 |
| 2026-10-12 | 9a6d3239 | (b) `b_stress_box_100` (synthetic, real NativeWriter), run 1 | 1118.4 | 40843 | 217820 | 100 | 3 nodal + 6 x 1-pt gauss | 21.2 | 2.05 / 0.82 | 1.16 / 0.70 | 3.9 | 957 KB in 1.82 ms | 0.53 + 0.05 | 274.9 | 393 (198 + 200) | 1090.6 |
| 2026-10-12 | 9a6d3239 | (b) `b_stress_box_100` (synthetic, real NativeWriter), run 2 | 1118.4 | 40843 | 217820 | 100 | 3 nodal + 6 x 1-pt gauss | 24.3 | 2.62 / 1.09 | 1.42 / 0.92 | 3.8 | 957 KB in 1.60 ms | 0.49 + 0.06 | 313.1 | 386 (196 + 201) | 1090.6 |
| 2026-10-12 | 9a6d3239 | (c) `c_maintainer_5000` (synthetic, real NativeWriter), run 1 | 2119.4 | 4312 | 7098 | 5000 | 3 nodal + 1 x 6-pt gauss | 22.8 | 3.15 / 0.99 | 0.75 / 0.36 | 36.8 | 101 KB in 0.40 ms | 0.06 + 0.02 | 989.1 | 239 (134 + 106) | 2118.1 |
| 2026-10-12 | 9a6d3239 | (c) `c_maintainer_5000` (synthetic, real NativeWriter), run 2 | 2119.4 | 4312 | 7098 | 5000 | 3 nodal + 1 x 6-pt gauss | 20.6 | 2.55 / 0.68 | 0.72 / 0.35 | 38.9 | 101 KB in 0.51 ms | 0.10 + 0.02 | 808.4 | 244 (134 + 111) | 2118.1 |

`detail:` lines, run 1 / run 2:

- (a): element step 42 588 values; node history warm 0.69 / 1.01 ms; scrub
  round trip 1.53 / 1.58 ms, of which read in main 0.76 / 0.87 ms; one gauss
  step (333 KB) transfers in 0.86 / 0.60 ms. Eager: nodal 9.9 MB + gauss
  195.0 MB.
- (b): element step 217 820 values; history warm 1.86 / 2.10 ms; round trip
  3.05 / 2.64 ms, read in main 1.23 / 1.04 ms; one gauss step (1 702 KB) in
  3.30 / 2.59 ms. Eager: nodal 93.5 MB + gauss 997.1 MB.
- (c): element step 42 588 values; history warm 36.50 / 36.69 ms; round trip
  0.93 / 1.12 ms, read in main 0.53 / 0.61 ms; one gauss step (333 KB) in
  0.52 / 0.65 ms. Eager: nodal 493.5 MB + gauss 1 624.6 MB.

**Repeatability (the card's 20 % rule), first two runs.** Within 20 % between
the two runs: open+index, the nodal cold read, the node history of (b) and
(c), the transfer, the scrub rate (largest gap: (c), 989 against 808 steps/s,
18 %), peak memory and every size. **Outside 20 %**: the warm nodal read of
(b) and (c) (0.82 / 1.09 and 0.99 / 0.68 ms), the element reads of (a) and
(b) (cold 0.77 / 0.61; warm 0.70 / 0.92 ms) and the 100-step history of (a)
(1.3 / 1.8 ms). Each of those is a single read under 2 ms, where one OS
scheduling slice is a third of the value; the apply columns are means of 100
steps for the same reason (the renderer's clock is coarsened to 0.1 ms). The
re-measure below shows the scrub rate of (c) does **not** repeat within 20 %
once the machine's load varies.

### Re-measure of (a) and (c), with the machine's load (review round 1, #1717)

The reviewer re-ran (c) four times while other sessions held the CPU at 87
to 100 %. They got a scrub of 475 to 561 steps/s and a 5 000-step history of
55 to 60 ms, against 808 to 989 steps/s and 37 to 39 ms above; peak memory
matched. These are four more cold runs of each, interleaved (a, c, a, c, …),
at `967f565a`, which changes only the version gate, so the read path is the
same. **CPU** is the machine's total processor time
(`\Processor(_Total)\% Processor Time`): the mean of 3 s just before the run,
and the mean and maximum of the run's first 5 s. Other sessions were running
on the machine throughout. The values are synthetic, written by the real
`NativeWriter`.

| run | file | CPU before / during mean / during max | open+index (ms) | nodal step cold / warm (ms) | element step cold / warm (ms) | node history (ms) | transfer | apply (ms) | scrub (steps/s) | peak MB main+renderer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (a) (synthetic, real NativeWriter) | 57 / 52 / 74 % | 24.9 | 3.05 / 0.76 | 0.88 / 0.43 | 1.4 | 101 KB in 0.55 ms | 0.08 + 0.03 | 726.7 | 234 (135 + 102) |
| 1 | (c) (synthetic, real NativeWriter) | 43 / 46 / 60 % | 23.7 | 3.06 / 0.84 | 0.58 / 0.27 | 38.4 | 101 KB in 0.46 ms | 0.10 + 0.01 | 792.4 | 237 (135 + 104) |
| 2 | (a) | 46 / 56 / 66 % | 26.5 | 2.81 / 0.85 | 0.61 / 0.43 | 2.1 | 101 KB in 0.58 ms | 0.10 + 0.02 | 704.7 | 234 (134 + 103) |
| 2 | (c) | 78 / 55 / 68 % | 26.4 | 2.81 / 0.92 | 0.80 / 0.45 | 46.3 | 101 KB in 0.52 ms | 0.07 + 0.02 | 825.1 | 239 (136 + 104) |
| 3 | (a) | 47 / 42 / 51 % | 22.4 | 3.01 / 0.73 | 0.64 / 0.30 | 1.3 | 101 KB in 0.68 ms | 0.09 + 0.03 | 670.2 | 234 (133 + 102) |
| 3 | (c) | 49 / 50 / 58 % | 20.3 | 1.91 / 0.90 | 1.24 / 0.36 | 43.6 | 101 KB in 0.85 ms | 0.11 + 0.03 | 536.2 | 240 (137 + 104) |
| 4 | (a) | 41 / 43 / 59 % | 29.2 | 3.04 / 0.86 | 1.50 / 0.41 | 1.7 | 101 KB in 0.55 ms | 0.08 + 0.02 | 766.9 | 234 (134 + 102) |
| 4 | (c) | 40 / 48 / 56 % | 22.8 | 2.52 / 1.04 | 0.85 / 0.39 | 40.3 | 101 KB in 0.43 ms | 0.07 + 0.02 | 885.0 | 237 (134 + 103) |

Ranges over every run of each file, including the two at `9a6d3239` and,
for (c), the reviewer's four. The **conservative** value is the worst of the
range, and it is what the evidence table uses:

| file | scrub (steps/s): range, conservative | node history (ms): range, conservative | within 20 %? |
|---|---|---|---|
| (a) | 584 to 767; **584** | 1.3 to 2.1; **2.1** | scrub: no (31 %) across both rounds, yes (14 %) within this round; history: no (sub-2 ms reads) |
| (c) | 475 to 989 (this round 536 to 885; reviewer 475 to 561); **475** | 37 to 60 (this round 38 to 46; reviewer 55 to 60); **60** | no: both move with the machine's load |

**The "(c) scrubs faster than (a)" reading is retracted.** The first rows put
(c) at 808 to 989 steps/s against 584 to 609 for (a), at the same per-step
payload (101 KB, 3 components of 4 312 nodes). In this round (a) runs 670 to
767 and (c) 536 to 885: the ranges overlap, and the (c) spread is about as
wide as the gap was. The per-step read is a hyperslab of one row, so it does
not depend on how many steps the dataset has. The scrub rate measures the
machine's load as much as the read path, and nothing here shows that (a) and
(c) differ. Peak memory, the transfer bytes and the open time do repeat, at
every load seen.

### MPCO data point (raw HDF5 open and one dataset, no MPCO reader)

`--mode=measure-results --file=<x.mpco> --dataset=<path>` opens the file with
h5wasm and reads one dataset; it decodes nothing.

| file | size (MB) | open (ms), run 1 / 2 | dataset | read (ms), run 1 / 2 | bytes |
|---|---|---|---|---|---|
| `STKO_to_python/src/STKO_to_python/examples/New/results_nodes.mpco` (the card's file) | 36.5 | 4.3 / 4.8: **refused** | none | n/a | n/a |
| `STKO_to_python/stko_results_examples/elasticFrame/QuadFrame_results/results.part-0.mpco` | 7.3 | 3.8 / 3.8 | `/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA/STEP_0`, (347, 3) float64 | 4.9 / 5.0 | 8 328 |

- The card's 37 MB file is **truncated** on this machine: HDF5 reads
  `eof = 38 273 024` against a stored end of `41 135 603` and refuses the
  superblock. h5py refuses it too, so the file is damaged, not the reader.
  The intact QuadFrame partition (one of two) stands in.
- **Finding:** h5wasm does not throw when HDF5 refuses a file. It returns a
  `File` whose `file_id` is `-1`, and the next call fails with an unrelated
  message. `results.ts` and the MPCO probe check the id and refuse by name
  (`test/results/results.test.ts`, "a truncated file is refused at open").
  `reader/read.ts`, the model reader, does not check it.

### Evidence for the two options (ADR 0112 D5)

What the rows say about each option. This is not a recommendation; the
maintainer decides. Every number comes from the **synthetic files written by
the real `NativeWriter`** above; no solve ran. Where runs differ by more than
20 %, the table gives the conservative (worst) value, with the range in
brackets.

| Question | In-process lazy read (main reads with h5wasm, IPC to the renderer), as measured (synthetic, real NativeWriter) | Thin local data server (a separate process that imports a reader) |
|---|---|---|
| Time to open the largest file | 20 to 29 ms for every file, 206 MB to 2.1 GB: the index reads no component data | not measured: the same index read, plus the server's start-up and a connection |
| One step on screen (read, transfer, colour, upload, draw) | conservative scrub, end to end: (a) **584** steps/s [584 to 767], 1.7 ms per step; (b) **275** [275 to 313], 3.6 ms; (c) **475** [475 to 989, under 40 to 100 % CPU load from other work], 2.1 ms. The read in main is 0.5 to 1.2 ms of a step | the same read, plus one more hop for the same bytes (101 KB for (a) and (c), 957 KB for (b) per nodal step; the measured main→renderer hop costs 0.4 to 1.8 ms for those sizes) |
| A node's history | 100 steps: (a) **2.1** ms [1.3 to 2.1], (b) **3.9** ms [3.8 to 3.9]; 5 000 steps: (c) **60** ms [37 to 60, under load]. A strided read of a contiguous `(T, N)` dataset; chunked datasets in the writer would change it | the same read, in another process |
| Memory | peak 234 to 393 MB, main plus renderer, on files of 206 MB to 2.1 GB: memory follows the step, not the file | the reader's memory moves out of the app's processes |
| Eager whole-history load, for comparison | 205 MB (a), 1.09 GB (b), 2.1 GB (c), against those peaks | the same bytes, in the server's process |
| What only a server adds | none measured | a reader the app cannot run: Python decoders (the MPCO layout, `.ladruno`, derived results such as von Mises, computed on read). Raw HDF5 is not one: h5wasm opened an MPCO file and read a dataset in 4 + 5 ms |
| Failure modes seen | h5wasm's silent open failure (above), refused by `results.ts` | not measured |

### D5 revisit trigger (V4b, #1728)

The maintainer ruled for the in-process lazy read (#1283, 2026-10-11), and
V4b builds on it. Reopen ADR 0112 D5, with the row that shows it, when a
**real** results file (a solve, not a synthetic one):

- scrubs below **30 steps/s** end to end (the "scrub" column above, measured
  with `--mode=measure-results`), or
- peaks above **2 GB** of memory, main plus renderer, while it is open.

Add that file's row to "Rows" above first, with its machine.

### Contour scrub (V4c1, #1756)

The end-to-end scrub with a nodal contour on, in the app's own window: the
model (`<stem>.h5`, written by `scripts/make_viewer_results_fixtures.py
--measure` from the same FEMData, so it pairs by `fem_hash`) with its
results beside it, coloured by `displacement` magnitude over the
whole-history range. Each step is `setResultStep`, the read in main, the IPC,
the step range, the vertex colours of the 2 860 drawn triangles, the upload
and a draw ended by `gl.finish()`; the scrub visits every step once. The
history scan (every step of the three components, read in chunks in main)
runs first and is timed apart. The CPU column is the whole machine's load
(`os.cpus()` ticks): over 1 s idle before the run, and over the scrub.
Same machine as "Results sizes (V4)" above. Runs 1 to 3 are at `fd13d8dd`
(vertex nodes matched by position); runs 4 to 6 at `db36d1eb` (vertex nodes
from `mesh/build.ts`, the code under review), which are the ones to quote.

```powershell
$env:PYTHONPATH = "src"
& C:\Users\nmb\venv\opensees_venv\Scripts\python.exe scripts/make_viewer_results_fixtures.py --measure apeGmshViewer/out/results --only a --only c
cd apeGmshViewer
npm run measure -- out/results/a_maintainer_100.h5     # prints the contour row after the P0 row
npm run measure -- out/results/c_maintainer_5000.h5
```

| run | file | steps | CPU before / during | history scan (ms) | scrub (steps/s) | step median / max (ms) | peak MB main+renderer |
|---|---|---|---|---|---|---|---|
| 1 | (a) | 100 | 32 / 42 % | 39 | 320.1 | 2.90 / 5.7 | 296 (148 + 148) |
| 1 | (c) | 5 000 | 32 / 43 % | 490 | 383.0 | 2.40 / 7.1 | 382 (170 + 213) |
| 2 | (a) | 100 | 46 / 49 % | 52 | 325.8 | 2.80 / 6.0 | 297 (146 + 151) |
| 2 | (c) | 5 000 | 40 / 44 % | 486 | 380.8 | 2.40 / 7.2 | 371 (158 + 214) |
| 3 | (a) | 100 | 48 / 60 % | 49 | 272.6 | 3.50 / 5.8 | 298 (151 + 148) |
| 3 | (c) | 5 000 | 40 / 46 % | 486 | 361.4 | 2.50 / 8.4 | 380 (169 + 211) |
| 4 | (a) | 100 | 59 / 67 % | 52 | 246.0 | 3.80 / 8.3 | 306 (150 + 156) |
| 4 | (c) | 5 000 | 62 / 68 % | 600 | 270.6 | 3.50 / 11.3 | 380 (166 + 215) |
| 5 | (a) | 100 | 54 / 59 % | 53 | 266.6 | 3.60 / 6.2 | 305 (148 + 156) |
| 5 | (c) | 5 000 | 69 / 66 % | 494 | 272.7 | 3.30 / 11.7 | 398 (163 + 235) |
| 6 | (a) | 100 | 59 / 71 % | 57 | 256.8 | 3.80 / 6.6 | 302 (148 + 154) |
| 6 | (c) | 5 000 | 61 / 63 % | 488 | 306.6 | 3.10 / 10.3 | 392 (170 + 222) |
| R | (a) | 100 | n/a / 91 % | n/a | 103 | n/a / 188 | 310 |

Row R is the reviewer's re-measure of (a) at `07e6fd03` (#1761 round 1), at 91 % CPU; the
reviewer reported the scrub, the worst step and the peak, not the other columns.

Ranges and the conservative (worst) value of each, which is the one to quote.
The worst (a) values are the reviewer's run R at 91 % CPU; the worst (c) values
are from runs 4 to 6, under a busier machine (54 to 71 % CPU against 42 to 60 %
for runs 1 to 3). The scrub follows the machine's load more than the code:

| file | scrub (steps/s): range over runs 1 to 6, **worst** | step max (ms), worst | history scan (ms), worst | peak MB, worst | CPU during |
|---|---|---|---|---|---|
| (a) | 103 to 326, **103** | 188 | 57 | 310 | 42 to 91 % |
| (c) | 271 to 383, **271** | 11.7 | 600 | 398 | 43 to 68 % |

- **The D5 revisit trigger does not fire**: the worst scrub, 103 steps/s
  at 91 % CPU, is three times the 30 steps/s floor (one step took 188 ms
  there, so a loaded machine shows single stalls), and the worst peak, 398 MB, is a fifth
  of 2 GB. These are synthetic files, not a solve (the trigger names a real
  file).
- With contours on, the worst scrub is below V4a's worst for the same files
  ((a) 103 against 584, (c) 271 against 475 steps/s): the step range, the
  per-vertex colours and a finished draw (`gl.finish()`, which V4a's
  raw-WebGL apply did not wait for) are now inside each step. Both are
  measured on a loaded machine, and the ranges say how much that moves them.
- The history scan of (c), 5 000 steps of three components (517 MB of
  values), takes 0.5 to 0.6 s, reading about 4 MB per chunk. The peak, 296 to
  398 MB, is above V4a's 234 to 240 MB: this window also holds the model,
  its mesh and its panels, which V4a's blank measuring page did not.

### Deformed contour scrub (V4d, #1765)

The contour scrub above, run again in the same window with the deformed
shape on at true scale: each step also waits for the displacement of every
vertex (through the vertex node ids), the new positions of the drawn faces,
segments and (runs 4 to 6) outline edges written in place, their normals and
bounds, and a finished draw (`gl.finish()`). `npm run measure` prints this
row after the contour row; the commands are the ones above. The files'
synthetic displacement is huge (`id + 0.001 k`), which costs nothing extra.
The history-scan column is not re-timed in the deformed pass (it reads 1 ms
there), so it is left out. Runs 1 to 3 are at `8f440c5d`, with the outline
edges hidden on the deformed mesh; runs 4 to 6 at `5b51405f`, where the edges
follow their nodes (the code under review, the ones to quote). Same machine,
with the contour-only row of the same run beside each:

| run | file | steps | CPU before / during | scrub (steps/s), contour → **deformed** | step median / max (ms), deformed | peak MB main+renderer, deformed |
|---|---|---|---|---|---|---|
| 1 | (a) | 100 | 41 / 40 % | 325.1 → **235.8** | 3.80 / 17.7 | 315 (154 + 161) |
| 1 | (c) | 5 000 | 41 / 40 % | 350.5 → **260.8** | 3.60 / 9.8 | 454 (235 + 218) |
| 2 | (a) | 100 | 33 / 42 % | 286.5 → **240.8** | 3.90 / 7.3 | 321 (154 + 167) |
| 2 | (c) | 5 000 | 42 / 48 % | 368.7 → **247.1** | 3.70 / 10.0 | 400 (169 + 231) |
| 3 | (a) | 100 | 47 / 46 % | 325.2 → **224.4** | 4.20 / 8.4 | 321 (152 + 169) |
| 3 | (c) | 5 000 | 40 / 52 % | 360.9 → **257.2** | 3.60 / 11.4 | 385 (160 + 225) |
| 4 | (a) | 100 | 33 / 33 % | 340.3 → **240.9** | 3.90 / 6.8 | 317 (159 + 159) |
| 4 | (c) | 5 000 | 30 / 40 % | 401.4 → **245.8** | 3.80 / 10.8 | 443 (230 + 213) |
| 5 | (a) | 100 | 37 / 47 % | 344.0 → **221.8** | 4.30 / 8.0 | 319 (157 + 162) |
| 5 | (c) | 5 000 | 34 / 39 % | 403.2 → **247.3** | 3.80 / 10.8 | 393 (157 + 236) |
| 6 | (a) | 100 | 34 / 44 % | 342.6 → **222.6** | 4.30 / 7.7 | 310 (152 + 158) |
| 6 | (c) | 5 000 | 37 / 45 % | 393.8 → **238.2** | 3.90 / 11.5 | 409 (196 + 213) |
| R | (a) | 100 | n/a / 47 % | n/a → **209.3** | n/a / 9.0 | 326 |

Row R is the reviewer's re-measure of (a) at `fca86de6` (#1820 round 1), at
47 % CPU; the reviewer reported the scrub, the worst step and the peak, not
the other columns. It is the worst (a) value, and the one quoted.

Worst over runs 4 to 6 and R (the code under review), and over every run:

| file | deformed scrub (steps/s): runs 4 to 6 and R, **worst** (every run) | step max (ms), worst | peak MB, worst | CPU during |
|---|---|---|---|---|
| (a) | 209 to 241, **209** (209) | 9.0 (17.7) | 326 (326) | 33 to 47 % |
| (c) | 238 to 247, **238** (238) | 11.5 (11.5) | 443 (454) | 39 to 45 % |

- **The D5 revisit trigger does not fire**: the worst deformed scrub, 209
  steps/s (the reviewer's run), is seven times the 30 steps/s floor, and
  the worst peak, 454 MB, under a quarter of 2 GB. The machine was at 33 to
  52 % CPU (V4c1's reviewer run fell to a third of the author's at 91 %).
- The deformed shape costs about 1 ms a step on these files (median 2.5 to
  3.3 ms → 3.6 to 4.3 ms): the per-vertex displacement, the positions of
  2 860 triangles, their normals and the outline edges, and the legend's
  "step loading" line.

### Element contour scrub (V4c2a, #1833)

The contour scrub above with an **element** field on: the first drawable
element field of a verified file (`stress` · `xx`), `gauss_cell_averaged`
(the mean of each element's 6 Gauss points, flat per cell), over the
whole-history range. Each step reads the nodal components and the one Gauss
component in main (one hyperslab per group), sends both over IPC, reduces
the Gauss block per element, joins it to the drawn cells by FEM id (the join
is cached per results read), measures the step range, colours every drawn
primitive and finishes a draw (`gl.finish()`). The second row runs it again
with the deformed shape on at true scale. `npm run measure` prints both rows
after the nodal ones when the results file is verified and has an element
field.

The file is the (a) mesh again, written **stamped** (`element_index_space =
"fem"`) by the real NativeWriter, 100 steps, `displacement_x/y/z` and
`stress_xx`, `stress_yy` at 6 Gauss points of 7 098 wedges (76 MB):

```powershell
$env:PYTHONPATH = "src"
& C:\Users\nmb\venv\opensees_venv\Scripts\python.exe scripts/make_viewer_results_fixtures.py --elements-measure apeGmshViewer/out/elements
cd apeGmshViewer; npm run measure -- out/elements/e_solid_100.h5
```

Same machine as the V4c1 and V4d rows, at `b8f69556`. The history scan is
timed in the first pass only (the deformed pass finds it done: 1 ms).

| run | field | CPU before / during | history scan (ms) | scrub (steps/s) | step median / max (ms) | peak MB main+renderer |
|---|---|---|---|---|---|---|
| 1 | element stress xx, gauss_cell_averaged | 31 / 34 % | 66 | 253.2 | 3.70 / 6.3 | 360 (164 + 196) |
| 1 | the same, deformed ×1 | 27 / 26 % | n/a | 201.1 | 4.70 / 7.8 | 360 (157 + 204) |
| 2 | element stress xx, gauss_cell_averaged | 20 / 23 % | 66 | 275.6 | 3.40 / 6.5 | 350 (166 + 185) |
| 2 | the same, deformed ×1 | 19 / 33 % | n/a | 184.5 | 4.90 / 9.5 | 355 (158 + 197) |
| 3 | element stress xx, gauss_cell_averaged | 20 / 27 % | 65 | 274.2 | 3.40 / 5.9 | 351 (170 + 181) |
| 3 | the same, deformed ×1 | 24 / 29 % | n/a | 192.1 | 4.90 / 8.2 | 357 (166 + 191) |

Worst over the three runs: element scrub **253** steps/s (worst step
6.5 ms), deformed element scrub **185** steps/s (worst step 9.5 ms), peak
**360 MB**, at 23 to 34 % CPU during the scrub. The nodal rows of the same
runs read 339 (nodal) and 230 (nodal, deformed) steps/s.

- **The D5 revisit trigger does not fire**: the worst element scrub, 185
  steps/s deformed, is six times the 30 steps/s floor, and the peak, 360 MB,
  under a fifth of 2 GB.
- An element field costs about 1 ms a step over the nodal contour on this
  file (median 2.8 → 3.4 to 3.7 ms): the Gauss block (7 098 × 6 values) over
  IPC, its per-element mean, and the per-cell colours.
