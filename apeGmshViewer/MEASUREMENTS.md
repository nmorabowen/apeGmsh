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
