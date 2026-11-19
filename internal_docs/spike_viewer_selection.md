# Spike: how apeGmshViewer should find what the mouse selects (2026-11-19)

Status: evidence for [ADR 0138](../architecture/decisions/0138-apegmshviewer-selection.md). Where the two differ, the ADR
decides. The throwaway harness and its result files are on the unpushed branch `viewer/selection-spike` under
`apeGmshViewer/spike/selection/`. They are not on `main`.

## Answer

An ID image read on the GPU, plus screen projection on the CPU, is fast enough for click, box and lasso selection of every
target kind on the models tested. Today's highlight is the one part that does not scale.

1. **A press costs a few milliseconds.** Drawing the element ID image and reading it back took 1 to 3 ms. Reading the whole
   image into memory took about 6 ms more. After that, a click or hover lookup took under 0.1 ms.
2. **Every region update fit well inside one frame.** Box and lasso, visible or through, for elements, nodes and 8 points per
   element: p95 at most 5.2 ms, and no single update over 7.2 ms.
3. **On the shell model, the ID image agreed with today's ray cast on all 120 sampled clicks.** On the slab model with beam
   lines, they disagreed on 34 of 40 clicks, for a reason not yet established (below).
4. **Today's highlight took 175 to 217 ms from selecting every element to the painted frame, and an indexed highlight took under 2 ms to build.**
   The indexed highlight shares the drawn positions and lets the halo pass draw the outline. Orbit frame time was the same
   with either.

## Method

The harness runs the real app. `build.mjs` bundles `src/` unchanged and swaps the measure harness at bundle time.
`harness.ts` reads the viewport's own drawn geometry and does not change it.

- **Build measured:** `main` at `341425c9c` plus the harness (`a2930a84c` on the spike branch).
- **Models:** `tims_c3b` (26,970 six-node shell triangles drawn as 26,970 triangles; 13,655 drawn nodes, 54,279 nodes in
  all) and `yoo_slab_N56_75` (21,672 shell triangles and 2,480 line elements; 11,245 nodes).
- **Machine:** Ryzen AI 7 PRO 350, Radeon 860M integrated GPU, canvas 1584 x 961 at device pixel ratio 1.
- **Each launch:** three camera poses. At each pose: 5 draws and reads of each ID image, 40 clicks sampled at the centres of
  drawn triangles and segments, and a drag of 40 updates growing from 5% to 95% of the canvas. Then the same at one pose
  with six section planes. The highlight is measured at 1%, 10% and 100% of the elements, through the real store.
- **Repeats:** 3 launches per model. Values are **median [lowest..highest launch]**. Within a launch, a figure is the median
  over the three poses.
- **Load:** a launch started only when no other measuring app ran, machine CPU was at most 60% busy, and enough memory was
  free (3 GB for these models). CPU was 46% to 59% busy before the `tims_c3b` launches and 4% to 59% before the slab ones.
- **Gauss points** are a stand-in: 8 random points per element on drawn triangles (215,760 on `tims_c3b`). Placing real Gauss
  points needs a results stage.
- **Frame times** use the time the callback ran, as ADR 0137 requires.

## Results

### The ID image

Draw time is CPU submission only. The first read waits for the GPU, so draw plus first read is the cost of a press.

| Image | `tims_c3b`: draw + first read | slab: draw + first read | Whole image into memory |
|---|---|---|---|
| elements | 0.3 + 1.6 ms | 0.2 + 1.1 ms | 5.6 to 5.8 ms (6 MB) |
| nodes | 0.3 + 2.9 ms | 0.2 + 1.6 ms | 5.0 to 6.0 ms |
| Gauss stand-in | 0.2 + 9.8 ms [8.4..12.0] | 0.3 + 3.6 ms | 4.8 to 5.1 ms |

Building the ID textures took 1.5 to 1.8 ms and 0.13 MB.

### Clicks

| | `tims_c3b` | slab |
|---|---|---|
| today's ray cast, p50 / p95 | 1.2 / 1.5 ms | 0.2 / 1.1 ms |
| lookup in a read ID image | under 0.1 ms | under 0.1 ms |
| same element, of 40 per pose | 40, 40, 40 in every launch | 4 to 7 |

With six section planes, `tims_c3b` gave 11 clicks the same, 28 where both missed, and 1 where only the ID image found an
element. That click was within the image's 2-pixel search radius of a cut edge.

**The slab disagreement is explained: the ray cast was wrong** (diagnostic launch, 2026-11-19). One launch on the slab
model, behind the same load gate (15% CPU, 3.05 GB available), recorded both hit distances for the first 12 differences
at each pose. The draw state was plain (no offsets, no extruded frames, opacity 1, undeformed), so the ID image drew what
the screen drew. The same element on 17 of 120 clicks again. In all 36 recorded differences the ray cast returned a beam;
in 34 the ID image returned the element drawn at the exact pixel, in 2 one found by its 2-pixel search. Two causes:

1. **A line within about 5 px of the cursor wins over the face under it.** Beams are drawn 3 px wide and the ray cast's
   line threshold is 7 px, so a line counts within (3 + 7) / 2 = 5 px, and `hitElement` tries lines before faces without
   comparing depths. 15 of the 36 were clicks on a slab triangle taken by a nearby beam, and 6 more by another beam
   within 5 px. In 6 of the 36 the beam was behind the face (`lineHitDist` > `faceHitDist`): hidden, but picked.
2. **Among the segments within that tolerance, the one nearest the eye wins**, not the one nearest the cursor: three
   sorts the hits by distance along the ray. On a beam split into segments shorter than about 10 px on screen, the
   neighbouring segment on the near side won: 13 of the 36 were a beam against the beam beside it. On a straight beam
   sloping toward the eye, alone, the ray cast returned the clicked segment on 100 of 100 clicks with 22 px segments,
   31 of 200 with 11 px, and 0 of 400 with 5.6 px.

`tims_c3b` draws no lines, which is why it agreed on every click. Not recorded: which element each click sampled (the
isolated reproduction stands in for it), and the 67 differences after the first 12 per pose. Evidence: the spike branch
at `504eaa05c`, `results/slabdiag-0.json` and `diag-ray.mjs`. Clicks moved to the ID image in SEL-2's first PR (ADR 0138,
amendment note of 2026-11-19).

### Region updates

p95 per update, over 40 updates, in ms. "Visible" uses the ID image read at the press. "Through" uses positions projected at
the press, and leaves out what section planes cut away.

| Update | `tims_c3b` | slab |
|---|---|---|
| box, visible, elements touching | 2.4 [2.4..2.7] | 1.6 |
| box, visible, elements fully inside | 1.8 | 1.4 |
| box, through, elements fully inside | 1.7 | 1.3 |
| box, through, elements touching (bounding test) | 0.6 | 0.4 |
| box, visible nodes | 2.4 | 1.7 |
| box, through, all nodes | 1.1 | 0.2 |
| box, visible Gauss stand-in | 4.4 [3.5..5.2] | 3.0 |
| box, through Gauss stand-in | 4.2 | 3.4 |
| lasso, visible, elements touching | 2.9 [2.5..3.7] | 1.9 |
| lasso, through, Gauss stand-in | 4.5 [4.5..5.0] | 4.1 |

Projecting at the press took under 2 ms for every kind. Filling the lasso took under 1.2 ms.

### Highlight

| Selected | Today: dispatch to painted | Indexed: build, then first draw |
|---|---|---|
| 1% (270 elements) | 41 ms [36..45] | 1.8 ms, then 20 ms |
| 10% | 48 ms [36..49] | 0.9 ms, then 14 ms |
| 100% (26,970) | 182 ms [175..217] | 0.4 ms, then 16 ms |

The slab gave 176 ms [105..204] for 100% today. Orbit frame p95 stayed at 17.0 to 17.4 ms with either highlight, against
16.8 to 17.2 ms with nothing selected. The indexed highlight shades the selected faces and lets the halo pass draw their
outline. It does not draw the outline today's code builds edge by edge, so a block of selected elements reads as one region.

### Selection data

On 26,970 elements, replacing, adding, removing half and inverting a membership array each took at most 0.5 ms. Today's
reducer, run in Node on a synthetic model, took 2.8 s to add one element to a selection of 26,970 and 86 s for 130,000,
because it looks up each existing pick for each declaration.

## Not established

- **`yoo_building_SLE_T2`** (128,923 cells) was not measured. The machine never had the 9 GB free memory gate set for it
  during the session, and this model hung the app under memory pressure before (ADR 0137).
- Device pixel ratio above 1, a discrete GPU, and other machines.
- The deformed shape. The ID image uses the drawn positions, so it should follow, but no launch tested it.
- Real Gauss point layouts, hinge markers and fibers.
- How large the pick radius must be for lines and points to feel right. The spike used 2 pixels.
- "Touching" through selection uses a bounding test. An exact test of each triangle edge against a lasso was not timed.
- Whether reading the whole image once per press is better than reading only the region's rectangle on each update.
