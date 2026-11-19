# Spike: how apeGmshViewer should host tools (2026-11-18)

Status: evidence for [ADR 0137](../architecture/decisions/0137-apegmshviewer-large-models-and-tool-hosting.md). Where the two
differ, the ADR decides; the thresholds, slices and contract below are this spike's proposals.
The throwaway prototype and its full result tables (`apeGmshViewer/spike/`) are on the local branch
`guppi/brave-chandrasekhar-1a3339`, commit `3fe38585c`. They are not on `main`, and paths under `spike/` refer to that commit.

## Answer

The direction holds, with three changes.

1. **Own windows should be separate renderers, not child windows drawn by the main page.** A tool that computes for
   40 ms per update delays a third of the viewport's frames from a child window and none from a separate renderer.
2. **A separate renderer does not protect the viewport from GPU work.** All windows share one GPU process. A chart that
   sends about 800,000 line segments to an accelerated canvas delays a third of the viewport's frames from every placement,
   separate renderer included. **This contradicts the assumption that an own renderer isolates a heavy tool.** The host must
   control how tools draw: reduce curves to screen resolution, and draw anything still heavy on the CPU.
3. **A second reader of the step costs the viewport 17%, and a small shared cache in main removes it.** A tool's request
   and the viewport's request for one step must share one read.

Eight light live tiles in one tool window cost the viewport nothing measurable. The 3D view can become a tile, but it
presents a blank frame on every resize today, and most of its overlays assume it fills the window. A second 3D view
should wait.

## Method

The prototype runs the real app. `spike/build.mjs` bundles `src/` unchanged, wraps the real main and preload with spike
IPC, and swaps the measure harness at bundle time. The same tile host (`spike/tiles.ts`) runs in all three placements:

| Name in tables | Placement | Thread | Step data comes from |
|---|---|---|---|
| `page` | a dock inside the main page | the viewport's | the page's own step |
| `child` | `window.open` child, drawn by the main page | the viewport's | the page's own step |
| `renderer` | separate sandboxed `BrowserWindow` | its own | a request to main over a `MessagePort` |

- **Build measured:** `cba797262`, which is `main` at `0c7db25a5` plus `origin/viewer/loads-summary-memo` (`5cbe603a`) and
  `origin/viewer/playback` (`01a7d05c`), the step-path fixes in #3041 and #3042. A later legend-only commit on the
  playback branch is not included.
- **Models:** `tims_c3b` (54,279 nodes, 26,970 cells, 51 steps, 653 MB `.ladruno`) for everything;
  `yoo_building_SLE_T2` (76,753 nodes, 128,923 cells, 1 step) for orbit and memory.
- **Machine:** Ryzen AI 7 PRO 350, Radeon 860M integrated GPU, 23.6 GB RAM, 60 Hz, canvas 1584 x 961.
- **Scenario:** contour and deformed shape on. Orbit 4 s, then the same orbit with tools updating 20 times a second, then
  scrub 8 s. The chart draws 4 series of 50,000 points, reduced to pixel columns unless a row says otherwise.
- **Stand-ins for a costly tool:** `cpu` blocks its thread 40 ms per update; `heavy` draws 4 series of 200,000 points
  unreduced; `heavy, software` is the same on a canvas kept off the GPU (`willReadFrequently`).
- **Repeats:** 3 launches on `tims_c3b`, scenario order rotated, a no-tool baseline at the start, middle and end of each.
  Values are **median [lowest..highest launch]**. "Relative" divides by the baselines of the same launch.
- **Load:** machine CPU was 7 to 30% before and after each launch, 14 to 59% during scrubs. These are the quietest runs
  of the session. Earlier runs at 60 to 100% moved the no-tool baseline by a factor of two within a single launch and are
  kept only for transport and security.

**A measurement trap.** `requestAnimationFrame`'s timestamp stays on the vsync grid when the page thread is late. Measured
that way, the 40 ms tool looked harmless. All frame times below use the time the callback actually ran.
`src/renderer/harness/measure.ts` uses the timestamp for its orbit, so its fps will not show page-thread stalls.

## Results: own-window technology

Viewport during an orbit while the tool updates 20 times a second. With no tool the frame p95 is 16.9 ms [16.8..35.1] and
the median share of frames over 33 ms is 0%.

| Tool | Child window: frame p95, share > 33 ms | In the page | Separate renderer |
|---|---|---|---|
| light chart | 16.9 ms, 0% | 16.8 ms, 0% | 16.9 ms, 0% |
| **cpu**, 40 ms per update | **42.8 ms, 32.9%** | **42.8 ms, 32.6%** | **16.9 ms, 0%** |
| **heavy** on GPU canvas | 90.2 ms, 39.7% | 103.0 ms, 32.5% | **97.3 ms, 32.7%** |
| heavy, software canvas | 87.9 ms, 32.8% | not run | **16.9 ms, 0%** |

Scrubbing, viewport step-to-paint relative to the same launch's baseline (baseline 47.0 ms [36.2..53.0]):

| Tool | Child | Page | Separate renderer (shared read) |
|---|---|---|---|
| light chart | 1.02 [0.98..1.02] | 1.01 [0.99..1.05] | 0.98 [0.92..1.09] |
| cpu | 1.73 [1.67..1.85] | 1.80 [1.65..2.04] | 1.08 [1.01..1.13] |
| heavy | 1.10 [1.03..1.19] | 1.42 [1.22..1.53] | 1.11 [1.07..1.13] |
| heavy, software | 2.51 [2.31..2.61] | not run | 1.11 [1.10..1.12] |

In the heavy GPU row the step itself barely slows, but the viewport's frame interval during a scrub grows by 2.21 (child),
1.57 (page) and 1.87 (renderer).

Step to painted in the tool (median), measured from the store dispatch:

| Tool | Child | Page | Separate renderer |
|---|---|---|---|
| light chart | 58.2 ms [45.0..61.2] | 55.9 ms [43.8..60.2] | 17.6 ms [13.7..17.7] |
| cpu | 167.1 ms [141.2..177.0] | 168.7 ms [159.3..178.0] | 58.6 ms [56.2..86.2] |

The separate renderer paints sooner because it does not wait for the viewport to finish its step.

Opening a tool, and the cost of the small state link:

| | Child | Page | Separate renderer |
|---|---|---|---|
| open to first painted chart | 84 ms [70..237] | 24 ms [23..25] | 111 ms [96..150] |
| state round trip | none | none | 0.4 ms [0.3..0.5] |
| state one way | none | none | 0.2 ms [0.1..0.3] |
| private memory added | page +1.3 MB, GPU process about +17 MB (noisy) | under 1 MB | **29 MB** [29..30] in the new process |

### Moving large arrays

From the earlier `tims_c3b` runs, all repeats pooled. Times are round trips, request or send to acknowledgement:

| Size | `ipcRenderer.invoke` from main | `MessagePort` from main | Page to tool, `ArrayBuffer` transferred | Page blocked by that transfer |
|---|---|---|---|---|
| 1 MB | 2.7 ms | 3.5 ms | 2.4 ms | 1.2 ms |
| 16 MB | 26.2 ms | 35.3 ms | 26.8 ms | 13.2 ms |
| 64 MB | 110.6 ms | 136.3 ms | 100.0 ms | 52.3 ms |

**Transfer across processes is still a copy.** The buffer detaches, but it costs about 1.6 ms per MB, and the sending thread
is blocked for about half of that. Ports are right for small, frequent state. Large arrays should stay where they are read,
and tools should ask main for the reduced values they draw.

### Second reader of the same step

This is the measurement viewer-speed asked this spike to own. One tool in a separate renderer, `tims_c3b`, scrubbing:

| | No tool | Tool, own read | Tool, read shared through a 4-entry cache in main |
|---|---|---|---|
| viewport step-to-paint | 47.0 ms [36.2..53.0] | 58.4 ms [41.4..58.8] | 48.9 ms [34.5..53.6] |
| relative to same launch | 1 | **1.17 [1.10..1.19]** | **0.98 [0.92..1.09]** |
| step reads in main, per step | 1 | 2 | 1 |
| main-process read time per step | 10.7 ms [7.5..11.8] | 20.4 ms [14.1..20.8] | 10.3 ms [7.2..10.8] |
| main event-loop lag, p99 | 24.4 ms | 33.0 ms | 22.4 ms |
| IPC to the page per step | 1,696 KB | 1,696 KB | 1,696 KB |
| IPC to the tool per step | 0 | 0.5 KB | 0.5 KB |

So a second read adds about 10 ms of blocking main-process time per step. The tool itself needs only 64 values (0.5 KB). The
page receives 1.7 MB per step; that is the larger transport cost, and it is not caused by tools.

## Results: tiling

Light live tiles in a split tree, with merged updates, paused hidden tiles, reduced curves and one shared request.

| Tiles | Placement | Viewport step, relative | Tile paint p50 | Tile paint p95 | Splitter drag p95 | Memory |
|---|---|---|---|---|---|---|
| 1 | tool window | 0.98 [0.92..1.09] | 17.6 ms | 27.3 ms | n/a | 29 MB |
| 4 | tool window | 1.02 [1.01..1.02] | 16.9 ms | 25.6 ms | 16.8 ms | 34 MB |
| 8 | tool window | 1.02 [1.01..1.03] | 18.0 ms | 24.3 ms | 16.8 ms | 36 MB |
| 4 | page | 1.03 [1.00..1.04] | 56.7 ms | 74.9 ms | 16.9 ms | about 4 MB |
| 8 | page | 1.10 [1.03..1.11] | 59.3 ms | 78.4 ms | 16.8 ms | about 7 MB |
| 8 | child window | 1.09 [1.03..1.23] | 62.6 ms | 94.0 ms | 16.8 ms | about 6 MB plus GPU |

What each rule is worth, with 8 tiles (4 hidden), a tool window, four state messages per step, and no main-process cache:

| Rule turned off | Viewport step, relative | Main lag p99, relative | Tile paint p50 | Tool read requests per step |
|---|---|---|---|---|
| none | 1.21 [1.00..1.28] | 1.32 | 29.1 ms | 2 |
| merged updates per frame | 1.32 [1.24..1.32] | 2.19 | 46.9 ms | 3.9 |
| paused hidden tiles | 1.24 [1.18..1.24] | 1.38 | 33.8 ms | 2 |
| curves reduced to screen | 1.16 step, **1.76 frame interval** | 1.11 | 102.6 ms | 2 |
| one shared request | 1.45 [1.37..1.45] | 2.14 | 78.3 ms | 6 |
| **all four** | frame interval **2.59**, p95 1,164 ms | **64.8** (1.4 s) | **4,573 ms** | 30.5 |

Paused hidden tiles matter little while tiles are cheap. They are worth keeping for tools that are not. With curves not
reduced, 32.7% of orbit frames exceed 33 ms.

**Limits.** Eight visible live tiles per tool window is the tested comfortable limit. The cost was flat from 1 to 8, so the
real ceiling may be higher, but nothing above 8 was measured. Past eight, keep further tiles as tabs or paused tiles that
update when shown. In the main page, allow only tools declared light.

## Results: the 3D view as a tile

The view went into a tile with spike CSS and no product change. Selection works: 30 to 35 of 121 grid points inside the
tile hit the model, and none to its left, because picking already uses the canvas rectangle.

**Resize.** `Viewport.resize` calls `renderer.setSize`, which clears the canvas, and asks for a render on the next frame. The
browser therefore presents the cleared canvas: in a 90-frame drag **all 90 resized frames were blank**, confirmed by a still.
Drawing at once in the resize callback gave **0 blank frames**, at about 0.1 ms of draw submission on top of 2 to 3 ms for
`setSize`. Frame p95 was 16.8 ms in both cases. The fix is small: `resize()` calls `renderNow()`.

**What must change.** The camera reserves room for the right rail (`railKeptPx`, 498 px, `viewport.ts:84` and `:1116`), so in
a tile the model sits left of centre (mean hit at 25% of the tile's width). The rail cannot be assumed to float over the
view once the view is a tile.

| Element (`style.css` line) | Today | Must do in a tiled layout |
|---|---|---|
| `#viewport` (24) | `fixed; inset: 0` | fill its tile |
| `#viewcube` (445) | fixed, offset from the window's right edge | **move into the 3D tile** |
| `#scale-stack` (531), `.viewport-scale[data-moved]` (526) | fixed; a dragged scale is clamped to the window (`ui/scaleChrome.ts:83`) | **move into the tile; clamp to the tile** |
| `#empty` (227) | centred on the window | centre on the tile |
| `#rail` (185), `#rail-toggle` (611) | fixed over the view | **become a shell column or a tile**; drop the camera reservation |
| `#left-rail` (87) | fixed over the view | shell column or tile |
| `#header` (28), `#phase` (255), `#banner` (60) | fixed, sized with `100vw` (lines 30, 36, 38; `panels/phase.ts:268`) | shell top bar, sized to the shell |
| `#windows-menu` (353), `#windows` (370) | fixed | shell; `#windows` becomes a tool placement later |
| `#spotlight` (688) | fixed, whole window | no change |

`effects.ts:538` and `:551` also size automatic markers from the window; they should read the view's size.

**A second 3D view should wait.** It is cheap to draw: a second `Viewport` on the same store mounted with no product change
in 53 to 80 ms on `tims_c3b`, and an orbit of both held a 16.9 ms p95. The obstacle is state. The store has one `camera`,
effects have one `frameTarget`, and selection and section planes assume one view. It needs a per-view state design first.

## The biggest-mesh crash

**The cause is not established. The evidence points to a hang during loading under severe memory pressure, not to a failed
allocation.**

- One of two final launches on `yoo_building_SLE_T2` ended after 123 s with exit code `0xCFFFFFFF` and no result. The
  other completed, as did both earlier launches. The final launch that failed had 1.9 GB free when it ended.
- Windows logged **Application Hang 1002** and a Windows Error Reporting `AppHangB1` for the spike's `electron.exe` at
  22:11:57. The event's fifth field reads 12, which I take to be the hang time in seconds. The exit code the runner saw is the main process's.
- There was no crash dump, no resource-exhaustion event, and no out-of-memory message in the log.
- I cannot tell what ended the hung process, a person answering the "not responding" prompt or Windows itself.
- Main reads files synchronously, on the thread that also owns the windows. A long read under paging would produce this
  hang. That link is an inference; no profile of the failing launch exists.

Memory was sampled every 2 s during the launch that **succeeded** immediately after (it was not sampled for the failure):

| Quantity | Value |
|---|---|
| RAM free, lowest | **33 MB** of 23.6 GB |
| Commit used, highest | **93.1 GB of a 97.9 GB limit (95%)** |
| Whole app, private memory, peak | 5.7 GB |
| Largest process (the page), private, peak | **5.3 GB**, falling to about 2.2 GB once loaded in this launch; the page held about 5.2 GB after loading in the earlier launches |
| Largest process, working set, peak | 4.3 GB |
| Everything else on the machine, private | **70.7 to 73.4 GB** |
| Largest other groups at the app's peak | `claude` 39 processes 16.0 GB, `node` 98 processes 11.0 GB, Revit 7.5 GB, `zen` 6.5 GB, WebView2 3.3 GB |

So the app was about 6% of committed memory; other programs, including the parallel agent sessions, held the rest. CPU was
quiet (11 to 18% before and after these launches).

### Memory limits to adopt

What each approach adds: separate renderer **29 to 37 MB** private for light tools (1 to 8 tiles), up to 43 MB with heavy
drawing; child window 1 to 3 MB in the page and about 17 MB in the GPU process (noisy); in-page tiles up to about 7 MB. **None
of these is significant next to a 5 GB load peak.** The load is what needs protecting.

These thresholds are a proposal. They rest on one model and one machine:

| Condition, checked before opening a file or a tool window | Behaviour |
|---|---|
| estimated load peak above available RAM plus safe commit headroom | **refuse the open**, naming the needed and available memory |
| commit headroom under 2 x the estimated peak, or available RAM under 2 GB | open, with a banner saying it may be slow or stop responding |
| available RAM under 1 GB while tools are open | pause all tool updates, refuse new tool windows, say why |
| a tool window's renderer dies or stops responding | keep the model window; show "tool stopped" with reopen |

Estimate the peak from cell count: this model used about **42 KB of page memory per cell** at peak (5,289 MB / 128,923 cells). The
figure needs checking on more models before it is used to refuse anything. Electron's `process.getSystemMemoryInfo()` gives
RAM and swap figures; check that it reports commit headroom on Windows before relying on it. The supported size stated today
should be what was tested: **about 130,000 cells with up to 8 live tiles on a 24 GB machine**, with memory available.

The more important fix is to **move file reading off the main process's UI thread** (a utility process), so a slow load
leaves windows responsive. This was not prototyped.

## Security

- Today the app has no `setWindowOpenHandler` and no `will-navigate` guard. With no policy added, `window.open` from the
  page **created windows** for `about:blank` and for the app's own `index.html`. They were sandboxed, context-isolated, had no
  preload, and ran in the page's process. External URLs were not tried in that mode.
- With the proposed policy, every open was refused except an explicitly named one, an attempt to open `https://example.invalid/`
  was refused, and a navigation attempt from the tool window was blocked. The tool window kept its page, opened nothing, and
  had no `viewer`, `require` or `process`.

Policy to adopt, on every `webContents` through `app.on("web-contents-created")`:

1. `setWindowOpenHandler` returns `deny` always. Tool windows are created only by main.
2. `will-navigate` and `will-redirect` are prevented unless the target is the page already loaded.
3. Tool windows use `sandbox: true`, `contextIsolation: true`, `nodeIntegration: false`, their own small preload, and a CSP
   with `default-src 'none'; script-src 'self'; style-src 'unsafe-inline'`.
4. The tool preload exposes only "ready" and port delivery, not the page's `viewer` bridge.
5. Main checks `event.sender` on every tool request, and validates each query as `results:readStep` does.

## What the shared tool contract must include

The options page's `ToolContext`, `ToolHost`, `ToolDefinition` and `ToolHandle` stand. The one tile host running unchanged in
all three placements supports that shape. The measurements add:

- **`cost` on `ToolDefinition`: `"light" | "heavy"`.** Heavy tools may not mount in the page or a child window.
- **Host-owned scheduling.** `update` runs at most once per frame with the newest context, and never for a hidden tool.
- **Host-owned canvases**, `host.canvas({ raster: "gpu" | "software" })`, with one shared chart that reduces curves to pixel
  columns. A tool that draws much more than its pixel width must use software raster.
- **One data broker.** `host.request` removes duplicates per step across tiles, newest wins, and returns reduced values.
- **A step read shared with the viewport**, in main, keyed by file, stage and step.
- **`ToolContext.step` carries `playing`**, so tools can lower their rate during playback (`viewer/playback`).
- **`visible` and `onLost`:** visibility from the host; a restart path when the tool's renderer dies.
- **No large arrays in context.** Context stays small and serializable; arrays go through `request`.

Everything in that contract is a design conclusion from the measurements, not separately tested code, apart from the merged
updates, pausing, reduction, shared request and main cache, which the prototype does implement.

## ADR changes

ADR 0137 records the decisions. This was the spike's proposal:

- **D7 "One window".** Replace with: one *model window* that owns the viewport, plus tool windows it owns. A tool has one
  contract and several placements: separate-renderer window, tiled window, dock in the model window for light tools, anchored
  card. Inspector, model-definition windows and owner approval of every screen stay.
- **D6 "One state store".** Keep the single store in the model window. Add: tool windows hold no store. They receive a small
  one-way `ToolContext`, send events back, and read data through main. "No view keeps a private copy of model data" gains the
  exception of reduced values drawn by a tool.
- **Add to D5 or D7:** the window and navigation policy above, and the tool cost rule.
- **Add:** one GPU process is shared, so tools have a drawing budget. An own renderer isolates CPU work only.

## Proposed slices

**H0, window policy (small, can land first).** `src/main/main.ts` or a new `src/main/windowPolicy.ts`, one test.

**H1, the host, first slice.** A placeholder tool in a separate-renderer window, driven by the real step, with a still. Land
it after #3040 and #3038, which touch the same reducer, types, `main.ts` and `preload.ts`.

| Owned | Files |
|---|---|
| New | `src/tools/contract.ts`, `src/tools/host.ts`, `src/tools/registry.ts`, `src/tools/placeholder.ts` |
| New | `src/main/toolWindows.ts`, `src/main/toolPreload.ts` |
| New | `src/renderer/toolWindow.html`, `src/renderer/toolWindow.ts` |
| New | `src/state/tools.ts`, `src/state/types/tools.ts`, `src/state/types/events/tools.ts`, `src/state/reduce/tools.ts` |
| New | `test/tools/*.test.ts`, `src/renderer/harness/tools.ts` |
| Small edits | `src/main/main.ts`, `src/main/preload.ts`, `src/renderer/app.ts`, `scripts/build.mjs`, `src/state/reduce.ts`, `src/state/reduce/initial.ts`, `src/state/types.ts`, `src/state/types/event.ts`, `src/state/types/state.ts`, `.dependency-cruiser.cjs`, `README.md` |

How current work fits:

- **`materialTest` (#3040).** State and the run effect stay in the page. Its S1b surface is the first real tool: it dispatches
  events, reads run status from `ToolContext`, and gets curves through `host.request`, since curves live in the page's
  `BlobStore`.
- **`hinges` (#3038).** Hinge picks join the shared targets. Its moment-rotation SVG in `panels/hinges.ts` migrates onto the
  shared chart in H2.
- **`viewer/playback`.** Its playing state feeds `ToolContext.step.playing`.

**Then:** H2 the shared chart (from `panels/windows/plot/series.ts`, plus #3038's plot) and data broker; H3 the shared step
read in `src/main/results.ts`, agreed with viewer-speed; H4 the tiled tool window; H5 the dock in the model window; H6 the
3D view as a tile (the resize fix can land alone at any time). Loading off the main UI thread is separate work.

## What this spike did not establish

- One machine with an integrated GPU. A discrete GPU may shift the GPU contention.
- The heavy tools are synthetic. A WebGL tool, such as a 3D section view, was not tested. It also uses the shared GPU process.
- `WebContentsView` was not tested. It could put a separate renderer inside the model window's dock, which would extend CPU
  isolation to docked tools. H5 should test it first.
- Nothing above 8 tiles, and no real tool code.
- `yoo_building_SLE_T2` has one valid launch with the corrected frame timing (the other is the failure above). Its orbit stayed
  at 16.8 ms p95 with up to 8 light tiles, and the child-window `cpu` tool again delayed 33.1% of orbit frames, against 0.4%
  from a separate renderer.
- GPU-process memory figures are noisy, and do not capture dedicated GPU memory.

## Reproduce

```powershell
cd apeGmshViewer
npm ci
node spike/build.mjs
node spike/run.mjs <model.h5> --repeats=3 --scrub-ms=8000 --out=<dir>
node spike/analyze.mjs <dir>
```

These commands need the prototype commit named at the top. To measure the build in this report, first merge the two
branches named under Method.
