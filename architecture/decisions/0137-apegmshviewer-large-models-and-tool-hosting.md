# ADR 0137 — apeGmshViewer: memory sets the supported model size, and tools run under one host with several placements

**Status:** Accepted (2026-11-18; approved by the owner). Merging the PR that carries this Status is the acceptance. The
pointer in ADR 0112 lands with it (ADR 0121 D5).

**Owner:** nmora

**Builds on** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md). **Keeps** the owner's ceiling from #2362
(2^24 nodes and 2^24 elements per stage, about 15M triangles). Its by-name refusal is planned in open #2373 and is not built
yet.
**Amends** ADR 0112 D5, D6 and D7 (D14
here). **Settles** the hosting question that ADR 0135 D5 (#3040) leaves open. **Supersedes** nothing.

**Evidence:** [`internal_docs/spike_viewer_tool_host.md`](../../internal_docs/spike_viewer_tool_host.md) for tool hosting;
`npm run profile` (#2705) and `apeGmshViewer/MEASUREMENTS.md` for the step path.

## Context

The viewer's step path was slow, and its playback stalled on a large results file. Profiling (`npm run profile`, #2705)
named the causes, and #3041 and #3042 fixed them. Loading a large model was the next cost, addressed by #3047, #3048 and
#3049. A separate spike tested tool windows and tiling beside the viewport.

Four tools are planned: a material tester, Mohr circles at Gauss points, a plotting tool and section views with fibres.
ADR 0112 D7 says "one window", which leaves them nowhere to go on a second monitor, and today's window cards rebuild on
every watched state change.

The measurements come from one machine (Ryzen AI 7 PRO 350, integrated Radeon 860M, 23.6 GB RAM, canvas 1584 x 961), so
they set a direction and not a general performance claim.

With #3041 and #3042, on `tims_c3b` (54,279 nodes, 26,970 cells, 51 steps, 653 MB `.ladruno`, contour and deformed shape on):

| Measure | Result |
|---|---|
| step change, click to drawn | 41 to 47 ms (274 ms before) |
| file read per step, in main | about 10 ms |
| data sent to the page per step | 1.7 MB |
| playback at x1 | all 51 steps drawn, 9.6 steps/s (2 of 51 before) |
| slider drag | all 41 requested steps drawn, p95 frame 55 to 60 ms |
| a second reader of each step | viewport step 1.17x slower |
| the same reader through a 4-entry cache in main | 0.98x, within noise |

On `yoo_building_SLE_T2` (76,753 nodes, 128,923 cells):

- The page process peaked at 5.3 GB of private memory while loading, about 42 KB per cell.
- Before the load fixes, the load spent about 200 s in the main process.
- One launch under severe memory pressure stopped responding, and Windows logged an application hang. In the next launch
  free RAM fell to 33 MB and commit reached 95% of its limit, with the viewer about 6% of committed memory. No profile of
  the failed launch exists, so its cause is not established. Synchronous file reading on the thread that owns the windows
  is the leading suspect.

In the tool spike, a tool in a separate renderer did not delay the viewport when it used the CPU heavily. A tool drawing
about 800,000 line segments on an accelerated canvas delayed about a third of the viewport's frames from every placement,
because all windows share one GPU process.

The spike ran the real app on `tims_c3b`, three launches, median of the launches. Viewport during an orbit while a tool
updates 20 times a second (p95 frame time, share of frames over 33 ms; with no tool, p95 is 16.9 ms):

| Tool | Child window | In the page | Separate renderer |
|---|---|---|---|
| light chart | 16.9 ms, 0% | 16.8 ms, 0% | 16.9 ms, 0% |
| blocks its thread 40 ms per update | 42.8 ms, 32.9% | 42.8 ms, 32.6% | 16.9 ms, 0% |
| about 800,000 segments, GPU canvas | 90.2 ms, 39.7% | 103.0 ms, 32.5% | 97.3 ms, 32.7% |
| the same, software canvas | 87.9 ms, 32.8% | not run | 16.9 ms, 0% |

With 1, 4 and 8 light tiles in one tool window, the viewport's step stayed within 0.98x to 1.02x of its no-tool baseline,
each tile painted a step about 18 ms after the store changed, and a splitter drag held a 16.8 ms p95. Opening a tool window
took about 110 ms and added 29 to 37 MB of private memory.

## Decision

### Large models

**D1. The stated support is what was tested.** About 130,000 cells, with up to 8 light live tool tiles, on a 24 GB machine
with memory available. The 2^24 ceiling stays as the absolute limit. It is not a promise that a model that size will open:
memory is expected to stop one first. It comes from V8's cap of 2^24 entries in a `Map` or `Set`, and #2373 will implement
its refusal at open; until then a model past it fails with an unnamed `RangeError`. The statement changes only with new
measurements (#3056).

**D2. Loading moves off the main process's UI thread.** Opening model and results files and scanning histories move to a
utility process (#3058), so that a slow load leaves the windows responsive. This is justified by the blocking itself. It is
not claimed to cure the observed hang. It has not been prototyped. The file work behind `model:open`, `geometry:open`,
`results:open`, `results:scan` and `results:scanCancel` moves. #3058 decides `particles:index`, `compare:model` and the
`.user.h5` reads. Windows, dialogs, drag-and-drop and open requests, the path allow-list, file watching, go-to-source, app
config and IPC routing stay in main.

**D3. Step reads stay in main until #3058 measures the hop.** A read costs about 10 ms. This is not a permanent placement:
`ResultsHost` holds the open file and index that `results:readStep` uses, so moving `results:open` needs either a second open
in main or step reads through the utility process. #3058 decides which from its numbers. They are also reconsidered when a
read regularly passes about 50 ms or measurably blocks the app.

**D4. Memory limits arrive in two stages.**

1. *Warn* (#3057). Before opening a model, show a banner when free RAM is under 2 GB, or commit headroom is under twice the
   estimated load peak. With tools open and free RAM under 1 GB, pause tool updates and refuse new tool windows.
2. *Refuse*, later and separately. An open is refused only after the peak estimate holds on several models and the commit
   headroom source is confirmed on Windows (#3056).

**D5. One read serves every consumer of a step.** A small cache in main, shared by the viewport and tools, is added with the
first separate-renderer tool (#3059), and not before.

**D6. No step prefetch and no step worker yet.** Playback follows a clock and skips steps when it falls behind, and the
newest request wins. Prefetch is reconsidered when reads, and not drawing, are shown to limit playback or scrubbing on a real
model. A step worker is reconsidered when a read regularly passes about 50 ms or measurably blocks the app. These triggers,
and the 50 ms, are judgement and not measured thresholds.

**D7. Step changes update existing GPU buffers.** They do not rebuild the scene.

### Tool hosting

**D8. Tools run in separate renderers, and the host limits what they draw.** A separate renderer isolates CPU work only, so
the host must reduce what a tool sends to the GPU. It does **not** isolate GPU work: every window shares one GPU process. The
one exception to separate renderers is the light tool of D9.

**D9. One model window, one tool contract, several placements.** The model window owns the viewport and the store. A tool is
written once and does not know where it is placed:

| Placement | Runs in | Accepts |
|---|---|---|
| tiled tool window, one or more, for example one per monitor | a separate sandboxed renderer | light and heavy tools |
| dock in the model window | the model window's page | light tools only |
| card anchored to a point of the model | the model window's page | light tools only |

**Child windows drawn by the main page are rejected.** They share the viewport's thread, so they give up the isolation that
justifies a window, and they are not cheaper in any way that matters: the separate renderer costs about 30 MB and opens in
about 110 ms.

**D10. A tool sees a small context and asks for data.**

- The model window sends each tool window a small, one-way, serializable `ToolContext`: step (with whether playback is
  running), targets (picks), the tool's parameters, and the theme.
- A tool changes anything only by sending an event back. The model window's store applies it (ADR 0112 D6).
- Data comes through main by request. Duplicate requests for a step are merged, and the newest request wins.
- **No large arrays travel in context.** A transferred `ArrayBuffer` is still copied between processes, about 1.6 ms per MB
  with the sender blocked for about half of it. Tools receive the reduced values they draw.

**D11. The host owns scheduling, visibility, canvases and the drawing budget.**

- `update` runs at most once per frame, with the newest context, and never for a hidden tool.
- A tool declares its cost, `light` or `heavy`, in its definition. Only `light` tools may mount in the model window. A light
  tool's update must fit well inside a frame on the page thread; the limit is set with H1 and checked by measurement.
- The host creates canvases and says how they are drawn. The shared chart reduces each curve to the canvas's pixel columns.
  A tool that still sends much more than its pixel width draws in software raster, kept off the GPU process. The spike tested
  one point on each side of this line: reduced curves (harmless) and about 800,000 segments (harmful). The threshold between
  them is set by measurement in H2.
- A tool whose renderer dies or stops responding shows as stopped, with a way to reopen it. The model window keeps running.

**D12. Eight visible live tiles per tool window is the tested limit.** Further tools stay as tabs or paused tiles that update
when shown. The cost was flat from 1 to 8 tiles; nothing above 8 was measured.

**D13. Window and navigation policy.** On every `webContents`:

1. `setWindowOpenHandler` denies every request. Only main creates windows.
2. `will-navigate` and `will-redirect` are prevented unless the target is the page already loaded.
3. A tool window has `sandbox: true`, `contextIsolation: true`, `nodeIntegration: false`, its own small preload, and a content
   security policy of `default-src 'none'; script-src 'self'; style-src 'unsafe-inline'`.
4. The tool preload exposes only what a tool needs to start and receive its ports. It does not expose the page's `viewer`
   bridge.
5. Main checks the sender of every tool request and validates each query, as `results:readStep` does.

The app has no such policy today. In the spike, `window.open` from the page created windows for `about:blank` and for the
app's own `index.html`.

**D14. What changes in ADR 0112.**

- **D5.** "HDF5 is read in-process" becomes: read in the app's own processes, which may include an Electron utility process
  (D2 here). No server is added, and ADR 0112 D4's rule on what the app may start is not widened: a utility process is the
  app's own Electron binary.
- **D6.** One store stays, in the model window. Tool windows hold no store. They receive `ToolContext`, send events and request
  data. The rule that no view keeps a private copy of model data gains one exception: the reduced values a tool is drawing.
- **D7.** "One window" becomes one model window plus the tool placements of D9. The contextual inspector, the model-definition
  windows and the owner's approval of every screen from a screenshot stay.

**D15. A second 3D view waits.** It needs a per-view state design first: the store has one `camera`, the effects have one
frame target, and selection and section planes assume one view. Drawing it is not the obstacle. A second `Viewport` on the
same store mounted without product changes and held a 16.9 ms p95 while both orbited.

**D16. Implementation order.**

| Order | Slice | Tracks |
|---|---|---|
| 1 | H0: the window and navigation policy (D13) | |
| 2 | Re-measure load time and peak memory once #3047 to #3049 have landed | #3056 |
| 3 | Memory warnings (D4 stage 1). Pausing tools and refusing tool windows land with H1, when a tool window first exists | #3057 |
| 4 | Loading and history scans in a utility process (D2) | #3058 |
| 5 | H1: the tool host, the contract and one separate-renderer window with a placeholder tool | |
| 6 | H2: the shared chart and the data broker | |
| 7 | H3: the shared step read (D5), no later than the first tool that reads step data | #3059 |
| 8 | H4: the tiled tool window | |
| 9 | H5: the dock in the model window | |
| 10 | H6: the 3D view as a tile | |

The material tester's surface (ADR 0135 S1b) is the first real tool after H1. The moment-rotation plot of #3038 moves onto the
shared chart in H2.

H6 has two known parts. `Viewport.resize` must draw in the resize callback: today every resized frame presents a cleared
canvas (90 of 90 in the spike), and drawing there removed them all at about 0.1 ms of draw submission. The overlays positioned
against the window must move to the tile or to the app shell, and the camera must stop reserving room for the right rail. The
spike report lists each one.

## Not established

- The cause of the hang, and whether #3047 to #3049 or D2 prevent it.
- Whether 42 KB per cell holds on other models, or after the load fixes.
- Whether Electron reports commit headroom on Windows in the form D4 needs.
- Any behaviour above about 130,000 cells. Scaling beyond the tested models is extrapolation.
- The cost of the extra process hop in D2, and how step reads (D3) reach the open file and index once `results:open` moves;
  `ResultsHost` holds both today (#3058).
- Discrete GPUs, WebGL tools, and machines other than the one above. A WebGL tool also uses the shared GPU process.
- Real tools: the spike's costly tools are synthetic stand-ins.
- More than 8 tiles.
- Whether a `WebContentsView` could place a separate renderer inside the dock. It was not tested, and H5 tests it first. If it
  works, the light-only rule for the dock can be reconsidered.

## Alternatives considered

- **Child windows.** Rejected in D9.
- **A separate renderer as complete isolation.** Rejected by measurement: it does not protect the viewport from GPU work (D8).
- **Every tool in the main page.** Simple and light on memory, but a costly tool delays the viewport (32.6% of orbit frames
  with a 40 ms update) and a second monitor goes unused. Kept for light tools only.
- **Separate applications.** They would duplicate readers, picking, time and packaging, and lose the live link to the model.
- **Tool data sent in context or as transferred arrays.** Rejected in D10: the copy is not avoided, and it blocks the sender.
- **Prefetch and a step worker.** Not adopted now (D6).

## Consequences

- Refusal by memory is deliberately absent until its estimate is proven, so a model that exhausts memory can still hang the
  app until D2 and D4 land.
- A utility process adds a boundary to the read path, which has to keep refusals, file-change checks and cancellation intact.
- New capacity work starts from a measurement with `npm run profile`, and `apeGmshViewer/MEASUREMENTS.md` keeps the record.
- Tool authors work inside the host's limits. They do not get a free canvas or the store.
- Each open tool window costs about 30 MB, small beside the load peak of a large model.
- State that a tool needs from the model window must go through `ToolContext` or a request, so new tool features may need a
  contract change. The contract is reviewed when each of the first tools lands.
- Frame timing must use the time the callback ran. The `requestAnimationFrame` timestamp stays on the vsync grid and hides
  page-thread stalls. `scripts/profile/probe.ts` (`npm run profile`, #2705) has the same flaw, and its fix follows #2705.
  `src/renderer/harness/measure.ts` (`npm run measure`) still needs one.
