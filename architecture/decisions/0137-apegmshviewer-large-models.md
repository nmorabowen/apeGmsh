# ADR 0137 — apeGmshViewer and large models: memory sets the supported size, and capacity is added at measured triggers

**Status:** Accepted (2026-11-18, by the owner, after the measurements below).

**Owner:** nmora

**Builds on** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md). **Keeps** the ceiling the owner set on #2362
(2^24 nodes or elements per stage, about 15M triangles, refused by name in #2373). **Supersedes** nothing.

## Context

The viewer's step path was slow, and its playback stalled on a large results file. Profiling (`npm run profile`, #2705)
named the causes, and #3041 and #3042 fixed them. Loading a large model was the next cost, addressed by #3047, #3048 and
#3049. A separate spike tested tool windows and tiling beside the viewport.

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

## Decision

**D1. The stated support is what was tested.** About 130,000 cells, with up to 8 light live tool tiles, on a 24 GB machine
with memory available. The 2^24 ceiling stays as the absolute limit. It is not a promise that a model that size will open:
memory is expected to stop one first. The statement changes only with new measurements (#3056).

**D2. Loading moves off the main process's UI thread.** Opening model and results files and scanning histories move to a
utility process (#3058), so that a slow load leaves the windows responsive. This is justified by the blocking itself. It is
not claimed to cure the observed hang.

**D3. Step reads stay in main for now.** A read costs about 10 ms. They are reconsidered with the numbers from #3058, or when
a read regularly passes about 50 ms or measurably blocks the app.

**D4. Memory limits arrive in two stages.**

1. *Warn* (#3057). Before opening a model, show a banner when free RAM is under 2 GB, or commit headroom is under twice the
   estimated load peak. With tools open and free RAM under 1 GB, pause tool updates and refuse new tool windows.
2. *Refuse*, later and separately. An open is refused only after the peak estimate holds on several models and the commit
   headroom source is confirmed on Windows (#3056).

**D5. One read serves every consumer of a step.** A small cache in main, shared by the viewport and tools, is added with the
first separate-renderer tool (#3059), and not before.

**D6. No step prefetch and no step worker yet.** Playback follows a clock and skips steps when it falls behind, and the
newest request wins. Either is reconsidered when reads, and not drawing, are shown to limit playback on a real model.

**D7. Step changes update existing GPU buffers.** They do not rebuild the scene.

**D8. Tools run in separate renderers, and the host limits what they draw.** A separate renderer isolates CPU work only, so
the host must reduce what a tool sends to the GPU. How tools are hosted is decided in its own ADR.

## Not established

- The cause of the hang, and whether #3047 to #3049 or D2 prevent it.
- Whether 42 KB per cell holds on other models, or after the load fixes.
- Any behaviour above about 130,000 cells. Scaling beyond the tested models is extrapolation.
- The cost of the extra process hop in D2.
- Discrete GPUs, WebGL tools, and machines other than the one above.

## Consequences

- Refusal by memory is deliberately absent until its estimate is proven, so a model that exhausts memory can still hang the
  app until D2 and D4 land.
- A utility process adds a boundary to the read path, which has to keep refusals, file-change checks and cancellation intact.
- New capacity work starts from a measurement with `npm run profile`, and `apeGmshViewer/MEASUREMENTS.md` keeps the record.
