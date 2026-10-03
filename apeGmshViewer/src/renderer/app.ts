// Renderer entry: wires the store, the viewport and the panels, then runs
// the mode main asked for (view, measure or capture).

import * as THREE from "three";
import { resolveChain, type ElementRef } from "../chain/resolve.ts";
import { buildMesh } from "../mesh/build.ts";
import type { ModelFile } from "../model/types.ts";
import { Store } from "../state/store.ts";
import { mountPanels } from "./panels.ts";
import { Viewport, sameRef } from "./viewport.ts";

interface Bridge {
  config(): Promise<{
    mode: "view" | "measure" | "capture";
    file: string | null;
    t0: number;
    appReadyMs: number;
    configMs: number;
    pick: string | null;
  }>;
  openModel(path: string): Promise<{ ok: true; model: ModelFile } | { ok: false; error: string }>;
  pathForFile(f: File): string;
  metrics(): Promise<{ mainMB: number; rendererMB: number; gpuProcessMB: number; gpuDevices: unknown[] }>;
  measureDone(result: unknown): Promise<void>;
  captureStill(suffix: string): Promise<string | null>;
  captureDone(): Promise<void>;
  fail(message: string): Promise<void>;
}
const bridge = (window as unknown as { viewer: Bridge }).viewer;

const store = new Store();
const viewport = new Viewport(document.getElementById("viewport")!, store);
mountPanels(store);

const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
/** Resolves once the frame after the current DOM change has been painted. */
const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function load(path: string): Promise<boolean> {
  const res = await bridge.openModel(path);
  if (!res.ok) {
    store.dispatch({ type: "model/failed", error: `${path}: ${res.error}` });
    return false;
  }
  for (const w of res.model.warnings) console.warn(w);
  const mesh = buildMesh(res.model);
  for (const w of mesh.warnings) console.warn(w);
  store.dispatch({ type: "model/loaded", model: res.model, mesh });
  return true;
}

const BEAM_TYPES = new Set(["dispBeamColumn", "forceBeamColumn", "elasticBeamColumn"]);

/**
 * The element a scripted click targets: a beam-column if any; else an element
 * whose chain resolves with at least one link; else the middle drawn element.
 * A bounded sample (from the middle outwards) keeps this cheap on big models.
 * With `pickType`, every drawn element is a candidate (not only the sample),
 * and the first of that OpenSees type whose chain resolves wins; none found
 * raises, naming the type.
 */
function demoTarget(pickType: string | null = null): ElementRef | null {
  const s = store.get();
  if (!s.model || !s.mesh) return null;
  if (pickType !== null) {
    for (const r of [...s.mesh.lineRefs, ...s.mesh.triRefs]) {
      const c = resolveChain(s.model, r);
      if (c.root.type === pickType && c.problems.length === 0) return r;
    }
    throw new Error(`--pick=${pickType}: no drawn element of that type with a resolved chain`);
  }
  const sample = (list: ElementRef[]) => {
    const step = Math.max(1, Math.floor(list.length / 200));
    const out: ElementRef[] = [];
    for (let i = Math.floor(list.length / 2); i < list.length; i += step) out.push(list[i]!);
    for (let i = 0; i < Math.floor(list.length / 2); i += step) out.push(list[i]!);
    return out;
  };
  const candidates = [...sample(s.mesh.lineRefs), ...sample(s.mesh.triRefs)];
  const chains = candidates.map((r) => ({ r, c: resolveChain(s.model!, r) }));
  const beam = chains.find(({ c }) => BEAM_TYPES.has(c.root.type) && c.problems.length === 0);
  const linked = chains.find(({ c }) => c.root.children.length > 0 && c.problems.length === 0);
  return (beam ?? linked)?.r ?? candidates[0] ?? null;
}

/** Click at the target's projected midpoint through the same pick path a mouse uses. */
async function scriptedClick(target: ElementRef) {
  const pt = viewport.screenPointOf(target);
  if (!pt) throw new Error("scripted click: target is not drawn");
  const t = performance.now();
  const ref = viewport.pickAt(pt.x, pt.y);
  if (!ref) throw new Error("scripted click: pick returned nothing");
  const tPick = performance.now();
  store.dispatch({ type: "select", ref });
  const tState = performance.now();
  await painted();
  const end = performance.now();
  return {
    ms: end - t,
    split: { pickMs: tPick - t, stateAndDomMs: tState - tPick, paintMs: end - tState },
    pickedTarget: sameRef(ref, target),
    ref,
  };
}

async function measure(t0: number, startup: { appReadyMs: number; configMs: number }) {
  await nextFrame();
  viewport.renderNow();
  await painted();
  const firstFrameMs = Date.now() - t0;
  const s = store.get();
  const model = s.model!;

  // Scripted orbit: one turn about the vertical (Z) axis through the fitted
  // centre, 6 s, through the same turntable code a right-drag uses.
  const ORBIT_MS = 6000;
  const target = viewport.nav.target.clone();
  const deltas: number[] = [];
  const renderMs: number[] = [];
  viewport.continuous = true;
  const start = await nextFrame();
  let last = start;
  let turned = 0;
  for (;;) {
    const now = await nextFrame();
    deltas.push(now - last);
    last = now;
    const ang = Math.min(1, (now - start) / ORBIT_MS) * Math.PI * 2;
    viewport.nav.orbit(target, ang - turned, 0);
    turned = ang;
    viewport.renderNow();
    renderMs.push(viewport.lastRenderMs);
    if (now - start >= ORBIT_MS) break;
  }
  viewport.continuous = false;
  viewport.renderNow();
  await painted();

  const median = (a: number[]) => {
    const b = [...a].sort((x, y) => x - y);
    return b.length ? b[Math.floor(b.length / 2)]! : NaN;
  };
  const tgt = demoTarget();
  const click = tgt ? await scriptedClick(tgt) : null;
  const chain = store.get().chain;
  const metrics = await bridge.metrics();
  const cells = model.blocks.reduce((a, b) => a + b.ids.length, 0);
  return {
    file: model.path,
    sizeBytes: model.sizeBytes,
    nodes: model.nodeIds.length,
    cells,
    opsOnlyElements: s.mesh!.counts.opsOnly,
    drawn: { segments: s.mesh!.lineRefs.length, triangles: s.mesh!.triRefs.length },
    readMs: model.readMs,
    firstFrameMs,
    startup: { electronReadyMs: startup.appReadyMs, rendererUpMs: startup.configMs },
    orbit: {
      durationMs: last - start,
      frames: deltas.length,
      medianFps: 1000 / median(deltas),
      medianRenderCpuMs: median(renderMs),
    },
    memory: metrics,
    inspector: click
      ? {
          fillMs: click.ms,
          split: click.split,
          pickedTarget: click.pickedTarget,
          root: chain?.root.name,
          type: chain?.root.type,
          links: chain ? countLinks(chain.root) : 0,
          problems: chain?.problems ?? [],
        }
      : null,
    gpu: viewport.gpuName(),
    pixelRatio: window.devicePixelRatio,
    viewport: [viewport.renderer.domElement.width, viewport.renderer.domElement.height],
    warnings: [...model.warnings, ...s.mesh!.warnings],
    three: THREE.REVISION,
  };
}

function countLinks(n: { children: { children: unknown[] }[] }): number {
  return n.children.reduce((a, c) => a + 1 + countLinks(c as typeof n), 0);
}

async function main() {
  const cfg = await bridge.config();
  if (cfg.mode === "view") {
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      const f = e.dataTransfer?.files[0];
      if (f) void load(bridge.pathForFile(f));
    });
    if (cfg.file) await load(cfg.file);
    return;
  }
  if (!cfg.file || !(await load(cfg.file))) {
    await bridge.fail(store.get().error ?? "no --file given");
    return;
  }
  if (cfg.mode === "measure") {
    await bridge.measureDone(await measure(cfg.t0, cfg));
    return;
  }
  // capture: select the scripted target and let main grab the page; when the
  // chain is taller than the window, a second still shows its end.
  const settle = () => new Promise((r) => setTimeout(r, 300));
  viewport.renderNow();
  const tgt = demoTarget(cfg.pick);
  if (tgt) store.dispatch({ type: "select", ref: tgt });
  viewport.renderNow();
  await settle();
  viewport.renderNow();
  await bridge.captureStill("");
  const insp = document.getElementById("inspector")!;
  if (!insp.hidden && insp.scrollHeight > insp.clientHeight + 4) {
    insp.scrollTop = insp.scrollHeight;
    await settle();
    viewport.renderNow();
    await bridge.captureStill("chain-end");
  }
  await bridge.captureDone();
}

main().catch((err) => {
  console.error(err);
  void bridge.fail(err instanceof Error ? `${err.message}\n${err.stack}` : String(err));
});
