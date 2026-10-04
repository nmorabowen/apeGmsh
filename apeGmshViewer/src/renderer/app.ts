// Renderer entry: wires the store, the BlobStore, the effects, the viewport
// and the panels, then runs the mode main asked for (view, measure or
// capture).

import * as THREE from "three";
import { Effects, type Bridge } from "../effects.ts";
import { mountBanner } from "../panels/banner.ts";
import { mountEmpty } from "../panels/empty.ts";
import { mountHeader } from "../panels/header.ts";
import { mountInspector } from "../panels/inspector.ts";
import { mountLegend } from "../panels/legend.ts";
import { mountPhase } from "../panels/phase.ts";
import { BlobStore } from "../state/blobs.ts";
import { chainOf } from "../state/selectors.ts";
import { Store } from "../state/store.ts";
import type { DeclPath } from "../state/types.ts";
import { Viewport } from "./viewport.ts";

const bridge = (window as unknown as { viewer: Bridge }).viewer;

const store = new Store();
const blobs = new BlobStore();
const effects = new Effects(store, blobs, bridge);
const viewport = new Viewport(document.getElementById("viewport")!, store, blobs);
const panels = [mountHeader(store), mountBanner(store), mountLegend(store), mountInspector(store), mountEmpty(store), mountPhase(store)];
window.addEventListener("beforeunload", () => {
  for (const d of panels) d();
  effects.dispose();
  viewport.dispose();
});

const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
/** Resolves once the frame after the current DOM change has been painted. */
const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

const BEAM_TYPES = new Set(["dispBeamColumn", "forceBeamColumn", "elasticBeamColumn"]);

/**
 * The element a scripted click targets: a beam-column if any; else an element
 * whose chain resolves with at least one link; else the middle drawn element.
 * A bounded sample (from the middle outwards) keeps this cheap on big models.
 * With `pickType`, every drawn element is a candidate (not only the sample),
 * and the first of that OpenSees type whose chain resolves wins; none found
 * raises, naming the type.
 */
function demoTarget(pickType: string | null = null): DeclPath | null {
  const s = store.get();
  if (!s.mesh) return null;
  const drawn = s.mesh.elements;
  if (pickType !== null) {
    for (const p of drawn) {
      if (s.decls[p]?.type !== pickType) continue;
      const c = chainOf(s, p);
      if (c && c.problems.length === 0) return p;
    }
    throw new Error(`--pick=${pickType}: no drawn element of that type with a resolved chain`);
  }
  const step = Math.max(1, Math.floor(drawn.length / 400));
  const candidates: DeclPath[] = [];
  for (let i = Math.floor(drawn.length / 2); i < drawn.length; i += step) candidates.push(drawn[i]!);
  for (let i = 0; i < Math.floor(drawn.length / 2); i += step) candidates.push(drawn[i]!);
  const chains = candidates.map((p) => ({ p, c: chainOf(s, p)! }));
  const beam = chains.find(({ c }) => BEAM_TYPES.has(c.root.type) && c.problems.length === 0);
  const linked = chains.find(({ c }) => c.root.children.length > 0 && c.problems.length === 0);
  return (beam ?? linked)?.p ?? candidates[0] ?? null;
}

/** Click at the target's projected midpoint through the same pick path a mouse uses. */
async function scriptedClick(target: DeclPath) {
  const pt = viewport.screenPointOf(target);
  if (!pt) throw new Error("scripted click: target is not drawn");
  const t = performance.now();
  const pick = viewport.pickAt(pt.x, pt.y);
  if (!pick) throw new Error("scripted click: pick returned nothing");
  const tPick = performance.now();
  store.dispatch({ type: "select", pick });
  const tState = performance.now();
  await painted();
  const end = performance.now();
  return {
    ms: end - t,
    split: { pickMs: tPick - t, stateAndDomMs: tState - tPick, paintMs: end - tState },
    pickedTarget: pick.decl === target,
    decl: pick.decl,
  };
}

async function measure(t0: number, startup: { appReadyMs: number; configMs: number }) {
  await nextFrame();
  viewport.renderNow();
  await painted();
  const firstFrameMs = Date.now() - t0;
  const s = store.get();
  const model = s.artifacts.model!;
  const mesh = s.mesh!;

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
  const chain = click ? chainOf(store.get(), click.decl) : null;
  const metrics = await bridge.metrics();
  return {
    file: model.path,
    sizeBytes: model.sizeBytes,
    nodes: model.counts.nodes,
    cells: model.counts.cells,
    opsOnlyElements: mesh.counts.opsOnly,
    drawn: { segments: mesh.linePositions.shape[0], triangles: mesh.triPositions.shape[0] },
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
    warnings: [...model.warnings],
    three: THREE.REVISION,
  };
}

function countLinks(n: { children: { children: unknown[] }[] }): number {
  return n.children.reduce((a, c) => a + 1 + countLinks(c as typeof n), 0);
}

async function main() {
  const cfg = await bridge.config();
  if (cfg.mode === "view") {
    effects.attach();
    // With V2f's `onOpen`, main replays the open set on subscription; loading
    // `config().file` here as well would read the model twice.
    if (cfg.file && !effects.hasOpenFeed()) await effects.open(cfg.file);
    return;
  }
  if (!cfg.file || !(await effects.open(cfg.file))) {
    await bridge.fail(store.get().artifacts.model?.error ?? "no --file given");
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
  if (tgt) store.dispatch({ type: "select", pick: viewport.pickOf(tgt, null) });
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
