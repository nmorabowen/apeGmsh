// Renderer entry: wires the store, the BlobStore, the effects, the viewport
// and the panels, then runs the mode main asked for (view here; measure and
// capture in ./harness/; ADR 0128's render and describe in ./agentRender.ts).

import * as THREE from "three";
import { Effects, userCombinationColumn, userResultColumn, type Bridge } from "../effects.ts";
import { mountBanner } from "../panels/banner.ts";
import { mountEmpty } from "../panels/empty.ts";
import { mountHeader } from "../panels/header.ts";
import { mountInspector } from "../panels/inspector.ts";
import { mountLegend } from "../panels/legend.ts";
import { mountLineDiagram } from "../panels/lineDiagram.ts";
import { mountLoads } from "../panels/loads.ts";
import { mountPrincipal } from "../panels/principal.ts";
import { mountReactions } from "../panels/reactions.ts";
import { mountSprings } from "../panels/springs.ts";
import { mountVectors } from "../panels/vectors.ts";
import { mountModalTable } from "../panels/modalTable.ts";
import { mountOfferBadge, mountOffers } from "../panels/offers.ts";
import { mountPhase } from "../panels/phase.ts";
import { mountPlanes } from "../panels/planes.ts";
import { mountSources } from "../panels/sources.ts";
import { mountSpotlight } from "../panels/spotlight.ts";
import { mountWindows } from "../panels/windows/frame.ts";
import { BlobStore } from "../state/blobs.ts";
import { userEventOf, userPathOf, userReadable } from "../state/reduce.ts";
import { Store } from "../state/store.ts";
import type { UserReadIn, VariantsIn } from "../state/types.ts";
import { DEFAULT_PANEL_IDS, mountCollapsibleChrome } from "../ui/collapse.ts";
import { OfferRunner, snapshotManifest } from "../spec/snapshot.ts";
import { APP_LAYERS, type AppLayer } from "../spec/spec.ts";
import type { OfferIn } from "../state/types.ts";
import { onScreenOf, runAgentMode, type Layout, type RenderBridge } from "./agentRender.ts";
import { CombinationFeed, UserResultFeed, VariantRunner, type VariantDoor } from "./contour.ts";
import { mountVariants } from "../panels/variants.ts";
import { mountSets } from "../panels/sets.ts";
import { runCapture } from "./harness/capture.ts";
import { painted, type HarnessContext } from "./harness/context.ts";
import { runMeasure } from "./harness/measure.ts";
import { mountViewCube } from "./viewcube.ts";
import { Viewport } from "./viewport.ts";

const bridge = (window as unknown as { viewer: Bridge }).viewer;

const store = new Store();
const blobs = new BlobStore();
const effects = new Effects(store, blobs, bridge);
// VA1: the theme is the store's; the effect styles the page with it (and does so now).
effects.setThemeTarget(document);
const viewport = new Viewport(document.getElementById("viewport")!, store, blobs);
effects.setFrameTarget(viewport);
// V4c1: the contour colours from the step the effects fetched (never kept in the store).
viewport.setStepSource(() => effects.currentStep());
// V4d: the deformed shape moves the drawn vertices by the displacement the effects computed for the step shown.
viewport.setDisplacementSource(() => effects.currentDisplacement());
// 3-D line diagrams: ribbon + profile + risers from the step the effects fetched.
viewport.setLineDiagramSource(() => effects.currentLineDiagram());
viewport.setStationsSource(() => effects.stationsOfStep());
// VU7b-2 (ADR 0130 D4): the User Combination shown, read through main's `user:readCase` and combined by VU4's
// evaluator (effects.ts `userCombinationColumn`); its values stay in the feed, as a step's stay in the effects.
const combinations = new CombinationFeed(store, (req) => userCombinationColumn(bridge, req));
viewport.setCombinationSource(() => combinations.held);
// VU16 (ADR 0130 D7): the user result picked, read through main's `user:readResult` (which refuses a stale entry);
// its column stays in renderer/contour.ts `userResultHeld`, which the viewport, the caps and the manifest read.
const userResults = new UserResultFeed(store, (req) => userResultColumn(bridge, req));
// VU10 (ADR 0132 D9): main's study, a switch, the compare set and the difference, through main's variant doors
// (preload.ts): a variant is named by row index and label, never a path. The difference is subtracted from the
// window's own step (the effects'), and its column stays in renderer/contour.ts `differenceHeld`.
const variantBridge = (window as unknown as { viewer: Partial<VariantDoor & StudyFeed> }).viewer;
// Fail closed: the preload of this build has every variant door, so a missing one is a broken build, never "no study".
if (!isVariantDoor(variantBridge)) throw new Error("apeGmshViewer: the preload has no variant doors (variants, onVariants, variantsSwitch, variantsCompare, readCompareStep, compareClose)");
const variantRunner = new VariantRunner(store, blobs, variantBridge, () => effects.currentStep());
const panels = [
  combinations.start(),
  userResults.start(),
  variantRunner.start(),
  // VA5 (ADR 0128 D5): first, so its box opens the right rail above the inspector.
  mountOffers(store),
  // VU10: below the offers, above the inspector.
  mountVariants(store),
  // VU9a (ADR 0132): the window set's named sets, below the variants; the effects carry its saves to main's workspace door.
  mountSets(store),
  // ADR 0130 D6 (VU7a): the .user.h5 modal table, below the offers, while a mode is shown.
  mountModalTable(store),
  mountHeader(store),
  mountBanner(store),
  mountLineDiagram(store, (fn) => viewport.onViewChange(fn)),
  mountLoads(store, (fn) => viewport.onViewChange(fn)),
  mountReactions(store, (fn) => viewport.onViewChange(fn)),
  mountPrincipal(store),
  mountVectors(store, (fn) => viewport.onViewChange(fn)),
  mountSprings(store, (fn) => viewport.onViewChange(fn)),
  mountLegend(store, (fn) => viewport.onViewChange(fn)),
  mountInspector(store),
  mountEmpty(store),
  mountPhase(store),
  mountSources(store),
  mountWindows(store),
  mountSpotlight(store),
];
// V4e1: the section planes panel; its slider box and the view direction come from the viewport.
panels.push(mountPlanes(store, { box: () => viewport.sliderBox(), viewNormal: () => viewport.viewNormal(), sliding: (on) => viewport.setSliding(on) }));
panels.push(
  mountViewCube({
    visible: () => {
      const s = store.get();
      return s.mesh !== null || s.geometry !== null;
    },
    heading: () => viewport.nav.heading,
    projection: () => viewport.projection,
    setProjection: (mode) => viewport.setProjection(mode),
    // VA1: the theme and every ViewCube click, drag and button go through the store (setTheme, setCamera).
    theme: () => store.get().theme,
    setTheme: (mode) => store.dispatch({ type: "setTheme", theme: mode }),
    snapTo: (h) => store.dispatch({ type: "setCamera", to: { yaw: h.yaw, tilt: h.tilt } }),
    orbit: (dYaw, dTilt) => {
      // A turntable orbit about the pivot is a turn to the heading it reaches (navigation.ts orbitAbout).
      const h = viewport.nav.heading;
      store.dispatch({ type: "setCamera", to: { yaw: h.yaw + dYaw, tilt: h.tilt + dTilt } });
    },
    home: () => store.dispatch({ type: "setCamera", to: "home" }),
    subscribe: (fn) => viewport.onViewChange(fn),
  }),
);
const chrome = mountCollapsibleChrome({
  panelIds: [...DEFAULT_PANEL_IDS, "line-diagram", "loads", "reactions", "vectors", "principal", "springs", "offers", "modal-table"],
  visible: () => {
    const s = store.get();
    return s.mesh !== null || s.provenance.length > 0;
  },
  onRailToggle: (collapsed) => viewport.setRailCollapsed(collapsed),
});
// VA5: a collapsed rail hides the offers panel; its toggle carries the count (the toggle exists once the chrome is mounted).
panels.push(mountOfferBadge(store));
const unsubChrome = store.subscribe(() => chrome.syncVisibility());
panels.push(() => {
  unsubChrome();
  chrome.dispose();
});
window.addEventListener("beforeunload", () => {
  for (const d of panels) d();
  effects.dispose();
  viewport.dispose();
});

/**
 * ADR 0128 D9: the page elements each app layer omits from the layout. A
 * layer that is off sets `data-still-omit` on its elements (style.css:
 * `display: none`); every panel is an overlay on the fixed viewport, so the
 * viewport keeps its position and size. Model mode omits every element of
 * the page but the viewport.
 */
const PANEL_LAYERS: Readonly<Record<AppLayer, readonly string[]>> = {
  header: ["header"],
  phase: ["phase"],
  viewcube: ["viewcube"],
  // #scale-stack holds the contour's and line diagram's colour bars, which #2701 moved off their left-rail cards.
  left_rail: ["left-rail", "scale-stack"],
  rail: ["rail", "rail-toggle"],
  windows: ["windows-menu", "windows"],
  banner: ["banner"],
};

const stillLayout: Layout = {
  apply(chrome, layers) {
    document.body.dataset["stillChrome"] = chrome;
    const omitted: HTMLElement[] = [];
    for (const layer of APP_LAYERS) {
      if (chrome === "app" && typeof layers[layer] !== "boolean") throw new Error(`render: app layer ${layer} has no state`);
      for (const id of PANEL_LAYERS[layer]) {
        const node = document.getElementById(id);
        if (!node) throw new Error(`render: the ${layer} layer's element #${id} is not on the page`);
        const off = chrome === "app" && layers[layer] === false;
        node.toggleAttribute("data-still-omit", off);
        if (off) omitted.push(node);
      }
    }
    // The DOM check: an omitted panel has no layout box (never merely covered or transparent).
    const left = omitted.filter((n) => n.getClientRects().length > 0);
    if (left.length) throw new Error(`render: ${left.map((n) => `#${n.id}`).join(", ")} still in the layout after being omitted`);
  },
  inLayout() {
    return [...document.body.children]
      .filter((n) => n.id !== "viewport" && n.getClientRects().length > 0)
      .map((n) => (n.id ? `#${n.id}` : n.tagName.toLowerCase()));
  },
};

/** ADR 0130 (VU3b): main's door for the set's `.user.h5` (main/preload.ts `readUser`, `user:read`); the Effects bridge does not carry it. */
interface UserBridge {
  readUser(req: { path: string; model: string | null; results: string | null }): Promise<UserReadIn>;
  onOpen(cb: (set: { model: string | null; results: string | null; user?: string | null }) => void): () => void;
  onFileChanged(cb: (path: string) => void): () => void;
}

/** One `.user.h5` read: the file and the set's model and results files its staleness is judged against (A6). */
interface UserTarget {
  path: string;
  model: string | null;
  results: string | null;
  /** main named the file in the set (`OpenSet.user`); otherwise `path` is where one would be, and may hold nothing */
  known: boolean;
}

const userDoor = bridge as unknown as UserBridge;
/** Counts `user:read` requests: an answer that is not the latest request's is dropped. */
let userSeq = 0;

/**
 * Read the set's `.user.h5` into the store; resolves once the answer is
 * there. No file there closes it (`userEventOf`); an answer the app cannot
 * take (an unknown target, a malformed reply) is a refusal with the reason,
 * never a partial listing. The read is announced (`userOpened`, the "reading"
 * badge) only for a file the set has: main named it, or it was read before.
 * A look where no file is known says nothing until a file answers (#2737
 * delta review, item 2).
 */
async function readUser(t: UserTarget): Promise<void> {
  const seq = ++userSeq;
  const announced = t.known || store.get().user?.path === t.path;
  if (announced) store.dispatch({ type: "userOpened", path: t.path });
  let read: UserReadIn;
  try {
    read = await userDoor.readUser(t);
  } catch (err) {
    read = { ok: false, error: `user:read failed: ${err instanceof Error ? err.message : String(err)}` };
  }
  if (seq !== userSeq) return;
  const event = userEventOf(t.path, read);
  if (!announced) {
    if (event.type === "userClosed") return;
    store.dispatch({ type: "userOpened", path: t.path });
  }
  try {
    store.dispatch(event);
  } catch (err) {
    store.dispatch({ type: "userRead", path: t.path, read: { ok: false, error: `the app cannot take user:read's answer: ${err instanceof Error ? err.message : String(err)}` } });
  }
}

/**
 * ADR 0130, view mode: each open set's `.user.h5` (main's `user` slot, else
 * `<stem>.user.h5`, which may appear later), read once the model's first read
 * has landed, so the first frame never waits on it; and again whenever the
 * user file, the model or the results change, because A6's staleness depends
 * on all three (#2729). The file appearing, being replaced or being deleted
 * arrives as `fileChanged` of its path, never as a reopen.
 */
function mountUserFeed(): void {
  let target: UserTarget | null = null;
  let pending = false;
  const go = () => {
    if (!pending || !target || !userReadable(store.get(), target.model)) return;
    pending = false;
    void readUser(target);
  };
  panels.push(store.subscribe(go));
  panels.push(
    userDoor.onOpen((set) => {
      const path = set.user ?? userPathOf(set);
      const moved = path !== target?.path;
      target = path === null ? null : { path, model: set.model, results: set.results, known: set.user !== null && set.user !== undefined };
      if (target === null) {
        userSeq++;
        pending = false;
        store.dispatch({ type: "userClosed" });
        return;
      }
      // Another set's user file: the last one's families leave at once, and an answer still on its way for it is dropped
      // (#2737 review, F1). Only the read itself waits for the model's first read; the same file again keeps waiting as before.
      if (moved) {
        userSeq++;
        store.dispatch(target.known ? { type: "userOpened", path: target.path } : { type: "userClosed" });
      }
      pending = true;
      go();
    }),
  );
  panels.push(
    userDoor.onFileChanged((path) => {
      if (!target || ![target.path, target.model, target.results].includes(path)) return;
      pending = true;
      go();
    }),
  );
}

/** VA5: main's door for `--present` and `--snapshot` (main/preload.ts); the Effects bridge does not carry them. */
interface OfferBridge {
  onOffers(cb: (offers: OfferIn[]) => void): () => void;
  onSnapshotRequest(cb: (req: { id: number; image: [number, number] }) => void): () => void;
  snapshotReply(id: number, reply: unknown): Promise<unknown>;
  offerOpen(path: string, needs: readonly string[]): Promise<unknown>;
}

/**
 * ADR 0128 D5, D6, view mode only: an offer that reaches the window is listed
 * and nothing else (INV-5); the runner carries out the human's clicks. A
 * snapshot paints nothing and dispatches nothing: main has captured the page,
 * and this reads the same view back as a spec and its manifest entry.
 */
function mountOfferFlow(): void {
  const door = bridge as unknown as OfferBridge;
  const runner = new OfferRunner({
    store,
    open: (path, needs) => door.offerOpen(path, needs),
    onScreen: (s) => onScreenOf(s, viewport, effects),
    pose: () => (store.get().mesh ? viewport.currentPose() : null),
    settle: async () => {
      viewport.renderNow();
      await painted();
      await painted();
    },
  });
  panels.push(runner.start());
  panels.push(door.onOffers((offers) => runner.receive(offers)));
  panels.push(
    door.onSnapshotRequest(({ id, image }) => {
      void door.snapshotReply(id, snapshotReplyOf(runner, image));
    }),
  );
}

/** The reply to one `--snapshot`; a view the manifest cannot state is refused (INV-6), never a still without its numbers. */
function snapshotReplyOf(runner: OfferRunner, image: [number, number]) {
  const result = runner.snapshot(image);
  if (!result.ok) return result;
  try {
    const canvas = viewport.renderer.domElement.getBoundingClientRect();
    const manifest = snapshotManifest(store.get(), result.snapshot, {
      image,
      fov: viewport.camera instanceof THREE.PerspectiveCamera ? viewport.camera.fov : null,
      layout: { viewport: [canvas.x, canvas.y, canvas.width, canvas.height], panels: stillLayout.inLayout() },
      blobs,
      step: effects.currentStep(),
    });
    return { ok: true, view: result.snapshot.view, manifest };
  } catch (err) {
    return { ok: false, reason: `the manifest cannot state this view: ${err instanceof Error ? err.message : String(err)}` };
  }
}

/** VU10: main's study feed (preload.ts `variants`, `onVariants`). */
interface StudyFeed {
  variants(): Promise<VariantsIn | null>;
  onVariants(cb: (study: VariantsIn | null) => void): () => void;
}

/** Whether the preload has every variant door. */
function isVariantDoor(b: Partial<VariantDoor & StudyFeed>): b is VariantDoor & StudyFeed {
  return ["variants", "onVariants", "variantsSwitch", "variantsCompare", "readCompareStep", "compareClose"].every((k) => typeof (b as Record<string, unknown>)[k] === "function");
}

/** VU10: the study main holds now, then each change; awaited before any shot compiles (a spec's `compare` names a row of it). */
async function followStudy(): Promise<void> {
  if (!isVariantDoor(variantBridge)) throw new Error("apeGmshViewer: the preload has no variant doors");
  variantBridge.onVariants((study) => store.dispatch({ type: "variantsRead", study }));
  store.dispatch({ type: "variantsRead", study: await variantBridge.variants() });
}

async function main() {
  const cfg = await bridge.config();
  await followStudy();
  // ADR 0128 D4: the agent modes (render, describe) run their own flow; the config's mode union predates them.
  const mode: string = cfg.mode;
  if (mode === "render" || mode === "describe") {
    const renderBridge = bridge as unknown as RenderBridge;
    if (mode === "render") {
      // ADR 0130: the spec's set has a `.user.h5` when one is beside its model; it is read before the shots, so every still states it.
      const job = await renderBridge.renderJob();
      const path = job.mode === "render" ? userPathOf(job) : null;
      if (job.mode === "render" && path !== null) await readUser({ path, model: job.model, results: job.results, known: false });
      // VU9a: the job's `<stem>.workspace.h5`, read (never written) before the shots, so a still shows its sets and verdicts.
      if (job.mode === "render") await effects.readWorkspace();
    }
    await runAgentMode({ store, blobs, effects, viewport, bridge: renderBridge, layout: stillLayout, combination: () => combinations.held });
    return;
  }
  if (cfg.mode === "view") {
    effects.attach();
    // After the effects' own onOpen, so a set's model read has started before the user feed looks at it.
    mountUserFeed();
    mountOfferFlow();
    // With V2f's `onOpen`, main replays the open set on subscription; loading
    // `config().file` here as well would read the model twice.
    if (cfg.file && !effects.hasOpenFeed()) await effects.open(cfg.file);
    return;
  }
  if (!cfg.file || !(await effects.open(cfg.file))) {
    await bridge.fail(store.get().artifacts.model?.error ?? "no --file given");
    return;
  }
  // The measure and capture harness (renderer/harness/): a throw inside it reaches the catch below.
  const harness: HarnessContext = { store, blobs, effects, viewport, bridge };
  if (cfg.mode === "measure") await runMeasure(harness, cfg);
  else await runCapture(harness, cfg, cfg.file);
}

main().catch((err) => {
  console.error(err);
  void bridge.fail(err instanceof Error ? `${err.message}\n${err.stack}` : String(err));
});
