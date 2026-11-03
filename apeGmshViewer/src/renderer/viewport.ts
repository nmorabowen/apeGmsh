// The 3-D viewport: renders the mesh from the state and the BlobStore, and
// turns a click into a `select` event. Its three.js objects are a
// derivation of the state keyed by the mesh's BlobRef ids and the
// visibility and selection records; it holds no model fact the store does
// not.
//
// Selection (R1 on #1283, and its follow-up on #1308): the selected element
// is drawn as a thick outline in the selection colour over everything (no
// depth test), its faces filled in that colour, a screen-space halo of fixed
// pixel width around all of it (selection.ts), and the rest of the model is
// dimmed. That reads at full-model zoom for a beam and for a shell alike;
// `F` frames the selection.

import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { BlobStore } from "../state/blobs.ts";
import type { DisplacementData } from "../effects.ts";
import type { LadrunoLineStationStep } from "../reader/ladruno.ts";
import { contourView, geometryPairing, gizmoPlanes, legendEntryColour, selectedElementsOf } from "../state/selectors.ts";
import { combinationColours, contourColours, contourShownKey, elementColours, noDataColours, toLinear, type CombinationColumnOf, type StepValues } from "./contour.ts";
import { CapLayer, releaseAware } from "./caps.ts";
import { SectionClip, sliderBox } from "./clip.ts";
import { GizmoInteractor, PlaneGizmos } from "./gizmo.ts";
import { applyGhostVisibility, basePositions, drawnPositions, edgesShown, frameFocus, gatherInto, ghostObjects, missingMarkerObjects, positionsFor, selectionOutline, writePositions, type FullPositions } from "./deform.ts";
import { LineDiagramLayer } from "./lineDiagram.ts";
import { EntitiesLayer } from "./entities.ts";
import { LoadsLayer, ReactionsLayer, SupportsLayer } from "./loads.ts";
import { SpringsLayer } from "./springs.ts";
import { makeNodePoints, nodePositions, setNodeSize, uniqueNodeVertices, type NodeVertices } from "./nodes.ts";
import { deformedPositions } from "../state/deform.ts";
import { dropUnplaced, gaussBasePositions, gaussDisplacements, gaussLayoutOf, type GaussLayout, type StationPoints } from "../state/gaussPoints.ts";
import type { State, Store } from "../state/store.ts";
import { roleColouring } from "../state/roles.ts";
import type { Box } from "../state/clip.ts";
import type { CameraPose, CameraRequest, DeclPath, GeometryInfo, LegendEntry, MeshInfo, Pick } from "../state/types.ts";
import { DARK, themeTokens, toHexNumber, type ThemeMode } from "../theme/tokens.ts";
import {
  DEFAULT_FOV_DEG,
  depthOf,
  headingOf,
  ISO_FORWARD,
  Navigator,
  nearestHit,
  poseOf,
  syncOrthoFrustum,
  VIEW_HEADINGS,
  type Heading,
  type ProjectionMode,
  type ViewCamera,
} from "./navigation.ts";
import { fitDistance, focusOf, type Focus, hitElement, SELECTION_COLOUR, SelectionHalo } from "./selection.ts";
import { edgeMask, elementMaskOf, sameDrawn, visibleMaps, type Drawn } from "./visible.ts";

/** The right rail's width plus its margins (style.css #rail: the inspector above the sources). */
const INSPECTOR_PX = 470 + 28;

/** The markers' size for the instant before the effects choose one. */
const NODE_SIZE_FALLBACK = 5;

export { SELECTION_COLOUR } from "./selection.ts";
/**
 * What the unselected model's colours are multiplied by while something is
 * selected. Material colours are linear; the renderer writes sRGB, so 0.07
 * reads as about 30 % brightness on screen.
 */
const DIM = 0.07;

/** The drawn primitives of one element. */
interface Prims {
  lines: number[];
  tris: number[];
}

/** Per-mesh indices the viewport derives once: element -> primitives, decl path -> element. */
interface MeshIndex {
  info: MeshInfo;
  byPath: Map<DeclPath, number>;
  prims: Map<number, Prims>;
}

/** VA1: `s` differs from `prev` in the camera alone (every other top-level slice is the same object). */
function onlyCameraChanged(s: State, prev: State | null): boolean {
  if (!prev || s.camera === prev.camera) return false;
  for (const k of Object.keys(s) as (keyof State)[]) if (k !== "camera" && s[k] !== prev[k]) return false;
  return true;
}

export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly perspCamera: THREE.PerspectiveCamera;
  readonly orthoCamera: THREE.OrthographicCamera;
  readonly nav: Navigator;
  private activeProjection: ProjectionMode = "perspective";
  private activeTheme: ThemeMode = "dark";
  private readonly hemiLight: THREE.HemisphereLight;
  private readonly keyLight: THREE.DirectionalLight;
  private readonly viewListeners = new Set<() => void>();
  private readonly scene = new THREE.Scene();
  private readonly raycaster = new THREE.Raycaster();
  private readonly host: HTMLElement;
  private readonly store: Store;
  private readonly blobs: BlobStore;
  private readonly content = new THREE.Group();
  private readonly highlight = new THREE.Group();
  /** V2f: the /geometry sibling (curves, faint surfaces, points), drawn on the geometry phase or alone. */
  private readonly geometryLayer = new THREE.Group();
  private geometryInfo: GeometryInfo | null = null;
  private lines: LineSegments2 | null = null;
  private drawnLineMap: Int32Array | null = null;
  private faces: THREE.Mesh | null = null;
  private edges: THREE.LineSegments | null = null;
  /** V4d: the original index of every drawn outline edge (what writePositions gathers the moved edges through). */
  private edgeMap: Int32Array | null = null;
  private frameFaces: THREE.Mesh | null = null;
  private frameEdges: THREE.LineSegments | null = null;
  private frameTriMap: Int32Array | null = null;
  private frameEdgeMap: Int32Array | null = null;
  private offsetLines: LineSegments2 | null = null;
  private offsetLineMap: Int32Array | null = null;
  private offsetJoints: THREE.Points | null = null;
  private offsetJointMap: Int32Array | null = null;
  private nodePoints: THREE.Points | null = null;
  private nodeVertices: NodeVertices | null = null;
  /** Outside `content`, so a rebuild of the mesh keeps the placed layout. */
  private readonly gaussGroup = new THREE.Group();
  private gaussPoints: THREE.Points | null = null;
  private gauss: { deps: unknown[]; layout: GaussLayout | null; base: Float32Array | null; posKey: string } | null = null;
  private stationsSource: () => (StationPoints & { key: string }) | null = () => null;
  private lineMaterials: LineMaterial[] = [];
  private index: MeshIndex | null = null;
  private drawn: Drawn | null = null;
  private readonly halo: SelectionHalo;
  private pending = false;
  private readonly observer: ResizeObserver;
  private readonly unsubscribe: () => void;
  /** CPU time of the last renderer.render call (ms). */
  lastRenderMs = 0;
  continuous = false;
  /** V4c1: the step data the effects fetched (effects.currentStep); the contour colours from it. */
  private stepSource: () => StepValues | null = () => null;
  /** VU7b-2: the User Combination column on hand (renderer/contour.ts `CombinationFeed.held`); app.ts wires it. */
  private combinationSource: () => CombinationColumnOf | null = () => null;
  /** V4c1: what the contour colours show now: the step, range and colormap key, or `no-data`; null when not colouring by contour. */
  contourApplied: string | null = null;
  /** V4d: the displacement the effects computed for the step shown (effects.currentDisplacement). */
  private dispSource: () => DisplacementData | null = () => null;
  /** 3-D line diagram step data from effects (effects.currentLineDiagram). */
  private lineDiagramSource: () => { key: string; step: LadrunoLineStationStep } | null = () => null;
  readonly lineDiagram: LineDiagramLayer;
  readonly loads = new LoadsLayer();
  readonly reactions = new ReactionsLayer();
  /** The Springs card: one force arrow per zero-length spring of the results step. */
  readonly springs = new SpringsLayer();
  /** The Display card's BC toggle: restraint glyphs at the supports. */
  readonly supports = new SupportsLayer();
  /** VU8: the `/entities` of the set's `.user.h5` (renderer/entities.ts). */
  readonly entities = new EntitiesLayer();
  /** V4d: every primitive's positions as drawn now (undeformed, or deformed at the step and scale its key names). */
  private full: FullPositions | null = null;
  /** V4d: the reference ghost (the undeformed mesh, faint) and the markers of drawn nodes with no displacement. */
  private readonly ghost = new THREE.Group();
  private readonly markers = new THREE.Group();
  /** V4e1: the section planes, on every drawn material before every render (renderer/clip.ts). */
  private readonly clip = new SectionClip();
  /** V4e2: the plane gizmos (clip-exempt, unpickable) and their pointer gestures, ahead of the navigator. */
  readonly gizmos = new PlaneGizmos();
  readonly gizmoInput: GizmoInteractor;
  /** V4e3: the cut-face caps (clip-exempt, unpickable), sliced on release and on a step change (renderer/caps.ts). */
  readonly caps = new CapLayer();
  /** V4e3: a panel offset slider is held: the caps wait for its release, as for a gizmo drag. */
  private sliding = false;
  private railCollapsed = false;
  /** V4d: what the drawn positions show now: `base`, or the displacement key and scale; null with no mesh. */
  get deformApplied(): string | null {
    return this.full?.key ?? null;
  }

  /** The active camera (perspective by default, or orthographic). */
  get camera(): ViewCamera {
    return this.activeProjection === "orthographic" ? this.orthoCamera : this.perspCamera;
  }

  /** Current camera projection mode. */
  get projection(): ProjectionMode {
    return this.activeProjection;
  }

  constructor(host: HTMLElement, store: Store, blobs: BlobStore) {
    this.host = host;
    this.store = store;
    this.blobs = blobs;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(toHexNumber(DARK.bg0), 0);
    // V4e1: the section planes are per material (renderer/clip.ts), never the renderer's global ones.
    this.renderer.localClippingEnabled = true;
    host.appendChild(this.renderer.domElement);

    this.perspCamera = new THREE.PerspectiveCamera(DEFAULT_FOV_DEG, 1, 0.01, 1000);
    this.orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);
    // Z up, always: the navigator's heading has no roll, so Z stays vertical on screen.
    this.perspCamera.up.set(0, 0, 1);
    this.orthoCamera.up.set(0, 0, 1);
    this.scene.add(this.perspCamera, this.orthoCamera);
    this.hemiLight = new THREE.HemisphereLight(toHexNumber(DARK.skyLight), toHexNumber(DARK.groundLight), 1.6);
    this.scene.add(this.hemiLight);
    this.keyLight = new THREE.DirectionalLight(toHexNumber(DARK.keyLight), 1.4);
    this.keyLight.position.set(0.5, 0.8, 1);
    this.perspCamera.add(this.keyLight);
    this.lineDiagram = new LineDiagramLayer(this.camera);
    this.scene.add(this.content, this.highlight, this.geometryLayer, this.lineDiagram.root, this.loads.root, this.reactions.root, this.springs.root, this.supports.root, this.entities.root);
    // V4d: the ghost and the missing-displacement markers (shown with the mesh, never on the geometry phase).
    this.scene.add(this.ghost, this.markers, this.gaussGroup);
    // V4e2: the gizmos (renderer/gizmo.ts): skipped by the clip stamping, never a raycast target.
    this.scene.add(this.gizmos.root);
    // V4e3: the caps, likewise skipped by the clip stamping (each is trimmed by the other planes itself).
    this.scene.add(this.caps.root);

    this.raycaster.params.Line2 = { threshold: 7 };
    this.halo = new SelectionHalo(this.renderer);
    const input = this.wireInput();
    this.gizmoInput = input.gizmo;
    this.nav = input.nav;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    this.unsubscribe = store.subscribe((s, prev) => this.onState(s, prev));
    this.onState(store.get(), null);
  }

  /** Subscribe to camera heading, projection, theme, or model visibility changes (for the ViewCube). */
  onViewChange(fn: () => void): () => void {
    this.viewListeners.add(fn);
    return () => this.viewListeners.delete(fn);
  }

  private notifyView(): void {
    for (const fn of this.viewListeners) fn();
  }

  /** Switch between perspective and orthographic projection, preserving target scale and pose. */
  setProjection(mode: ProjectionMode): void {
    if (mode === this.activeProjection) return;
    const m = this.store.get().mesh ?? this.geometryInfo;
    const r = m?.radius ?? 1;
    if (mode === "orthographic") {
      this.nav.orthoDistance = Math.max(depthOf(this.perspCamera, this.nav.target), 1e-3 * r);
      this.orthoCamera.position.copy(this.perspCamera.position);
      this.orthoCamera.quaternion.copy(this.perspCamera.quaternion);
      this.orthoCamera.updateMatrixWorld();
    } else {
      this.perspCamera.position.copy(this.orthoCamera.position);
      this.perspCamera.quaternion.copy(this.orthoCamera.quaternion);
      const back = new THREE.Vector3(0, 0, 1).applyQuaternion(this.perspCamera.quaternion);
      const curDepth = depthOf(this.perspCamera, this.nav.target);
      this.perspCamera.position.addScaledVector(back, this.nav.orthoDistance - curDepth);
      this.perspCamera.updateMatrixWorld();
    }
    this.activeProjection = mode;
    this.camera.add(this.keyLight);
    this.lineDiagram.setCamera(this.camera);
    this.syncCameraFrustum();
    this.nav.updated();
  }

  /** Show the dark or light (report-ready white paper) viewport theme: the store's `theme` (VA1; `onState` calls it). */
  private showTheme(mode: ThemeMode): void {
    if (mode === this.activeTheme) return;
    this.activeTheme = mode;
    const t = themeTokens(mode);
    this.renderer.setClearColor(toHexNumber(t.bg0), 0);
    this.hemiLight.color.setHex(toHexNumber(t.skyLight));
    this.hemiLight.groundColor.setHex(toHexNumber(t.groundLight));
    this.hemiLight.intensity = mode === "light" ? 1.05 : 1.6;
    this.keyLight.color.setHex(toHexNumber(t.keyLight));
    this.keyLight.intensity = mode === "light" ? 0.95 : 1.4;
    this.halo.setTheme(mode);
    this.lineDiagram.setTheme(mode);
    this.loads.setTheme(mode);
    this.reactions.setTheme(mode);
    this.springs.setTheme(mode);
    this.supports.setTheme(mode);
    this.entities.setTheme(mode);
    const s = this.store.get();
    if (this.geometryInfo) {
      this.setGeometry(this.geometryInfo);
      const onGeometry = s.phase.at?.kind === "geometry";
      this.geometryLayer.visible = geometryPairing(s).draw && (onGeometry || !s.mesh);
    }
    if (s.mesh) {
      this.rebuild(s, s.mesh);
      if (s.selection.decls.length) this.setHighlight(s, false);
      this.syncLineDiagram(s);
      this.syncLoads(s);
    }
    this.syncEntities(s);
    this.notifyView();
    this.requestRender();
  }

  /** Reset the view to the default isometric (or 2-D plan) orientation and fit the whole model (the `home` view). */
  private home(): void {
    const s = this.store.get();
    if (this.geometryInfo && (s.phase.at?.kind === "geometry" || !s.mesh)) this.frameGeometry(this.geometryInfo, null);
    else if (s.mesh) this.frame(s.mesh, null, null);
  }

  /**
   * VA1: fit the whole drawn scene at `heading` (the `fit` view and the fixed
   * named views): the geometry on the geometry phase or with no mesh, as `F`
   * does there; else the whole drawn (deformed) model, as `F` with nothing
   * selected does (`frameTo(null)`).
   */
  private fitScene(heading: Heading): void {
    const s = this.store.get();
    if (this.geometryInfo && (s.phase.at?.kind === "geometry" || !s.mesh)) this.frameGeometry(this.geometryInfo, heading);
    else if (s.mesh) this.frame(s.mesh, heading, frameFocus(this.full, s.mesh, this.blobs, this.drawn, null, null));
  }

  /**
   * VA1: apply a `setCamera` request (ADR 0128 D1). The reducer has checked
   * it; the camera moves only here, and the move is written back to the
   * store as `cameraMoved` with the next frame.
   */
  private applyCamera(s: State, to: CameraRequest): void {
    if (typeof to === "string") {
      if (to === "home") this.home();
      else if (to === "fit") this.fitScene(this.nav.heading);
      else if (to === "fit-selection") {
        const focus = s.mesh ? focusOf(s, this.blobs) : null;
        // Nothing of the selection drawn: a defined no-op, as for an empty selection.
        if (focus) this.frameTo(focus);
      } else this.fitScene(VIEW_HEADINGS[to]);
    } else if ("eye" in to) this.nav.place(to);
    else this.nav.snapTo({ yaw: to.yaw, tilt: to.tilt });
  }

  /** VA1: camera moves so far (`changed` counts them), and the count last written back. */
  private cameraMoves = 0;
  private cameraWritten = 0;
  private cameraWritePending = false;

  /**
   * VA1: write the camera back to the store once it has held still for a
   * whole frame: the move count is read at one animation frame and must be
   * the same at the next. A drag or the scripted orbit (one move a frame)
   * writes once, when it stops, not every frame: each dispatch runs every
   * panel's listener, and a write per frame cost the measured orbit about
   * 15 MB of renderer working set and about a fifth of its uncapped frame
   * rate. The write runs from an animation frame, so never inside a dispatch.
   */
  private scheduleCameraWrite(): void {
    if (this.cameraWritePending || this.cameraMoves === this.cameraWritten) return;
    this.cameraWritePending = true;
    let seen = -1;
    const tick = () => {
      // Moved since the last look (or this is the first): look again next frame.
      if (this.cameraMoves !== seen) {
        seen = this.cameraMoves;
        requestAnimationFrame(tick);
        return;
      }
      this.cameraWritePending = false;
      const pose = this.currentPose();
      // A refused write raises (loudly, from this frame) and stays unwritten:
      // the count advances only once the store holds the pose, so the next move writes again.
      this.store.dispatch({ type: "cameraMoved", pose });
      this.cameraWritten = seen;
    };
    requestAnimationFrame(tick);
  }

  /**
   * ADR 0128 D4: the camera shown now, as the write-back above records it,
   * read at once. The render mode records a still's camera from the frame it
   * took; a hidden window's animation frames come about once a second, so it
   * does not wait for the write-back.
   */
  currentPose(): CameraPose {
    const m = this.store.get().mesh ?? this.geometryInfo;
    // The target lies on the view axis at the pivot's depth, never closer than 1e-3 of the scene radius.
    return poseOf(this.camera, this.nav.target, 1e-3 * (m?.radius ?? 1), this.activeProjection, this.nav.orthoDistance);
  }

  /**
   * V4e2: the canvas's pointer listeners, the gizmo's first: listeners on one
   * target run in the order they were added, so a press the gizmo claims
   * (`stopImmediatePropagation`) never reaches the navigator, and a miss
   * reaches it untouched. A click on a gizmo goes to the same pick as a click
   * anywhere else (`selectAt`).
   */
  private hoverCleanup: (() => void) | null = null;

  private wireInput(): { gizmo: GizmoInteractor; nav: Navigator } {
    const element = this.renderer.domElement;
    const self = this;
    const gizmo = new GizmoInteractor({
      // V4e3: a release ends the drag in the gizmo's own handler, and moves no plane; the caps are sliced right after it.
      element: releaseAware(element, () => this.settleCaps()),
      rayAt: (x, y) => {
        this.aim(x, y);
        const r = this.raycaster.ray;
        return { origin: r.origin.toArray(), direction: r.direction.toArray() };
      },
      hit: (ray) => (this.gizmos.root.visible ? this.gizmos.hit(ray) : null),
      geometry: (id) => this.gizmos.geometries().get(id),
      plane: (id) => this.store.get().clip.planes.find((p) => p.id === id),
      dispatch: (e) => this.store.dispatch(e),
      // HTML chrome over the canvas (the legend, a panel) wins the press.
      chromeAt: (x, y) => {
        const top = typeof document === "undefined" ? element : document.elementFromPoint(x, y);
        return top !== null && top !== element;
      },
      select: (x, y, add) => this.selectAt(x, y, add),
    });
    const nav = new Navigator({
      get camera(): ViewCamera {
        return self.camera;
      },
      element: this.renderer.domElement,
      keys: window,
      bounds: () => {
        const m = this.store.get().mesh ?? this.geometryInfo;
        return m ? { center: new THREE.Vector3(...m.center), radius: m.radius } : null;
      },
      hitAt: (x, y) => this.pointAt(x, y),
      rayAt: (x, y) => {
        this.aim(x, y);
        return this.raycaster.ray.clone();
      },
      select: (x, y, add) => this.selectAt(x, y, add),
      fit: () => {
        const s = this.store.get();
        // On the geometry phase (or with no mesh), F fits the geometry; else the frame effect
        // frames the selection, or the whole model (the frameSelection store event).
        if (this.geometryInfo && (s.phase.at?.kind === "geometry" || !s.mesh)) this.frameGeometry(this.geometryInfo, nav.heading);
        else this.store.dispatch({ type: "frameSelection" });
      },
      clearSelection: () => this.store.dispatch({ type: "clearSelection" }),
      toggleEdges: () => {
        const s = this.store.get();
        if (s.mesh && (s.mesh.counts.faceCells > 0 || s.mesh.counts.solidCells > 0) && s.visibility.opacity > 0) {
          this.store.dispatch({ type: "setEdges", on: !s.visibility.edges });
        }
      },
      changed: () => {
        this.cameraMoves++;
        this.scheduleCameraWrite();
        this.syncCameraFrustum();
        this.notifyView();
        this.lineDiagram.onCameraChange();
        this.requestRender();
      },
    });
    const onPointerMove = (ev: PointerEvent) => {
      if (ev.buttons !== 0 || this.gizmoInput?.dragging || this.sliding) return;
      const top = typeof document === "undefined" ? element : document.elementFromPoint(ev.clientX, ev.clientY);
      if (top !== null && top !== element) {
        if (this.store.get().hover !== null) this.store.dispatch({ type: "hover", at: null });
        return;
      }
      this.hoverAt(ev.clientX, ev.clientY);
    };
    const onPointerLeave = () => {
      if (this.store.get().hover !== null) this.store.dispatch({ type: "hover", at: null });
    };
    element.addEventListener("pointermove", onPointerMove);
    element.addEventListener("pointerleave", onPointerLeave);
    this.hoverCleanup = () => {
      element.removeEventListener("pointermove", onPointerMove);
      element.removeEventListener("pointerleave", onPointerLeave);
    };
    return { gizmo, nav };
  }

  /** Project a hover raycast at a client point when line diagrams are active and update `s.hover`. */
  hoverAt(x: number, y: number): void {
    const s = this.store.get();
    if (!s.lineDiagram.on) {
      if (s.hover !== null) this.store.dispatch({ type: "hover", at: null });
      return;
    }
    const pick = this.pickAt(x, y);
    const prev = s.hover;
    if (pick === null && prev === null) return;
    if (
      pick !== null &&
      prev !== null &&
      typeof prev !== "string" &&
      prev.decl === pick.decl &&
      prev.at?.[0] === pick.at?.[0] &&
      prev.at?.[1] === pick.at?.[1] &&
      prev.at?.[2] === pick.at?.[2]
    ) {
      return;
    }
    this.store.dispatch({ type: "hover", at: pick });
  }

  /** A click at a client point: select (or toggle in the selection when `add` is true) what it picks; on a plain miss, clear the selection. */
  private selectAt(x: number, y: number, add = false): void {
    const pick = this.pickAt(x, y);
    if (add) {
      if (pick) this.store.dispatch({ type: "selectAdd", pick });
      return;
    }
    this.store.dispatch(pick ? { type: "select", pick } : { type: "clearSelection" });
  }

  /** Release the listeners, the observer, the subscription and the GPU objects. */
  dispose(): void {
    this.unsubscribe();
    this.observer.disconnect();
    this.hoverCleanup?.();
    this.hoverCleanup = null;
    this.nav.dispose();
    this.gizmoInput.dispose();
    this.gizmos.dispose();
    this.caps.dispose();
    this.lineDiagram.dispose();
    this.loads.dispose();
    this.reactions.dispose();
    this.springs.dispose();
    this.supports.dispose();
    this.entities.dispose();
    this.halo.dispose();
    this.clear(this.content);
    this.clear(this.highlight);
    this.clear(this.geometryLayer);
    this.clear(this.ghost);
    this.clear(this.markers);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private onState(s: State, prev: State | null): void {
    // VA1: the theme first, so a mesh built below takes its colours.
    if (s.theme !== this.activeTheme) this.showTheme(s.theme);
    // The camera write-back (once a frame while the view moves) changes nothing the scene is built from.
    if (!onlyCameraChanged(s, prev)) this.onScene(s, prev);
    // VA1: each new setCamera request once, after the scene it frames is built.
    const req = s.camera.request;
    if (req && req.seq !== prev?.camera.request?.seq) this.applyCamera(s, req.to);
  }

  private onScene(s: State, prev: State | null): void {
    // V4e1: the planes move in place; every material already shares them.
    if (!prev || s.clip !== prev.clip) {
      if (this.clip.sync(s.clip)) {
        this.lineDiagram.setClippingPlanes(this.clip.planes);
        this.loads.setClippingPlanes(this.clip.planes);
        this.reactions.setClippingPlanes(this.clip.planes);
        this.springs.setClippingPlanes(this.clip.planes);
        this.supports.setClippingPlanes(this.clip.planes);
        this.requestRender();
      }
    }
    this.syncGizmos(s, prev);
    this.settleCaps(s);
    this.onGeometry(s, prev);
    if (!prev || s.mesh !== prev.mesh) {
      this.setMesh(s.mesh);
      this.applyVisibility(s);
      this.setHighlight(s);
      this.syncGauss(s);
      this.syncLineDiagram(s);
      this.syncLoads(s);
      this.syncEntities(s);
      return;
    }
    if (
      (s.visibility.colourBy !== prev.visibility.colourBy ||
        s.visibility.extrudeFrames !== prev.visibility.extrudeFrames ||
        s.visibility.showOffsets !== prev.visibility.showOffsets) &&
      s.mesh
    ) {
      this.rebuild(s, s.mesh);
    } else if (s.visibility !== prev.visibility || s.phase !== prev.phase || s.lineDiagram.on !== prev.lineDiagram.on) {
      this.applyVisibility(s);
    }
    this.syncNodes(s);
    if (s.selection !== prev.selection) this.setHighlight(s);
    if (s.visibility.colourBy === "contour") this.recolour(s);
    this.redeform(s);
    this.syncGauss(s);
    this.syncLineDiagram(s);
    this.syncLoads(s);
    this.syncEntities(s);
  }

  /** The node markers: built on first use, then only shown, hidden and resized. */
  private syncNodes(s: State): void {
    if (!s.mesh) return;
    const { on: nodes, size: nodeSize } = s.nodes;
    if (nodes && !this.nodePoints) {
      // Only the markers: what is drawn has not changed, so the scene is not rebuilt.
      if (!this.drawn) return;
      this.buildNodePoints(s, s.mesh, this.drawn, this.full ?? basePositions(s.mesh, this.blobs));
      this.requestRender();
      return;
    }
    if (!this.nodePoints) return;
    const size = nodeSize ?? NODE_SIZE_FALLBACK;
    const m = this.nodePoints.material as THREE.PointsMaterial;
    if (this.nodePoints.visible === nodes && m.size === size) return;
    this.nodePoints.visible = nodes;
    setNodeSize(this.nodePoints, size);
    this.requestRender();
  }

  /** Where the beam stations for the Gauss markers come from (app.ts wires the effects). */
  setStationsSource(source: () => (StationPoints & { key: string }) | null): void {
    this.stationsSource = source;
  }

  /**
   * The Gauss point and beam station markers: the layout is placed once per
   * stage, hidden set and model; the positions follow the deformed shape by
   * the layout's own weights over the step's nodal displacements.
   */
  private syncGauss(s: State): void {
    const at = s.phase.at;
    const want = s.gaussPoints.on && at?.kind === "results" && s.mesh !== null;
    if (!want) {
      if (this.gaussPoints?.visible) {
        this.gaussPoints.visible = false;
        this.requestRender();
      }
      return;
    }
    const info = s.mesh!;
    const stations = this.stationsSource();
    const deps = [s.results, at.stage, s.visibility.hidden, s.mesh, s.decls, s.stageElements, s.sequence, stations?.key ?? null, this.activeTheme];
    let changed = false;
    if (!this.gauss || this.gauss.deps.length !== deps.length || this.gauss.deps.some((d, i) => d !== deps[i])) {
      if (this.gaussPoints) {
        this.gaussGroup.remove(this.gaussPoints);
        this.gaussPoints.geometry.dispose();
        (this.gaussPoints.material as THREE.Material).dispose();
        this.gaussPoints = null;
      }
      let layout = info.nodes ? gaussLayoutOf(s, this.blobs, stations) : null;
      let base = layout && info.nodes ? gaussBasePositions(layout, this.blobs.get(info.nodes.ids) as Float64Array, this.blobs.get(info.nodes.xyz) as Float64Array) : null;
      if (layout && base) ({ layout, base } = dropUnplaced(layout, base));
      this.gauss = { deps, layout, base, posKey: "" };
      if (layout && base && layout.count) {
        const t = themeTokens(this.activeTheme);
        this.gaussPoints = makeNodePoints(base.slice(), s.gaussPoints.size ?? NODE_SIZE_FALLBACK, toHexNumber(t.accent), toHexNumber(t.bg0));
        this.gaussGroup.add(this.gaussPoints);
      }
      changed = true;
      // The Display row states what was placed and why the rest was not.
      const stage = s.results?.stages[at.stage]?.id ?? "";
      if (layout) {
        const measured = { stage, count: layout.count, refused: layout.refused, notes: layout.notes };
        queueMicrotask(() => this.store.dispatch({ type: "gaussMeasured", ...measured }));
      }
    }
    const g = this.gauss;
    if (this.gaussPoints && g.layout && g.base) {
      const full = this.full;
      const disp = full && full.key !== "base" && full.disp?.nodal ? full.disp : null;
      const posKey = disp ? `${full!.key}` : "base";
      if (posKey !== g.posKey) {
        g.posKey = posKey;
        const a = this.gaussPoints.geometry.getAttribute("position") as THREE.BufferAttribute;
        const out = a.array as Float32Array;
        if (disp) deformedPositions(g.base, gaussDisplacements(g.layout, disp.nodal!.column, disp.nodal!.rows), s.deform.scale, out);
        else out.set(g.base);
        a.needsUpdate = true;
        this.gaussPoints.geometry.computeBoundingSphere();
        changed = true;
      }
      const size = s.gaussPoints.size ?? NODE_SIZE_FALLBACK;
      if (!this.gaussPoints.visible || (this.gaussPoints.material as THREE.PointsMaterial).size !== size) {
        this.gaussPoints.visible = true;
        setNodeSize(this.gaussPoints, size);
        changed = true;
      }
    }
    if (changed) this.requestRender();
  }

  /** The markers of the drawn nodes, at the drawn (deformed while it shows) positions. */
  private buildNodePoints(s: State, info: MeshInfo, drawn: { lineMap: Int32Array; triMap: Int32Array }, full: { line: Float32Array; tri: Float32Array }): void {
    this.nodeVertices = uniqueNodeVertices(
      this.blobs.get(info.vertexNodes.line) as Float64Array,
      this.blobs.get(info.vertexNodes.tri) as Float64Array,
      drawn.lineMap,
      drawn.triMap,
    );
    const t = themeTokens(this.activeTheme);
    this.nodePoints = makeNodePoints(nodePositions(this.nodeVertices, full.line, full.tri), s.nodes.size ?? NODE_SIZE_FALLBACK, toHexNumber(t.text), toHexNumber(t.bg0));
    this.content.add(this.nodePoints);
  }

  /** Where the 3-D line diagram's step data comes from (app.ts wires the effects). */
  setLineDiagramSource(source: () => { key: string; step: LadrunoLineStationStep } | null): void {
    this.lineDiagramSource = source;
    this.syncLineDiagram(this.store.get());
  }

  private syncLineDiagram(s: State = this.store.get()): void {
    if (
      this.lineDiagram.sync(
        s,
        this.blobs,
        this.lineDiagramSource(),
        this.full ?? this.dispSource(),
        false,
        this.drawn?.lineMap,
      )
    ) {
      this.requestRender();
    }
  }

  private syncEntities(s: State = this.store.get()): void {
    if (this.entities.sync(s, this.blobs)) this.requestRender();
  }

  private syncLoads(s: State = this.store.get()): void {
    // The Reactions and Springs cards' arrows follow the same inputs, plus the step's data.
    const loads = this.loads.sync(s, this.blobs, this.full);
    const supports = this.supports.sync(s, this.blobs, this.full);
    const springs = this.springs.sync(s, this.blobs, this.full, this.stepSource());
    if (this.reactions.sync(s, this.blobs, this.full, this.stepSource()) || loads || supports || springs) {
      this.requestRender();
    }
  }

  /**
   * V4e2: the gizmos follow the planes, the eyes and the footer toggle, over
   * the reference (undeformed) box the slider maps; hidden with no mesh and on
   * the geometry phase.
   */
  private syncGizmos(s: State, prev: State | null): void {
    const show = s.mesh !== null && s.phase.at?.kind !== "geometry";
    let changed = show !== this.gizmos.root.visible;
    this.gizmos.root.visible = show;
    if (!prev || s.clip !== prev.clip || s.mesh !== prev.mesh) changed = this.gizmos.sync(gizmoPlanes(s.clip), this.sliderBox()) || changed;
    if (changed) this.requestRender();
  }

  /**
   * V4e3: the caps follow the state, hidden while a plane is dragged (a gizmo
   * or a panel slider) and sliced again once it is released: on every state
   * change, after a gizmo release (`releaseAware`) and a slider release.
   */
  private settleCaps(s: State = this.store.get()): void {
    if (this.caps.sync(s, this.blobs, this.stepSource(), this.gizmoInput.dragging || this.sliding, this.combinationSource())) this.requestRender();
  }

  /** V4e3: a panel offset slider is held (true) or released (app.ts wires the planes panel). */
  setSliding(on: boolean): void {
    if (this.sliding === on) return;
    this.sliding = on;
    this.settleCaps();
  }

  /** V4d: where the deformed shape's displacement comes from (app.ts wires the effects). */
  setDisplacementSource(source: () => DisplacementData | null): void {
    this.dispSource = source;
    const s = this.store.get();
    this.redeform(s);
    this.syncLineDiagram(s);
    this.syncLoads(s);
  }

  /** The ghost shows behind a deformed mesh, when the user asked for it. */
  private ghostWanted(s: State): boolean {
    return s.deform.ghost && this.full !== null && this.full.key !== "base";
  }

  /**
   * Move the drawn vertices when what the deformed shape shows changed (a
   * step, a scale, a re-read, the toggle): every primitive's positions are
   * recomputed once, then written in place into the drawn faces and lines
   * (no rebuild), with the markers, the ghost and the selection outline.
   */
  private redeform(s: State): void {
    const info = s.mesh;
    if (!info || !this.drawn) return;
    const next = positionsFor(s, info, this.blobs, this.dispSource(), this.full);
    let changed = false;
    if (next !== null && next !== this.full) {
      this.full = next;
      const lineSrc = s.visibility.showOffsets && next.clearLine ? next.clearLine : next.line;
      const drawnLines = { lineMap: this.drawnLineMap ?? this.drawn.lineMap, triMap: this.drawn.triMap };
      writePositions(
        this.faces,
        this.lines,
        { line: lineSrc, tri: next.tri, edge: next.edge },
        drawnLines,
        this.edges && this.edgeMap ? { object: this.edges, map: this.edgeMap } : null,
      );
      if (this.frameFaces && this.frameTriMap && next.frameTri) {
        const a = this.frameFaces.geometry.getAttribute("position") as THREE.BufferAttribute;
        gatherInto(next.frameTri, this.frameTriMap, 9, a.array as Float32Array);
        a.needsUpdate = true;
        this.frameFaces.geometry.computeVertexNormals();
        this.frameFaces.geometry.computeBoundingBox();
        this.frameFaces.geometry.computeBoundingSphere();
      }
      if (this.frameEdges && this.frameEdgeMap && next.frameEdge) {
        const a = this.frameEdges.geometry.getAttribute("position") as THREE.BufferAttribute;
        gatherInto(next.frameEdge, this.frameEdgeMap, 6, a.array as Float32Array);
        a.needsUpdate = true;
        this.frameEdges.geometry.computeBoundingSphere();
      }
      if (this.offsetLines && this.offsetLineMap && next.offsetLine) {
        const a = this.offsetLines.geometry.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute;
        gatherInto(next.offsetLine, this.offsetLineMap, 6, a.data.array as Float32Array);
        a.data.needsUpdate = true;
        this.offsetLines.geometry.computeBoundingBox();
        this.offsetLines.geometry.computeBoundingSphere();
      }
      if (this.offsetJoints && this.offsetJointMap && next.offsetJoint) {
        const a = this.offsetJoints.geometry.getAttribute("position") as THREE.BufferAttribute;
        gatherInto(next.offsetJoint, this.offsetJointMap, 3, a.array as Float32Array);
        a.needsUpdate = true;
        this.offsetJoints.geometry.computeBoundingSphere();
      }
      if (this.nodePoints && this.nodeVertices) {
        const a = this.nodePoints.geometry.getAttribute("position") as THREE.BufferAttribute;
        nodePositions(this.nodeVertices, next.line, next.tri, a.array as Float32Array);
        a.needsUpdate = true;
        this.nodePoints.geometry.computeBoundingSphere();
      }
      this.setMarkers(info);
      if (this.edges) this.edges.visible = edgesShown(s.visibility.edges, next, s.visibility.opacity);
      if (this.frameEdges) this.frameEdges.visible = edgesShown(s.visibility.edges, next, s.visibility.opacity);
      if (this.ghost) applyGhostVisibility(this.ghost, s.visibility.edges, next, s.visibility.opacity);
      // The selection outline follows the moved vertices (no pulse: the selection did not change).
      if (selectedElementsOf(s).length) this.setHighlight(s, false);
      changed = true;
    }
    const ghost = this.ghostWanted(s) && s.phase.at?.kind !== "geometry";
    if (ghost !== this.ghost.visible) {
      this.ghost.visible = ghost;
      changed = true;
    }
    if (changed) this.requestRender();
  }

  /** The markers of drawn nodes with no displacement at the step shown (none when undeformed). */
  private setMarkers(info: MeshInfo): void {
    this.clear(this.markers);
    if (this.full && this.drawn) for (const o of missingMarkerObjects(this.full, info, this.blobs, this.drawn, this.activeTheme)) this.markers.add(o);
  }

  /** V4c1: where the contour's step data comes from (app.ts wires the effects). */
  setStepSource(source: () => StepValues | null): void {
    this.stepSource = source;
    this.recolour(this.store.get());
    this.settleCaps();
  }

  /** VU7b-2: where a User Combination's contour values come from (app.ts wires the feed). */
  setCombinationSource(source: () => CombinationColumnOf | null): void {
    this.combinationSource = source;
    this.recolour(this.store.get());
    this.settleCaps();
  }

  /**
   * The contour colours of the drawn primitives: the step on screen once its
   * data and range are here, no data everywhere when nothing can be coloured
   * (the legend says why), else null (the step is still loading).
   */
  private contourFor(s: State, info: MeshInfo, drawn: Drawn): { line: Float32Array; tri: Float32Array; key: string } | null {
    const v = contourView(s);
    if (v.state === "off") throw new Error("viewport: contour colours while not colouring by contour");
    // V4c2a: a refused element field (an unverified file, disagreeing groups, rows that do not join) and a topology
    // not built yet are drawn as no data, never coloured "with a warning".
    if (v.state === "unavailable" || v.state === "refused" || v.state === "not-yet") {
      return { line: noDataColours(drawn.lineMap, 2), tri: noDataColours(drawn.triMap, 3), key: `no-data:${v.reason}` };
    }
    if (v.state === "element") return elementColours(s, v, info, this.blobs, this.stepSource(), drawn.lineMap, drawn.triMap);
    if (v.state === "combination") return combinationColours(v, info, this.blobs, this.combinationSource(), drawn.lineMap, drawn.triMap);
    return contourColours(s, v, info, this.blobs, this.stepSource(), drawn.lineMap, drawn.triMap);
  }

  /** Map per-segment start/end contour colours (`lineColours` on `drawnLineMap`) onto extruded prism triangles (`frameTriMap`). */
  private frameTriContourColours(info: MeshInfo, drawnLineMap: Int32Array, lineColours: Float32Array, frameTriMap: Int32Array): Float32Array {
    const fe = info.frameExtrusion!;
    const nSegs = info.linePositions.shape[0]!;
    const segToDrawn = new Int32Array(nSegs).fill(-1);
    drawnLineMap.forEach((seg, d) => {
      segToDrawn[seg] = d;
    });
    const lineNodes = this.blobs.get(info.vertexNodes.line) as Float64Array;
    const triSegs = this.blobs.i32(fe.triSegments);
    const triNodes = this.blobs.get(fe.triNodes) as Float64Array;
    const out = new Float32Array(frameTriMap.length * 9);
    for (let dTri = 0; dTri < frameTriMap.length; dTri++) {
      const t = frameTriMap[dTri]!;
      const seg = triSegs[t]!;
      const d = segToDrawn[seg]!;
      if (d < 0) continue;
      const nodeI = lineNodes[2 * seg]!;
      for (let v = 0; v < 3; v++) {
        const isI = triNodes[3 * t + v] === nodeI;
        const srcOff = 6 * d + (isI ? 0 : 3);
        const dstOff = 9 * dTri + 3 * v;
        out[dstOff] = lineColours[srcOff]!;
        out[dstOff + 1] = lineColours[srcOff + 1]!;
        out[dstOff + 2] = lineColours[srcOff + 2]!;
      }
    }
    return out;
  }

  private subsetLineColours(info: MeshInfo, drawnLineMap: Int32Array, subLineMap: Int32Array, lineColours: Float32Array): Float32Array {
    if (subLineMap.length === drawnLineMap.length) return lineColours;
    const nSegs = info.linePositions.shape[0]!;
    const segToDrawn = new Int32Array(nSegs).fill(-1);
    drawnLineMap.forEach((seg, d) => {
      segToDrawn[seg] = d;
    });
    const out = new Float32Array(subLineMap.length * 6);
    for (let i = 0; i < subLineMap.length; i++) {
      const d = segToDrawn[subLineMap[i]!]!;
      if (d >= 0) out.set(lineColours.subarray(6 * d, 6 * d + 6), 6 * i);
    }
    return out;
  }

  /** Update the colour attributes in place when what the contour shows changed (a scrub: no rebuild). */
  private recolour(s: State): void {
    const info = s.mesh;
    if (!info || !this.drawn || s.visibility.colourBy !== "contour") return;
    // Most state changes (a hover, a panel) leave what the contour shows alone: compare keys before computing colours.
    const v = contourView(s);
    if (contourShownKey(v) === this.contourApplied) return;
    const c = this.contourFor(s, info, this.drawn);
    if (!c || c.key === this.contourApplied) return;
    if (this.faces) {
      const a = this.faces.geometry.getAttribute("color") as THREE.BufferAttribute;
      (a.array as Float32Array).set(c.tri);
      a.needsUpdate = true;
    }
    if (this.frameFaces && this.frameTriMap) {
      const a = this.frameFaces.geometry.getAttribute("color") as THREE.BufferAttribute;
      (a.array as Float32Array).set(this.frameTriContourColours(info, this.drawn.lineMap, c.line, this.frameTriMap));
      a.needsUpdate = true;
    }
    if (this.lines) {
      // LineSegmentsGeometry keeps the colours as one interleaved buffer, 6 floats per segment (start rgb, end rgb).
      const attr = this.lines.geometry.getAttribute("instanceColorStart") as THREE.InterleavedBufferAttribute;
      (attr.data.array as Float32Array).set(this.subsetLineColours(info, this.drawn.lineMap, this.drawnLineMap ?? this.drawn.lineMap, c.line));
      attr.data.needsUpdate = true;
    }
    this.contourApplied = c.key;
    this.requestRender();
  }

  /**
   * The geometry layer (V2f): rebuilt when the geometry changes; shown when
   * it may be drawn (it pairs with the model, or there is no model) and the
   * phase is geometry or there is no mesh. The mesh and its selection are
   * hidden on the geometry phase.
   */
  private onGeometry(s: State, prev: State | null): void {
    if (!prev || s.geometry !== prev.geometry) this.setGeometry(s.geometry);
    const onGeometry = s.phase.at?.kind === "geometry";
    const show = this.geometryInfo !== null && geometryPairing(s).draw && (onGeometry || !s.mesh);
    const wasShown = this.geometryLayer.visible && this.geometryLayer.children.length > 0;
    this.geometryLayer.visible = show;
    this.content.visible = !onGeometry;
    this.highlight.visible = !onGeometry;
    this.markers.visible = !onGeometry;
    this.ghost.visible = !onGeometry && this.ghostWanted(s);
    // A geometry opened alone is framed when it first shows.
    if (show && !wasShown && !s.mesh && this.geometryInfo) this.frameGeometry(this.geometryInfo);
    if (!prev || s.geometry !== prev.geometry || s.phase !== prev.phase || s.artifacts !== prev.artifacts) {
      this.notifyView();
      this.requestRender();
    }
  }

  private setGeometry(g: GeometryInfo | null): void {
    this.clear(this.geometryLayer);
    this.geometryLayer.visible = false;
    this.geometryInfo = g;
    if (!g) return;
    const t = themeTokens(this.activeTheme);
    const curves = this.blobs.f32(g.curvePositions);
    if (curves.length) {
      const cg = new THREE.BufferGeometry();
      cg.setAttribute("position", new THREE.BufferAttribute(curves, 3));
      this.geometryLayer.add(new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: toHexNumber(t.geometryCurve) })));
    }
    const tris = this.blobs.f32(g.surfacePositions);
    if (tris.length) {
      const sg = new THREE.BufferGeometry();
      sg.setAttribute("position", new THREE.BufferAttribute(tris, 3));
      sg.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({
        color: toHexNumber(t.geometrySurface), side: THREE.DoubleSide, flatShading: true, transparent: true, opacity: 0.35, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      });
      this.geometryLayer.add(new THREE.Mesh(sg, m));
    }
    const pts = this.blobs.f32(g.pointPositions);
    if (pts.length) {
      const pg = new THREE.BufferGeometry();
      pg.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      this.geometryLayer.add(new THREE.Points(pg, new THREE.PointsMaterial({ color: toHexNumber(t.geometryPoint), size: 5, sizeAttenuation: false })));
    }
  }

  /** Collapse or expand the right rail, updating the camera frustum offset immediately. */
  setRailCollapsed(collapsed: boolean): void {
    if (this.railCollapsed === collapsed) return;
    this.railCollapsed = collapsed;
    this.syncCameraFrustum();
    this.requestRender();
  }

  /** Width kept clear for the right rail (inspector, sources): the model is framed left of it. */
  private panelWidth(): number {
    if (this.railCollapsed) return 0;
    const w = this.host.clientWidth || 1;
    return w > 1000 ? INSPECTOR_PX : 0;
  }

  private syncCameraFrustum(): void {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    // A virtual frame P px wider, viewed from x = P: its centre lands at (w - P) / 2.
    const p = this.panelWidth();
    this.perspCamera.aspect = (w + p) / h;
    this.perspCamera.setViewOffset(w + p, h, p, 0, w, h);
    this.perspCamera.updateProjectionMatrix();
    if (this.nav) syncOrthoFrustum(this.orthoCamera, this.nav.orthoDistance, w + p, h, p, w, h, this.perspCamera.fov);
  }

  private resize(): void {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.syncCameraFrustum();
    for (const m of this.lineMaterials) m.resolution.set(w, h);
    this.halo.resize(w, h, this.renderer.getPixelRatio());
    this.requestRender();
  }

  private makeLines(pos: Float32Array, col: Float32Array | null, width: number, colour?: number): LineSegments2 {
    const g = new LineSegmentsGeometry();
    g.setPositions(pos);
    if (col) g.setColors(col);
    const m = new LineMaterial({
      linewidth: width,
      vertexColors: col !== null,
      color: colour ?? toHexNumber(themeTokens(this.activeTheme).keyLight),
      worldUnits: false,
    });
    m.resolution.set(this.host.clientWidth || 1, this.host.clientHeight || 1);
    this.lineMaterials.push(m);
    return new LineSegments2(g, m);
  }

  private clear(group: THREE.Group): void {
    group.traverse((o) => {
      const any = o as THREE.Mesh;
      any.geometry?.dispose();
      const mat = any.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat?.dispose();
    });
    group.clear();
  }

  /** The viewport's own index of a mesh, built once per MeshInfo. */
  private indexOf(info: MeshInfo): MeshIndex {
    if (this.index && this.index.info === info) return this.index;
    const byPath = new Map<DeclPath, number>();
    info.elements.forEach((p, i) => byPath.set(p, i));
    const prims = new Map<number, Prims>();
    const at = (e: number): Prims => prims.get(e) ?? prims.set(e, { lines: [], tris: [] }).get(e)!;
    this.blobs.i32(info.lineElement).forEach((e, i) => at(e).lines.push(i));
    this.blobs.i32(info.triElement).forEach((e, i) => at(e).tris.push(i));
    this.index = { info, byPath, prims };
    return this.index;
  }

  /**
   * What colours the primitives: by physical group (the loader's legend and
   * per-primitive group blobs) or by the structural role the file records
   * (state/roles.ts, per element, mapped to the primitives here).
   */
  private colourSource(s: State, info: MeshInfo): { legend: readonly LegendEntry[]; line: Int32Array; tri: Int32Array; edge: number } {
    const t = themeTokens(this.activeTheme);
    if (s.visibility.colourBy === "group") {
      return { legend: info.legend, line: this.blobs.i32(info.lineGroup), tri: this.blobs.i32(info.triGroup), edge: toHexNumber(t.edge) };
    }
    const r = roleColouring(s);
    if (!r) throw new Error("viewport: colour by role with no mesh");
    const perPrim = (element: Int32Array) => Int32Array.from(element, (e) => r.byElement[e]!);
    return { legend: r.legend, line: perPrim(this.blobs.i32(info.lineElement)), tri: perPrim(this.blobs.i32(info.triElement)), edge: toHexNumber(t.edgeRole) };
  }

  /** Vertex colours of the drawn primitives from a legend and a per-primitive row index. */
  private colours(legend: readonly LegendEntry[], group: Int32Array, map: Int32Array, perPrim: number): Float32Array {
    const out = new Float32Array(map.length * perPrim * 3);
    // LineMaterial (perPrim === 2) is unlit and runs linearToSRGB on output; linearising in light mode keeps 1-D beams true to their legend chip.
    const linearise = perPrim === 2 && this.activeTheme === "light";
    for (let d = 0; d < map.length; d++) {
      const entry = legend[group[map[d]!]!];
      if (!entry) throw new Error(`viewport: primitive ${map[d]} has no legend row`);
      const srgb = legendEntryColour(entry, this.activeTheme);
      const c = linearise ? [toLinear(srgb[0]), toLinear(srgb[1]), toLinear(srgb[2])] : srgb;
      for (let v = 0; v < perPrim; v++) out.set(c, (d * perPrim + v) * 3);
    }
    return out;
  }

  /** Which original primitives the hidden set leaves visible (visible.ts). */
  private visible(s: State, info: MeshInfo): Drawn {
    const { lineMap, triMap } = visibleMaps(info.legend, s.visibility.hidden, this.blobs.i32(info.lineGroup), this.blobs.i32(info.triGroup), elementMaskOf(s, info, this.blobs));
    const nTri = info.triPositions.shape[0]!;
    const mask = triMap.length === nTri ? null : edgeMask(this.blobs.f32(info.triPositions), this.blobs.f32(info.edgePositions), triMap);
    return { lineMap, triMap, edgeMask: mask };
  }

  private static gather(src: Float32Array, map: Int32Array, stride: number): Float32Array {
    if (map.length * stride === src.length) return src;
    const out = new Float32Array(map.length * stride);
    map.forEach((o, d) => out.set(src.subarray(o * stride, (o + 1) * stride), d * stride));
    return out;
  }

  private setMesh(info: MeshInfo | null): void {
    this.clear(this.content);
    this.clear(this.highlight);
    this.lineMaterials = [];
    this.lines = null;
    this.drawnLineMap = null;
    this.faces = null;
    this.edges = null;
    this.edgeMap = null;
    this.frameFaces = null;
    this.frameEdges = null;
    this.frameTriMap = null;
    this.frameEdgeMap = null;
    this.offsetLines = null;
    this.offsetLineMap = null;
    this.offsetJoints = null;
    this.offsetJointMap = null;
    this.nodePoints = null;
    this.nodeVertices = null;
    this.drawn = null;
    this.index = null;
    this.full = null;
    this.clear(this.ghost);
    this.clear(this.markers);
    this.notifyView();
    if (!info) return this.requestRender();
    this.indexOf(info);
    const s = this.store.get();
    this.rebuild(s, info);
    // VA1: a new model opens framed; a re-read keeps the camera the store holds (and the view with it).
    if (s.camera.pose === null) this.frame(info);
    else this.nav.updated();
  }

  /** (Re)build the content group for the current hidden set. */
  private rebuild(s: State, info: MeshInfo): void {
    this.clear(this.content);
    this.lineMaterials = this.lineMaterials.filter((m) => this.highlight.children.some((o) => (o as THREE.Mesh).material === m));
    this.lines = null;
    this.drawnLineMap = null;
    this.faces = null;
    this.edges = null;
    this.edgeMap = null;
    this.frameFaces = null;
    this.frameEdges = null;
    this.frameTriMap = null;
    this.frameEdgeMap = null;
    this.offsetLines = null;
    this.offsetLineMap = null;
    this.offsetJoints = null;
    this.offsetJointMap = null;
    this.nodePoints = null;
    this.nodeVertices = null;
    const drawn = this.visible(s, info);
    this.drawn = drawn;
    const t = themeTokens(this.activeTheme);
    // V4c1: a contour colours per vertex from the step on screen; until its
    // data is here, every drawn primitive is no data (never a group colour).
    const contour = s.visibility.colourBy === "contour";
    const cc = contour ? (this.contourFor(s, info, drawn) ?? { line: noDataColours(drawn.lineMap, 2), tri: noDataColours(drawn.triMap, 3), key: "no-data:loading" }) : null;
    this.contourApplied = cc?.key ?? null;
    const source = contour ? { legend: [], line: new Int32Array(0), tri: new Int32Array(0), edge: toHexNumber(t.edge) } : this.colourSource(s, info);
    // V4d: the positions drawn are the deformed ones when the deformed shape
    // shows (copied out: writePositions later moves them in place, so they
    // never alias a mesh blob), and the ghost is the undeformed mesh.
    const full = positionsFor(s, info, this.blobs, this.dispSource(), this.full) ?? this.full ?? basePositions(info, this.blobs);
    this.full = full;

    const tri = gatherInto(full.tri, drawn.triMap, 9);
    if (tri.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(tri, 3));
      g.setAttribute("color", new THREE.BufferAttribute(cc ? cc.tri : this.colours(source.legend, source.tri, drawn.triMap, 3), 3));
      g.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        flatShading: true,
        roughness: 0.75,
        metalness: 0.0,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      this.faces = new THREE.Mesh(g, m);
      this.content.add(this.faces);
      const ep = this.blobs.f32(info.edgePositions);
      const edgeMap = drawn.edgeMask
        ? Int32Array.from(drawn.edgeMask.reduce<number[]>((acc, v, i) => (v ? (acc.push(i), acc) : acc), []))
        : Int32Array.from({ length: ep.length / 6 }, (_, i) => i);
      // The drawn edges are a copy of the drawn (possibly deformed) positions,
      // moved in place with the faces (V4d); the ghost reads the blobs itself.
      this.edgeMap = edgeMap;
      const edge = gatherInto(full.edge, edgeMap, 6);
      if (edge.length) {
        const eg = new THREE.BufferGeometry();
        eg.setAttribute("position", new THREE.BufferAttribute(edge, 3));
        const edgeOpacity = this.activeTheme === "light" ? 0.72 : 0.45;
        const em = new THREE.LineBasicMaterial({ color: source.edge, transparent: true, opacity: edgeOpacity });
        this.edges = new THREE.LineSegments(eg, em);
        this.content.add(this.edges);
      }
    }

    const fe = info.frameExtrusion;
    const visibleSegSet = new Set<number>(drawn.lineMap);
    const extMask = fe && s.visibility.extrudeFrames ? (this.blobs.get(fe.extrudedSegmentMask) as Uint8Array) : null;
    const drawnLineMap = extMask ? drawn.lineMap.filter((seg) => extMask[seg] === 0) : drawn.lineMap;
    this.drawnLineMap = drawnLineMap;

    if (fe && s.visibility.extrudeFrames && full.frameTri && full.frameEdge) {
      const triSegs = this.blobs.i32(fe.triSegments);
      const fTriIdx: number[] = [];
      for (let i = 0; i < triSegs.length; i++) {
        if (visibleSegSet.has(triSegs[i]!)) fTriIdx.push(i);
      }
      const frameTriMap = Int32Array.from(fTriIdx);
      this.frameTriMap = frameTriMap;
      const fTri = gatherInto(full.frameTri, frameTriMap, 9);
      if (fTri.length) {
        const fg = new THREE.BufferGeometry();
        fg.setAttribute("position", new THREE.BufferAttribute(fTri, 3));
        let fCol: Float32Array;
        if (cc) {
          fCol = this.frameTriContourColours(info, drawn.lineMap, cc.line, frameTriMap);
        } else if (s.visibility.colourBy === "role") {
          const r = roleColouring(s)!;
          const roleTri = Int32Array.from(this.blobs.i32(fe.triElement), (e) => r.byElement[e]!);
          fCol = this.colours(r.legend, roleTri, frameTriMap, 3);
        } else {
          fCol = this.colours(source.legend, this.blobs.i32(fe.triGroup), frameTriMap, 3);
        }
        fg.setAttribute("color", new THREE.BufferAttribute(fCol, 3));
        fg.computeVertexNormals();
        const fm = new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          flatShading: true,
          roughness: 0.72,
          metalness: 0.05,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        });
        this.frameFaces = new THREE.Mesh(fg, fm);
        this.content.add(this.frameFaces);

        const edgeSegs = this.blobs.i32(fe.edgeSegments);
        const fEdgeIdx: number[] = [];
        for (let i = 0; i < edgeSegs.length; i++) {
          if (visibleSegSet.has(edgeSegs[i]!)) fEdgeIdx.push(i);
        }
        const frameEdgeMap = Int32Array.from(fEdgeIdx);
        this.frameEdgeMap = frameEdgeMap;
        const fEdge = gatherInto(full.frameEdge, frameEdgeMap, 6);
        if (fEdge.length) {
          const feg = new THREE.BufferGeometry();
          feg.setAttribute("position", new THREE.BufferAttribute(fEdge, 3));
          const edgeOpacity = this.activeTheme === "light" ? 0.72 : 0.45;
          const fem = new THREE.LineBasicMaterial({ color: source.edge, transparent: true, opacity: edgeOpacity });
          this.frameEdges = new THREE.LineSegments(feg, fem);
          this.content.add(this.frameEdges);
        }
      }
    }

    const lineSrc = s.visibility.showOffsets && full.clearLine ? full.clearLine : full.line;
    const line = gatherInto(lineSrc, drawnLineMap, 6);
    if (line.length) {
      const lineCol = cc
        ? this.subsetLineColours(info, drawn.lineMap, drawnLineMap, cc.line)
        : this.colours(source.legend, source.line, drawnLineMap, 2);
      this.lines = this.makeLines(line, lineCol, 3);
      this.content.add(this.lines);
    }

    if (fe && s.visibility.showOffsets) {
      const offsetColour = toHexNumber(themeTokens(this.activeTheme).warn);
      if (full.offsetLine) {
        const offLineSegs = this.blobs.i32(fe.offsetLineSegments);
        const offLineIdx: number[] = [];
        for (let i = 0; i < offLineSegs.length; i++) {
          if (visibleSegSet.has(offLineSegs[i]!)) offLineIdx.push(i);
        }
        const offsetLineMap = Int32Array.from(offLineIdx);
        this.offsetLineMap = offsetLineMap;
        const oLine = gatherInto(full.offsetLine, offsetLineMap, 6);
        if (oLine.length) {
          this.offsetLines = this.makeLines(oLine, null, 4, offsetColour);
          this.content.add(this.offsetLines);
        }
      }
      if (full.offsetJoint) {
        const offJointSegs = this.blobs.i32(fe.offsetJointSegments);
        const offJointIdx: number[] = [];
        for (let i = 0; i < offJointSegs.length; i++) {
          if (visibleSegSet.has(offJointSegs[i]!)) offJointIdx.push(i);
        }
        const offsetJointMap = Int32Array.from(offJointIdx);
        this.offsetJointMap = offsetJointMap;
        const oJoint = gatherInto(full.offsetJoint, offsetJointMap, 3);
        if (oJoint.length) {
          const jg = new THREE.BufferGeometry();
          jg.setAttribute("position", new THREE.BufferAttribute(oJoint, 3));
          const jm = new THREE.PointsMaterial({ color: offsetColour, size: 6, sizeAttenuation: false });
          this.offsetJoints = new THREE.Points(jg, jm);
          this.content.add(this.offsetJoints);
        }
      }
    }

    // Built only while wanted: off, the markers cost nothing.
    if (s.nodes.on) this.buildNodePoints(s, info, drawn, full);

    // V4d: the ghost of what is drawn, and the markers of its nodes with no displacement.
    this.clear(this.ghost);
    for (const o of ghostObjects(info, this.blobs, drawn, this.edgeMap, this.activeTheme, this.frameTriMap, this.frameEdgeMap, this.drawnLineMap, s.visibility.showOffsets)) this.ghost.add(o);
    this.ghost.visible = this.ghostWanted(s) && s.phase.at?.kind !== "geometry";
    this.setMarkers(info);
    this.applyVisibility(s, false);
    this.applyDim(selectedElementsOf(s).length > 0);
    this.requestRender();
  }

  /** Edges on or off, opacity, and the hidden set (a rebuild when it changed). */
  private applyVisibility(s: State, rebuild = true): void {
    const info = s.mesh;
    if (!info || !this.drawn) return;
    if (rebuild) {
      // Compare what would be drawn with what is drawn, element by element:
      // isolating one 22-element group after another changes no count.
      const want = visibleMaps(info.legend, s.visibility.hidden, this.blobs.i32(info.lineGroup), this.blobs.i32(info.triGroup), elementMaskOf(s, info, this.blobs));
      if (!sameDrawn(want, this.drawn)) {
        this.rebuild(s, info);
        return;
      }
    }
    if (this.edges) this.edges.visible = edgesShown(s.visibility.edges, this.full, s.visibility.opacity);
    if (this.frameEdges) this.frameEdges.visible = edgesShown(s.visibility.edges, this.full, s.visibility.opacity);
    if (this.ghost) applyGhostVisibility(this.ghost, s.visibility.edges, this.full, s.visibility.opacity);
    if (this.faces) {
      const opacity = s.visibility.opacity;
      this.faces.visible = opacity > 0;
      const m = this.faces.material as THREE.MeshStandardMaterial;
      m.transparent = opacity < 1;
      m.opacity = opacity;
      m.depthWrite = opacity >= 1;
      m.needsUpdate = true;
    }
    if (this.frameFaces) {
      const baseOpacity = s.visibility.opacity;
      const effOpacity = s.lineDiagram?.on ? Math.min(baseOpacity, 0.25) : baseOpacity;
      this.frameFaces.visible = effOpacity > 0;
      const m = this.frameFaces.material as THREE.MeshStandardMaterial;
      m.transparent = effOpacity < 1;
      m.opacity = effOpacity;
      m.depthWrite = effOpacity >= 1;
      m.needsUpdate = true;
    }
    this.requestRender();
  }

  /** Dim or restore the unselected model. */
  private applyDim(on: boolean): void {
    const k = on ? DIM : 1;
    if (this.faces) (this.faces.material as THREE.MeshStandardMaterial).color.setScalar(k);
    if (this.frameFaces) (this.frameFaces.material as THREE.MeshStandardMaterial).color.setScalar(k);
    if (this.lines) (this.lines.material as LineMaterial).color.setScalar(k);
    if (this.offsetLines) (this.offsetLines.material as LineMaterial).color.setScalar(k);
    if (this.offsetJoints) (this.offsetJoints.material as THREE.PointsMaterial).color.setScalar(k);
    const restOpacity = this.activeTheme === "light" ? 0.72 : 0.45;
    if (this.edges) (this.edges.material as THREE.LineBasicMaterial).opacity = on ? 0.2 : restOpacity;
    if (this.frameEdges) (this.frameEdges.material as THREE.LineBasicMaterial).opacity = on ? 0.2 : restOpacity;
    this.caps.setDim(k);
  }

  /**
   * Fit the camera to the model, or to `focus` (the selection's bounds) when
   * given. With no heading (on load): plan view for a planar model, else
   * isometric. `F` passes the current heading, so only the distance and the
   * target change.
   */
  frame(info: MeshInfo, heading: Heading | null = null, focus: Focus | null = null): void {
    const tri = this.blobs.f32(info.triPositions);
    this.fit((focus ?? info).center, (focus ?? info).radius, tri.length ? tri : this.blobs.f32(info.linePositions), heading);
  }

  /** Fit the camera to the geometry sibling, the same way (V2f). */
  frameGeometry(g: GeometryInfo, heading: Heading | null = null): void {
    const tri = this.blobs.f32(g.surfacePositions);
    this.fit(g.center, g.radius, tri.length ? tri : this.blobs.f32(g.curvePositions), heading);
  }

  /** Fit a bounding sphere; with no heading, plan view when `pos` is planar, else isometric. */
  private fit(center: readonly [number, number, number], r: number, pos: Float32Array, heading: Heading | null): void {
    const c = new THREE.Vector3(...center);
    if (!heading) {
      let z0 = Infinity, z1 = -Infinity;
      for (let i = 2; i < pos.length; i += 3) {
        z0 = Math.min(z0, pos[i]!);
        z1 = Math.max(z1, pos[i]!);
      }
      const planar = z1 - z0 <= 1e-6 * r;
      // Plan: looking down -Z with +Y up the screen. Isometric: from (1, -1.3, 0.9).
      heading = planar ? { yaw: 0, tilt: 0 } : headingOf(ISO_FORWARD.clone());
    }
    this.nav.setHeading(heading);
    // Fit the bounding sphere in the free area, vertically and horizontally.
    const dist = fitDistance(r, this.perspCamera.fov, (this.host.clientWidth || 1) - this.panelWidth(), this.host.clientHeight || 1);
    this.nav.orthoDistance = dist;
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(this.camera.quaternion);
    this.camera.position.copy(c).addScaledVector(back, dist);
    this.nav.target.copy(c);
    this.nav.updated();
  }

  /** Aim the raycaster through a client-space point. */
  private aim(clientX: number, clientY: number): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
  }

  /** The nearest drawn mesh point under a client-space point, or null on a miss. */
  pointAt(clientX: number, clientY: number): THREE.Vector3 | null {
    const targets: THREE.Object3D[] = [];
    if (this.lines && this.lines.visible) targets.push(this.lines);
    if (this.faces && this.faces.visible) targets.push(this.faces);
    if (this.frameFaces && this.frameFaces.visible) targets.push(this.frameFaces);
    if (!targets.length) return null;
    this.aim(clientX, clientY);
    return nearestHit(this.raycaster, targets);
  }

  /** The frame effect's target: frame `focus` (the selection's bounds), or the whole model, keeping the heading. */
  frameTo(focus: Focus | null): void {
    const s = this.store.get();
    const info = s.mesh;
    if (!info) return;
    // V4d: frame what is drawn: the selection where it is drawn, or the whole (deformed) model (renderer/deform.ts `frameFocus`).
    const idx = this.indexOf(info);
    const selected = selectedElementsOf(s).flatMap((d) => {
      const e = idx.byPath.get(d);
      const p = e === undefined ? undefined : idx.prims.get(e);
      return p ? [p] : [];
    });
    this.frame(info, this.nav.heading, frameFocus(this.full, info, this.blobs, this.drawn, selected, focus));
  }

  private setHighlight(s: State, pulse = true): void {
    this.clear(this.highlight);
    this.lineMaterials = this.lineMaterials.filter((m) => this.lines?.material === m || this.offsetLines?.material === m);
    const info = s.mesh;
    const selected = selectedElementsOf(s);
    this.applyDim(selected.length > 0);
    if (!info || selected.length === 0) {
      this.halo.release();
      return this.requestRender();
    }
    const idx = this.indexOf(info);
    const prims = selected.flatMap((path) => {
      const e = idx.byPath.get(path);
      const p = e === undefined ? undefined : idx.prims.get(e);
      return p ? [p] : [];
    });
    // V4d: the outline follows the drawn (possibly deformed) positions.
    for (const o of selectionOutline(this.full, info, this.blobs, prims, (pos, width) => this.makeLines(pos, null, width, SELECTION_COLOUR), s.visibility.showOffsets)) this.highlight.add(o);
    if (pulse) this.halo.pulse();
    this.requestRender();
  }

  /** The element under a client-space point, or null. Lines win over faces. */
  pickAt(clientX: number, clientY: number): Pick | null {
    const info = this.store.get().mesh;
    const drawn = this.drawn;
    if (!info || !drawn) return null;
    const lineElement = this.blobs.i32(info.lineElement), triElement = this.blobs.i32(info.triElement);
    const drawnForPick = { lineMap: this.drawnLineMap ?? drawn.lineMap, triMap: drawn.triMap };
    const frameTriElement = info.frameExtrusion ? this.blobs.i32(info.frameExtrusion.triElement) : null;
    this.aim(clientX, clientY);
    // The objects hold the drawn positions (deformed when the deformed shape shows, V4d).
    // V4e1: a hit the section planes cut away is a miss (C-PICK).
    const hit = hitElement(
      this.raycaster,
      this.lines,
      this.faces,
      drawnForPick,
      info.elements,
      lineElement,
      triElement,
      info.radius,
      (p) => this.clip.clipped(p, 1e-6 * info.radius),
      this.frameFaces,
      this.frameTriMap,
      frameTriElement,
    );
    if (!hit) return null;
    const decl = info.elements[hit.element];
    if (decl === undefined) throw new Error(`viewport: element index ${hit.element} is not in the mesh`);
    return this.pickOf(decl, [hit.point.x, hit.point.y, hit.point.z]);
  }

  /**
   * The pick of a declaration: the hit point and, for a cell, its node tags
   * read from the block's connectivity blob (the state holds the range, not
   * a copy per element).
   */
  pickOf(decl: DeclPath, at: readonly [number, number, number] | null): Pick {
    const s = this.store.get();
    const cell = s.decls[decl]?.element?.cell;
    if (!cell) return { decl, at, nodes: null };
    const block = s.blocks[cell.block];
    if (!block) throw new Error(`viewport: ${decl} names block ${cell.block}, which the state does not hold`);
    const conn = this.blobs.get(block.connectivity);
    return { decl, at, nodes: Array.from(conn.subarray(cell.row * block.npe, (cell.row + 1) * block.npe)) };
  }

  /** Client-space point at the middle of a declaration's first drawn primitive. */
  screenPointOf(decl: DeclPath): { x: number; y: number } | null {
    const s = this.store.get();
    const info = s.mesh;
    if (!info) return null;
    const idx = this.indexOf(info);
    const e = idx.byPath.get(decl);
    const prims = e === undefined ? undefined : idx.prims.get(e);
    if (!prims) return null;
    const p = new THREE.Vector3();
    const pos = drawnPositions(this.full, info, this.blobs);
    const linePos = s.visibility.showOffsets && pos.clearLine ? pos.clearLine : pos.line;
    if (prims.lines.length) {
      const a = linePos, li = prims.lines[0]!;
      p.set((a[6 * li]! + a[6 * li + 3]!) / 2, (a[6 * li + 1]! + a[6 * li + 4]!) / 2, (a[6 * li + 2]! + a[6 * li + 5]!) / 2);
    } else if (prims.tris.length) {
      const a = pos.tri, ti = prims.tris[0]!;
      p.set(
        (a[9 * ti]! + a[9 * ti + 3]! + a[9 * ti + 6]!) / 3,
        (a[9 * ti + 1]! + a[9 * ti + 4]! + a[9 * ti + 7]!) / 3,
        (a[9 * ti + 2]! + a[9 * ti + 5]! + a[9 * ti + 8]!) / 3,
      );
    } else return null;
    p.project(this.camera);
    const rect = this.renderer.domElement.getBoundingClientRect();
    return { x: rect.left + ((p.x + 1) / 2) * rect.width, y: rect.top + ((1 - p.y) / 2) * rect.height };
  }

  requestRender(): void {
    if (this.pending || this.continuous) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.renderNow();
    });
  }

  renderNow(): void {
    const t = performance.now();
    this.lineDiagram?.flushCameraChange();
    this.clip.beforeRender(this.scene);
    this.renderer.render(this.scene, this.camera);
    if (this.highlight.children.length) this.halo.render(this.scene, this.camera);
    this.lastRenderMs = performance.now() - t;
    if (this.halo.pulsing) this.requestRender();
  }

  /** V4e1: the box the offset slider maps, the reference (undeformed) mesh (renderer/clip.ts sliderBox); null with no mesh. */
  sliderBox(): Box | null {
    const info = this.store.get().mesh;
    return info ? sliderBox(this.full, info, this.blobs) : null;
  }

  /** V4e1: the unit direction the camera looks along (a plane with this normal keeps what lies beyond it). */
  viewNormal(): [number, number, number] {
    const d = this.camera.getWorldDirection(new THREE.Vector3());
    return [d.x, d.y, d.z];
  }

  /** V4e3: how many cut-face caps are drawn (and shown). */
  get capCount(): number {
    return this.caps.count;
  }

  /** V4e3: the caps show what the state asks for now (measure and capture wait on it). */
  capsSettled(): boolean {
    return this.caps.settledFor(this.store.get());
  }

  /** V4e2: how many plane gizmos are drawn (and shown). */
  get gizmoCount(): number {
    return this.gizmos.root.visible ? this.gizmos.count : 0;
  }

  /** V4e1: how many section planes cut now. */
  get clipCount(): number {
    return this.clip.count;
  }

  gpuName(): string {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
  }
}
