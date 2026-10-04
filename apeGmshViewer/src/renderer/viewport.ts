// The 3-D viewport: renders the mesh from the state and the BlobStore, and
// turns a click into a `select` event. Its three.js objects are a
// derivation of the state keyed by the mesh's BlobRef ids and the
// visibility and selection records; it holds no model fact the store does
// not.
//
// Selection (R1 on #1283): the selected element is drawn as a thick outline
// in the selection colour over everything (no depth test), its faces filled
// in that colour, and the rest of the model is dimmed. That reads at
// full-model zoom for a beam and for a shell alike.

import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { BlobStore } from "../state/blobs.ts";
import { geometryPairing } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { DeclPath, GeometryInfo, MeshInfo, Pick } from "../state/types.ts";
import { headingOf, Navigator, nearestHit, type Heading } from "./navigation.ts";
import { edgeMask, sameDrawn, visibleMaps, type Drawn } from "./visible.ts";

/** Inspector width plus its margins (style.css #inspector). */
const INSPECTOR_PX = 470 + 28;

export const SELECTION_COLOUR = 0xffd400;
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

export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  readonly nav: Navigator;
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
  private faces: THREE.Mesh | null = null;
  private edges: THREE.LineSegments | null = null;
  private lineMaterials: LineMaterial[] = [];
  private index: MeshIndex | null = null;
  private drawn: Drawn | null = null;
  private pending = false;
  private readonly observer: ResizeObserver;
  private readonly unsubscribe: () => void;
  /** CPU time of the last renderer.render call (ms). */
  lastRenderMs = 0;
  continuous = false;

  constructor(host: HTMLElement, store: Store, blobs: BlobStore) {
    this.host = host;
    this.store = store;
    this.blobs = blobs;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x000000, 0);
    host.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
    // Z up, always: the navigator's heading has no roll, so Z stays vertical on screen.
    this.camera.up.set(0, 0, 1);
    this.scene.add(this.camera);
    this.scene.add(new THREE.HemisphereLight(0xdfe6f0, 0x30343c, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(0.5, 0.8, 1);
    this.camera.add(key);
    this.scene.add(this.content, this.highlight, this.geometryLayer);

    this.raycaster.params.Line2 = { threshold: 7 };
    this.nav = new Navigator({
      camera: this.camera,
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
      select: (x, y) => {
        const pick = this.pickAt(x, y);
        this.store.dispatch(pick ? { type: "select", pick } : { type: "clearSelection" });
      },
      fit: () => {
        const s = this.store.get();
        // On the geometry phase (or with no mesh), F fits the geometry.
        if (this.geometryInfo && (s.phase.at?.kind === "geometry" || !s.mesh)) this.frameGeometry(this.geometryInfo, this.nav.heading);
        else if (s.mesh) this.frame(s.mesh, this.nav.heading);
      },
      changed: () => this.requestRender(),
    });
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    this.unsubscribe = store.subscribe((s, prev) => this.onState(s, prev));
    this.onState(store.get(), null);
  }

  /** Release the listeners, the observer, the subscription and the GPU objects. */
  dispose(): void {
    this.unsubscribe();
    this.observer.disconnect();
    this.nav.dispose();
    this.clear(this.content);
    this.clear(this.highlight);
    this.clear(this.geometryLayer);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private onState(s: State, prev: State | null): void {
    this.onGeometry(s, prev);
    if (!prev || s.mesh !== prev.mesh) {
      this.setMesh(s.mesh);
      this.applyVisibility(s);
      this.setHighlight(s);
      return;
    }
    if (s.visibility !== prev.visibility) this.applyVisibility(s);
    if (s.selection !== prev.selection) this.setHighlight(s);
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
    // A geometry opened alone is framed when it first shows.
    if (show && !wasShown && !s.mesh && this.geometryInfo) this.frameGeometry(this.geometryInfo);
    if (!prev || s.geometry !== prev.geometry || s.phase !== prev.phase || s.artifacts !== prev.artifacts) this.requestRender();
  }

  private setGeometry(g: GeometryInfo | null): void {
    this.clear(this.geometryLayer);
    this.geometryLayer.visible = false;
    this.geometryInfo = g;
    if (!g) return;
    const curves = this.blobs.f32(g.curvePositions);
    if (curves.length) {
      const cg = new THREE.BufferGeometry();
      cg.setAttribute("position", new THREE.BufferAttribute(curves, 3));
      this.geometryLayer.add(new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: 0xd8dde6 })));
    }
    const tris = this.blobs.f32(g.surfacePositions);
    if (tris.length) {
      const sg = new THREE.BufferGeometry();
      sg.setAttribute("position", new THREE.BufferAttribute(tris, 3));
      sg.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({
        color: 0x7d8796, side: THREE.DoubleSide, flatShading: true, transparent: true, opacity: 0.35, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      });
      this.geometryLayer.add(new THREE.Mesh(sg, m));
    }
    const pts = this.blobs.f32(g.pointPositions);
    if (pts.length) {
      const pg = new THREE.BufferGeometry();
      pg.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      this.geometryLayer.add(new THREE.Points(pg, new THREE.PointsMaterial({ color: 0xffffff, size: 5, sizeAttenuation: false })));
    }
  }

  /** Width kept clear for the inspector: the model is framed left of it. */
  private panelWidth(): number {
    const w = this.host.clientWidth || 1;
    return w > 1000 ? INSPECTOR_PX : 0;
  }

  private resize(): void {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h);
    // A virtual frame P px wider, viewed from x = P: its centre lands at (w - P) / 2.
    const p = this.panelWidth();
    this.camera.aspect = (w + p) / h;
    this.camera.setViewOffset(w + p, h, p, 0, w, h);
    this.camera.updateProjectionMatrix();
    for (const m of this.lineMaterials) m.resolution.set(w, h);
    this.requestRender();
  }

  private makeLines(pos: Float32Array, col: Float32Array | null, width: number, colour?: number): LineSegments2 {
    const g = new LineSegmentsGeometry();
    g.setPositions(pos);
    if (col) g.setColors(col);
    const m = new LineMaterial({
      linewidth: width,
      vertexColors: col !== null,
      color: colour ?? 0xffffff,
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

  /** Vertex colours of the drawn primitives from the legend (R2) and the per-primitive group index. */
  private colours(info: MeshInfo, group: Int32Array, map: Int32Array, perPrim: number): Float32Array {
    const out = new Float32Array(map.length * perPrim * 3);
    for (let d = 0; d < map.length; d++) {
      const c = info.legend[group[map[d]!]!]?.color;
      if (!c) throw new Error(`viewport: primitive ${map[d]} has no legend row`);
      for (let v = 0; v < perPrim; v++) out.set(c, (d * perPrim + v) * 3);
    }
    return out;
  }

  /** Which original primitives the hidden set leaves visible (visible.ts). */
  private visible(s: State, info: MeshInfo): Drawn {
    const { lineMap, triMap } = visibleMaps(info.legend, s.visibility.hidden, this.blobs.i32(info.lineGroup), this.blobs.i32(info.triGroup));
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
    this.faces = null;
    this.edges = null;
    this.drawn = null;
    this.index = null;
    if (!info) return this.requestRender();
    this.indexOf(info);
    this.rebuild(this.store.get(), info);
    this.frame(info);
  }

  /** (Re)build the content group for the current hidden set. */
  private rebuild(s: State, info: MeshInfo): void {
    this.clear(this.content);
    this.lineMaterials = this.lineMaterials.filter((m) => this.highlight.children.some((o) => (o as THREE.Mesh).material === m));
    this.lines = null;
    this.faces = null;
    this.edges = null;
    const drawn = this.visible(s, info);
    this.drawn = drawn;

    const tri = Viewport.gather(this.blobs.f32(info.triPositions), drawn.triMap, 9);
    if (tri.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(tri, 3));
      g.setAttribute("color", new THREE.BufferAttribute(this.colours(info, this.blobs.i32(info.triGroup), drawn.triMap, 3), 3));
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
      const edge = Viewport.gather(ep, edgeMap, 6);
      if (edge.length) {
        const eg = new THREE.BufferGeometry();
        eg.setAttribute("position", new THREE.BufferAttribute(edge, 3));
        const em = new THREE.LineBasicMaterial({ color: 0x0b0c0f, transparent: true, opacity: 0.45 });
        this.edges = new THREE.LineSegments(eg, em);
        this.content.add(this.edges);
      }
    }
    const line = Viewport.gather(this.blobs.f32(info.linePositions), drawn.lineMap, 6);
    if (line.length) {
      this.lines = this.makeLines(line, this.colours(info, this.blobs.i32(info.lineGroup), drawn.lineMap, 2), 3);
      this.content.add(this.lines);
    }
    this.applyVisibility(s, false);
    this.applyDim(s.selection.decls.length > 0);
    this.requestRender();
  }

  /** Edges on or off, opacity, and the hidden set (a rebuild when it changed). */
  private applyVisibility(s: State, rebuild = true): void {
    const info = s.mesh;
    if (!info || !this.drawn) return;
    if (rebuild) {
      // Compare what would be drawn with what is drawn, element by element:
      // isolating one 22-element group after another changes no count.
      const want = visibleMaps(info.legend, s.visibility.hidden, this.blobs.i32(info.lineGroup), this.blobs.i32(info.triGroup));
      if (!sameDrawn(want, this.drawn)) {
        this.rebuild(s, info);
        return;
      }
    }
    if (this.edges) this.edges.visible = s.visibility.edges;
    if (this.faces) {
      const m = this.faces.material as THREE.MeshStandardMaterial;
      // Line elements inside solids (embedded rebar) would be hidden: see through the solids.
      const auto = info.counts.solidCells > 0 && info.linePositions.shape[0]! > 0 ? 0.35 : 1;
      const opacity = Math.min(auto, s.visibility.opacity);
      m.transparent = opacity < 1;
      m.opacity = opacity;
      m.depthWrite = opacity >= 1;
      m.needsUpdate = true;
    }
    this.requestRender();
  }

  /** Dim or restore the unselected model. */
  private applyDim(on: boolean): void {
    const k = on ? DIM : 1;
    if (this.faces) (this.faces.material as THREE.MeshStandardMaterial).color.setScalar(k);
    if (this.lines) (this.lines.material as LineMaterial).color.setScalar(k);
    if (this.edges) (this.edges.material as THREE.LineBasicMaterial).opacity = on ? 0.2 : 0.45;
  }

  /**
   * Fit the camera to the model. With no heading (on load): plan view for a
   * planar model, else isometric. `F` passes the current heading, so only the
   * distance and the target change.
   */
  frame(info: MeshInfo, heading: Heading | null = null): void {
    const tri = this.blobs.f32(info.triPositions);
    this.fit(info.center, info.radius, tri.length ? tri : this.blobs.f32(info.linePositions), heading);
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
      heading = planar ? { yaw: 0, tilt: 0 } : headingOf(new THREE.Vector3(-1.0, 1.3, -0.9));
    }
    this.nav.setHeading(heading);
    // Fit the bounding sphere in the free area, vertically and horizontally.
    const half = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const free = Math.max(1, (this.host.clientWidth || 1) - this.panelWidth());
    const halfH = Math.atan(Math.tan(half) * (free / (this.host.clientHeight || 1)));
    const dist = (r / Math.sin(Math.min(half, halfH))) * 1.08;
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
    if (this.lines) targets.push(this.lines);
    if (this.faces) targets.push(this.faces);
    if (!targets.length) return null;
    this.aim(clientX, clientY);
    return nearestHit(this.raycaster, targets);
  }

  /** The outline of an element's triangles: the edges that belong to one of them only. */
  private static outline(tp: Float32Array, tris: number[]): Float32Array {
    const count = new Map<string, [number, number, number, number, number, number]>();
    const seen = new Map<string, number>();
    for (const t of tris) {
      for (let k = 0; k < 3; k++) {
        const a = 9 * t + 3 * k, b = 9 * t + 3 * ((k + 1) % 3);
        const pa = [tp[a]!, tp[a + 1]!, tp[a + 2]!], pb = [tp[b]!, tp[b + 1]!, tp[b + 2]!];
        const ka = pa.join(","), kb = pb.join(",");
        const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
        seen.set(key, (seen.get(key) ?? 0) + 1);
        count.set(key, [...pa, ...pb] as [number, number, number, number, number, number]);
      }
    }
    const out: number[] = [];
    for (const [key, n] of seen) if (n === 1) out.push(...count.get(key)!);
    return new Float32Array(out);
  }

  private setHighlight(s: State): void {
    this.clear(this.highlight);
    this.lineMaterials = this.lineMaterials.filter((m) => this.lines?.material === m);
    const info = s.mesh;
    const selected = s.selection.decls;
    this.applyDim(selected.length > 0);
    if (!info || selected.length === 0) return this.requestRender();
    const idx = this.indexOf(info);
    const lp = this.blobs.f32(info.linePositions), tp = this.blobs.f32(info.triPositions);
    const seg: number[] = [], tri: number[] = [], outline: number[] = [];
    for (const path of selected) {
      const e = idx.byPath.get(path);
      const prims = e === undefined ? undefined : idx.prims.get(e);
      if (!prims) continue;
      for (const i of prims.lines) for (let k = 0; k < 6; k++) seg.push(lp[6 * i + k]!);
      for (const i of prims.tris) for (let k = 0; k < 9; k++) tri.push(tp[9 * i + k]!);
      outline.push(...Viewport.outline(tp, prims.tris));
    }
    const thick = (pos: Float32Array, width: number) => {
      const hl = this.makeLines(pos, null, width, SELECTION_COLOUR);
      hl.renderOrder = 10;
      (hl.material as LineMaterial).depthTest = false;
      return hl;
    };
    if (seg.length) this.highlight.add(thick(new Float32Array(seg), 9));
    if (outline.length) this.highlight.add(thick(new Float32Array(outline), 6));
    if (tri.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tri), 3));
      const m = new THREE.MeshBasicMaterial({
        color: SELECTION_COLOUR, side: THREE.DoubleSide, transparent: true, opacity: 0.6, depthTest: false,
      });
      const fill = new THREE.Mesh(g, m);
      fill.renderOrder = 9;
      this.highlight.add(fill);
    }
    this.requestRender();
  }

  /** The element under a client-space point, or null. Lines win over faces. */
  pickAt(clientX: number, clientY: number): Pick | null {
    const info = this.store.get().mesh;
    const drawn = this.drawn;
    if (!info || !drawn) return null;
    const lineElement = this.blobs.i32(info.lineElement), triElement = this.blobs.i32(info.triElement);
    this.aim(clientX, clientY);
    const pick = (e: number, p: THREE.Vector3): Pick => {
      const decl = info.elements[e];
      if (decl === undefined) throw new Error(`viewport: element index ${e} is not in the mesh`);
      return this.pickOf(decl, [p.x, p.y, p.z]);
    };
    if (this.lines) {
      const hits = this.raycaster
        .intersectObject(this.lines, false)
        .filter((h) => h.faceIndex !== undefined && h.faceIndex !== null);
      if (hits.length) {
        // An OpenSees-only element (no cell) drawn on the same nodes as a cell
        // would be unreachable behind it; among coincident hits, prefer it.
        const near = hits[0]!.distance + 1e-6 * info.radius;
        const coincident = hits.filter((h) => h.distance <= near);
        const ops = coincident.find((h) => info.elements[lineElement[drawn.lineMap[h.faceIndex!]!]!]!.startsWith("opensees/element/"));
        const h = ops ?? coincident[0]!;
        return pick(lineElement[drawn.lineMap[h.faceIndex!]!]!, h.point);
      }
    }
    if (this.faces) {
      const hit = this.raycaster.intersectObject(this.faces, false)[0];
      if (hit && hit.faceIndex !== undefined && hit.faceIndex !== null) return pick(triElement[drawn.triMap[hit.faceIndex]!]!, hit.point);
    }
    return null;
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
    const info = this.store.get().mesh;
    if (!info) return null;
    const idx = this.indexOf(info);
    const e = idx.byPath.get(decl);
    const prims = e === undefined ? undefined : idx.prims.get(e);
    if (!prims) return null;
    const p = new THREE.Vector3();
    if (prims.lines.length) {
      const a = this.blobs.f32(info.linePositions), li = prims.lines[0]!;
      p.set((a[6 * li]! + a[6 * li + 3]!) / 2, (a[6 * li + 1]! + a[6 * li + 4]!) / 2, (a[6 * li + 2]! + a[6 * li + 5]!) / 2);
    } else if (prims.tris.length) {
      const a = this.blobs.f32(info.triPositions), ti = prims.tris[0]!;
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
    this.renderer.render(this.scene, this.camera);
    this.lastRenderMs = performance.now() - t;
  }

  gpuName(): string {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
  }
}
