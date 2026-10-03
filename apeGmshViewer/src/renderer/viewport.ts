// The 3-D viewport: renders the mesh from the store and turns a click
// into a `select` event. It holds three.js objects, never model data.

import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { ElementRef } from "../chain/resolve.ts";
import type { MeshBuffers } from "../mesh/build.ts";
import type { State, Store } from "../state/store.ts";
import { headingOf, Navigator, nearestHit, type Heading } from "./navigation.ts";

/** Inspector width plus its margins (style.css #inspector). */
const INSPECTOR_PX = 470 + 28;

export const sameRef =(a: ElementRef, b: ElementRef): boolean =>
  a.kind === b.kind &&
  a.row === b.row &&
  (a.kind === "fem" ? a.blockIndex === (b as typeof a).blockIndex : a.metaIndex === (b as typeof a).metaIndex);

export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  readonly nav: Navigator;
  private readonly scene = new THREE.Scene();
  private readonly raycaster = new THREE.Raycaster();
  private readonly host: HTMLElement;
  private readonly store: Store;
  private content = new THREE.Group();
  private highlight = new THREE.Group();
  private lines: LineSegments2 | null = null;
  private faces: THREE.Mesh | null = null;
  private lineMaterials: LineMaterial[] = [];
  private pending = false;
  /** CPU time of the last renderer.render call (ms). */
  lastRenderMs = 0;
  continuous = false;

  constructor(host: HTMLElement, store: Store) {
    this.host = host;
    this.store = store;
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
    this.scene.add(this.content, this.highlight);

    this.raycaster.params.Line2 = { threshold: 7 };
    this.nav = new Navigator({
      camera: this.camera,
      element: this.renderer.domElement,
      bounds: () => {
        const m = this.store.get().mesh;
        return m ? { center: new THREE.Vector3(...m.center), radius: m.radius } : null;
      },
      hitAt: (x, y) => this.pointAt(x, y),
      rayAt: (x, y) => {
        this.aim(x, y);
        return this.raycaster.ray.clone();
      },
      select: (x, y) => {
        const ref = this.pickAt(x, y);
        this.store.dispatch(ref ? { type: "select", ref } : { type: "clear-selection" });
      },
      fit: () => {
        const m = this.store.get().mesh;
        if (m) this.frame(m, this.nav.heading);
      },
      changed: () => this.requestRender(),
    });
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    store.subscribe((s, prev) => this.onState(s, prev));
  }

  private onState(s: State, prev: State): void {
    if (s.mesh !== prev.mesh) this.setMesh(s.mesh);
    if (s.selection !== prev.selection) this.setHighlight(s.selection, s.mesh);
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

  private setMesh(mesh: MeshBuffers | null): void {
    this.clear(this.content);
    this.clear(this.highlight);
    this.lineMaterials = [];
    this.lines = null;
    this.faces = null;
    if (!mesh) return this.requestRender();

    if (mesh.triPositions.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(mesh.triPositions, 3));
      g.setAttribute("color", new THREE.BufferAttribute(mesh.triColors, 3));
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
      // Line elements inside solids (embedded rebar) would be hidden: see through the solids.
      if (mesh.counts.solidCells > 0 && mesh.linePositions.length > 0) {
        m.transparent = true;
        m.opacity = 0.35;
        m.depthWrite = false;
      }
      this.faces = new THREE.Mesh(g, m);
      this.content.add(this.faces);
      if (mesh.edgePositions.length) {
        const eg = new THREE.BufferGeometry();
        eg.setAttribute("position", new THREE.BufferAttribute(mesh.edgePositions, 3));
        const em = new THREE.LineBasicMaterial({ color: 0x0b0c0f, transparent: true, opacity: 0.45 });
        this.content.add(new THREE.LineSegments(eg, em));
      }
    }
    if (mesh.linePositions.length) {
      this.lines = this.makeLines(mesh.linePositions, mesh.lineColors, 3);
      this.content.add(this.lines);
    }
    this.frame(mesh);
  }

  /**
   * Fit the camera to the model. With no heading (on load): plan view for a
   * planar model, else isometric. `F` passes the current heading, so only the
   * distance and the target change.
   */
  frame(mesh: MeshBuffers, heading: Heading | null = null): void {
    const c = new THREE.Vector3(...mesh.center);
    const r = mesh.radius;
    if (!heading) {
      const pos = mesh.triPositions.length ? mesh.triPositions : mesh.linePositions;
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

  private setHighlight(sel: ElementRef | null, mesh: MeshBuffers | null): void {
    this.clear(this.highlight);
    this.lineMaterials = this.lineMaterials.filter((m) => this.lines?.material === m);
    if (!sel || !mesh) return this.requestRender();
    const seg: number[] = [];
    mesh.lineRefs.forEach((r, i) => {
      if (sameRef(r, sel)) for (let k = 0; k < 6; k++) seg.push(mesh.linePositions[6 * i + k]!);
    });
    if (seg.length) {
      const hl = this.makeLines(new Float32Array(seg), null, 7, 0xf5c542);
      hl.renderOrder = 10;
      (hl.material as LineMaterial).depthTest = false;
      this.highlight.add(hl);
    }
    const tri: number[] = [];
    mesh.triRefs.forEach((r, i) => {
      if (sameRef(r, sel)) for (let k = 0; k < 9; k++) tri.push(mesh.triPositions[9 * i + k]!);
    });
    if (tri.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tri), 3));
      const m = new THREE.MeshBasicMaterial({
        color: 0xf5c542, side: THREE.DoubleSide, transparent: true, opacity: 0.85,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
      });
      this.highlight.add(new THREE.Mesh(g, m));
    }
    this.requestRender();
  }

  /** The element under a client-space point, or null. Lines win over faces. */
  pickAt(clientX: number, clientY: number): ElementRef | null {
    const mesh = this.store.get().mesh;
    if (!mesh) return null;
    this.aim(clientX, clientY);
    if (this.lines) {
      const hits = this.raycaster
        .intersectObject(this.lines, false)
        .filter((h) => h.faceIndex !== undefined && h.faceIndex !== null);
      if (hits.length) {
        // An OpenSees-only element (no cell) drawn on the same nodes as a cell
        // would be unreachable behind it; among coincident hits, prefer it.
        const near = hits[0]!.distance + 1e-6 * mesh.radius;
        const coincident = hits.filter((h) => h.distance <= near).map((h) => mesh.lineRefs[h.faceIndex!]!);
        return coincident.find((r) => r.kind === "ops") ?? coincident[0] ?? null;
      }
    }
    if (this.faces) {
      const hit = this.raycaster.intersectObject(this.faces, false)[0];
      if (hit && hit.faceIndex !== undefined && hit.faceIndex !== null) return mesh.triRefs[hit.faceIndex] ?? null;
    }
    return null;
  }

  /** Client-space point at the middle of a ref's first drawn primitive. */
  screenPointOf(ref: ElementRef): { x: number; y: number } | null {
    const mesh = this.store.get().mesh;
    if (!mesh) return null;
    const p = new THREE.Vector3();
    const li = mesh.lineRefs.findIndex((r) => sameRef(r, ref));
    if (li >= 0) {
      const a = mesh.linePositions;
      p.set((a[6 * li]! + a[6 * li + 3]!) / 2, (a[6 * li + 1]! + a[6 * li + 4]!) / 2, (a[6 * li + 2]! + a[6 * li + 5]!) / 2);
    } else {
      const ti = mesh.triRefs.findIndex((r) => sameRef(r, ref));
      if (ti < 0) return null;
      const a = mesh.triPositions;
      p.set(
        (a[9 * ti]! + a[9 * ti + 3]! + a[9 * ti + 6]!) / 3,
        (a[9 * ti + 1]! + a[9 * ti + 4]! + a[9 * ti + 7]!) / 3,
        (a[9 * ti + 2]! + a[9 * ti + 5]! + a[9 * ti + 8]!) / 3,
      );
    }
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
