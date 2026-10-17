// Revit-style ViewCube and camera projection switcher (Perspective / Orthographic).
//
// Geometry: a unit cube [-1, 1]^3 with Z up. Each of its 6 faces is tiled into
// a 3x3 grid (inner square [-INNER, INNER]^2, 4 edge strips, 4 corner squares).
// Every sub-quad's outward sign triple s = n + i*u + j*v in {-1, 0, 1}^3 \ {0}
// names one of the 26 ViewCube zones:
//   - |sx| + |sy| + |sz| = 1: 6 faces   (top, bottom, front, right, back, left)
//   - |sx| + |sy| + |sz| = 2: 12 edges  (e.g. top-front, front-right)
//   - |sx| + |sy| + |sz| = 3: 8 corners (e.g. top-front-right)
// Adjacent faces share edge and corner zone ids, so hovering an edge or corner
// highlights the meeting strips across all visible faces at once.

import * as THREE from "three";
import { el } from "../ui/dom.ts";
import { CLICK_SLOP_PX } from "./bindings.ts";
import {
  headingOf,
  ISO_FORWARD,
  orientationOf,
  PROJECTION_MODES,
  type Heading,
  type ProjectionMode,
} from "./navigation.ts";

export type Sign = -1 | 0 | 1;
export type SignTriple = readonly [Sign, Sign, Sign];
export type ZoneKind = "face" | "edge" | "corner";
export type FaceId = "top" | "bottom" | "front" | "right" | "back" | "left";

/** Half-width of the inner face square on [-1, 1]^2; the border (1 - INNER) is the edge/corner bevel band. */
export const INNER = 0.56;
/** Half-size of the cube in SVG viewBox [-100, 100]^2 units. */
export const CUBE_SCALE = 48;

export interface CubeFace {
  readonly id: FaceId;
  readonly label: string;
  readonly normal: SignTriple;
  /** In-plane right axis when looking straight at the face. */
  readonly u: SignTriple;
  /** In-plane up axis when looking straight at the face (`u x v = normal`). */
  readonly v: SignTriple;
}

export const CUBE_FACES: readonly CubeFace[] = [
  { id: "top", label: "TOP", normal: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
  { id: "bottom", label: "BOTTOM", normal: [0, 0, -1], u: [1, 0, 0], v: [0, -1, 0] },
  { id: "front", label: "FRONT", normal: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  { id: "right", label: "RIGHT", normal: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] },
  { id: "back", label: "BACK", normal: [0, 1, 0], u: [-1, 0, 0], v: [0, 0, 1] },
  { id: "left", label: "LEFT", normal: [-1, 0, 0], u: [0, -1, 0], v: [0, 0, 1] },
];

export interface ViewCubeZone {
  readonly id: string;
  readonly kind: ZoneKind;
  readonly sign: SignTriple;
}

/** Canonical zone id from a non-zero sign triple, ordered z (top/bottom), y (front/back), x (left/right). */
export function zoneIdOf(s: SignTriple): string {
  const parts: string[] = [];
  if (s[2] === 1) parts.push("top");
  else if (s[2] === -1) parts.push("bottom");
  if (s[1] === -1) parts.push("front");
  else if (s[1] === 1) parts.push("back");
  if (s[0] === -1) parts.push("left");
  else if (s[0] === 1) parts.push("right");
  if (parts.length === 0) throw new RangeError("viewcube: zero sign triple is not a surface zone");
  return parts.join("-");
}

/** All 26 surface zones of the ViewCube (6 faces, 12 edges, 8 corners). */
export const VIEWCUBE_ZONES: readonly ViewCubeZone[] = (() => {
  const out: ViewCubeZone[] = [];
  const signs: readonly Sign[] = [-1, 0, 1];
  for (const sz of signs) {
    for (const sy of signs) {
      for (const sx of signs) {
        const norm1 = Math.abs(sx) + Math.abs(sy) + Math.abs(sz);
        if (norm1 === 0) continue;
        const sign: SignTriple = [sx, sy, sz];
        const kind: ZoneKind = norm1 === 1 ? "face" : norm1 === 2 ? "edge" : "corner";
        out.push({ id: zoneIdOf(sign), kind, sign });
      }
    }
  }
  return out;
})();

const ZONE_BY_ID: ReadonlyMap<string, ViewCubeZone> = new Map(VIEWCUBE_ZONES.map((z) => [z.id, z]));

export function zoneById(id: string): ViewCubeZone | undefined {
  return ZONE_BY_ID.get(id);
}

/** Snap `yaw` to the nearest 90-degree cardinal step around world Z. */
export function snapCardinalYaw(yaw: number): number {
  const step = Math.PI / 2;
  const k = Math.round(yaw / step);
  const snapped = k * step;
  return Object.is(snapped, -0) ? 0 : snapped;
}

/**
 * Target camera heading when clicking `zone`.
 * For the 24 non-polar zones, the camera stands at `zone.sign` and looks at the
 * origin along `-zone.sign`. For `top` and `bottom`, `yaw` snaps to the nearest
 * 90-degree step so plan views stay axis-aligned with the previous view.
 */
export function headingForZone(zone: ViewCubeZone, currentYaw = 0): Heading {
  const [sx, sy, sz] = zone.sign;
  if (sx === 0 && sy === 0) {
    return { yaw: snapCardinalYaw(currentYaw), tilt: sz > 0 ? 0 : Math.PI };
  }
  return headingOf(new THREE.Vector3(-sx, -sy, -sz));
}

export interface ProjectedPatch {
  readonly zoneId: string;
  readonly kind: ZoneKind;
  readonly points: readonly (readonly [number, number])[];
}

export interface ProjectedFace {
  readonly face: CubeFace;
  /** Outward normal component toward the camera in (0, 1]. */
  readonly facing: number;
  /** Outer 4 corners of the face in SVG coordinates. */
  readonly outline: readonly (readonly [number, number])[];
  /** The 9 interactive sub-quads (1 face center, 4 edges, 4 corners). */
  readonly patches: readonly ProjectedPatch[];
  /** SVG 2D affine matrix `matrix(a, b, c, d, e, f)` that maps face-local text onto the projected face. */
  readonly labelMatrix: readonly [number, number, number, number, number, number];
}

const BANDS: readonly (readonly [Sign, number, number])[] = [
  [-1, -1, -INNER],
  [0, -INNER, INNER],
  [1, INNER, 1],
];

/**
 * Project the visible faces of the ViewCube for camera `heading` into SVG
 * coordinates (x right, y down, origin at cube centre, scale `scale`).
 * Because a cube is convex and projected orthographically, front-face culling
 * (`facing > 1e-6`) is exact and visible faces never overlap.
 */
export function projectCube(heading: Heading, scale = CUBE_SCALE): ProjectedFace[] {
  const invQ = orientationOf(heading).invert();
  const toCam = (x: number, y: number, z: number): THREE.Vector3 =>
    new THREE.Vector3(x, y, z).applyQuaternion(invQ);
  const pt2 = (v: THREE.Vector3): [number, number] => [v.x * scale, -v.y * scale];

  const visible: ProjectedFace[] = [];
  for (const face of CUBE_FACES) {
    const nc = toCam(face.normal[0], face.normal[1], face.normal[2]);
    if (nc.z <= 1e-6) continue;
    const uc = toCam(face.u[0], face.u[1], face.u[2]);
    const vc = toCam(face.v[0], face.v[1], face.v[2]);

    const corner = (u: number, v: number): [number, number] =>
      pt2(
        new THREE.Vector3(
          face.normal[0] + u * face.u[0] + v * face.v[0],
          face.normal[1] + u * face.u[1] + v * face.v[1],
          face.normal[2] + u * face.u[2] + v * face.v[2],
        ).applyQuaternion(invQ),
      );

    const outline: [number, number][] = [
      corner(-1, -1),
      corner(1, -1),
      corner(1, 1),
      corner(-1, 1),
    ];

    const patches: ProjectedPatch[] = [];
    for (const [iu, u0, u1] of BANDS) {
      for (const [iv, v0, v1] of BANDS) {
        const sign: SignTriple = [
          (face.normal[0] + iu * face.u[0] + iv * face.v[0]) as Sign,
          (face.normal[1] + iu * face.u[1] + iv * face.v[1]) as Sign,
          (face.normal[2] + iu * face.u[2] + iv * face.v[2]) as Sign,
        ];
        const id = zoneIdOf(sign);
        const norm1 = Math.abs(sign[0]) + Math.abs(sign[1]) + Math.abs(sign[2]);
        const kind: ZoneKind = norm1 === 1 ? "face" : norm1 === 2 ? "edge" : "corner";
        patches.push({
          zoneId: id,
          kind,
          points: [corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)],
        });
      }
    }

    // In SVG (y down), face-local (+u right, +v up) maps to (+u_cam.x, -u_cam.y) and (-v_cam.x, v_cam.y).
    const [cx, cy] = pt2(nc);
    const labelMatrix: [number, number, number, number, number, number] = [
      uc.x,
      -uc.y,
      -vc.x,
      vc.y,
      cx,
      cy,
    ];

    visible.push({ face, facing: nc.z, outline, patches, labelMatrix });
  }
  visible.sort((a, b) => a.facing - b.facing);
  return visible;
}

export interface ViewCubeHost {
  visible(): boolean;
  heading(): Heading;
  projection(): ProjectionMode;
  setProjection(mode: ProjectionMode): void;
  snapTo(heading: Heading): void;
  orbit(dYaw: number, dTilt: number): void;
  home(): void;
  subscribe(fn: () => void): () => void;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string> = {}): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

const ptsAttr = (pts: readonly (readonly [number, number])[]): string =>
  pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");

/**
 * Mount the Revit-style ViewCube and its Perspective / Orthographic dropdown
 * into `#viewcube`. Returns the cleanup function.
 */
export function mountViewCube(host: ViewCubeHost): () => void {
  const root = document.getElementById("viewcube");
  if (!root) return () => {};
  root.replaceChildren();

  const homeBtn = el("button", "vc-btn", "\u2302");
  homeBtn.type = "button";
  homeBtn.title = "Home isometric view";
  const ccwBtn = el("button", "vc-btn", "\u21b6");
  ccwBtn.type = "button";
  ccwBtn.title = "Rotate 90\u00b0 counter-clockwise around Z";
  const cwBtn = el("button", "vc-btn", "\u21b7");
  cwBtn.type = "button";
  cwBtn.title = "Rotate 90\u00b0 clockwise around Z";
  const bar = el("div", "vc-bar");
  bar.append(homeBtn, ccwBtn, cwBtn);

  const svg = svgEl("svg", {
    class: "vc-svg",
    viewBox: "-100 -100 200 200",
    role: "img",
    "aria-label": "ViewCube navigation",
  });

  const optPersp = el("option", undefined, "Perspective");
  optPersp.value = "perspective";
  const optOrtho = el("option", undefined, "Orthographic");
  optOrtho.value = "orthographic";
  const select = el("select", "vc-projection");
  select.title = "Camera projection";
  select.append(optPersp, optOrtho);

  root.append(bar, svg, select);

  let hoveredZone: string | null = null;
  let drag: { id: number; x0: number; y0: number; x: number; y: number; zoneId: string | null } | null = null;

  const syncHover = () => {
    for (const poly of svg.querySelectorAll<SVGPolygonElement>("polygon.vc-patch")) {
      poly.classList.toggle("hovered", hoveredZone !== null && poly.dataset["zone"] === hoveredZone);
    }
  };

  const render = () => {
    const show = host.visible();
    root.hidden = !show;
    if (!show) return;
    select.value = host.projection();
    const h = host.heading();
    const faces = projectCube(h);
    svg.replaceChildren();

    // Compass ring under the cube in the z = -1 plane.
    const invQ = orientationOf(h).invert();
    const ringPts: [number, number][] = [];
    const RING_R = 1.44;
    for (let k = 0; k < 36; k++) {
      const a = (k / 36) * Math.PI * 2;
      const v = new THREE.Vector3(RING_R * Math.cos(a), RING_R * Math.sin(a), -1).applyQuaternion(invQ);
      ringPts.push([v.x * CUBE_SCALE, -v.y * CUBE_SCALE]);
    }
    svg.appendChild(svgEl("polygon", { class: "vc-ring", points: ptsAttr(ringPts) }));
    for (const [label, dx, dy] of [["N", 0, 1], ["E", 1, 0], ["S", 0, -1], ["W", -1, 0]] as const) {
      const v = new THREE.Vector3(dx * 1.65, dy * 1.65, -1).applyQuaternion(invQ);
      const t = svgEl("text", {
        class: "vc-compass",
        x: (v.x * CUBE_SCALE).toFixed(2),
        y: (-v.y * CUBE_SCALE).toFixed(2),
        "text-anchor": "middle",
        "dominant-baseline": "central",
      });
      t.textContent = label;
      svg.appendChild(t);
    }

    for (const pf of faces) {
      const g = svgEl("g", { class: `vc-face vc-face-${pf.face.id}` });
      g.appendChild(svgEl("polygon", { class: "vc-face-bg", points: ptsAttr(pf.outline) }));

      for (const patch of pf.patches) {
        const poly = svgEl("polygon", {
          class: `vc-patch vc-patch-${patch.kind}${hoveredZone === patch.zoneId ? " hovered" : ""}`,
          points: ptsAttr(patch.points),
          "data-zone": patch.zoneId,
        });
        poly.addEventListener("pointerenter", () => {
          if (drag) return;
          hoveredZone = patch.zoneId;
          syncHover();
        });
        g.appendChild(poly);
      }

      g.appendChild(svgEl("polygon", { class: "vc-face-outline", points: ptsAttr(pf.outline) }));

      if (pf.facing > 0.18) {
        const [a, b, c, d, e, f] = pf.labelMatrix;
        const label = svgEl("text", {
          class: "vc-label",
          transform: `matrix(${a.toFixed(4)} ${b.toFixed(4)} ${c.toFixed(4)} ${d.toFixed(4)} ${e.toFixed(2)} ${f.toFixed(2)})`,
          "text-anchor": "middle",
          "dominant-baseline": "central",
        });
        label.textContent = pf.face.label;
        g.appendChild(label);
      }

      svg.appendChild(g);
    }
  };

  svg.addEventListener("pointerleave", () => {
    if (drag) return;
    hoveredZone = null;
    syncHover();
  });

  svg.addEventListener("pointerdown", (e: PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return;
    e.preventDefault();
    const target = e.target as Element | null;
    const zoneId = target?.closest?.("[data-zone]")?.getAttribute("data-zone") ?? hoveredZone;
    drag = { id: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, zoneId };
    svg.setPointerCapture(e.pointerId);
  });

  svg.addEventListener("pointermove", (e: PointerEvent) => {
    if (!drag || e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    if (dx === 0 && dy === 0) return;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > CLICK_SLOP_PX) {
      hoveredZone = null;
      host.orbit((-Math.PI * dx) / 140, (-Math.PI * dy) / 140);
    }
  });

  const endDrag = (e: PointerEvent, clickAllowed: boolean) => {
    if (!drag || e.pointerId !== drag.id) return;
    const d = drag;
    drag = null;
    if (svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId);
    if (clickAllowed && d.zoneId && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) <= CLICK_SLOP_PX) {
      const zone = zoneById(d.zoneId);
      if (zone) host.snapTo(headingForZone(zone, host.heading().yaw));
    }
  };

  svg.addEventListener("pointerup", (e: PointerEvent) => endDrag(e, true));
  svg.addEventListener("pointercancel", (e: PointerEvent) => endDrag(e, false));
  svg.addEventListener("lostpointercapture", (e: PointerEvent) => endDrag(e, false));
  svg.addEventListener("contextmenu", (e: Event) => e.preventDefault());

  homeBtn.addEventListener("click", () => host.home());
  ccwBtn.addEventListener("click", () => {
    const h = host.heading();
    host.snapTo({ yaw: h.yaw - Math.PI / 2, tilt: h.tilt });
  });
  cwBtn.addEventListener("click", () => {
    const h = host.heading();
    host.snapTo({ yaw: h.yaw + Math.PI / 2, tilt: h.tilt });
  });
  select.addEventListener("change", () => {
    const v = select.value as ProjectionMode;
    if ((PROJECTION_MODES as readonly string[]).includes(v)) host.setProjection(v);
  });

  const unsub = host.subscribe(render);
  render();
  return () => {
    unsub();
    root.replaceChildren();
    root.hidden = true;
  };
}

/** Default 3-D isometric heading (`ISO_FORWARD`). */
export function homeHeading(): Heading {
  return headingOf(ISO_FORWARD.clone());
}
