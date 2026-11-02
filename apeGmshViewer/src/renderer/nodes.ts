// The node markers: one filled circle per drawn node, in one draw call. A point
// costs 12 bytes and a single vertex; the circle is cut in the fragment shader.
// Positions are read from the drawn vertices, so a deformed step needs no
// displacement of its own: the markers copy where the mesh already moved to.

import * as THREE from "three";

/** One float offset per node into the drawn segment (`line`) and triangle (`tri`) positions. */
export interface NodeVertices {
  line: Int32Array;
  tri: Int32Array;
}

/**
 * The first drawn vertex of every distinct node of what is drawn. `lineNodes`
 * has 2 tags per segment and `triNodes` 3 per triangle; the maps hold the drawn
 * primitives' indices.
 */
export function uniqueNodeVertices(lineNodes: Float64Array, triNodes: Float64Array, lineMap: Int32Array, triMap: Int32Array): NodeVertices {
  const seen = new Set<number>();
  const line: number[] = [];
  const tri: number[] = [];
  for (let d = 0; d < lineMap.length; d++) {
    const seg = lineMap[d]!;
    for (let k = 0; k < 2; k++) {
      const id = lineNodes[seg * 2 + k]!;
      if (seen.has(id)) continue;
      seen.add(id);
      line.push(seg * 6 + k * 3);
    }
  }
  for (let d = 0; d < triMap.length; d++) {
    const t = triMap[d]!;
    for (let k = 0; k < 3; k++) {
      const id = triNodes[t * 3 + k]!;
      if (seen.has(id)) continue;
      seen.add(id);
      tri.push(t * 9 + k * 3);
    }
  }
  return { line: Int32Array.from(line), tri: Int32Array.from(tri) };
}

/** The markers' xyz from the drawn positions, into `out` (a new array when absent). */
export function nodePositions(v: NodeVertices, line: Float32Array, tri: Float32Array, out: Float32Array = new Float32Array((v.line.length + v.tri.length) * 3)): Float32Array {
  let o = 0;
  for (const [offsets, src] of [[v.line, line], [v.tri, tri]] as const) {
    for (let i = 0; i < offsets.length; i++) {
      const p = offsets[i]!;
      out[o++] = src[p]!;
      out[o++] = src[p + 1]!;
      out[o++] = src[p + 2]!;
    }
  }
  return out;
}

/** Under perspective, the share of its eye distance a marker is drawn nearer the eye, so it wins against the face it sits on. */
export const NODE_DEPTH_PULL = 5e-4;
/**
 * The least perspective pull in normalised depth, about 2 steps of a 24-bit depth buffer: far off, the share alone
 * rounds away. Being a fixed NDC offset it covers about d^2 * 2.5e-7 / (2 near) metres at distance d (0.5 m at
 * 100 m with near 5 mm), where the depth buffer barely resolves more anyway.
 */
export const NODE_DEPTH_FLOOR = 2.5e-7;
/** Under an orthographic camera, the fixed pull in normalised depth: there depth is linear, so a fixed offset is the same everywhere. */
export const NODE_DEPTH_ORTHO = 2e-4;

/** The markers' depth pull, injected after `project_vertex`. `nodeClipZ` is the same arithmetic, for the tests. */
export const NODE_DEPTH_GLSL = `
  if (isPerspectiveMatrix(projectionMatrix)) {
    vec4 nodePull = projectionMatrix * vec4(mvPosition.xyz * (1.0 - ${NODE_DEPTH_PULL}), 1.0);
    gl_Position.z = min(nodePull.z / nodePull.w, gl_Position.z / gl_Position.w - ${NODE_DEPTH_FLOOR}) * gl_Position.w;
  } else {
    gl_Position.z -= ${NODE_DEPTH_ORTHO} * gl_Position.w;
  }`;

/** The normalised depth a marker at view-space `mv` is drawn at under `projection` (column-major, as three.js stores it). */
export function nodeClipZ(projection: ArrayLike<number>, mv: readonly [number, number, number]): number {
  const p = projection;
  const clip = (x: number, y: number, z: number): [number, number] => [p[2]! * x + p[6]! * y + p[10]! * z + p[14]!, p[3]! * x + p[7]! * y + p[11]! * z + p[15]!];
  const [z, w] = clip(mv[0], mv[1], mv[2]);
  // three.js's isPerspectiveMatrix: the last row's third entry is -1 under perspective.
  if (p[11] === -1) {
    const k = 1 - NODE_DEPTH_PULL;
    const [pz, pw] = clip(mv[0] * k, mv[1] * k, mv[2] * k);
    return Math.min(pz / pw, z / w - NODE_DEPTH_FLOOR);
  }
  return (z - NODE_DEPTH_ORTHO * w) / w;
}

/** Thickness of the dark rim in pixels; at or under 3 px a marker is all fill, because a rim would swallow it. */
const RIM_PX = 1.25;
const rimAt = (size: number): number => (size > 3 ? 1 - (2 * RIM_PX) / size : 2);

export function makeNodePoints(positions: Float32Array, size: number, fill: number, rim: number): THREE.Points {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const uniforms = { nodeRim: { value: new THREE.Color(rim) }, nodeRimAt: { value: rimAt(size) } };
  const m = new THREE.PointsMaterial({ color: fill, size, sizeAttenuation: false });
  m.alphaToCoverage = true;
  m.userData["nodeUniforms"] = uniforms;
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    // A small depth pull toward the eye: a node on a face sits exactly at its depth and would flicker through
    // it. Under perspective it is a share of the eye distance, because a fixed offset in clip depth grows with
    // depth^2 / near and shows markers through walls once the camera is inside the model. Under an orthographic
    // camera depth is linear and the eye can sit inside the model (negative near), where scaling the view
    // position would push the marker away, so the pull there is a fixed offset.
    shader.vertexShader = shader.vertexShader.replace("#include <project_vertex>", `#include <project_vertex>${NODE_DEPTH_GLSL}`);
    shader.fragmentShader = shader.fragmentShader
      .replace("void main() {", "uniform vec3 nodeRim;\nuniform float nodeRimAt;\nvoid main() {")
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  float nodeR = length(gl_PointCoord - 0.5) * 2.0;
  float nodeW = fwidth(nodeR);
  diffuseColor.a = 1.0 - smoothstep(1.0 - nodeW, 1.0, nodeR);
  if (diffuseColor.a <= 0.0) discard;
  diffuseColor.rgb = mix(diffuseColor.rgb, nodeRim, smoothstep(nodeRimAt - nodeW, nodeRimAt, nodeR));`,
      );
  };
  const points = new THREE.Points(g, m);
  // The markers are drawn, not picked: a click keeps choosing the element under it.
  points.raycast = () => {};
  return points;
}

export function setNodeSize(points: THREE.Points, size: number): void {
  const m = points.material as THREE.PointsMaterial;
  m.size = size;
  (m.userData["nodeUniforms"] as { nodeRimAt: { value: number } }).nodeRimAt.value = rimAt(size);
}
