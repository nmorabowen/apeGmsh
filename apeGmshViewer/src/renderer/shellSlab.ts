// The thickness slab of the drawn shell and membrane faces: each face
// triangle with a section thickness t is drawn as a top and a bottom
// triangle at +-t/2 along the surface normal, closed by side walls where the
// surface ends (a free edge, a junction with a panel at an angle, or a change
// of thickness). Pure: no three.js and no DOM, so the tests build it.
//
// Cheap by construction: the topology (which drawn corner each slab vertex
// copies, and its signed offset) is built once per rebuild, only while the
// extrude toggle is on; the positions are recomputed from the drawn face
// positions, so the slab follows the deformed shape (a scrub writes the
// faces in place, then `slabPositions` moves the slab with them) and its
// colours are the faces' own (each slab vertex copies its corner's ramp
// coordinate or vertex colour).
//
// The offset direction at a corner is the mean of the unit normals of the
// extruded triangles at that node that lie within SMOOTH_DEG of this
// triangle's plane, signed to agree with it. Two neighbours in one plane (or
// on a gently curved shell) so share their offset and meet without a gap,
// whichever way each one winds; a panel meeting another at an angle keeps its
// own normal and gets a wall at the junction.

/** Neighbouring triangles within this angle of each other share their corner normals. */
export const SMOOTH_DEG = 30;
const COS_SMOOTH = Math.cos((SMOOTH_DEG * Math.PI) / 180);

export interface SlabTopology {
  /** per slab vertex: the drawn face corner (3 * drawn triangle + k) it is offset from */
  readonly src: Int32Array;
  /** per slab vertex: its signed offset along the corner normal (+t/2 top, -t/2 bottom) */
  readonly off: Float32Array;
  /** per slab triangle: the drawn face triangle it belongs to */
  readonly tri: Int32Array;
  /** per slab outline vertex: the slab vertex it sits on (2 per segment: the top and bottom rims of each wall) */
  readonly edge: Int32Array;
  /** per drawn corner: its node, as a dense index */
  readonly cornerNode: Int32Array;
  /** CSR from dense node index to the extruded drawn triangles at that node */
  readonly nodeStart: Int32Array;
  readonly nodeTris: Int32Array;
  /** per drawn triangle: 1 when it is extruded */
  readonly extruded: Uint8Array;
  /** how many drawn triangles are extruded, and how many wall quads close them */
  readonly faces: number;
  readonly walls: number;
}

/** The unit normal of drawn triangle `t` (zeros when it is degenerate). */
function triNormal(pos: ArrayLike<number>, t: number, out: Float64Array, o = 0): void {
  const b = 9 * t;
  const ux = pos[b + 3]! - pos[b]!, uy = pos[b + 4]! - pos[b + 1]!, uz = pos[b + 5]! - pos[b + 2]!;
  const vx = pos[b + 6]! - pos[b]!, vy = pos[b + 7]! - pos[b + 1]!, vz = pos[b + 8]! - pos[b + 2]!;
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz);
  if (len > 0) {
    out[o] = nx / len;
    out[o + 1] = ny / len;
    out[o + 2] = nz / len;
  } else out[o] = out[o + 1] = out[o + 2] = 0;
}

function allNormals(pos: ArrayLike<number>, nTri: number): Float64Array {
  const n = new Float64Array(3 * nTri);
  for (let t = 0; t < nTri; t++) triNormal(pos, t, n, 3 * t);
  return n;
}

const dot = (n: Float64Array, a: number, b: number): number => n[3 * a]! * n[3 * b]! + n[3 * a + 1]! * n[3 * b + 1]! + n[3 * a + 2]! * n[3 * b + 2]!;

/**
 * The slab topology of the drawn faces. `pos` holds the drawn face positions
 * (9 floats per triangle), `cornerTags` the node tag at each drawn corner
 * (3 per triangle) and `thickness` each drawn triangle's thickness (0: not
 * extruded). Null when no drawn triangle has a thickness.
 */
export function slabTopology(pos: ArrayLike<number>, cornerTags: ArrayLike<number>, thickness: ArrayLike<number>): SlabTopology | null {
  const nTri = thickness.length;
  if (cornerTags.length !== 3 * nTri || pos.length !== 9 * nTri) {
    throw new RangeError(`slabTopology: ${pos.length / 9} triangles of positions, ${cornerTags.length / 3} of node tags, ${nTri} thicknesses`);
  }
  const extruded = new Uint8Array(nTri);
  let faces = 0;
  for (let t = 0; t < nTri; t++) {
    const th = thickness[t]!;
    if (th > 0 && Number.isFinite(th)) {
      extruded[t] = 1;
      faces++;
    }
  }
  if (!faces) return null;

  // Dense node indices and the node -> extruded triangles CSR.
  const nodeOf = new Map<number, number>();
  const cornerNode = new Int32Array(3 * nTri).fill(-1);
  for (let c = 0; c < 3 * nTri; c++) {
    if (!extruded[(c / 3) | 0]) continue;
    const tag = cornerTags[c]!;
    let k = nodeOf.get(tag);
    if (k === undefined) nodeOf.set(tag, (k = nodeOf.size));
    cornerNode[c] = k;
  }
  const nNode = nodeOf.size;
  const nodeStart = new Int32Array(nNode + 1);
  for (let c = 0; c < 3 * nTri; c++) if (cornerNode[c]! >= 0) nodeStart[cornerNode[c]! + 1]!++;
  for (let k = 0; k < nNode; k++) nodeStart[k + 1]! += nodeStart[k]!;
  const nodeTris = new Int32Array(nodeStart[nNode]!);
  const fill = nodeStart.slice(0, nNode);
  for (let c = 0; c < 3 * nTri; c++) if (cornerNode[c]! >= 0) nodeTris[fill[cornerNode[c]!]!++] = (c / 3) | 0;

  // A wall closes edge (a, b) of triangle t unless another extruded triangle
  // on that edge lies in t's plane (within SMOOTH_DEG) with the same thickness.
  const normals = allNormals(pos, nTri);
  const wallEdges: number[] = []; // (t, k): the edge from corner k to corner k+1 of t
  for (let t = 0; t < nTri; t++) {
    if (!extruded[t]) continue;
    for (let k = 0; k < 3; k++) {
      const a = cornerNode[3 * t + k]!, b = cornerNode[3 * t + ((k + 1) % 3)]!;
      let joined = false;
      for (let i = nodeStart[a]!; i < nodeStart[a + 1]! && !joined; i++) {
        const u = nodeTris[i]!;
        if (u === t) continue;
        let hasB = false;
        for (let j = 0; j < 3; j++) if (cornerNode[3 * u + j] === b) hasB = true;
        if (!hasB) continue;
        const sameT = Math.abs(thickness[u]! - thickness[t]!) <= 1e-6 * Math.max(thickness[u]!, thickness[t]!);
        joined = sameT && Math.abs(dot(normals, t, u)) >= COS_SMOOTH;
      }
      if (!joined) wallEdges.push(t, k);
    }
  }
  const walls = wallEdges.length / 2;

  const nV = 6 * faces + 6 * walls;
  const src = new Int32Array(nV);
  const off = new Float32Array(nV);
  const tri = new Int32Array(nV / 3);
  const edge = new Int32Array(4 * walls);
  let v = 0;
  const put = (corner: number, o: number): number => {
    src[v] = corner;
    off[v] = o;
    return v++;
  };
  for (let t = 0; t < nTri; t++) {
    if (!extruded[t]) continue;
    const h = thickness[t]! / 2;
    const c = 3 * t;
    tri[v / 3] = t;
    put(c, h), put(c + 1, h), put(c + 2, h);
    tri[v / 3] = t;
    put(c, -h), put(c + 2, -h), put(c + 1, -h);
  }
  for (let w = 0; w < walls; w++) {
    const t = wallEdges[2 * w]!, k = wallEdges[2 * w + 1]!;
    const h = thickness[t]! / 2;
    const ca = 3 * t + k, cb = 3 * t + ((k + 1) % 3);
    // Two triangles per wall (the geometry is not indexed, so each has its own vertices).
    tri[v / 3] = t;
    const aTop = put(ca, h), bBot = put(cb, -h), bTop = put(cb, h);
    tri[v / 3] = t;
    put(ca, h);
    const aBot = put(ca, -h);
    put(cb, -h);
    edge.set([aTop, bTop, aBot, bBot], 4 * w);
  }
  return { src, off, tri, edge, cornerNode, nodeStart, nodeTris, extruded, faces, walls };
}

/**
 * The offset direction at every extruded drawn corner (3 floats per corner;
 * zeros elsewhere): the mean of the unit normals of the extruded triangles at
 * its node within SMOOTH_DEG of its own triangle, each signed to agree with it.
 */
export function cornerNormals(topo: SlabTopology, pos: ArrayLike<number>): Float32Array {
  const nTri = topo.extruded.length;
  const normals = allNormals(pos, nTri);
  const out = new Float32Array(9 * nTri);
  for (let c = 0; c < 3 * nTri; c++) {
    const t = (c / 3) | 0;
    if (!topo.extruded[t]) continue;
    const node = topo.cornerNode[c]!;
    let x = 0, y = 0, z = 0;
    for (let i = topo.nodeStart[node]!; i < topo.nodeStart[node + 1]!; i++) {
      const u = topo.nodeTris[i]!;
      const d = dot(normals, t, u);
      if (Math.abs(d) < COS_SMOOTH) continue;
      const sgn = d < 0 ? -1 : 1;
      x += sgn * normals[3 * u]!;
      y += sgn * normals[3 * u + 1]!;
      z += sgn * normals[3 * u + 2]!;
    }
    const len = Math.hypot(x, y, z);
    if (len > 0) {
      out[3 * c] = x / len;
      out[3 * c + 1] = y / len;
      out[3 * c + 2] = z / len;
    }
  }
  return out;
}

/** The slab's vertex positions (3 floats per slab vertex) from the drawn face positions, into `out` when given. */
export function slabPositions(topo: SlabTopology, pos: ArrayLike<number>, out: Float32Array = new Float32Array(3 * topo.src.length)): Float32Array {
  if (out.length !== 3 * topo.src.length) throw new RangeError(`slabPositions: ${out.length} floats for ${topo.src.length} slab vertices`);
  const n = cornerNormals(topo, pos);
  for (let v = 0; v < topo.src.length; v++) {
    const c = topo.src[v]!, o = topo.off[v]!;
    out[3 * v] = pos[3 * c]! + o * n[3 * c]!;
    out[3 * v + 1] = pos[3 * c + 1]! + o * n[3 * c + 1]!;
    out[3 * v + 2] = pos[3 * c + 2]! + o * n[3 * c + 2]!;
  }
  return out;
}

/** The slab rim lines (6 floats per segment) from the slab positions. */
export function slabEdgePositions(topo: SlabTopology, slab: ArrayLike<number>, out: Float32Array = new Float32Array(3 * topo.edge.length)): Float32Array {
  for (let i = 0; i < topo.edge.length; i++) {
    const v = topo.edge[i]!;
    out[3 * i] = slab[3 * v]!;
    out[3 * i + 1] = slab[3 * v + 1]!;
    out[3 * i + 2] = slab[3 * v + 2]!;
  }
  return out;
}

/** A per-face-corner attribute (`size` floats per drawn corner: a ramp coordinate, a colour) copied onto the slab vertices. */
export function slabAttribute(topo: SlabTopology, perCorner: ArrayLike<number>, size: number): Float32Array {
  const out = new Float32Array(size * topo.src.length);
  for (let v = 0; v < topo.src.length; v++) {
    const c = topo.src[v]!;
    for (let k = 0; k < size; k++) out[size * v + k] = perCorner[size * c + k]!;
  }
  return out;
}
