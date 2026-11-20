// Layer results of a `.ladruno`'s layered-shell buckets: `section.fiber.<token>` on an
// ASDShellQ4 / ASDShellT3 whose LayeredShell section reaches each layer's material
// (crackPattern, damage, stress, strain). reader/ladruno.ts indexes a stage; this module
// turns each such `RESULTS/ON_ELEMENTS/<result>/<classTag>-<ClassName>[g:h]` bucket into a
// `LayerGroupInfo` and reads it back as (E, G, layers, components), element-major.
// The layout is the fork's: `Ladruno_implementation/ladruno_schema_v1.md` §6
// (MODEL/SECTION_ASSIGNMENTS) and §7.2 (COLUMN_MAP).
//
// Rules this file keeps:
// - a bucket is a layer bucket when its element group is a surface (GP_PARAM has 2
//   columns), every COLUMN_MAP row has a GAUSS_ID >= 0 and no FIBER_ID, and a Gauss point
//   owns more than one block or a block is repeated (MULTIPLICITY > 1). The recorder writes
//   one block run per Gauss point: a run of layers that answer the token (NUM_COMP > 0),
//   then a run that does not (rebar: NUM_COMP 0, an empty COMP_NAMES line), and so on, so
//   MULTIPLICITY is a count of section layers. A Gauss bucket (one block per Gauss point) is
//   never one; reader/ladrunoElements.ts reads it;
// - the responding layers are the blocks with NUM_COMP > 0, expanded by MULTIPLICITY, in
//   section order (the first is on the -e3 side). Every Gauss point must have the same
//   block run and every responding block the same labels; a bucket that does not is
//   refused by name, never read in part;
// - each responding layer is matched to its section layer by position along the run
//   (the run's MULTIPLICITY must sum to the section's layer count), and its through-
//   thickness coordinate and thickness are FIBER_DATA's columns 1 and 2 (schema §6: y, z,
//   area | thickness). An element whose Gauss points carry different sections, or with no
//   assignment, or a run that does not cover its section, leaves the group's geometry null
//   with the reason in `geometry`, never a guessed position. The values still read;
// - components are the file's own (`<result>/<label>`, ADR 0122 D7), as for Gauss groups;
// - a bucket that disagrees with itself (step axes, element rows, NUM_COLUMNS, a Gauss
//   point count that is not GP_PARAM's) is refused by name in `refused`, and never read.
//
// Browser-safe: no Node import, and no import of reader/ladruno.ts (it imports this).

import { intOf, ints, modelGroupFor, oneValue, sameTime, type ECtors, type EDataset, type EFile, type EGroup, type ElementsFail } from "./ladrunoElements.ts";
import type { LayerGroupInfo } from "./results.ts";

const H5T_INTEGER = 0;
const H5T_FLOAT = 1;

interface LayerBucket {
  info: LayerGroupInfo;
  dataPath: string;
  ids: Float64Array;
  /** DATA column of (Gauss point g, responding layer k, component c): `cols[(g * L + k) * C + c]`. */
  cols: Int32Array;
  labels: string[];
}

/** One block of a COLUMN_MAP, as the layer read needs it. */
interface Block {
  gauss: number;
  mult: number;
  names: string[];
  colStart: number;
}

/**
 * Whether a COLUMN_MAP's rows are a layer run (see the header): a Gauss point owning more
 * than one block, or a block repeated. `gauss` and `mult` are its GAUSS_ID and MULTIPLICITY rows.
 */
export function isLayerRun(gauss: readonly number[], mult: readonly number[] | null): boolean {
  return new Set(gauss).size !== gauss.length || (mult !== null && mult.some((m) => m > 1));
}

export class LadrunoLayers {
  private readonly h5: ECtors;
  private readonly file: EFile;
  private readonly fail: ElementsFail;
  private readonly buckets = new Map<string, Map<string, LayerBucket>>();

  constructor(h5: ECtors, file: EFile, fail: ElementsFail) {
    this.h5 = h5;
    this.file = file;
    this.fail = fail;
  }

  /**
   * Index one stage's layer buckets: the groups (read or refused), and the
   * `<result>/<bucket>` keys this module took, which the Gauss index then leaves alone.
   */
  indexStage(stageId: string, onElements: EGroup, model: EGroup | null, time: Float64Array, warnings: string[]): { layers: LayerGroupInfo[]; claimed: Set<string> } {
    const byId = new Map<string, LayerBucket>();
    const claimed = new Set<string>();
    const modelElements = model ? this.groupOf(model, "ELEMENTS") : null;
    let sections: SectionIndex | null | undefined;
    for (const result of onElements.keys()) {
      if (result === "LOCAL_AXES") continue;
      const rg = this.groupOf(onElements, result);
      if (!rg) continue;
      for (const key of rg.keys()) {
        const bg = this.groupOf(rg, key);
        if (!bg || !this.isLayerBucket(key, bg, modelElements)) continue;
        if (sections === undefined) sections = model ? this.sectionIndex(model, warnings) : null;
        const b = this.bucket(result, key, bg, modelElements, sections, time);
        byId.set(b.info.id, b);
        claimed.add(`${result}/${key}`);
      }
    }
    this.buckets.set(stageId, byId);
    return { layers: [...byId.values()].map((b) => b.info), claimed };
  }

  /**
   * Steps `k0` to `k1` (exclusive) of `components` of one layer group, at the responding
   * layers `layers` (all when null): ((k1 - k0) * E * G * nLayers * nComponents,), step-major,
   * then element, Gauss point, layer (in the order asked) and component (in the order asked).
   */
  read(stage: { id: string; nSteps: number }, groupId: string, components: readonly string[], layers: readonly number[] | null, k0: number, k1: number): Float64Array {
    if (!Number.isInteger(k0) || !Number.isInteger(k1) || k0 < 0 || k1 > stage.nSteps || k0 >= k1) {
      throw this.fail("argument", `steps ${k0}..${k1} are not a range inside stage ${stage.id} (0..${stage.nSteps})`);
    }
    const b = this.bucketOf(stage.id, groupId);
    if (b.info.refused.length) throw this.fail("argument", `layer group ${groupId} is refused: ${b.info.refused[0]!.reason}`);
    const { nGauss: G, nLayers: L, nElements: E } = b.info;
    const C = b.labels.length;
    const comp = components.map((name) => {
      const c = b.info.components.indexOf(name);
      if (c < 0) throw this.fail("argument", `layer group ${groupId} has no component ${JSON.stringify(name)}; it has ${b.info.components.join(", ")}`);
      return c;
    });
    const lay = layers ?? Array.from({ length: L }, (_, k) => k);
    for (const k of lay) if (!Number.isInteger(k) || k < 0 || k >= L) throw this.fail("argument", `layer group ${groupId} has layers 0..${L - 1}, not ${k}`);
    const ds = this.file.get(b.dataPath);
    if (!(ds instanceof this.h5.Dataset)) throw this.fail("layout", `${b.dataPath} is not a dataset`);
    const v = (ds as unknown as EDataset).slice([[k0, k1], [], []]);
    if (!(v instanceof Float64Array || v instanceof Float32Array)) throw this.fail("dtype", `${b.dataPath}: read returned ${Object.prototype.toString.call(v)}, not float32 or float64`);
    const width = v.length / ((k1 - k0) * E);
    const out = new Float64Array((k1 - k0) * E * G * lay.length * comp.length);
    let o = 0;
    for (let t = 0; t < k1 - k0; t++) {
      for (let e = 0; e < E; e++) {
        const row = (t * E + e) * width;
        for (let g = 0; g < G; g++) {
          for (const k of lay) {
            for (const c of comp) out[o++] = v[row + b.cols[(g * L + k) * C + c]!]!;
          }
        }
      }
    }
    return out;
  }

  /** A layer group's element rows (the bucket's ID), in read order. */
  elementRows(stageId: string, groupId: string): Float64Array {
    return this.bucketOf(stageId, groupId).ids;
  }

  private bucketOf(stageId: string, groupId: string): LayerBucket {
    const b = this.buckets.get(stageId)?.get(groupId);
    if (!b) throw this.fail("argument", `stage ${stageId} has no layer group ${JSON.stringify(groupId)}`);
    return b;
  }

  private isLayerBucket(key: string, bg: EGroup, modelElements: EGroup | null): boolean {
    const cm = this.groupOf(bg, "COLUMN_MAP");
    const gidDs = cm ? this.datasetOf(cm, "GAUSS_ID") : null;
    if (!cm || !gidDs || gidDs.metadata.type !== H5T_INTEGER) return false;
    const gids = ints(gidDs);
    const fiber = this.intsOf(cm, "FIBER_ID");
    if (gids.length === 0 || gids.some((g) => g < 0) || fiber?.some((x) => x >= 0)) return false;
    if (!isLayerRun(gids, this.intsOf(cm, "MULTIPLICITY"))) return false;
    const model = modelGroupFor(modelElements, key, (n) => this.groupOf(modelElements!, n));
    const quad = model ? this.groupOf(model, "QUADRATURE") : null;
    const gp = quad ? this.datasetOf(quad, "GP_PARAM") : null;
    return gp?.shape?.length === 2 && gp.shape[1] === 2;
  }

  /** A layer bucket, read or refused. */
  private bucket(result: string, key: string, bg: EGroup, modelElements: EGroup | null, sections: SectionIndex | null, time: Float64Array): LayerBucket {
    const id = `${result}/${key}`;
    const model = modelGroupFor(modelElements, key, (n) => this.groupOf(modelElements!, n))!;
    const gp = this.datasetOf(this.groupOf(model, "QUADRATURE")!, "GP_PARAM")!;
    const G = gp.shape![0]!;
    const cm = this.groupOf(bg, "COLUMN_MAP")!;
    const gids = ints(this.datasetOf(cm, "GAUSS_ID")!);
    const mult = this.intsOf(cm, "MULTIPLICITY") ?? gids.map(() => 1);
    const ids = this.elementIdsIn(bg) ?? new Float64Array(0);
    const info: LayerGroupInfo = {
      id,
      path: bg.path,
      classTag: Number(/^(\d+)-/.exec(key)?.[1] ?? -1),
      intRule: intOf(oneValue(model.attrs["INTEGRATION_RULE"]?.value)) ?? -1,
      nElements: ids.length,
      nGauss: G,
      nLayers: 0,
      components: [],
      refused: [],
      naturalCoords: null,
      geometry: { state: "absent", reason: "not read" },
    };
    const refuse = (reason: string): LayerBucket => {
      info.refused = [{ component: result, reason }];
      return { info, dataPath: `${bg.path}/DATA`, ids, cols: new Int32Array(0), labels: [] };
    };

    // The datasets, and the step and element axes (a partial write is never read short).
    const ds = ["ID", "DATA", "TIME", "STEP"].map((k) => this.datasetOf(bg, k));
    const missing = ["ID", "DATA", "TIME", "STEP"].filter((_, i) => !ds[i]);
    if (missing.length) return refuse(`${bg.path} has no dataset ${missing.join(", ")}`);
    const [idDs, dataDs, timeDs, stepDs] = ds as EDataset[];
    if (!this.elementIdsIn(bg)) return refuse(`${idDs!.path} is not an (n, 1) integer dataset`);
    const shape = dataDs!.shape ?? [];
    if (shape.length !== 3 || dataDs!.metadata.type !== H5T_FLOAT) return refuse(`${dataDs!.path} is not a 3-D floating-point dataset (T, elements, columns)`);
    const [T, nE, width] = shape as [number, number, number];
    const nTime = timeDs!.shape?.length === 1 ? timeDs!.shape[0]! : -1;
    const nStep = stepDs!.shape?.length === 1 ? stepDs!.shape[0]! : -1;
    if (T !== nTime || T !== nStep) return refuse(`${dataDs!.path} holds ${T} steps, TIME ${nTime} and STEP ${nStep}: the step axes disagree (a recorder stopped mid-step)`);
    if (nE !== ids.length) return refuse(`${dataDs!.path} has ${nE} element rows and ${idDs!.path} ${ids.length} ids (a partial write)`);
    if (timeDs!.metadata.type !== H5T_FLOAT || !sameTime(timeDs!.value, time)) return refuse(`${timeDs!.path} is not the stage's time (${time.length} steps)`);

    // The column map against itself and DATA (schema §7.2's invariant).
    // An empty COMP_NAMES line is a run of layers that do not answer (NUM_COMP 0), not a malformed label.
    const names = compNameLines(cm.attrs["COMP_NAMES"]?.value, gids.length);
    if (typeof names === "string") return refuse(`${cm.path}: ${names}`);
    return this.withBlocks(info, refuse, result, bg, cm, gids, mult, names, width, G, model, sections, ids, gp);
  }

  private withBlocks(
    info: LayerGroupInfo,
    refuse: (reason: string) => LayerBucket,
    result: string,
    bg: EGroup,
    cm: EGroup,
    gids: number[],
    mult: number[],
    names: string[][],
    width: number,
    G: number,
    model: EGroup,
    sections: SectionIndex | null,
    ids: Float64Array,
    gp: EDataset,
  ): LayerBucket {
    const numComp = this.intsOf(cm, "NUM_COMP");
    if (!numComp || numComp.length !== gids.length) return refuse(`${cm.path}/NUM_COMP is not an integer dataset with one row per GAUSS_ID row (${gids.length})`);
    const bad = numComp.findIndex((n, i) => n !== names[i]!.length);
    if (bad >= 0) return refuse(`${cm.path}: NUM_COMP[${bad}] = ${numComp[bad]} but COMP_NAMES line ${bad} names ${names[bad]!.length} components`);
    if (mult.length !== gids.length || mult.some((m) => !Number.isInteger(m) || m < 1)) return refuse(`${cm.path}/MULTIPLICITY is not a positive integer per GAUSS_ID row (${gids.length})`);
    const sum = numComp.reduce((a, n, i) => a + mult[i]! * n, 0);
    const numColumns = intOf(oneValue(bg.attrs["NUM_COLUMNS"]?.value));
    if (numColumns === null) return refuse(`${bg.path} has no integer NUM_COLUMNS attribute`);
    if (numColumns !== sum || numColumns !== width) return refuse(`${bg.path}@NUM_COLUMNS = ${numColumns}, the column map sums MULTIPLICITY * NUM_COMP to ${sum}, and DATA has ${width} columns: they must agree`);

    // One block run per Gauss point, each GP_PARAM row once, every run alike.
    const blocks: Block[] = [];
    let col = 0;
    gids.forEach((g, i) => {
      blocks.push({ gauss: g, mult: mult[i]!, names: names[i]!, colStart: col });
      col += mult[i]! * names[i]!.length;
    });
    const runs = Array.from({ length: G }, () => [] as Block[]);
    for (const b of blocks) {
      if (b.gauss >= G) return refuse(`${cm.path}/GAUSS_ID holds ${b.gauss}, past the ${G} rows of ${gp.path}`);
      runs[b.gauss]!.push(b);
    }
    const empty = runs.findIndex((r) => r.length === 0);
    if (empty >= 0) return refuse(`${cm.path}/GAUSS_ID never names Gauss point ${empty} of the ${G} rows of ${gp.path}`);
    const shapeOf = (r: Block[]) => r.map((b) => `${b.mult}x[${b.names.join(",")}]`).join(" ");
    const odd = runs.findIndex((r) => shapeOf(r) !== shapeOf(runs[0]!));
    if (odd >= 0) return refuse(`${cm.path}: Gauss point ${odd} has the layer run ${shapeOf(runs[odd]!)}, Gauss point 0 ${shapeOf(runs[0]!)}; every Gauss point must have the same one`);
    const answering = runs[0]!.filter((b) => b.names.length > 0);
    if (answering.length === 0) return refuse(`${cm.path}: no layer answers ${result} (every block has NUM_COMP 0)`);
    const labels = answering[0]!.names;
    if (answering.some((b) => b.names.join(",") !== labels.join(","))) return refuse(`${cm.path}@COMP_NAMES: the layers name different components (${answering.map((b) => b.names.join(",")).join(" | ")})`);

    const L = answering.reduce((a, b) => a + b.mult, 0);
    const C = labels.length;
    const cols = new Int32Array(G * L * C);
    runs.forEach((run, g) => {
      let k = 0;
      for (const b of run) {
        if (b.names.length === 0) continue;
        for (let m = 0; m < b.mult; m++, k++) for (let c = 0; c < C; c++) cols[(g * L + k) * C + c] = b.colStart + m * C + c;
      }
    });

    const domain = oneValue(model.attrs["PARAM_DOMAIN"]?.value);
    if (domain === "[-1,1]" && gp.metadata.type === H5T_FLOAT) {
      const flat = gp.value instanceof Float64Array ? gp.value : Float64Array.from(gp.value as ArrayLike<number>);
      info.naturalCoords = Array.from({ length: G }, (_, r) => [flat[2 * r]!, flat[2 * r + 1]!]);
    }
    info.nLayers = L;
    info.components = labels.map((l) => `${result}/${l}`);
    info.geometry = layerGeometry(runs[0]!, L, ids, G, sections);
    return { info, dataPath: `${bg.path}/DATA`, ids, cols, labels };
  }

  /** Every MODEL/SECTION_ASSIGNMENTS section: its layers, and the (element, Gauss point) it is assigned to. */
  private sectionIndex(model: EGroup, warnings: string[]): SectionIndex | null {
    const sa = this.groupOf(model, "SECTION_ASSIGNMENTS");
    if (!sa) return null;
    const out: SectionIndex = { at: new Map(), sections: [] };
    for (const name of sa.keys()) {
      const s = this.groupOf(sa, name);
      if (!s) continue;
      const fd = this.datasetOf(s, "FIBER_DATA");
      const as = this.datasetOf(s, "ASSIGNMENT");
      if (!fd || !as) continue;
      const fs = fd.shape ?? [];
      const ash = as.shape ?? [];
      if (fs.length !== 2 || fs[1] !== 3 || fd.metadata.type !== H5T_FLOAT || ash.length !== 2 || ash[1] !== 2 || as.metadata.type !== H5T_INTEGER) {
        warnings.push(`${s.path}: FIBER_DATA is not (layers, 3) floating point or ASSIGNMENT not (rows, 2) integer; its layers are not placed`);
        continue;
      }
      const flat = fd.value instanceof Float64Array ? fd.value : Float64Array.from(fd.value as ArrayLike<number>);
      const z = Float64Array.from({ length: fs[0]! }, (_, i) => flat[3 * i + 1]!);
      const t = Float64Array.from({ length: fs[0]! }, (_, i) => flat[3 * i + 2]!);
      const index = out.sections.length;
      out.sections.push({ path: s.path, z, t });
      const rows = ints(as);
      for (let r = 0; r + 1 < rows.length; r += 2) out.at.set(`${rows[r]}:${rows[r + 1]}`, index);
    }
    return out;
  }

  private elementIdsIn(bg: EGroup): Float64Array | null {
    const ds = this.datasetOf(bg, "ID");
    const shape = ds?.shape ?? [];
    if (!ds || ds.metadata.type !== H5T_INTEGER || !(shape.length === 1 || (shape.length === 2 && shape[1] === 1))) return null;
    return Float64Array.from(ints(ds));
  }

  private groupOf(parent: EGroup, name: string): EGroup | null {
    if (!parent.keys().includes(name)) return null;
    const g = parent.get(name);
    return g instanceof this.h5.Group ? (g as unknown as EGroup) : null;
  }

  private datasetOf(parent: EGroup, name: string): EDataset | null {
    if (!parent.keys().includes(name)) return null;
    const d = parent.get(name);
    return d instanceof this.h5.Dataset ? (d as unknown as EDataset) : null;
  }

  private intsOf(parent: EGroup, name: string): number[] | null {
    const d = this.datasetOf(parent, name);
    return d && d.metadata.type === H5T_INTEGER ? ints(d) : null;
  }
}

interface SectionIndex {
  /** `${element}:${gauss}` -> index into `sections` */
  at: Map<string, number>;
  sections: { path: string; z: Float64Array; t: Float64Array }[];
}

/** COMP_NAMES lines with empty lines kept (a run of non-answering layers), or why not. */
function compNameLines(attr: unknown, nBlocks: number): string[][] | string {
  const raw = oneValue(attr);
  if (typeof raw !== "string") return "no string COMP_NAMES attribute";
  const lines = raw.split("\n");
  if (lines.length !== nBlocks) return `${lines.length} COMP_NAMES lines for ${nBlocks} rows`;
  const out: string[][] = [];
  for (const line of lines) {
    if (line.trim() === "") {
      out.push([]);
      continue;
    }
    const names = line.split(",").map((t) => t.trim());
    if (names.some((t) => t === "") || new Set(names).size !== names.length) return `COMP_NAMES line ${JSON.stringify(line)} has an empty or repeated label`;
    out.push(names);
  }
  return out;
}

/**
 * Each responding layer's through-thickness coordinate and thickness per element, and each
 * element's section thickness, from the element's section (the same at every Gauss point);
 * or why not (see the header).
 */
function layerGeometry(run: readonly Block[], L: number, ids: Float64Array, G: number, sections: SectionIndex | null): LayerGroupInfo["geometry"] {
  if (!sections) return { state: "absent", reason: "the stage's MODEL has no SECTION_ASSIGNMENTS, so its layers are not placed through the thickness" };
  const span = run.reduce((a, b) => a + b.mult, 0);
  // Section layer index of each responding layer, by position along the run.
  const at: number[] = [];
  let s = 0;
  for (const b of run) {
    for (let m = 0; m < b.mult; m++, s++) if (b.names.length > 0) at.push(s);
  }
  const E = ids.length;
  const z = new Float64Array(E * L);
  const t = new Float64Array(E * L);
  const thickness = new Float64Array(E);
  for (let e = 0; e < E; e++) {
    const id = ids[e]!;
    const sec = sections.at.get(`${id}:0`);
    if (sec === undefined) return { state: "absent", reason: `element ${id} has no SECTION_ASSIGNMENTS row for Gauss point 0, so its layers are not placed` };
    for (let g = 1; g < G; g++) {
      if (sections.at.get(`${id}:${g}`) !== sec) return { state: "absent", reason: `element ${id}'s Gauss points carry different sections, so one through-thickness layout does not hold for it` };
    }
    const S = sections.sections[sec]!;
    if (S.z.length !== span) return { state: "absent", reason: `${S.path} has ${S.z.length} layers and the column map's run covers ${span} (element ${id}), so its layers are not placed` };
    at.forEach((sl, k) => {
      z[e * L + k] = S.z[sl]!;
      t[e * L + k] = S.t[sl]!;
    });
    thickness[e] = S.t.reduce((a, x) => a + x, 0);
  }
  return { state: "present", z, t, thickness };
}
