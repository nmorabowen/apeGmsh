// The particle results of a `.ladruno` (ADR 0134 D3; chain M, M2-S3 #2982).
//
// The layout is the fork's Contract 2, `Ladruno_implementation/232a_mpm_contracts.md`
// §2 on the fork's `ladruno` branch: per stage, `MODEL/PARTICLES` (the grid, the
// bodies and the registry) and `RESULTS/ON_PARTICLES` (the outputs, ragged, as CSR:
// `OFFSETS[k]..OFFSETS[k+1]` rows of `ID` and of each field's `DATA` are output k).
// reader/ladruno.ts gates the file and indexes its FE stages; it hands each stage
// group here when `INFO/FORMAT_MINOR_VERSION >= 2` (§2.1: the minor is the feature
// flag), and refuses a part file that holds particles (§2.2, C2-PART).
//
// Rules this file keeps:
// - the index is read at open, the values on request: STEP, TIME, OFFSETS and the
//   registry are read whole (one entry per output, one per particle); `ID` and each
//   `DATA` are read one output at a time, by hyperslab, never the whole history (D5);
// - the crash rule of §2.5: the writer flushes after each output, so a killed run
//   leaves a file whose committed outputs all read and whose tail may be torn. The
//   outputs are `T' = min(len(STEP), len(TIME), len(OFFSETS) - 1)`, then only the
//   prefix whose `OFFSETS[k+1]` is within `len(ID)` and every field's `len(DATA)`.
//   Rows past `OFFSETS[T']` are never read, and no full-file shape is taken as the
//   row count `S`; `torn` says what was dropped;
// - no chunk shape is assumed: the reads are hyperslabs, which read a contiguous
//   dataset, one chunk per output or 1 MiB chunks alike (§0 "Storage", §2.8);
// - the fork's attribute encodings (§0): a scalar or a 1-element array; a string
//   fixed-length (`NULLTERM`, NULs trimmed) or variable-length; an absent string
//   attribute is `""` where the table says "absent ≡ ''";
// - exact dtypes (§0): an int64 `ID` read as `Number` with the safe-integer check, and
//   any other dtype is refused by name, never cast;
// - a stage that breaks the contract is refused by name in `refused`, with the rule
//   code, and the file's other stages still read; a field that breaks it is refused
//   in its stage's `refusedFields`, and a stage whose POSITION is refused is refused;
// - `NaN` is read as stored: `STATE_*` writes it where a material does not answer
//   (§2.6). Values are not scanned for C2-FINITE (that is the fork's validator).
//
// Browser-safe: no Node import, and no import of reader/ladruno.ts (it imports this).

/** HDF5 type classes, as h5wasm reports `metadata.type`. */
const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;

/** Field names: §2.5 C2-NAME. */
const FIELD_NAME = /^[A-Z][A-Z0-9_]*$/;
const STATE_NAME = /^STATE_[A-Za-z0-9_]+$/;
/** The index datasets directly in ON_PARTICLES (§2.5). */
const INDEX_DATASETS = ["STEP", "TIME", "OFFSETS", "ID"] as const;
const MODES = { explicit: "transient", implicit: "transient", "quasi-static": "static" } as const;
/** §2.6: the reserved fields' COMPONENTS, by spatial dimension (C2-CAT). */
const RESERVED: Readonly<Record<string, Readonly<Record<2 | 3, string | null>>>> = {
  POSITION: { 2: "x,y", 3: "x,y,z" },
  VELOCITY: { 2: "Vx,Vy", 3: "Vx,Vy,Vz" },
  ACCELERATION: { 2: "Ax,Ay", 3: "Ax,Ay,Az" },
  STRESS: { 2: "sigma11,sigma22,sigma12", 3: "sigma11,sigma22,sigma33,sigma12,sigma23,sigma13" },
  STRESS33: { 2: "sigma33", 3: null },
  STRAIN: { 2: "eps11,eps22,gamma12", 3: "eps11,eps22,eps33,gamma12,gamma23,gamma13" },
  VOLUME: { 2: "V", 3: "V" },
  GIMP_HALF_LENGTH: { 2: "lx,ly", 3: "lx,ly,lz" },
  DEFORMATION_GRADIENT: { 2: "F11,F12,F21,F22", 3: "F11,F12,F13,F21,F22,F23,F31,F32,F33" },
};

// -- the h5wasm surface this module reads (structural, as reader/ladrunoElements.ts) --

export interface PAttr {
  value: unknown;
  metadata: { type: number; size?: number; signed?: boolean };
}
export interface PNode {
  path: string;
  attrs: Record<string, PAttr>;
}
export interface PGroup extends PNode {
  keys(): string[];
  get(name: string): unknown;
}
export interface PDataset extends PNode {
  shape: number[] | null;
  metadata: { type: number; size: number; signed: boolean; littleEndian: boolean };
  value: unknown;
  slice(ranges: ([] | [number | null] | [number | null, number | null])[]): unknown;
}
export interface PCtors {
  Group: abstract new (...args: never[]) => unknown;
  Dataset: abstract new (...args: never[]) => unknown;
}

/**
 * A stage torn at opening (the fork's contract follow-up, #2983): `MODEL/PARTICLES` is
 * written once, when the stage opens, so registry columns of unequal length mean the
 * writer stopped inside that write, before any output could commit. Such a stage is
 * skipped, by name, and the file's other stages still read; it is not a contract break.
 */
export class ParticleStageTorn extends Error {}

/** A contract break: the rule code of the fork's page, and what was found. */
export class ParticleContractError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(`${message} (${code})`);
    this.code = code;
  }
}

// -- what the points layer (M2-S4) draws from ------------------------------------

/** `GRID` (§1.2, §2.4): nodes at `origin + i * cellSize`, `i = 0 .. numCells` per axis. */
export interface ParticleGrid {
  origin: number[];
  cellSize: number[];
  numCells: number[];
  basis: "gimp" | "linear";
}

/** One body: the id particles carry, and its label in apeGmsh (§1.3). */
export interface ParticleBody {
  id: number;
  name: string;
}

/** The registry (§2.4): one row per particle that appears in any output of the stage, `ids` strictly increasing. */
export interface ParticleRegistry {
  ids: Float64Array;
  bodyIds: Int32Array;
  materialTags: Int32Array;
  mass: Float64Array;
}

/** One field group of ON_PARTICLES (§2.5, §2.6), spelled as the file spells it (ADR 0122 D7). */
export interface ParticleField {
  name: string;
  displayName: string;
  components: string[];
  /** absent ≡ "" (dimensionless) */
  dimension: string;
  /** absent ≡ "" */
  description: string;
  /** 0 scalar, 1 vectorial, 2 tensorial */
  dataType: number;
  /** the stored precision */
  dtype: "f64" | "f32";
}

/** One stage that holds particles, as read at open (metadata only). */
export interface ParticleStage {
  /** the stage id reader/ladruno.ts gives the same MODEL_STAGE (`stage_<i>`) */
  stageId: string;
  /** the stage group, `/MODEL_STAGE[<id>]` */
  group: string;
  kind: "static" | "transient";
  epoch: number;
  mode: keyof typeof MODES;
  patternTag: number;
  /** `SOURCE_SET_ID` (the Contract-1 `SET_ID`), "" when absent (no file loaded) */
  sourceSetId: string;
  ndm: 2 | 3;
  /** `PLANE_STRAIN_THICKNESS` when ndm = 2, else null */
  thickness: number | null;
  units: { length: string; force: string; time: string; system: string };
  grid: ParticleGrid;
  bodies: ParticleBody[];
  registry: ParticleRegistry;
  /** the committed outputs (`T'` of the crash rule) */
  nOutputs: number;
  /** (nOutputs,) the Domain commit tag of each output; outputs are keyed by index or step, never time (§2.5) */
  steps: Float64Array;
  /** (nOutputs,) the Domain time; it may restart inside a stage */
  time: Float64Array;
  /** (nOutputs + 1,) output k is rows `offsets[k] .. offsets[k+1]` */
  offsets: Float64Array;
  fields: ParticleField[];
  /** a field group that breaks the contract, by name, with why; never read */
  refusedFields: { field: string; reason: string }[];
  /** what the crash rule dropped (a torn tail), or null for a closed file */
  torn: string | null;
}

/** One output of one field: the particles present, their registry rows, and their values row-major. */
export interface ParticleOutput {
  /** (n,) particle ids, strictly increasing */
  ids: Float64Array;
  /** (n,) each id's row in the stage registry (body, material, mass) */
  rows: Int32Array;
  /** (n * nComp,) row-major, as stored (float32 is widened) */
  values: Float64Array;
  nComp: number;
  step: number;
  time: number;
}

export interface ParticleRead {
  stages: ParticleStage[];
  /** a stage whose particle groups break the contract, by name, with why */
  refused: { stageId: string; group: string; reason: string }[];
  /** a stage torn at opening (`ParticleStageTorn`): skipped, by name, with why; it holds no committed output */
  skipped: { stageId: string; group: string; reason: string }[];
}

/** Whether a stage group names a particle group, of any HDF5 kind (either half of the pair, C2-PAIR): a malformed one is refused, never skipped. */
export function hasParticleGroups(stage: PGroup, ctors: PCtors): boolean {
  return childNamed(stage, ["MODEL", "PARTICLES"], ctors) || childNamed(stage, ["RESULTS", "ON_PARTICLES"], ctors);
}

/**
 * The particle stages of one `.ladruno`. `stages` are its MODEL_STAGE groups with the
 * stage ids reader/ladruno.ts gave them, `ndm` its `INFO@SPATIAL_DIM`, and `fileKinds`
 * each stage's `KIND`. Call only at `FORMAT_MINOR_VERSION >= 2`.
 */
export class LadrunoParticles {
  readonly stages: ParticleStage[];
  readonly refused: ParticleRead["refused"];
  readonly skipped: ParticleRead["skipped"];
  private readonly ctors: PCtors;
  private readonly root: PGroup;

  constructor(ctors: PCtors, root: PGroup, stages: readonly { id: string; group: PGroup; fileKind: string }[], ndm: number | null) {
    this.ctors = ctors;
    this.root = root;
    const read: ParticleStage[] = [];
    const refused: ParticleRead["refused"] = [];
    const skipped: ParticleRead["skipped"] = [];
    for (const s of stages) {
      if (!hasParticleGroups(s.group, ctors)) continue;
      try {
        read.push(this.readStage(s.id, s.group, s.fileKind, ndm));
      } catch (err) {
        if (err instanceof ParticleStageTorn) skipped.push({ stageId: s.id, group: s.group.path, reason: err.message });
        else if (err instanceof ParticleContractError) refused.push({ stageId: s.id, group: s.group.path, reason: err.message });
        else throw err;
      }
    }
    this.stages = read;
    this.refused = refused;
    this.skipped = skipped;
  }

  stage(stageId: string): ParticleStage {
    const s = this.stages.find((x) => x.stageId === stageId);
    if (!s) {
      const why = this.refused.find((r) => r.stageId === stageId);
      const torn = this.skipped.find((r) => r.stageId === stageId);
      throw new RangeError(
        why ? `stage ${stageId}'s particles are refused: ${why.reason}` : torn ? `stage ${stageId}'s particles are skipped: ${torn.reason}` : `stage ${stageId} holds no particles; particle stages: ${this.stages.map((x) => x.stageId).join(", ") || "none"}`,
      );
    }
    return s;
  }

  /** The ids of output `k` (its `ID` hyperslab), checked as `readOutput` checks them (C2-ID). */
  outputIds(stageId: string, k: number): Float64Array {
    const s = this.stage(stageId);
    const [a, b] = this.rows(s, k);
    return this.checkedIds(s, k, a, b).ids;
  }

  /** Output `k` of `field` (as the file spells it: `POSITION`, `STATE_<token>`): one hyperslab of `ID` and one of `DATA`. */
  readOutput(stageId: string, k: number, field: string): ParticleOutput {
    const s = this.stage(stageId);
    const f = s.fields.find((x) => x.name === field);
    if (!f) {
      const why = s.refusedFields.find((x) => x.field === field);
      throw new RangeError(why ? `${field} is refused: ${why.reason}` : `stage ${stageId} records no particle field ${JSON.stringify(field)}; it has ${s.fields.map((x) => x.name).join(", ")}`);
    }
    const [a, b] = this.rows(s, k);
    const on = `${s.group}/RESULTS/ON_PARTICLES`;
    const { ids, rows } = this.checkedIds(s, k, a, b);
    const nComp = f.components.length;
    const dataDs = this.dataset(`${on}/${field}/DATA`);
    const values = b === a ? new Float64Array(0) : floats(dataDs.slice([[a, b], [0, nComp]]), dataDs.path);
    if (values.length !== (b - a) * nComp) throw new ParticleContractError("C2-DATA", `${dataDs.path}: rows ${a}..${b} read ${values.length} values, not ${(b - a) * nComp}`);
    return { ids, rows, values, nComp, step: s.steps[k]!, time: s.time[k]! };
  }

  /** Output `k`'s row range `[OFFSETS[k], OFFSETS[k+1])`. */
  private rows(s: ParticleStage, k: number): [number, number] {
    if (!Number.isInteger(k) || k < 0 || k >= s.nOutputs) {
      throw new RangeError(`output ${k} is outside stage ${s.stageId} (${s.nOutputs} outputs${s.torn ? `; ${s.torn}` : ""})`);
    }
    return [s.offsets[k]!, s.offsets[k + 1]!];
  }

  /** Rows `a..b` of `ID` (output `k`): each a safe integer, strictly increasing within the output, and in the stage registry (C2-ID). */
  private checkedIds(s: ParticleStage, k: number, a: number, b: number): { ids: Float64Array; rows: Int32Array } {
    const idDs = this.dataset(`${s.group}/RESULTS/ON_PARTICLES/ID`);
    const ids = readIds(idDs, a, b);
    for (let i = 1; i < ids.length; i++) {
      if (!(ids[i]! > ids[i - 1]!)) throw new ParticleContractError("C2-ID", `${idDs.path}: output ${k}'s ids are not strictly increasing at row ${a + i}`);
    }
    return { ids, rows: registryRows(s.registry.ids, ids, `${idDs.path} (output ${k})`) };
  }

  private dataset(path: string): PDataset {
    const d = this.root.get(path);
    if (!(d instanceof this.ctors.Dataset)) throw new ParticleContractError("C2-DATA", `${path} is not a dataset`);
    return d as unknown as PDataset;
  }

  private readStage(stageId: string, g: PGroup, fileKind: string, fileNdm: number | null): ParticleStage {
    const hasModel = childNamed(g, ["MODEL", "PARTICLES"], this.ctors);
    const hasOn = childNamed(g, ["RESULTS", "ON_PARTICLES"], this.ctors);
    if (!hasModel || !hasOn) {
      throw new ParticleContractError("C2-PAIR", `${g.path} holds ${hasModel ? "MODEL/PARTICLES" : "RESULTS/ON_PARTICLES"} without ${hasModel ? "RESULTS/ON_PARTICLES" : "MODEL/PARTICLES"}; both or neither`);
    }
    const model = requiredGroup(requiredGroup(g, "MODEL", this.ctors, "C2-PAIR"), "PARTICLES", this.ctors, "C2-PAIR");
    const on = requiredGroup(requiredGroup(g, "RESULTS", this.ctors, "C2-PAIR"), "ON_PARTICLES", this.ctors, "C2-PAIR");
    const kind = fileKind === "static" || fileKind === "transient" ? fileKind : null;
    if (kind === null) throw new ParticleContractError("C2-KIND", `${g.path}@KIND = ${JSON.stringify(fileKind)}; a particle stage is "transient" or "static"`);
    if (fileNdm !== 2 && fileNdm !== 3) throw new ParticleContractError("C2-NDM", `INFO@SPATIAL_DIM = ${JSON.stringify(fileNdm)}; a file with particles says 2 or 3`);
    const ndm = fileNdm;
    const epoch = requiredInt(g, "PARTICLE_EPOCH", "C2-EPOCH");
    const mode = requiredText(model, "MODE", "C2-MODE");
    if (!Object.hasOwn(MODES, mode)) throw new ParticleContractError("C2-MODE", `${model.path}@MODE = ${JSON.stringify(mode)}; expected explicit, implicit or quasi-static`);
    const m = mode as keyof typeof MODES;
    if (MODES[m] !== kind) throw new ParticleContractError("C2-MODEKIND", `${model.path}@MODE = ${mode} needs KIND ${MODES[m]}, and ${g.path}@KIND is ${kind}`);
    const thickness = ndm === 2 ? requiredFloat(model, "PLANE_STRAIN_THICKNESS", "C2-THK") : null;
    if (ndm === 3 && model.attrs["PLANE_STRAIN_THICKNESS"] !== undefined) throw new ParticleContractError("C2-THK", `${model.path}@PLANE_STRAIN_THICKNESS is set on a 3-D stage`);
    if (thickness !== null && !(thickness > 0 && Number.isFinite(thickness))) throw new ParticleContractError("C2-THK", `${model.path}@PLANE_STRAIN_THICKNESS = ${thickness}; expected finite and > 0`);
    const units = {
      length: requiredText(model, "UNIT_LENGTH", "C2-UNIT"),
      force: requiredText(model, "UNIT_FORCE", "C2-UNIT"),
      time: requiredText(model, "UNIT_TIME", "C2-UNIT"),
      system: optionalText(model, "UNIT_SYSTEM"),
    };
    for (const [k, v] of Object.entries(units)) if (k !== "system" && v === "") throw new ParticleContractError("C2-UNIT", `${model.path}@UNIT_${k.toUpperCase()} is empty`);
    const grid = readGrid(requiredGroup(model, "GRID", this.ctors, "C2-GRID"), ndm, "C2-GRID");
    const bodies = readBodies(requiredGroup(model, "BODIES", this.ctors, "C2-BODY"), this.ctors, "C2-BODY");
    const registry = this.readRegistry(model, bodies, on);
    const outputs = this.readOutputs(on, ndm);
    return {
      stageId,
      group: g.path,
      kind,
      epoch,
      mode: m,
      patternTag: requiredInt(model, "PATTERN_TAG", "C2-PATTERN"),
      sourceSetId: optionalText(model, "SOURCE_SET_ID"),
      ndm,
      thickness,
      units,
      grid,
      bodies,
      registry,
      ...outputs,
    };
  }

  private readRegistry(model: PGroup, bodies: ParticleBody[], on: PGroup): ParticleRegistry {
    const ds = (name: string) => requiredDataset(model, name, this.ctors, "C2-REG");
    const idDs = ds("ID");
    exactInt(idDs, 8, "C2-REG");
    // The registry is written once, at the stage's opening: columns of unequal length are an opening the writer
    // never finished, so the stage holds no output and is skipped (ParticleStageTorn). The same columns beside a
    // committed output cannot come from that writer, so they still refuse the stage (C2-REG).
    const lengths = (["ID", "BODY_ID", "MATERIAL_TAG", "MASS"] as const).map((n) => ({ n, rows: oneD(ds(n), "C2-REG") }));
    if (lengths.some((l) => l.rows !== lengths[0]!.rows)) {
      const said = lengths.map((l) => `${l.n} ${l.rows}`).join(", ");
      const committed = committedIndex(on, this.ctors);
      if (committed > 0) {
        throw new ParticleContractError("C2-REG", `${model.path}: the registry columns differ in length (${said}), and the stage holds ${committed} committed output${committed === 1 ? "" : "s"}, so it is no torn opening`);
      }
      throw new ParticleStageTorn(`${model.path}: torn at opening: the registry columns differ in length (${said}) and no output was committed`);
    }
    const ids = readIds(idDs, 0, oneD(idDs, "C2-REG"));
    for (let i = 1; i < ids.length; i++) if (!(ids[i]! > ids[i - 1]!)) throw new ParticleContractError("C2-REG", `${idDs.path} is not strictly increasing at row ${i}`);
    const bodyIds = int32s(ds("BODY_ID"), ids.length, "C2-REG");
    const materialTags = int32s(ds("MATERIAL_TAG"), ids.length, "C2-REG");
    const massDs = ds("MASS");
    exactFloat(massDs, 8, "C2-REG");
    if (oneD(massDs, "C2-REG") !== ids.length) throw new ParticleContractError("C2-REG", `${massDs.path} has ${oneD(massDs, "C2-REG")} rows and the registry ID ${ids.length}`);
    const mass = ids.length === 0 ? new Float64Array(0) : floats(massDs.value, massDs.path);
    const known = new Set(bodies.map((b) => b.id));
    bodyIds.forEach((b, i) => {
      if (!known.has(b)) throw new ParticleContractError("C2-REG", `${model.path}/BODY_ID[${i}] = ${b} is not in BODIES/ID`);
    });
    return { ids, bodyIds, materialTags, mass };
  }

  /** The index datasets and the fields, under the crash rule of §2.5. */
  private readOutputs(on: PGroup, ndm: 2 | 3): Pick<ParticleStage, "nOutputs" | "steps" | "time" | "offsets" | "fields" | "refusedFields" | "torn"> {
    const reduction = requiredText(on, "PARTITION_REDUCTION", "C2-RED");
    if (reduction !== "NONE") throw new ParticleContractError("C2-RED", `${on.path}@PARTITION_REDUCTION = ${JSON.stringify(reduction)}; expected "NONE" (serial only in minor 2)`);
    const stepDs = requiredDataset(on, "STEP", this.ctors, "C2-AXES");
    const timeDs = requiredDataset(on, "TIME", this.ctors, "C2-AXES");
    const offDs = requiredDataset(on, "OFFSETS", this.ctors, "C2-OFF");
    const idDs = requiredDataset(on, "ID", this.ctors, "C2-ID");
    exactInt(stepDs, 4, "C2-AXES");
    exactFloat(timeDs, 8, "C2-AXES");
    exactInt(offDs, 8, "C2-OFF");
    exactInt(idDs, 8, "C2-ID");
    const nStep = oneD(stepDs, "C2-AXES");
    const nTime = oneD(timeDs, "C2-AXES");
    const nOff = oneD(offDs, "C2-OFF");
    const nId = oneD(idDs, "C2-ID");
    const offsets = nOff === 0 ? new Float64Array(0) : readIds(offDs, 0, nOff);
    // §2.5's crash rule, the fork's final text (fork commit 7b949e58d), c = len(OFFSETS). The writer
    // appends DATA, ID, STEP and TIME, then OFFSETS last, flushing after each output, so only the
    // last OFFSETS entry can be torn (extended, still the fill value 0). A torn entry equal to the
    // one before it (every output so far empty) is no decrease: it reads as one more empty output.
    // (1) OFFSETS[0] != 0 refuses.
    if (nOff > 0 && offsets[0] !== 0) throw new ParticleContractError("C2-OFF", `${offDs.path}[0] = ${offsets[0]}; expected 0`);
    // (2) c >= 2 and OFFSETS[c-1] < OFFSETS[c-2]: the last entry is a torn extension; that output is dropped.
    let nOffUsed = nOff;
    let inFlight: string | null = null;
    if (nOff >= 2 && offsets[nOff - 1]! < offsets[nOff - 2]!) {
      nOffUsed = nOff - 1;
      inFlight = `${offDs.path}[${nOff - 1}] = ${offsets[nOff - 1]} is below the entry before it, so it is the output being written and is not counted`;
    }
    // (3) OFFSETS still decreasing anywhere in what remains refuses the stage.
    for (let i = 1; i < nOffUsed; i++) if (offsets[i]! < offsets[i - 1]!) throw new ParticleContractError("C2-OFF", `${offDs.path} decreases at ${i}`);
    // Then T' = min(len(STEP), len(TIME), len(OFFSETS) - 1), and the ID and DATA bounds apply to what remains.
    // The fields, checked against the contract; each one's DATA length bounds the outputs.
    const fields: ParticleField[] = [];
    const refusedFields: { field: string; reason: string }[] = [];
    const bounds: { what: string; rows: number }[] = [{ what: idDs.path, rows: nId }];
    for (const name of on.keys()) {
      const item = on.get(name);
      if (item instanceof this.ctors.Dataset) {
        if (!(INDEX_DATASETS as readonly string[]).includes(name)) refusedFields.push({ field: name, reason: `${on.path}/${name} is a dataset, and only STEP, TIME, OFFSETS and ID may sit in ON_PARTICLES (C2-UNKNOWN)` });
        continue;
      }
      if (!(item instanceof this.ctors.Group)) {
        refusedFields.push({ field: name, reason: `${on.path}/${name} is neither a group nor a dataset (C2-UNKNOWN)` });
        continue;
      }
      // Every field group's DATA bounds the committed outputs, a refused field's too, as the fork's
      // `committed_outputs` reads it (#2985 review follow-up): a group that breaks the contract by an
      // attribute still had its rows appended by the writer, and its length still says where it stopped.
      const g = item as unknown as PGroup;
      const dataRows = groupDataRows(g, this.ctors);
      if (dataRows !== null) bounds.push({ what: `${on.path}/${name}/DATA`, rows: dataRows });
      try {
        fields.push(readField(g, name, ndm, this.ctors));
      } catch (err) {
        if (!(err instanceof ParticleContractError)) throw err;
        refusedFields.push({ field: name, reason: err.message });
      }
    }
    if (!fields.some((f) => f.name === "POSITION")) {
      const why = refusedFields.find((f) => f.field === "POSITION");
      throw new ParticleContractError("C2-POS", why ? `${on.path}/POSITION is refused: ${why.reason}` : `${on.path} has no POSITION field`);
    }
    // §2.5: T' from the index datasets, then the committed prefix every dataset holds.
    const tPrime = Math.max(0, Math.min(nStep, nTime, nOffUsed - 1));
    let n = tPrime;
    let short: { what: string; rows: number } | null = null;
    while (n > 0) {
      const end = offsets[n]!;
      const lacking = bounds.find((b) => end > b.rows);
      if (!lacking) break;
      short = lacking;
      n--;
    }
    const notes: string[] = [];
    if (nOff === 0) notes.push(`${offDs.path} is empty: no output was committed`);
    if (inFlight !== null) notes.push(inFlight);
    if (nStep !== tPrime || nTime !== tPrime || nOffUsed - 1 !== tPrime) notes.push(`STEP holds ${nStep}, TIME ${nTime} and OFFSETS ${nOff} entries, so ${tPrime} output${tPrime === 1 ? " is" : "s are"} committed`);
    if (n < tPrime) notes.push(`${short!.what} holds ${short!.rows} rows, short of output ${n}'s end ${offsets[n + 1]}, so ${tPrime - n} committed output${tPrime - n === 1 ? " is" : "s are"} dropped`);
    const used = nOff === 0 ? 0 : offsets[n]!;
    const longer = bounds.filter((b) => b.rows > used);
    if (longer.length > 0) notes.push(`rows past ${used} (${longer.map((b) => `${b.what}: ${b.rows}`).join(", ")}) are a partial output and are not read`);
    const steps = n === 0 ? new Float64Array(0) : readIds(stepDs, 0, n);
    for (let k = 1; k < n; k++) if (!(steps[k]! > steps[k - 1]!)) throw new ParticleContractError("C2-AXES", `${stepDs.path} is not strictly increasing at output ${k} (${steps[k - 1]}, then ${steps[k]})`);
    const time = n === 0 ? new Float64Array(0) : floats(timeDs.slice([[0, n]]), timeDs.path);
    return {
      nOutputs: n,
      steps,
      time,
      offsets: nOff === 0 ? new Float64Array([0]) : offsets.slice(0, n + 1),
      fields,
      refusedFields,
      torn: notes.length > 0 ? `a torn tail (§2.5 crash rule): ${notes.join("; ")}` : null,
    };
  }
}

/** `min(len(STEP), len(TIME), len(OFFSETS) - 1)` of ON_PARTICLES as stored (0 for a missing dataset): whether any output may have committed. */
function committedIndex(on: PGroup, ctors: PCtors): number {
  const rows = (name: string): number => {
    if (!on.keys().includes(name)) return 0;
    const d = on.get(name);
    if (!(d instanceof ctors.Dataset)) return 0;
    const shape = (d as unknown as PDataset).shape;
    return shape && shape.length >= 1 ? shape[0]! : 0;
  };
  return Math.max(0, Math.min(rows("STEP"), rows("TIME"), rows("OFFSETS") - 1));
}

/** The row count of a field group's `DATA`, whatever else the group breaks; null when it has no DATA dataset with a shape. */
function groupDataRows(g: PGroup, ctors: PCtors): number | null {
  if (!g.keys().includes("DATA")) return null;
  const d = g.get("DATA");
  if (!(d instanceof ctors.Dataset)) return null;
  const shape = (d as unknown as PDataset).shape;
  return shape && shape.length >= 1 ? shape[0]! : null;
}

/** One field group: its attributes, checked (C2-ATTR, C2-COMP, C2-NAME, C2-CAT, C2-DTYPE). */
function readField(g: PGroup, name: string, ndm: 2 | 3, ctors: PCtors): ParticleField {
  if (!FIELD_NAME.test(name) && !STATE_NAME.test(name)) throw new ParticleContractError("C2-NAME", `${g.path}: a field name matches ^[A-Z][A-Z0-9_]*$ or ^STATE_[A-Za-z0-9_]+$`);
  for (const a of ["DISPLAY_NAME", "COMPONENTS", "TYPE", "DATA_TYPE", "PARTITION_REDUCTION"]) {
    if (g.attrs[a] === undefined) throw new ParticleContractError("C2-ATTR", `${g.path} has no @${a}`);
  }
  const displayName = requiredText(g, "DISPLAY_NAME", "C2-ATTR");
  if (displayName === "") throw new ParticleContractError("C2-ATTR", `${g.path}@DISPLAY_NAME is empty`);
  const compText = requiredText(g, "COMPONENTS", "C2-COMP");
  const components = compText.split(",");
  if (components.some((c) => c === "") || new Set(components).size !== components.length) {
    throw new ParticleContractError("C2-COMP", `${g.path}@COMPONENTS = ${JSON.stringify(compText)}: a label is empty or repeated`);
  }
  const reserved = RESERVED[name];
  if (reserved !== undefined && reserved[ndm] !== compText) {
    throw new ParticleContractError("C2-CAT", `${g.path}@COMPONENTS = ${JSON.stringify(compText)}; the reserved ${name} is ${reserved[ndm] === null ? `refused in ${ndm}-D` : JSON.stringify(reserved[ndm])}`);
  }
  const reduction = requiredText(g, "PARTITION_REDUCTION", "C2-RED");
  if (reduction !== "NONE") throw new ParticleContractError("C2-RED", `${g.path}@PARTITION_REDUCTION = ${JSON.stringify(reduction)}; expected "NONE"`);
  const dataType = requiredInt(g, "DATA_TYPE", "C2-ATTR");
  requiredInt(g, "TYPE", "C2-ATTR");
  const data = requiredDataset(g, "DATA", ctors, "C2-DATA");
  const shape = data.shape ?? [];
  if (shape.length !== 2 || shape[1] !== components.length) {
    throw new ParticleContractError("C2-DATA", `${data.path} has shape (${shape.join(", ")}); expected (S, ${components.length}), one column per COMPONENTS label`);
  }
  if (data.metadata.type !== H5T_FLOAT || (data.metadata.size !== 8 && data.metadata.size !== 4) || data.metadata.littleEndian === false) {
    throw new ParticleContractError("C2-DTYPE", `${data.path} is not a little-endian float64 or float32 dataset (class ${data.metadata.type}, size ${data.metadata.size})`);
  }
  if (name === "POSITION" && data.metadata.size !== 8) throw new ParticleContractError("C2-DTYPE", `${data.path} is float32; POSITION is always float64`);
  return {
    name,
    displayName,
    components,
    dimension: optionalText(g, "DIMENSION"),
    description: optionalText(g, "DESCRIPTION"),
    dataType,
    dtype: data.metadata.size === 8 ? "f64" : "f32",
  };
}

// -- shared with reader/particleSet.ts (Contract 1 has the same encodings) -------

/** The grid's four attributes (§1.2). `code` is the rule the contract names for it. */
export function readGrid(g: PGroup, ndm: 2 | 3, code: string): ParticleGrid {
  const origin = floatVector(g, "ORIGIN", ndm, code);
  const cellSize = floatVector(g, "CELL_SIZE", ndm, code);
  const numCells = intVector(g, "NUM_CELLS", ndm, code);
  if (origin.some((x) => !Number.isFinite(x))) throw new ParticleContractError(code, `${g.path}@ORIGIN is not finite`);
  if (cellSize.some((x) => !(x > 0) || !Number.isFinite(x))) throw new ParticleContractError(code, `${g.path}@CELL_SIZE must be finite and > 0 on each axis`);
  if (numCells.some((x) => x < 1)) throw new ParticleContractError(code, `${g.path}@NUM_CELLS must be >= 1 on each axis`);
  const basis = requiredText(g, "BASIS", code === "C1-GRID" ? "C1-BASIS" : code);
  if (basis !== "gimp" && basis !== "linear") throw new ParticleContractError(code === "C1-GRID" ? "C1-BASIS" : code, `${g.path}@BASIS = ${JSON.stringify(basis)}; expected gimp or linear`);
  return { origin, cellSize, numCells, basis };
}

/** `BODIES/{ID, NAME}` (§1.3, §2.4): ids unique, names unique and non-empty; NAME fixed- or variable-length. */
export function readBodies(g: PGroup, ctors: PCtors, code: string): ParticleBody[] {
  const idDs = requiredDataset(g, "ID", ctors, code);
  const nameDs = requiredDataset(g, "NAME", ctors, code);
  const n = oneD(idDs, code);
  const ids = int32s(idDs, n, code);
  if (oneD(nameDs, code) !== n) throw new ParticleContractError(code, `${nameDs.path} has ${oneD(nameDs, code)} names and ${idDs.path} ${n} ids`);
  if (nameDs.metadata.type !== H5T_STRING) throw new ParticleContractError(code, `${nameDs.path} is not a string dataset`);
  const raw = nameDs.value;
  const names = (Array.isArray(raw) ? raw : [raw]).map((v) => trimText(v));
  if (names.length !== n) throw new ParticleContractError(code, `${nameDs.path} read ${names.length} names, not ${n}`);
  if (n < 1) throw new ParticleContractError(code, `${g.path} names no body`);
  if (new Set(ids).size !== n || ids.some((x) => x < 0)) throw new ParticleContractError(code, `${idDs.path} must be unique and >= 0`);
  if (names.some((x) => x === "") || new Set(names).size !== n) throw new ParticleContractError(code, `${nameDs.path} must be unique and non-empty`);
  return Array.from(ids, (id, i) => ({ id, name: names[i]! }));
}

/** A required string attribute, scalar or 1-element, fixed (NULs trimmed) or variable length. */
export function requiredText(g: PNode, key: string, code: string): string {
  const a = g.attrs[key];
  if (a === undefined) throw new ParticleContractError(code, `${g.path} has no @${key}`);
  if (a.metadata.type !== H5T_STRING) throw new ParticleContractError(code, `${g.path}@${key} is not a string attribute`);
  const v = one(a.value, `${g.path}@${key}`, code);
  return trimText(v);
}

/** An optional string attribute: absent ≡ "" (§0, "Empty strings are absent"). */
export function optionalText(g: PNode, key: string): string {
  return g.attrs[key] === undefined ? "" : requiredText(g, key, "C2-ATTR");
}

/** A required integer attribute, scalar or 1-element. */
export function requiredInt(g: PNode, key: string, code: string): number {
  const a = g.attrs[key];
  if (a === undefined) throw new ParticleContractError(code, `${g.path} has no @${key}`);
  if (a.metadata.type !== H5T_INTEGER) throw new ParticleContractError(code, `${g.path}@${key} is not an integer attribute`);
  const v = one(a.value, `${g.path}@${key}`, code);
  const n = typeof v === "bigint" ? safe(v, `${g.path}@${key}`) : v;
  if (typeof n !== "number" || !Number.isInteger(n)) throw new ParticleContractError(code, `${g.path}@${key} is not an integer`);
  return n;
}

/** A required float attribute, scalar or 1-element. */
export function requiredFloat(g: PNode, key: string, code: string): number {
  const a = g.attrs[key];
  if (a === undefined) throw new ParticleContractError(code, `${g.path} has no @${key}`);
  if (a.metadata.type !== H5T_FLOAT) throw new ParticleContractError(code, `${g.path}@${key} is not a floating-point attribute`);
  const v = one(a.value, `${g.path}@${key}`, code);
  if (typeof v !== "number") throw new ParticleContractError(code, `${g.path}@${key} is not a number`);
  return v;
}

export function requiredGroup(parent: PGroup, name: string, ctors: PCtors, code: string): PGroup {
  if (!parent.keys().includes(name)) throw new ParticleContractError(code, `${parent.path} has no group ${name}`);
  const g = parent.get(name);
  if (!(g instanceof ctors.Group)) throw new ParticleContractError(code, `${parent.path}/${name} is not a group`);
  return g as unknown as PGroup;
}

export function requiredDataset(parent: PGroup, name: string, ctors: PCtors, code: string): PDataset {
  if (!parent.keys().includes(name)) throw new ParticleContractError(code, `${parent.path} has no dataset ${name}`);
  const d = parent.get(name);
  if (!(d instanceof ctors.Dataset)) throw new ParticleContractError(code, `${parent.path}/${name} is not a dataset`);
  return d as unknown as PDataset;
}

/** The length of a 1-D dataset; any other rank is refused. */
export function oneD(ds: PDataset, code: string): number {
  const shape = ds.shape ?? [];
  if (shape.length !== 1) throw new ParticleContractError(code, `${ds.path} has shape (${shape.join(", ")}); expected 1-D`);
  return shape[0]!;
}

/** Refuse anything but a little-endian signed integer of `size` bytes (§0 "Exact dtypes"). */
export function exactInt(ds: PDataset, size: 4 | 8, code: string): void {
  const m = ds.metadata;
  if (m.type !== H5T_INTEGER || m.size !== size || !m.signed || m.littleEndian === false) {
    throw new ParticleContractError(code, `${ds.path} is not int${size * 8} (class ${m.type}, size ${m.size}, ${m.signed ? "signed" : "unsigned"})`);
  }
}

/** Refuse anything but a little-endian float of `size` bytes. */
export function exactFloat(ds: PDataset, size: 4 | 8, code: string): void {
  const m = ds.metadata;
  if (m.type !== H5T_FLOAT || m.size !== size || m.littleEndian === false) throw new ParticleContractError(code, `${ds.path} is not float${size * 8} (class ${m.type}, size ${m.size})`);
}

/** Rows `a..b` of an int64 (or int32) 1-D dataset as Numbers, each one a safe integer. */
export function readIds(ds: PDataset, a: number, b: number): Float64Array {
  if (b === a) return new Float64Array(0);
  const v = ds.slice([[a, b]]);
  if (v instanceof BigInt64Array) return Float64Array.from(v, (x, i) => safe(x, `${ds.path}[${a + i}]`));
  if (v instanceof Int32Array) return Float64Array.from(v);
  throw new ParticleContractError("C2-DTYPE", `${ds.path} is not an int64 or int32 dataset`);
}

/** A whole int32 1-D dataset of `n` rows. */
export function int32s(ds: PDataset, n: number, code: string): Int32Array {
  exactInt(ds, 4, code);
  if (oneD(ds, code) !== n) throw new ParticleContractError(code, `${ds.path} has ${oneD(ds, code)} rows; expected ${n}`);
  if (n === 0) return new Int32Array(0);
  const v = ds.value;
  if (!(v instanceof Int32Array)) throw new ParticleContractError(code, `${ds.path} did not read as int32`);
  return v;
}

/** float64 passes through, float32 is widened, anything else is refused. */
export function floats(v: unknown, where: string): Float64Array {
  if (v instanceof Float64Array) return v;
  if (v instanceof Float32Array) return Float64Array.from(v);
  throw new ParticleContractError("C2-DTYPE", `${where}: read returned ${Object.prototype.toString.call(v)}, not float32 or float64`);
}

/** Each id's row in a sorted registry; an id the registry lacks is refused (C2-ID). */
export function registryRows(registry: Float64Array, ids: Float64Array, where: string): Int32Array {
  const rows = new Int32Array(ids.length);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i]!;
    let lo = 0;
    let hi = registry.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (registry[mid]! < id) lo = mid + 1;
      else hi = mid;
    }
    if (lo >= registry.length || registry[lo] !== id) throw new ParticleContractError("C2-ID", `${where}: particle ${id} is not in the stage registry`);
    rows[i] = lo;
  }
  return rows;
}

function floatVector(g: PNode, key: string, n: number, code: string): number[] {
  const a = g.attrs[key];
  if (a === undefined) throw new ParticleContractError(code, `${g.path} has no @${key}`);
  if (a.metadata.type !== H5T_FLOAT) throw new ParticleContractError(code, `${g.path}@${key} is not a floating-point attribute`);
  const v = vector(a.value);
  if (v.length !== n || v.some((x) => typeof x !== "number")) throw new ParticleContractError(code, `${g.path}@${key} must hold ${n} numbers`);
  return v as number[];
}

function intVector(g: PNode, key: string, n: number, code: string): number[] {
  const a = g.attrs[key];
  if (a === undefined) throw new ParticleContractError(code, `${g.path} has no @${key}`);
  if (a.metadata.type !== H5T_INTEGER || a.metadata.size !== 4) throw new ParticleContractError(code, `${g.path}@${key} is not int32`);
  const v = vector(a.value).map((x) => (typeof x === "bigint" ? Number(x) : x));
  if (v.length !== n || v.some((x) => typeof x !== "number" || !Number.isInteger(x))) throw new ParticleContractError(code, `${g.path}@${key} must hold ${n} integers`);
  return v as number[];
}

function vector(v: unknown): unknown[] {
  if (Array.isArray(v)) return v;
  if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v as unknown as ArrayLike<unknown>);
  return [v];
}

/** A scalar, or the one element of a 1-element array (§0); a longer array is refused. */
function one(v: unknown, where: string, code: string): unknown {
  const xs = vector(v);
  if (xs.length !== 1) throw new ParticleContractError(code, `${where} holds ${xs.length} values; expected one`);
  return xs[0];
}

/** A string value with the NUL padding of a fixed-length string removed. */
function trimText(v: unknown): string {
  if (typeof v !== "string") throw new ParticleContractError("C2-ATTR", `a string attribute read as ${typeof v}`);
  return v.replace(/\0+$/, "");
}

function safe(x: bigint, where: string): number {
  if (x > BigInt(Number.MAX_SAFE_INTEGER) || x < BigInt(Number.MIN_SAFE_INTEGER)) throw new ParticleContractError("C2-DTYPE", `${where} = ${x} is not a safe integer`);
  return Number(x);
}

/** Whether `parent/<a>/<b>` is named (the last link of any kind, so a malformed one is refused, not skipped). */
function childNamed(parent: PGroup, [a, b]: readonly [string, string], ctors: PCtors): boolean {
  if (!parent.keys().includes(a)) return false;
  const g = parent.get(a);
  return g instanceof ctors.Group && (g as unknown as PGroup).keys().includes(b);
}
