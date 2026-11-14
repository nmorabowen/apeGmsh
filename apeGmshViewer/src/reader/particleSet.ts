// The seeded particle set, `<stem>.particles.h5` (ADR 0134 D1; chain M, M2-S3 #2982).
//
// The layout is the fork's Contract 1, `Ladruno_implementation/232a_mpm_contracts.md`
// §1 on the fork's `ladruno` branch: one file per deck, written by apeGmsh's deck
// emitter (or a user's own script) and read by the fork's `mpm particles -file`. The
// fork owns it, and its integer `FORMAT_VERSION` is gated exactly (`(1,)`, no floor:
// ADR 0134 D1 records the exception to ADRs 0023 and 0113). This reader gives the
// points layer (M2-S4) the shape it draws before a run: positions, bodies, the grid.
//
// Rules this file keeps:
// - the gate runs first: `FORMAT = "ladruno-mpm-particles"` and `FORMAT_VERSION = 1`;
//   anything else is refused by name, never read as a guess (C1-FMT, C1-VER);
// - exact dtypes (§0): `NUM_PARTICLES` and `ID` are int64, read as `Number` with the
//   safe-integer check; any other dtype is refused (C1-DTYPE);
// - every per-particle dataset has `NUM_PARTICLES` rows (C1-N), and a dataset the
//   contract does not name refuses the file (C1-UNKNOWN: an unknown per-particle
//   field could carry meaning);
// - the attribute encodings of reader/ladrunoParticles.ts (scalar or 1-element;
//   fixed or variable-length strings), which this file shares;
// - the set is small next to a run's history (one row per particle), so it is read
//   whole at open.
//
// Browser-safe: no Node import, beyond h5open's own.

import { openH5 } from "./h5open.ts";
import {
  exactFloat,
  exactInt,
  floats,
  int32s,
  oneD,
  optionalText,
  ParticleContractError,
  readBodies,
  readGrid,
  readIds,
  requiredDataset,
  requiredFloat,
  requiredGroup,
  requiredInt,
  requiredText,
  type ParticleBody,
  type ParticleGrid,
  type PCtors,
  type PDataset,
  type PGroup,
} from "./ladrunoParticles.ts";
import type { ResultsH5 } from "./results.ts";

/** `FORMAT` of a Contract-1 file. */
export const PARTICLE_SET_FORMAT = "ladruno-mpm-particles";
/** The `FORMAT_VERSION` values this reader opens: the exact integer, no floor (ADR 0134 D1). */
export const PARTICLE_SET_VERSIONS: readonly number[] = [1] as const;

/** The per-particle datasets of §1.4 and whether each is required; any other name is C1-UNKNOWN. */
const DATASETS: Readonly<Record<string, "required" | "optional" | "mass">> = {
  ID: "required",
  POSITION: "required",
  VOLUME: "required",
  MASS: "mass",
  DENSITY: "mass",
  GIMP_HALF_LENGTH: "required",
  VELOCITY: "optional",
  STRESS: "optional",
  STRESS33: "optional",
  BODY_ID: "required",
  MATERIAL_TAG: "required",
  FLAGS: "optional",
};

/** One Contract-1 file, read whole. Arrays are row-major, one row per particle. */
export interface ParticleSet {
  path: string;
  setId: string;
  generator: string;
  ndm: 2 | 3;
  /** N */
  count: number;
  /** `PLANE_STRAIN_THICKNESS` when ndm = 2, else null */
  thickness: number | null;
  units: { length: string; force: string; time: string; system: string };
  grid: ParticleGrid;
  bodies: ParticleBody[];
  /** (N,) strictly increasing */
  ids: Float64Array;
  /** (N * ndm,) */
  positions: Float64Array;
  /** (N,) each one of `bodies[].id` */
  bodyIds: Int32Array;
  /** (N,) */
  materialTags: Int32Array;
  /** (N,) */
  volume: Float64Array;
  /** (N * ndm,) */
  gimpHalfLength: Float64Array;
  /** which of MASS and DENSITY the file carries, and its (N,) values */
  massSource: "mass" | "density";
  mass: Float64Array;
  /** `PROVENANCE`'s attributes, as written (readers ignore the ones they do not know) */
  provenance: Record<string, string>;
}

/** Open and read `<stem>.particles.h5`; a ParticleContractError names the rule a file breaks. */
export function openParticleSet(h5: ResultsH5, path: string): ParticleSet {
  const f = openH5(h5, path) as unknown as PGroup & { file_id: bigint | number; close(): unknown };
  if (Number(f.file_id) < 0) throw new ParticleContractError("C1-REQ", `HDF5 could not open ${path} (truncated, or not an HDF5 file)`);
  try {
    return readSet(h5 as unknown as PCtors, f, path);
  } finally {
    f.close();
  }
}

function readSet(ctors: PCtors, f: PGroup, path: string): ParticleSet {
  const root = { path: `${path}:/`, attrs: f.attrs };
  const format = f.attrs["FORMAT"] === undefined ? null : requiredText(root, "FORMAT", "C1-FMT");
  if (format !== PARTICLE_SET_FORMAT) throw new ParticleContractError("C1-FMT", `${path}: FORMAT is ${format === null ? "absent" : JSON.stringify(format)}, not ${JSON.stringify(PARTICLE_SET_FORMAT)}`);
  const version = requiredInt(root, "FORMAT_VERSION", "C1-VER");
  if (!PARTICLE_SET_VERSIONS.includes(version)) throw new ParticleContractError("C1-VER", `${path}: FORMAT_VERSION is ${version}; this app reads ${PARTICLE_SET_VERSIONS.join(", ")}`);
  if (f.attrs["FORMAT_VERSION"]!.metadata.size !== 4) throw new ParticleContractError("C1-DTYPE", `${path}: FORMAT_VERSION is not int32`);
  const generator = requiredText(root, "GENERATOR", "C1-GEN");
  if (generator === "") throw new ParticleContractError("C1-GEN", `${path}: GENERATOR is empty`);
  const setId = requiredText(root, "SET_ID", "C1-SETID");
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(setId)) throw new ParticleContractError("C1-SETID", `${path}: SET_ID ${JSON.stringify(setId)} is not a canonical lowercase uuid4`);
  const ndm = requiredInt(root, "NDM", "C1-NDM");
  if (ndm !== 2 && ndm !== 3) throw new ParticleContractError("C1-NDM", `${path}: NDM is ${ndm}; expected 2 or 3`);
  const nAttr = f.attrs["NUM_PARTICLES"];
  if (nAttr !== undefined && nAttr.metadata.size !== 8) throw new ParticleContractError("C1-DTYPE", `${path}: NUM_PARTICLES is not int64`);
  const count = requiredInt(root, "NUM_PARTICLES", "C1-N");
  if (count < 1) throw new ParticleContractError("C1-N", `${path}: NUM_PARTICLES is ${count}; expected >= 1`);
  let thickness: number | null = null;
  if (ndm === 2) {
    thickness = requiredFloat(root, "PLANE_STRAIN_THICKNESS", "C1-THK");
    if (!(thickness > 0) || !Number.isFinite(thickness)) throw new ParticleContractError("C1-THK", `${path}: PLANE_STRAIN_THICKNESS is ${thickness}; expected finite and > 0`);
  } else if (f.attrs["PLANE_STRAIN_THICKNESS"] !== undefined) throw new ParticleContractError("C1-THK", `${path}: PLANE_STRAIN_THICKNESS is set when NDM = 3`);
  const units = {
    length: requiredText(root, "UNIT_LENGTH", "C1-UNIT"),
    force: requiredText(root, "UNIT_FORCE", "C1-UNIT"),
    time: requiredText(root, "UNIT_TIME", "C1-UNIT"),
    system: optionalText(root, "UNIT_SYSTEM"),
  };
  if (units.length === "" || units.force === "" || units.time === "") throw new ParticleContractError("C1-UNIT", `${path}: a UNIT_* attribute is empty`);
  const massSource = requiredText(root, "MASS_SOURCE", "C1-MASS");
  if (massSource !== "mass" && massSource !== "density") throw new ParticleContractError("C1-MASS", `${path}: MASS_SOURCE is ${JSON.stringify(massSource)}; expected "mass" or "density"`);
  const sign = requiredText(root, "STRESS_SIGN", "C1-SIGN");
  if (sign !== "tension-positive") throw new ParticleContractError("C1-SIGN", `${path}: STRESS_SIGN is ${JSON.stringify(sign)}; this app reads "tension-positive"`);
  const voigt = requiredText(root, "VOIGT_ORDER", "C1-VOIGT");
  const wantVoigt = ndm === 3 ? "11,22,33,12,23,13" : "11,22,12";
  if (voigt !== wantVoigt) throw new ParticleContractError("C1-VOIGT", `${path}: VOIGT_ORDER is ${JSON.stringify(voigt)}; expected ${JSON.stringify(wantVoigt)} for NDM ${ndm}`);

  const grid = readGrid(requiredGroup(f, "GRID", ctors, "C1-REQ"), ndm, "C1-GRID");
  const bodies = readBodies(requiredGroup(f, "BODIES", ctors, "C1-REQ"), ctors, "C1-BODY");
  const particles = requiredGroup(f, "PARTICLES", ctors, "C1-REQ");
  for (const name of particles.keys()) {
    if (!Object.hasOwn(DATASETS, name)) throw new ParticleContractError("C1-UNKNOWN", `${particles.path}/${name} is not a Contract-1 dataset; a new field needs a FORMAT_VERSION bump`);
  }
  const want = massSource === "mass" ? "MASS" : "DENSITY";
  const other = massSource === "mass" ? "DENSITY" : "MASS";
  if (particles.keys().includes(other)) throw new ParticleContractError("C1-MASS", `${particles.path} carries ${other}, and MASS_SOURCE says ${massSource}`);
  const ds = (name: string) => requiredDataset(particles, name, ctors, "C1-REQ");
  const rows = (d: PDataset, width: number | null): void => {
    const shape = d.shape ?? [];
    const ok = width === null ? shape.length === 1 : shape.length === 2 && shape[1] === width;
    if (!ok) throw new ParticleContractError(width === null ? "C1-N" : "C1-SHAPE", `${d.path} has shape (${shape.join(", ")}); expected (${count}${width === null ? "" : `, ${width}`})`);
    if (shape[0] !== count) throw new ParticleContractError("C1-N", `${d.path} has ${shape[0]} rows; NUM_PARTICLES is ${count}`);
  };
  const f64 = (name: string, width: number | null): Float64Array => {
    const d = ds(name);
    exactFloat(d, 8, "C1-DTYPE");
    rows(d, width);
    const v = floats(d.value, d.path);
    if (v.some((x) => !Number.isFinite(x))) throw new ParticleContractError("C1-FINITE", `${d.path} holds a value that is not finite`);
    return v;
  };
  const idDs = ds("ID");
  exactInt(idDs, 8, "C1-DTYPE");
  rows(idDs, null);
  const ids = readIds(idDs, 0, oneD(idDs, "C1-N"));
  for (let i = 0; i < ids.length; i++) {
    if (ids[i]! < 0 || (i > 0 && !(ids[i]! > ids[i - 1]!))) throw new ParticleContractError("C1-ID", `${idDs.path} is not strictly increasing and >= 0 at row ${i}`);
  }
  const positions = f64("POSITION", ndm);
  const volume = f64("VOLUME", null);
  if (volume.some((v) => !(v > 0))) throw new ParticleContractError("C1-VOL", `${particles.path}/VOLUME holds a value that is not > 0`);
  const mass = f64(want, null);
  if (mass.some((v) => !(v > 0))) throw new ParticleContractError("C1-MASS", `${particles.path}/${want} holds a value that is not > 0`);
  const gimpHalfLength = f64("GIMP_HALF_LENGTH", ndm);
  const bodyIds = int32s(ds("BODY_ID"), count, "C1-DTYPE");
  const known = new Set(bodies.map((b) => b.id));
  bodyIds.forEach((b, i) => {
    if (!known.has(b)) throw new ParticleContractError("C1-BODYREF", `${particles.path}/BODY_ID[${i}] = ${b} is not in BODIES/ID`);
  });
  const materialTags = int32s(ds("MATERIAL_TAG"), count, "C1-DTYPE");
  if (materialTags.some((t) => t <= 0)) throw new ParticleContractError("C1-MAT", `${particles.path}/MATERIAL_TAG holds a tag that is not > 0`);
  for (const opt of ["VELOCITY", "STRESS", "STRESS33"] as const) if (particles.keys().includes(opt)) f64(opt, opt === "VELOCITY" ? ndm : opt === "STRESS" ? (ndm === 3 ? 6 : 3) : null);
  if (ndm === 3 && particles.keys().includes("STRESS33")) throw new ParticleContractError("C1-STRESS33", `${particles.path}/STRESS33 is set when NDM = 3 (sigma33 is in STRESS)`);
  if (particles.keys().includes("FLAGS")) {
    const d = ds("FLAGS");
    const m = d.metadata;
    if (m.type !== 0 || m.size !== 4 || m.signed) throw new ParticleContractError("C1-DTYPE", `${d.path} is not uint32`);
    rows(d, null);
    const v = d.value as ArrayLike<number>;
    for (let i = 0; i < v.length; i++) if (v[i] !== 0) throw new ParticleContractError("C1-FLAGS", `${d.path}[${i}] sets a bit; v1 defines none`);
  }
  const provenance: Record<string, string> = {};
  if (f.keys().includes("PROVENANCE")) {
    const p = requiredGroup(f, "PROVENANCE", ctors, "C1-PROV");
    // §1.5: readers ignore the attributes they do not know, so a non-string one is skipped.
    for (const [k, a] of Object.entries(p.attrs)) if (a.metadata.type === 3) provenance[k] = requiredText(p, k, "C1-PROV");
  }
  return { path, setId, generator, ndm, count, thickness, units, grid, bodies, ids, positions, bodyIds, materialTags, volume, gimpHalfLength, massSource, mass, provenance };
}
