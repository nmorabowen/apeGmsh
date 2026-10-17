// Consistent unit systems from `baseUnits` (baseUnits/_factors.py and
// baseUnits/_make_system.py): exact NIST HB 44 / CGPM 1901 double-precision
// constants for converting contour values and labelling display units.
//
// Every conversion is a positive linear factor `k = (F_src / F_dst)^f * (L_src / L_dst)^l * (T_src / T_dst)^t`.
// Because the normalized colour ramp maps `[min, max] -> [0, 1]`, scaling by
// `k > 0` leaves the GPU vertex colours unchanged and updates only the legend
// numbers and unit symbol.

import type { ContourSource, UnitDisplay, UnitSystemId } from "./types.ts";

export interface UnitSystemSpec {
  readonly id: UnitSystemId;
  /** `baseUnits.BASE` label */
  readonly label: string;
  /** Base force in SI Newtons (N) */
  readonly F: number;
  /** Base length in SI metres (m) */
  readonly L: number;
  /** Base time in SI seconds (s) */
  readonly T: number;
  /** Unit symbols for basic quantities in this system */
  readonly symbols: {
    readonly force: string;
    readonly length: string;
    readonly time: string;
    readonly pressure: string;
  };
}

export const UNIT_SYSTEM_IDS: readonly UnitSystemId[] = [
  "kN_m_s",
  "tf_m_s",
  "kgf_m_s",
  "N_mm_s",
  "N_m_s",
  "kip_in_s",
  "dyne_cm_s",
];

export const UNIT_DISPLAY_OPTIONS: readonly { readonly value: UnitDisplay; readonly label: string }[] = [
  { value: "as_stored", label: "as stored (raw)" },
  { value: "tf_m_s", label: "tf-m-s (tf, m, s)" },
  { value: "kN_m_s", label: "kN-m-tonne-s (kN, m, s)" },
  { value: "kgf_m_s", label: "kgf-m-s (kgf, m, s)" },
  { value: "N_mm_s", label: "N-mm-tonne-s (N, mm, s)" },
  { value: "N_m_s", label: "N-m-kg-s (N, m, s)" },
  { value: "kip_in_s", label: "kip-in-s (kip, in, s)" },
  { value: "dyne_cm_s", label: "dyne-cm-g-s (dyne, cm, s)" },
];

export const UNIT_SYSTEMS: Readonly<Record<UnitSystemId, UnitSystemSpec>> = {
  kN_m_s: {
    id: "kN_m_s",
    label: "kN-m-tonne-s",
    F: 1e3,
    L: 1.0,
    T: 1.0,
    symbols: { force: "kN", length: "m", time: "s", pressure: "kPa" },
  },
  tf_m_s: {
    id: "tf_m_s",
    label: "tf-m-s",
    F: 9806.65,
    L: 1.0,
    T: 1.0,
    symbols: { force: "tf", length: "m", time: "s", pressure: "tf/m²" },
  },
  kgf_m_s: {
    id: "kgf_m_s",
    label: "kgf-m-s",
    F: 9.80665,
    L: 1.0,
    T: 1.0,
    symbols: { force: "kgf", length: "m", time: "s", pressure: "kgf/m²" },
  },
  N_mm_s: {
    id: "N_mm_s",
    label: "N-mm-tonne-s",
    F: 1.0,
    L: 1e-3,
    T: 1.0,
    symbols: { force: "N", length: "mm", time: "s", pressure: "MPa" },
  },
  N_m_s: {
    id: "N_m_s",
    label: "N-m-kg-s",
    F: 1.0,
    L: 1.0,
    T: 1.0,
    symbols: { force: "N", length: "m", time: "s", pressure: "Pa" },
  },
  kip_in_s: {
    id: "kip_in_s",
    label: "kip-in-s",
    F: 4448.2216152605,
    L: 0.0254,
    T: 1.0,
    symbols: { force: "kip", length: "in", time: "s", pressure: "ksi" },
  },
  dyne_cm_s: {
    id: "dyne_cm_s",
    label: "dyne-cm-g-s",
    F: 1e-5,
    L: 1e-2,
    T: 1.0,
    symbols: { force: "dyne", length: "cm", time: "s", pressure: "dyne/cm²" },
  },
};

export type FieldDimensionKind =
  | "dimensionless"
  | "angle"
  | "frequency"
  | "length"
  | "velocity"
  | "acceleration"
  | "force"
  | "moment"
  | "shell_force_per_length"
  | "shell_moment_per_length"
  | "pressure";

interface DimensionExponents {
  readonly kind: FieldDimensionKind;
  /** Exponents in F^f * L^l * T^t */
  readonly f: number;
  readonly l: number;
  readonly t: number;
}

const DIM_SPECS: Readonly<Record<FieldDimensionKind, DimensionExponents>> = {
  dimensionless: { kind: "dimensionless", f: 0, l: 0, t: 0 },
  angle: { kind: "angle", f: 0, l: 0, t: 0 },
  frequency: { kind: "frequency", f: 0, l: 0, t: -1 },
  length: { kind: "length", f: 0, l: 1, t: 0 },
  velocity: { kind: "velocity", f: 0, l: 1, t: -1 },
  acceleration: { kind: "acceleration", f: 0, l: 1, t: -2 },
  force: { kind: "force", f: 1, l: 0, t: 0 },
  moment: { kind: "moment", f: 1, l: 1, t: 0 },
  shell_force_per_length: { kind: "shell_force_per_length", f: 1, l: -1, t: 0 },
  shell_moment_per_length: { kind: "shell_moment_per_length", f: 1, l: 0, t: 0 },
  pressure: { kind: "pressure", f: 1, l: -2, t: 0 },
};

/**
 * Classify a contour field into its physical dimension in `(F, L, T)`.
 * Mode shapes (`stageKind === "mode"`) are unscaled eigenvectors and stay dimensionless.
 */
export function classifyFieldDimension(
  field: string,
  source: ContourSource,
  stageKind?: "static" | "transient" | "mode",
): DimensionExponents {
  if (stageKind === "mode") return DIM_SPECS.dimensionless;
  const name = field.toLowerCase();
  if (name === "rotation") return DIM_SPECS.angle;
  if (name.endsWith("_hz") || name === "frequency") return DIM_SPECS.frequency;
  if (
    name === "footfall_r" ||
    name.startsWith("footfall_r_") ||
    name === "footfall_utilization" ||
    name.startsWith("strain") ||
    name.startsWith("damage")
  ) {
    return DIM_SPECS.dimensionless;
  }
  if (name === "displacement" || name.startsWith("displacement_")) return DIM_SPECS.length;
  if (name === "velocity" || name.startsWith("velocity_")) return DIM_SPECS.velocity;
  if (name === "acceleration" || name.startsWith("acceleration_")) return DIM_SPECS.acceleration;
  if (
    name === "reaction_force" ||
    name === "axial_force" ||
    name === "shear" ||
    name.startsWith("shear_") ||
    name === "rib_shear" ||
    name.startsWith("rib_shear_")
  ) {
    return DIM_SPECS.force;
  }
  if (
    name === "reaction_moment" ||
    name === "torsion" ||
    name === "rib_moment_pos" ||
    name === "rib_moment_neg" ||
    name.startsWith("rib_moment_")
  ) {
    return DIM_SPECS.moment;
  }
  if (name === "wood_armer_pos" || name === "wood_armer_neg" || name.startsWith("wood_armer_")) {
    return DIM_SPECS.shell_moment_per_length;
  }
  if (name === "membrane_force" || name === "transverse_shear") {
    return DIM_SPECS.shell_force_per_length;
  }
  if (name === "bending_moment") {
    return source === "element" ? DIM_SPECS.shell_moment_per_length : DIM_SPECS.moment;
  }
  if (name === "stress" || name.startsWith("stress_") || name.includes("pressure")) {
    return DIM_SPECS.pressure;
  }
  return DIM_SPECS.dimensionless;
}

/** Format the unit symbol for `kind` in `sys`. */
export function unitSymbolFor(kind: FieldDimensionKind, sys: UnitSystemSpec): string | null {
  const { force: F, length: L, time: T, pressure: P } = sys.symbols;
  switch (kind) {
    case "dimensionless":
      return null;
    case "angle":
      return "rad";
    case "frequency":
      return "Hz";
    case "length":
      return L;
    case "velocity":
      return `${L}/${T}`;
    case "acceleration":
      return `${L}/${T}²`;
    case "force":
      return F;
    case "moment":
      return `${F}·${L}`;
    case "shell_force_per_length":
      return `${F}/${L}`;
    case "shell_moment_per_length":
      return `${F}·${L}/${L}`;
    case "pressure":
      return P;
  }
}

/** Parse a `/meta/@units` or `INFO/PROVENANCE@units` attribute value into a canonical `UnitSystemId`, or null. */
export function parseDeclaredUnits(raw: unknown): UnitSystemId | null {
  if (typeof raw === "string" && UNIT_SYSTEM_IDS.includes(raw as UnitSystemId)) {
    return raw as UnitSystemId;
  }
  return null;
}

/**
 * Compute the multiplicative conversion factor and display unit symbol for a
 * contour field when converting from `sourceUnits` to `targetUnits`.
 */
export function resolveFieldUnits(
  field: string,
  source: ContourSource,
  sourceUnits: UnitSystemId | null,
  targetUnits: UnitDisplay,
  stageKind?: "static" | "transient" | "mode",
  declaredUnits: UnitSystemId | null = null,
): { factor: number; unit: string | null; note: string } {
  if (targetUnits === "as_stored") {
    return {
      factor: 1,
      unit: null,
      note: declaredUnits
        ? `Units: raw values (declared as ${UNIT_SYSTEMS[declaredUnits].label} in file).`
        : "Units: not stored in the results file.",
    };
  }
  const effectiveSource = declaredUnits ?? sourceUnits;
  if (effectiveSource === null) {
    return {
      factor: 1,
      unit: null,
      note: "Units: not declared in file — select the source unit system (from …) to convert.",
    };
  }
  const src = UNIT_SYSTEMS[effectiveSource];
  const dst = UNIT_SYSTEMS[targetUnits];
  const dim = classifyFieldDimension(field, source, stageKind);
  const factor =
    Math.pow(src.F / dst.F, dim.f) *
    Math.pow(src.L / dst.L, dim.l) *
    Math.pow(src.T / dst.T, dim.t);
  const unit = unitSymbolFor(dim.kind, dst);
  let note: string;
  if (declaredUnits !== null) {
    note =
      effectiveSource === targetUnits
        ? `Units: ${dst.label} (declared in file).`
        : `Units: ${dst.label} (converted from declared ${src.label}).`;
  } else {
    note =
      effectiveSource === targetUnits
        ? `Units: ${dst.label} (file native).`
        : `Units: ${dst.label} (converted from ${src.label}).`;
  }
  return { factor, unit, note };
}
