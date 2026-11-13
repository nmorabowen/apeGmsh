// What an open set offers a view spec (ADR 0128 D4 `--describe`): its files
// and their pairing, the results stages with their steps, times, fields and
// components, the physical groups, the bounding box, and the spec's own
// closed sets (cameras, layers, colormaps). An agent reads it to write a
// spec that will not refuse; ./compile.ts checks a shot against the same
// description, so the two can never disagree. Pure: it reads the store's
// State (and, for the box, the drawn positions), never the DOM.

import { pairingNotice } from "../state/results.ts";
import { elementContourFields, fieldsOf, MAGNITUDE, type ElementComponent } from "../state/contour.ts";
import type { BlobStore } from "../state/blobs.ts";
import { drawnNotices, noticesOf, refusalsOf, resultsPairing, warningsOf } from "../state/selectors.ts";
import { CMAP_NAMES, type State } from "../state/types.ts";
import { THEME_MODES } from "../theme/tokens.ts";
import { APP_LAYERS, BACKGROUNDS, CHROMES, GAUSS_MODES, MODEL_LAYERS, SPEC_CAMERAS, SPEC_VERSION } from "./spec.ts";

export interface DescribedComponent {
  /** the dataset name (`displacement_z`, `DISPLACEMENT/Uz`), or `magnitude` */
  key: string;
  /** what the legend shows (`z`, `magnitude`, `Uz`, `[0]`) */
  label: string;
  /**
   * VA6d (#2766): set on a scalar derived from the field's tensor
   * (`von_mises_stress`), not a dataset: what it is computed from, and how
   */
  derived?: string;
}

export interface DescribedField {
  name: string;
  source: "nodal" | "element";
  components: DescribedComponent[];
  /** why the field cannot be drawn (a refused component, element groups that disagree), or null */
  refused: string | null;
  /** VA6d: the derived scalars the field's tensor cannot give (incomplete or ambiguous), each with why; absent when none */
  refusedComponents?: { key: string; reason: string }[];
}

export interface DescribedStage {
  index: number;
  id: string;
  /** the name shown: a `.ladruno`'s display name, else the program name */
  name: string;
  /** the program's stage name (W4d #2843); a spec names a stage by either, see compile.ts `stageOf` */
  programName: string;
  kind: "static" | "transient" | "mode";
  steps: number;
  times: number[];
  fields: DescribedField[];
}

export interface SetDescription {
  spec: typeof SPEC_VERSION;
  files: { model: string | null; geometry: string | null; results: string | null };
  /** the results pairing (ADR 0112): `paired` draws them; any other state names why not */
  pairing: { state: string; reason: string | null };
  /** results stages: only for a paired file (an unpaired file's stages cannot be drawn on this model) */
  stages: DescribedStage[];
  groups: { path: string; name: string }[];
  /** the axis-aligned box of every drawn vertex (undeformed), or null with nothing drawn */
  bbox: { min: [number, number, number]; max: [number, number, number] } | null;
  /** the banner's lines when the set is opened, before any shot */
  banner: string[];
  offers: {
    cameras: readonly string[];
    themes: readonly string[];
    chromes: readonly string[];
    layers: { app: readonly string[]; model: readonly string[] };
    backgrounds: readonly string[];
    cmaps: readonly string[];
    gauss: readonly string[];
  };
}

/** The magnitude's name in a spec: the legend's word, never the internal key. */
export const SPEC_MAGNITUDE = "magnitude";

/** The banner lines the state words alone, in the banner's order (panels/banner.ts): refusals, notices, warnings. */
export function stateBannerLines(s: State): string[] {
  return [...refusalsOf(s), ...noticesOf(s), ...warningsOf(s)];
}

/**
 * Every banner line of a drawn view, in the banner's order: the state's
 * lines, with the notices only the drawn contour's element ids can word
 * (VF #2940, selectors `drawnNotices`: a derived contour's plane-recovery
 * warning) after the state's notices. A still's banner and its manifest
 * entry carry these; the set's description (before any shot) and a refused
 * shot (nothing drawn) carry `stateBannerLines`.
 */
export function bannerLines(s: State, blobs: BlobStore): string[] {
  return [...refusalsOf(s), ...noticesOf(s), ...drawnNotices(s, blobs), ...warningsOf(s)];
}

/** The box of a set of position arrays (3 floats per vertex); null when they hold no finite vertex. */
export function boxOf(positions: readonly ArrayLike<number>[]): SetDescription["bbox"] {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let any = false;
  for (const p of positions) {
    for (let i = 0; i + 2 < p.length; i += 3) {
      const x = p[i]!, y = p[i + 1]!, z = p[i + 2]!;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      any = true;
      min[0] = Math.min(min[0], x), min[1] = Math.min(min[1], y), min[2] = Math.min(min[2], z);
      max[0] = Math.max(max[0], x), max[1] = Math.max(max[1], y), max[2] = Math.max(max[2], z);
    }
  }
  return any ? { min, max } : null;
}

const componentKey = (key: string): string => (key === MAGNITUDE ? SPEC_MAGNITUDE : key);

/** What a derived component says it is (Python's `_derived._von_mises`, on v1's `plane="auto"` tensor). */
function derivedNote(c: ElementComponent): string {
  const d = c.derived!;
  const strain = d.tensor.family === "strain" ? ", the shear components halved (engineering shear)" : "";
  // F1: v1's `plane="auto"`, per element; the manifest counts the elements left at 0 (`contour.derived.plane`).
  const zz = `${d.tensor.family}_zz`;
  const plane = d.tensor.groups.some((g) => g.slots[2] === null)
    ? `; ${zz}, not recorded, is recovered per element from its plane type and Poisson's ratio (${d.tensor.family === "stress" ? "ν(σxx + σyy) for plane strain, 0 for plane stress" : "-ν/(1-ν)(εxx + εyy) for plane stress, 0 for plane strain"}), and left at 0 for an element the model does not classify`
    : "";
  return `the von Mises equivalent of the ${d.tensor.family} tensor (${d.tensor.components.join(", ")}) at each Gauss point${strain}${plane}`;
}

export function describeSet(s: State, positions: readonly ArrayLike<number>[]): SetDescription {
  const p = resultsPairing(s);
  const ra = s.artifacts.results;
  // A results file that could not be read pairs with nothing; its read error is the reason.
  const pairing =
    ra?.status === "failed"
      ? { state: "failed", reason: `${ra.path} could not be read: ${ra.error ?? "unknown error"}` }
      : { state: p.state, reason: s.results ? pairingNotice(p, s.results.path.split(/[\\/]/).pop() ?? s.results.path) : null };
  const stages: DescribedStage[] =
    p.state === "paired" && s.results
      ? s.results.stages.map((st, index) => ({
          index,
          id: st.id,
          name: st.name,
          programName: st.programName,
          kind: st.kind,
          steps: st.time.length,
          times: [...st.time],
          fields: [
            ...fieldsOf(st).map((f) => ({
              name: f.name,
              source: "nodal" as const,
              components: f.components.map((c) => ({ key: componentKey(c.key), label: c.label })),
              refused: f.magnitudeRefused,
            })),
            ...elementContourFields(st.gauss).map((f) => ({
              name: f.name,
              source: "element" as const,
              components: f.components.map((c) => (c.derived ? { key: c.key, label: c.label, derived: derivedNote(c) } : { key: c.key, label: c.label })),
              refused: f.refused,
              ...(f.derivedRefused?.length ? { refusedComponents: f.derivedRefused.map((r) => ({ key: r.name, reason: r.reason })) } : {}),
            })),
          ],
        }))
      : [];
  return {
    spec: SPEC_VERSION,
    files: { model: s.artifacts.model?.path ?? null, geometry: s.artifacts.geometry?.path ?? null, results: s.artifacts.results?.path ?? null },
    pairing,
    stages,
    groups: Object.values(s.decls)
      .filter((d) => d.kind === "physical_group")
      .map((d) => ({ path: d.path, name: d.name })),
    bbox: boxOf(positions),
    banner: stateBannerLines(s),
    offers: {
      cameras: SPEC_CAMERAS,
      themes: THEME_MODES,
      chromes: CHROMES,
      layers: { app: APP_LAYERS, model: MODEL_LAYERS },
      backgrounds: BACKGROUNDS,
      cmaps: CMAP_NAMES,
      gauss: GAUSS_MODES,
    },
  };
}
