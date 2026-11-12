// The view spec `apegmsh-view/1` (ADR 0128 D1, D2, D8, D9): its closed key
// set, and the validator that turns a parsed JSON document into a normalised
// spec or refuses it. Pure: no DOM, no Electron, no file access; main runs it
// before any window opens, and the tests run it in Node.
//
// D2: the whole spec is validated before any shot is taken. An unknown key,
// a wrong type, a value outside its closed set or an unknown version refuses
// the whole spec; the refusal names the key (a JSON path such as
// `shots[1].contour.cmap`) and no PNG is written. What depends on the files
// (a stage the results do not have, a field a stage lacks) is a shot
// refusal instead, decided by ./compile.ts against the set's description.
//
// Every key has one state it maps to: a key left out takes its documented
// default here, so the normalised spec states every choice explicitly.

import { CMAP_NAMES, type ContourCmap, type GaussMode, type Vec3 } from "../state/types.ts";
import { THEME_MODES, type ThemeMode } from "../theme/tokens.ts";

export const SPEC_VERSION = "apegmsh-view/1";

/**
 * The named camera views a spec may ask for (ADR 0128 D1). `home` is the
 * app's own. `fit-selection` is not in v1: a spec cannot make a selection,
 * so the app would draw the `fit` view under that name (INV-4; review of #2660).
 */
export const SPEC_CAMERAS = ["iso", "+x", "-x", "+y", "-y", "top", "bottom", "fit"] as const;
export type SpecCameraName = (typeof SPEC_CAMERAS)[number];

/** D9: the app panels a `chrome: "app"` shot can leave out of the layout, each by name. */
export const APP_LAYERS = ["header", "phase", "viewcube", "left_rail", "rail", "windows", "banner"] as const;
/** D9: the still-only layers a `chrome: "model"` shot can draw into the image. */
export const MODEL_LAYERS = ["scale_bar", "axes", "banner", "caption"] as const;
export type AppLayer = (typeof APP_LAYERS)[number];
export type ModelLayer = (typeof MODEL_LAYERS)[number];

export const CHROMES = ["app", "model"] as const;
export type Chrome = (typeof CHROMES)[number];
export const BACKGROUNDS = ["theme", "transparent"] as const;
export type Background = (typeof BACKGROUNDS)[number];

export const GAUSS_MODES: readonly GaussMode[] = ["cell", "cell_averaged", "node", "node_discrete"];
/** VU16: `user` is a state-free ON_NODES user result of the set's `.user.h5` (`field` its name, `component` one of its columns). */
export const CONTOUR_SOURCES = ["nodal", "element", "user"] as const;

/** The size bounds of a still, in CSS pixels (the render window is at device scale 1). */
export const MIN_SIZE = 64;
export const MAX_SIZE = 8192;
export const DEFAULT_SIZE: readonly [number, number] = [1600, 1000];
/** The app's limit of active section planes (state/clip.ts MAX_ACTIVE_PLANES). */
export const MAX_CLIP = 6;

/** A shot's name is its PNG's file name: letters, digits, `_`, `.`, `-`, starting with a letter or digit. */
const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/;

export type SpecStep = "first" | "last" | number;
export type SpecCamera = SpecCameraName | { eye: Vec3; target: Vec3; up: Vec3 };
export type SpecRange = { mode: "history" } | { mode: "step" } | { mode: "fixed"; min: number; max: number };

export interface SpecContour {
  field: string;
  component: string;
  /**
   * null: the field's own kind (a nodal field, else an element field);
   * ./compile.ts refuses a name both kinds have. VU16: `user` names a
   * state-free user result of the set's `.user.h5`; it is never inferred, and
   * the shot shows no stage, step, combination, compare or deformed shape.
   */
  source: "nodal" | "element" | "user" | null;
  range: SpecRange;
  cmap: ContourCmap;
  gauss: GaussMode;
}

export interface SpecDeform {
  /**
   * The deformed shape's scale; null when the spec leaves it out (G3, VA6c
   * #2765): the render mode then takes the auto scale of the step drawn
   * (state/deform.ts `autoDeformScale`), and the manifest records it with
   * `scale_auto`.
   */
  scale: number | null;
  ghost: boolean;
}

export interface SpecClip {
  /** a unit-free direction; an axis name is normalised to its unit vector */
  normal: Vec3;
  /** the plane is { x : normal . x = offset }, with the normal as given (state/clip.ts) */
  offset: number;
  flip: boolean;
}

/** VU18: a manifest row as a shot names it: both must match the row main reads (D9.2). */
export interface SpecVariant {
  index: number;
  label: string;
}

export interface SpecShot {
  name: string;
  caption: string | null;
  /** a results stage, by name or by 0-based index into the results file's stages; null is the mesh */
  stage: string | number | null;
  /**
   * VU7b-2 (ADR 0130 D1): a User Combination of the set's `.user.h5`, by name,
   * as the state shown instead of a stage; it has no step. Absent or null: none.
   * A snapshot never writes one (spec/snapshot.ts names a combination phase as
   * outside what it can state).
   */
  combination?: string | null;
  /**
   * VU10 (ADR 0132 D9.5): a variant of `files.variants`, by label, opened as
   * the compare set; the contour then shows the window variant minus it, at
   * the shot's stage and step. Absent or null: none. Needs a nodal contour of a
   * stage; never with a combination or a deformed shape.
   */
  compare?: string | null;
  /**
   * VU18 (ADR 0132 D9.2): a row of `files.variants`, named by its index and
   * label as the page names one (never a path); the shot draws that row's set
   * instead of the spec's. Main re-reads the manifest for every shot that
   * names one (D9.3) and refuses the shot when the row no longer answers to
   * both. Absent or null: the spec's own set.
   */
  variant?: SpecVariant;
  step: SpecStep;
  camera: SpecCamera;
  colour_by: "group" | "role" | "contour";
  contour: SpecContour | null;
  deform: SpecDeform | null;
  clip: SpecClip[];
  /** physical group paths (`/physical/Columns`); [] shows every group */
  isolate: string[];
  chrome: Chrome;
  /** every layer of the chrome, resolved (D9): what the still draws, and what the render result records */
  layers: Record<string, boolean>;
  background: Background;
}

export interface ViewSpec {
  spec: typeof SPEC_VERSION;
  /**
   * As written in the spec; main resolves a relative path against the spec
   * file's folder. VU10: `variants` is a `<study>.variants.json` whose rows
   * include `model`; the window set is that row, and a shot may compare.
   */
  files: { model: string; results: string | null; variants?: string | null };
  size: [number, number];
  theme: ThemeMode;
  shots: SpecShot[];
}

export type Validated = { ok: true; spec: ViewSpec } | { ok: false; key: string; reason: string };

class SpecError extends Error {
  readonly key: string;
  readonly reason: string;
  constructor(key: string, reason: string) {
    super(`${key}: ${reason}`);
    this.key = key;
    this.reason = reason;
  }
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const show = (v: unknown): string => (v === undefined ? "nothing" : JSON.stringify(v));

function object(v: unknown, key: string, allowed: readonly string[]): Record<string, unknown> {
  if (!isObject(v)) throw new SpecError(key, `must be an object; got ${show(v)}`);
  for (const k of Object.keys(v)) {
    if (!allowed.includes(k)) throw new SpecError(key ? `${key}.${k}` : k, `is not a key of ${SPEC_VERSION} here; the keys are ${allowed.join(", ")}`);
  }
  return v;
}

function oneOf<T extends string>(v: unknown, key: string, set: readonly T[]): T {
  if (typeof v !== "string" || !(set as readonly string[]).includes(v)) throw new SpecError(key, `must be one of ${set.join(", ")}; got ${show(v)}`);
  return v as T;
}

function finite(v: unknown, key: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) throw new SpecError(key, `must be a finite number; got ${show(v)}`);
  return v;
}

function bool(v: unknown, key: string): boolean {
  if (typeof v !== "boolean") throw new SpecError(key, `must be true or false; got ${show(v)}`);
  return v;
}

function text(v: unknown, key: string): string {
  if (typeof v !== "string" || v.length === 0) throw new SpecError(key, `must be a non-empty string; got ${show(v)}`);
  return v;
}

function vec3(v: unknown, key: string): Vec3 {
  if (!Array.isArray(v) || v.length !== 3) throw new SpecError(key, `must be three numbers [x, y, z]; got ${show(v)}`);
  return [finite(v[0], `${key}[0]`), finite(v[1], `${key}[1]`), finite(v[2], `${key}[2]`)];
}

const AXIS_NORMALS: Readonly<Record<string, Vec3>> = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

function camera(v: unknown, key: string): SpecCamera {
  if (typeof v === "string") return oneOf(v, key, SPEC_CAMERAS);
  const o = object(v, key, ["eye", "target", "up"]);
  for (const k of ["eye", "target", "up"]) if (!(k in o)) throw new SpecError(`${key}.${k}`, "is required in an explicit camera");
  const pose = { eye: vec3(o["eye"], `${key}.eye`), target: vec3(o["target"], `${key}.target`), up: vec3(o["up"], `${key}.up`) };
  if (pose.eye.every((x, i) => x === pose.target[i])) throw new SpecError(`${key}.eye`, "must differ from the target");
  if (pose.up.every((x) => x === 0)) throw new SpecError(`${key}.up`, "must not be the zero vector");
  return pose;
}

function range(v: unknown, key: string): SpecRange {
  if (v === "history" || v === "step") return { mode: v };
  if (typeof v === "string") throw new SpecError(key, `must be "history", "step" or { "min": a, "max": b }; got ${show(v)}`);
  const o = object(v, key, ["min", "max"]);
  const min = finite(o["min"], `${key}.min`);
  const max = finite(o["max"], `${key}.max`);
  if (!(max > min)) throw new SpecError(`${key}.max`, `must be greater than min (${min}); got ${max}`);
  return { mode: "fixed", min, max };
}

function contour(v: unknown, key: string): SpecContour {
  const o = object(v, key, ["field", "component", "source", "range", "cmap", "gauss"]);
  return {
    field: text(o["field"], `${key}.field`),
    component: text(o["component"], `${key}.component`),
    source: o["source"] === undefined ? null : oneOf(o["source"], `${key}.source`, CONTOUR_SOURCES),
    range: o["range"] === undefined ? { mode: "history" } : range(o["range"], `${key}.range`),
    cmap: o["cmap"] === undefined ? "jet" : oneOf(o["cmap"], `${key}.cmap`, CMAP_NAMES),
    gauss: o["gauss"] === undefined ? "cell" : oneOf(o["gauss"], `${key}.gauss`, GAUSS_MODES),
  };
}

function deform(v: unknown, key: string): SpecDeform {
  const o = object(v, key, ["scale", "ghost"]);
  // G3: a scale left out is the auto scale (null), never 1.
  const scale = o["scale"] === undefined ? null : finite(o["scale"], `${key}.scale`);
  if (scale !== null && !(scale > 0)) throw new SpecError(`${key}.scale`, `must be positive; got ${scale}`);
  const ghost = o["ghost"] === undefined ? false : bool(o["ghost"], `${key}.ghost`);
  return { scale, ghost };
}

function clip(v: unknown, key: string): SpecClip[] {
  if (!Array.isArray(v)) throw new SpecError(key, `must be a list of planes; got ${show(v)}`);
  if (v.length > MAX_CLIP) throw new SpecError(key, `holds ${v.length} planes; the app cuts with at most ${MAX_CLIP}`);
  return v.map((p, i) => {
    const k = `${key}[${i}]`;
    const o = object(p, k, ["normal", "offset", "flip"]);
    const raw = o["normal"];
    const normal = typeof raw === "string" ? AXIS_NORMALS[oneOf(raw, `${k}.normal`, ["x", "y", "z"])]! : vec3(raw, `${k}.normal`);
    if (normal.every((x) => x === 0)) throw new SpecError(`${k}.normal`, "must not be the zero vector");
    return { normal, offset: finite(o["offset"], `${k}.offset`), flip: o["flip"] === undefined ? false : bool(o["flip"], `${k}.flip`) };
  });
}

function step(v: unknown, key: string): SpecStep {
  if (v === "first" || v === "last") return v;
  if (typeof v === "number" && Number.isInteger(v) && v >= 0) return v;
  throw new SpecError(key, `must be "first", "last" or a step index (an integer from 0); got ${show(v)}`);
}

/** VU18: `{ index, label }`, both required; whether the row answers to them is main's to say, per shot (D9.2, D9.3). */
function variantRow(v: unknown, key: string): SpecVariant {
  const o = object(v, key, ["index", "label"]);
  for (const k of ["index", "label"]) if (!(k in o)) throw new SpecError(`${key}.${k}`, "is required: a variant row is named by its index and its label");
  const index = o["index"];
  if (typeof index !== "number" || !Number.isInteger(index) || index < 0) throw new SpecError(`${key}.index`, `must be a row index of files.variants (an integer from 0); got ${show(index)}`);
  return { index, label: text(o["label"], `${key}.label`) };
}

function stage(v: unknown, key: string): string | number {
  if (typeof v === "string") return v;
  if (typeof v === "number" && Number.isInteger(v) && v >= 0) return v;
  throw new SpecError(key, `must be a results stage name or its index (an integer from 0); got ${show(v)}`);
}

/** D9: the layer keys a chrome accepts. `banner` is a panel in app mode and a still-only line in model mode. */
const layersOf = (chrome: Chrome): readonly string[] => (chrome === "app" ? APP_LAYERS : MODEL_LAYERS);

function layerKeys(v: unknown, key: string): Record<string, boolean> {
  const o = object(v, key, [...new Set([...APP_LAYERS, ...MODEL_LAYERS])]);
  const out: Record<string, boolean> = {};
  for (const [k, x] of Object.entries(o)) out[k] = bool(x, `${key}.${k}`);
  return out;
}

/**
 * D9: every layer of `chrome`, on or off. App panels default on. In model
 * mode the scale bar defaults on whenever a contour is drawn (a contour
 * without its scale misleads), the axes on, the banner and caption off.
 */
export function resolveLayers(chrome: Chrome, given: Record<string, boolean>, key: string, drawsContour: boolean, hasCaption: boolean): Record<string, boolean> {
  const accepted = layersOf(chrome);
  for (const k of Object.keys(given)) {
    if (!accepted.includes(k)) throw new SpecError(`${key}.${k}`, `is not a layer of chrome "${chrome}"; its layers are ${accepted.join(", ")}`);
  }
  if (chrome === "app") return Object.fromEntries(APP_LAYERS.map((k) => [k, given[k] ?? true]));
  const out: Record<ModelLayer, boolean> = {
    scale_bar: given["scale_bar"] ?? drawsContour,
    axes: given["axes"] ?? true,
    banner: given["banner"] ?? false,
    caption: given["caption"] ?? false,
  };
  if (out.scale_bar && !drawsContour) throw new SpecError(`${key}.scale_bar`, "is on, but the shot draws no contour to scale");
  if (out.caption && !hasCaption) throw new SpecError(`${key}.caption`, "is on, but the shot has no caption");
  return out;
}

interface Defaults {
  chrome: Chrome;
  layers: Record<string, boolean> | null;
  layersKey: string;
  background: Background;
  /** VU10: whether the spec names `files.variants` (a shot may compare only then) */
  variants: boolean;
}

function shot(v: unknown, key: string, d: Defaults): SpecShot {
  const o = object(v, key, ["name", "caption", "stage", "combination", "compare", "variant", "step", "camera", "colour_by", "contour", "deform", "clip", "isolate", "chrome", "layers", "background"]);
  const name = text(o["name"], `${key}.name`);
  if (!NAME.test(name)) throw new SpecError(`${key}.name`, `is the PNG's file name: letters, digits, _, . and -, starting with a letter or digit, at most 100 characters; got ${show(name)}`);
  const c = o["contour"] === undefined ? null : contour(o["contour"], `${key}.contour`);
  const colourBy = o["colour_by"] === undefined ? (c ? "contour" : "group") : oneOf(o["colour_by"], `${key}.colour_by`, ["group", "role", "contour"]);
  if (colourBy === "contour" && !c) throw new SpecError(`${key}.colour_by`, `is "contour", but the shot has no contour key naming the field`);
  if (colourBy !== "contour" && c) throw new SpecError(`${key}.colour_by`, `is "${colourBy}", but the shot has a contour; a contour colours by "contour"`);
  const st = o["stage"] === undefined ? null : stage(o["stage"], `${key}.stage`);
  const comb = o["combination"] === undefined ? null : text(o["combination"], `${key}.combination`);
  if (comb !== null && st !== null) throw new SpecError(`${key}.combination`, "names the state shown, as stage does: a shot shows a stage or a User Combination, not both");
  if (o["step"] !== undefined && st === null) {
    throw new SpecError(`${key}.step`, comb !== null ? "a User Combination is one state: it has no step" : "needs a stage: a step is a position in a results stage");
  }
  // VU10: a compare names a variant; the shot then shows a difference, of a nodal contour at a stage, never a shape.
  const cmp = o["compare"] === undefined || o["compare"] === null ? null : text(o["compare"], `${key}.compare`);
  if (cmp !== null) {
    if (!d.variants) throw new SpecError(`${key}.compare`, "names a variant, and the spec has no files.variants to find it in");
    if (comb !== null) throw new SpecError(`${key}.compare`, "a combination across variants is refused: a User Combination is of the window variant's own states");
    if (st === null) throw new SpecError(`${key}.compare`, "needs a stage: a difference is of a results step");
    if (!c) throw new SpecError(`${key}.compare`, "needs a contour: a difference is shown as a contour");
    if (c.source === "element") throw new SpecError(`${key}.compare`, "a difference is of a nodal component; contour.source is \"element\"");
    if (o["deform"] !== undefined) throw new SpecError(`${key}.deform`, "a difference is never a deformed shape");
  }
  // VU18: a variant names the row whose set the shot draws; it needs the manifest, and is never its own compare set.
  const row = o["variant"] === undefined || o["variant"] === null ? null : variantRow(o["variant"], `${key}.variant`);
  if (row !== null) {
    if (!d.variants) throw new SpecError(`${key}.variant`, "names a variant row, and the spec has no files.variants to find it in");
    if (cmp !== null && cmp === row.label) throw new SpecError(`${key}.compare`, `is ${show(cmp)}, the variant the shot draws; a compare set is another variant`);
  }
  // VU16: a user result is one state of its own file: it is shown at no stage or step, over no combination, never subtracted, never a shape.
  if (c?.source === "user") {
    if (st !== null) throw new SpecError(`${key}.stage`, "a user result has no state: a shot of one names no stage");
    if (o["step"] !== undefined) throw new SpecError(`${key}.step`, "a user result has no state: it has no step");
    if (comb !== null) throw new SpecError(`${key}.combination`, "a shot shows a User Combination or a user result, not both");
    if (cmp !== null) throw new SpecError(`${key}.compare`, "a user result is not subtracted across variants");
    if (o["deform"] !== undefined) throw new SpecError(`${key}.deform`, "a user result is shown as a contour only, never as a deformed shape");
  }
  const caption = o["caption"] === undefined ? null : text(o["caption"], `${key}.caption`);
  const chrome = o["chrome"] === undefined ? d.chrome : oneOf(o["chrome"], `${key}.chrome`, CHROMES);
  const background = o["background"] === undefined ? d.background : oneOf(o["background"], `${key}.background`, BACKGROUNDS);
  // An app still is the window as the human sees it, panels and all: a transparent page under the panels is not defined.
  if (chrome === "app" && background === "transparent") {
    throw new SpecError(o["background"] === undefined ? `${key}.chrome` : `${key}.background`, `background "transparent" needs chrome "model"; an app still keeps the theme's background`);
  }
  const ownLayers = o["layers"] === undefined ? null : layerKeys(o["layers"], `${key}.layers`);
  const layersKey = ownLayers ? `${key}.layers` : d.layersKey;
  const layers = resolveLayers(chrome, ownLayers ?? d.layers ?? {}, layersKey, c !== null, caption !== null);
  return {
    name,
    caption,
    stage: st,
    ...(comb === null ? {} : { combination: comb }),
    ...(cmp === null ? {} : { compare: cmp }),
    ...(row === null ? {} : { variant: row }),
    step: o["step"] === undefined ? "last" : step(o["step"], `${key}.step`),
    camera: o["camera"] === undefined ? "iso" : camera(o["camera"], `${key}.camera`),
    colour_by: colourBy,
    contour: c,
    deform: o["deform"] === undefined ? null : deform(o["deform"], `${key}.deform`),
    clip: o["clip"] === undefined ? [] : clip(o["clip"], `${key}.clip`),
    isolate: o["isolate"] === undefined ? [] : isolate(o["isolate"], `${key}.isolate`),
    chrome,
    layers,
    background,
  };
}

function isolate(v: unknown, key: string): string[] {
  if (!Array.isArray(v) || v.length === 0) throw new SpecError(key, `must be a non-empty list of physical group paths; got ${show(v)}`);
  const out = v.map((p, i) => text(p, `${key}[${i}]`));
  const dup = out.find((p, i) => out.indexOf(p) !== i);
  if (dup !== undefined) throw new SpecError(key, `names ${dup} twice`);
  return out;
}

function size(v: unknown, key: string): [number, number] {
  if (!Array.isArray(v) || v.length !== 2) throw new SpecError(key, `must be [width, height] in pixels; got ${show(v)}`);
  return v.map((x, i) => {
    if (typeof x !== "number" || !Number.isInteger(x) || x < MIN_SIZE || x > MAX_SIZE) {
      throw new SpecError(`${key}[${i}]`, `must be an integer from ${MIN_SIZE} to ${MAX_SIZE}; got ${show(x)}`);
    }
    return x;
  }) as [number, number];
}

/** Validate a parsed spec document (D2). The whole spec or nothing: the first problem refuses it, naming its key. */
export function validateSpec(raw: unknown): Validated {
  try {
    return { ok: true, spec: validate(raw) };
  } catch (err) {
    if (err instanceof SpecError) return { ok: false, key: err.key, reason: err.reason };
    throw err;
  }
}

function validate(raw: unknown): ViewSpec {
  if (!isObject(raw)) throw new SpecError("spec", `the document must be a JSON object; got ${show(raw)}`);
  // The version first: a spec of another version is refused for that, not for a key this version lacks.
  if (raw["spec"] !== SPEC_VERSION) throw new SpecError("spec", `must be "${SPEC_VERSION}"; got ${show(raw["spec"])}`);
  const o = object(raw, "", ["spec", "files", "size", "theme", "chrome", "layers", "background", "shots"]);
  const files = object(o["files"], "files", ["model", "results", "variants"]);
  const model = text(files["model"], "files.model");
  const results = files["results"] === undefined ? null : text(files["results"], "files.results");
  const variants = files["variants"] === undefined ? null : text(files["variants"], "files.variants");
  if (variants !== null && !/.\.variants\.json$/i.test(variants)) throw new SpecError("files.variants", `must name a <study>.variants.json; got ${show(variants)}`);
  const chrome = o["chrome"] === undefined ? "app" : oneOf(o["chrome"], "chrome", CHROMES);
  const background = o["background"] === undefined ? "theme" : oneOf(o["background"], "background", BACKGROUNDS);
  const layers = o["layers"] === undefined ? null : layerKeys(o["layers"], "layers");
  if (!Array.isArray(o["shots"]) || o["shots"].length === 0) throw new SpecError("shots", `must be a non-empty list of shots; got ${show(o["shots"])}`);
  const shots = o["shots"].map((s, i) => shot(s, `shots[${i}]`, { chrome, layers, layersKey: "layers", background, variants: variants !== null }));
  const seen = new Map<string, number>();
  for (const [i, s] of shots.entries()) {
    // Windows file names ignore case: two names that differ only in case would write one PNG.
    const k = s.name.toLowerCase();
    if (seen.has(k)) throw new SpecError(`shots[${i}].name`, `${show(s.name)} is already the name of shots[${seen.get(k)}] (names are file names, compared ignoring case)`);
    seen.set(k, i);
  }
  // VU18: the render mode reads the spec set's `.user.h5` once, before the shots, and never a variant row's. A spec
  // that names a row draws every shot with no user file, so no shot of it may show a User Combination or a user result.
  const switching = shots.findIndex((s) => s.variant !== undefined);
  if (switching >= 0) {
    const why = `shots[${switching}] names a variant row, and a spec that draws variant rows reads no .user.h5 (the render mode does not read a row's user file)`;
    for (const [i, s] of shots.entries()) {
      if (s.combination != null) throw new SpecError(`shots[${i}].combination`, `${why}: a User Combination cannot be shown`);
      if (s.contour?.source === "user") throw new SpecError(`shots[${i}].contour.source`, `${why}: a user result cannot be shown`);
    }
  }
  return {
    spec: SPEC_VERSION,
    files: { model, results, ...(variants === null ? {} : { variants }) },
    size: o["size"] === undefined ? [DEFAULT_SIZE[0], DEFAULT_SIZE[1]] : size(o["size"], "size"),
    // D8: batch stills default to the light, white-paper theme.
    theme: o["theme"] === undefined ? "light" : oneOf(o["theme"], "theme", THEME_MODES),
    shots,
  };
}
