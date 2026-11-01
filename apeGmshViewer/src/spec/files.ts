// The files of a view spec (ADR 0128 D1): `files` names the set the way an
// open does, and the results are found by ADR 0112's pairing rules
// (main/pairing.ts), never named into a pairing they would not make. A
// relative path is read against the spec file's folder. Main runs this
// before any window opens; a refusal names its key, as ./spec.ts does.

import { isAbsolute, resolve } from "node:path";
import { pairSet, samePath, type OpenSet } from "../main/pairing.ts";
import type { ViewSpec } from "./spec.ts";

export type ResolvedFiles = { ok: true; set: OpenSet & { model: string } } | { ok: false; key: string; reason: string };

export function resolveSpecFiles(spec: ViewSpec, specDir: string, exists: (p: string) => boolean, list?: (dir: string) => readonly string[]): ResolvedFiles {
  const abs = (p: string) => (isAbsolute(p) ? resolve(p) : resolve(specDir, p));
  const model = abs(spec.files.model);
  if (!exists(model)) return { ok: false, key: "files.model", reason: `no such file: ${model}` };
  const fromModel = pairSet(model, exists, list);
  if (!samePath(fromModel.model, model)) {
    return { ok: false, key: "files.model", reason: `${model} is not a model file; the model of its set is ${fromModel.model ?? "absent"}` };
  }
  if (spec.files.results === null) return { ok: true, set: { ...fromModel, model } };
  const results = abs(spec.files.results);
  if (!exists(results)) return { ok: false, key: "files.results", reason: `no such file: ${results}` };
  // Opening the results file opens its set: its model must be the spec's model, and it must be the set's results.
  const fromResults = pairSet(results, exists, list);
  if (!samePath(fromResults.results, results)) return { ok: false, key: "files.results", reason: `${results} is not a results file` };
  if (!samePath(fromResults.model, model)) {
    return { ok: false, key: "files.results", reason: `${results} is not in the set of ${model}: opening it opens the model ${fromResults.model ?? "(none)"}` };
  }
  return { ok: true, set: { ...fromResults, model } };
}
