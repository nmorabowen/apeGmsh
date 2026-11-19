// One quantity on one state, and the join of several (ADR 0130 D4, A3).
//
// The app's twin of `Field` and `_align` in src/apeGmsh/results/user/_evaluate.py.
// A row is named by its key: `id` for a node; `id, sub` for a Gauss point,
// station or element node; `kind, row` for an entity; one fixed row for a
// domain value. Terms join on the key and the component label, and a row or a
// label a term lacks is NaN in that term. NaN means "not defined here" and
// propagates through every rule.

import { byCodePoint } from "../reader/rf1.ts";

export type Key = number | string;

export interface Field {
  /** what a key column names (`id`; `id, sub`; `kind, row`) */
  readonly keyNames: readonly string[];
  /** one key per row, one cell per key name */
  readonly keys: readonly (readonly Key[])[];
  readonly components: readonly string[];
  /** row-major, `keys.length × components.length`; NaN is "not defined here" */
  readonly values: Float64Array;
}

/** A field that breaks its own shape: a bug in its producer, never a refusal of one combination. */
export class FieldError extends Error {}

/** Two keys in the order numpy sorts a structured array: column by column, text by code point. */
export function compareKeys(a: readonly Key[], b: readonly Key[]): number {
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    const d = typeof x === "number" ? (x as number) - (y as number) : byCodePoint(x, y as string);
    if (d !== 0) return d;
  }
  return 0;
}

/** The type of each key column, from its first row (`null` for an empty field). */
function columnTypes(f: Field): ("number" | "string")[] | null {
  const first = f.keys[0];
  if (first === undefined) return null;
  return first.map((k) => (typeof k === "number" ? "number" : "string"));
}

/** A checked field: Python's `Field.__post_init__` (one row per key, no repeated key or label). */
export function makeField(keyNames: readonly string[], keys: readonly (readonly Key[])[], components: readonly string[], values: ArrayLike<number>): Field {
  if (keyNames.length === 0 || new Set(keyNames).size !== keyNames.length) throw new FieldError(`a field's key names must be distinct and non-empty: ${JSON.stringify(keyNames)}`);
  if (new Set(components).size !== components.length) throw new FieldError(`a field repeats a label: ${JSON.stringify(components)}`);
  if (values.length !== keys.length * components.length) {
    throw new FieldError(`a field has ${values.length} values, expected ${keys.length} rows × ${components.length} labels`);
  }
  const types = keys[0]?.map((k) => typeof k);
  for (const key of keys) {
    if (key.length !== keyNames.length) throw new FieldError(`a key ${JSON.stringify(key)} has ${key.length} cells for ${keyNames.length} key names`);
    key.forEach((k, i) => {
      if (typeof k !== types![i] || (typeof k === "number" && !Number.isFinite(k))) throw new FieldError(`key column ${keyNames[i]} mixes types or is not finite: ${JSON.stringify(key)}`);
    });
  }
  const sorted = [...keys].sort(compareKeys);
  for (let i = 1; i < sorted.length; i++) {
    if (compareKeys(sorted[i - 1]!, sorted[i]!) === 0) throw new FieldError(`a field repeats the row ${JSON.stringify(sorted[i])}`);
  }
  return { keyNames: [...keyNames], keys: keys.map((k) => [...k]), components: [...components], values: Float64Array.from(values) };
}

/** `factor` times every value. */
export function scaled(f: Field, factor: number): Field {
  if (factor === 1) return f;
  return { ...f, values: f.values.map((v) => factor * v) };
}

/** Why the terms cannot be joined (they are on different targets), or null. */
export function targetProblem(fields: readonly Field[]): string | null {
  const first = fields[0]!;
  let types = columnTypes(first);
  for (const f of fields.slice(1)) {
    if (f.keyNames.length !== first.keyNames.length || f.keyNames.some((n, i) => n !== first.keyNames[i])) {
      return `its terms are on different targets (${first.keyNames.join(", ")} and ${f.keyNames.join(", ")})`;
    }
    const t = columnTypes(f);
    if (t === null) continue;
    if (types !== null && t.some((x, i) => x !== types![i])) return `its terms' key columns differ in type (${types.join(", ")} and ${t.join(", ")})`;
    types ??= t;
  }
  return null;
}

/** The join of `fields`: Python's `_align`. The caller has checked `targetProblem`. */
export interface Aligned {
  keyNames: readonly string[];
  keys: readonly (readonly Key[])[];
  components: readonly string[];
  /** one `keys.length × components.length` block per field, NaN where the field has no cell */
  stack: Float64Array[];
}

export function align(fields: readonly Field[]): Aligned {
  const first = fields[0]!;
  const components: string[] = [];
  for (const f of fields) for (const c of f.components) if (!components.includes(c)) components.push(c);
  const sameRows = fields.every((f) => f.keys.length === first.keys.length && f.keys.every((k, i) => compareKeys(k, first.keys[i]!) === 0));
  // Python keeps the first field's order when every term has the same rows, else `np.unique`'s sorted union.
  let keys: readonly (readonly Key[])[] = first.keys;
  if (!sameRows) {
    const all = fields.flatMap((f) => f.keys).sort(compareKeys);
    keys = all.filter((k, i) => i === 0 || compareKeys(all[i - 1]!, k) !== 0);
  }
  const n = keys.length;
  const c = components.length;
  const stack = fields.map((f) => {
    const out = new Float64Array(n * c).fill(NaN);
    const cols = f.components.map((label) => components.indexOf(label));
    const fc = f.components.length;
    f.keys.forEach((key, r) => {
      const row = sameRows ? r : search(keys, key);
      for (let j = 0; j < fc; j++) out[row * c + cols[j]!] = f.values[r * fc + j]!;
    });
    return out;
  });
  return { keyNames: first.keyNames, keys, components, stack };
}

/** The position of `key` in the sorted, distinct `keys` (it is there: they are the union). */
function search(keys: readonly (readonly Key[])[], key: readonly Key[]): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (compareKeys(keys[mid]!, key) < 0) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
