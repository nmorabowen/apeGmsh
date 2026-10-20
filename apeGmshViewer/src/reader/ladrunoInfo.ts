// `INFO@SPATIAL_DIM` of a `.ladruno`, read under Python's rule (#2205).
//
// Python `_ladruno_element_io._spatial_dim_attr` and `_file_spatial_dim` (#2098,
// #2174): a present attribute holding one value is read as `float(value)`, and
// only 1.0, 2.0 or 3.0 is accepted (so `2.0` reads as 2). Any other value (7, 0,
// -1, a truncated 2.9, an array of several values) raises `SpatialDimInvalid`,
// naming the file and the value. An absent attribute, or one holding no value,
// is unknown: the caller infers the dimension from MODEL/NODES/COORDINATES.
//
// The HDF5 type is read before the value (#2324), because h5wasm's value hides
// what Python sees. Python reads an integer, an enum over one, a float, or a
// string; it refuses a compound (even one member, which h5wasm returns as that
// member) and an opaque. h5wasm decodes no float but float32 and float64 (it
// throws a plain Error), so any other width is refused here by name; Python
// reads a float16 2.0 as 2, the one place this reader is stricter. A
// fixed-length string reaches Python as bytes: any byte outside ASCII is refused,
// where h5wasm decodes ASCII as windows-1252 (so b'\xa02' read " 2", then 2).
// h5wasm also cuts the string at its first NUL and exposes no bytes past it. For
// a NULLTERM string that cut is HDF5's own, as for Python. For a NULLPAD or
// SPACEPAD string Python keeps the bytes after the NUL (b'2\x003' is refused,
// b'2\x00' reads as 2), which this reader cannot see, so a NUL there is refused.
//
// Browser-safe: no Node import. The caller passes its own error factory, as
// reader/ladrunoElements.ts does, so this module imports nothing from ladruno.ts.

/**
 * Python `float()` of a str: one ASCII decimal literal, `_` only between digits,
 * inside the whitespace `float()` itself skips. That set was enumerated over every
 * code point: JS `trim()`'s set without U+FEFF, plus U+0085 (#2205 review).
 * Non-ASCII digits, which `float()` reads, are refused on purpose.
 */
const PY_SPACE = String.raw`[\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*`;
const DIGITS = String.raw`[0-9](?:_?[0-9])*`;
const PY_FLOAT = new RegExp(String.raw`^${PY_SPACE}([+-]?(?:${DIGITS}(?:\.(?:${DIGITS})?)?|\.${DIGITS})(?:[eE][+-]?${DIGITS})?)${PY_SPACE}$`);

/** The attribute's values, flattened, as `np.asarray(raw).reshape(-1)` lists them. */
function valuesOf(v: unknown): unknown[] {
  if (Array.isArray(v)) return v.flatMap(valuesOf);
  if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v as unknown as ArrayLike<unknown>);
  return [v];
}

/** Python `float(x)` of one attribute value, or NaN where Python's `float` raises. */
function floatOf(x: unknown): number {
  if (typeof x === "number") return x;
  if (typeof x === "bigint") return Number(x);
  if (typeof x === "boolean") return x ? 1 : 0;
  if (typeof x === "string") {
    const literal = PY_FLOAT.exec(x)?.[1];
    return literal === undefined ? NaN : Number(literal.replace(/_/g, ""));
  }
  return NaN;
}

const shown = (x: unknown): string => (typeof x === "string" ? JSON.stringify(x) : String(x));

/**
 * The h5wasm attribute metadata (h5wasm `Metadata`) this rule reads. `size`,
 * `strpad` and `enum_type` are optional only because the caller's structural
 * type leaves them out. h5wasm always reports them, and a missing one that this
 * rule needs throws instead of skipping the check.
 */
export interface SpatialDimMetadata {
  type: number;
  vlen: boolean;
  shape: number[] | null;
  size?: number;
  strpad?: number;
  enum_type?: { type: number };
}

// HDF5 H5T_class_t and H5T_str_t values, as h5wasm reports them.
const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;
const H5T_ENUM = 8;
const H5T_STR_NULLTERM = 0;
const CLASS_NAMES: Record<number, string> = {
  2: "a time value",
  4: "a bitfield",
  5: "an opaque value",
  6: "a compound",
  7: "a reference",
  9: "a variable-length sequence",
  10: "an array type",
};

function need<K extends "size" | "strpad" | "enum_type">(m: SpatialDimMetadata, key: K): NonNullable<SpatialDimMetadata[K]> {
  const v = m[key];
  if (v === undefined || v === null) {
    throw new Error(`INFO@SPATIAL_DIM: the attribute metadata has no ${key}, which h5wasm always reports for HDF5 type class ${m.type}`);
  }
  return v as NonNullable<SpatialDimMetadata[K]>;
}

/** Why Python refuses this attribute's HDF5 type, or why h5wasm cannot decode it; null when the value may be read. */
function typeRefusal(m: SpatialDimMetadata): string | null {
  if (m.type === H5T_INTEGER || m.type === H5T_STRING) return null;
  if (m.type === H5T_ENUM) return need(m, "enum_type").type === H5T_INTEGER ? null : "an enum over no integer type";
  if (m.type === H5T_FLOAT) {
    const size = need(m, "size");
    return size === 4 || size === 8 ? null : `a ${size * 8}-bit float, which this reader cannot decode`;
  }
  return `${CLASS_NAMES[m.type] ?? "a value"} (HDF5 type class ${m.type}), not a number or a string`;
}

/** Why Python reads this h5wasm-decoded fixed-length string otherwise, or null when it reads the same text. */
function fixedStringRefusal(m: SpatialDimMetadata, s: string): string | null {
  // windows-1252 and UTF-8 both decode every byte outside ASCII to a code point outside it.
  if (/[^\x00-\x7f]/.test(s)) return `the fixed-length string ${shown(s)}, which holds a byte outside ASCII`;
  // All ASCII, so one char per byte: a shorter string was cut at a NUL.
  const size = need(m, "size");
  if (s.length < size && need(m, "strpad") !== H5T_STR_NULLTERM) {
    return `a ${size}-byte fixed-length string with a NUL after ${shown(s)}, past which this reader cannot see`;
  }
  return null;
}

/**
 * `INFO@SPATIAL_DIM` as 1, 2 or 3; null when absent or empty. Any other value
 * throws `refuse(message)`, the message naming `path`, the attribute and the value
 * (Python `SpatialDimInvalid`).
 */
export function spatialDimOf(attr: { value: unknown; metadata: SpatialDimMetadata } | undefined, path: string, refuse: (message: string) => Error): number | null {
  if (attr === undefined) return null;
  const m = attr.metadata;
  const refused = (what: string) => refuse(`${path}: INFO@SPATIAL_DIM is ${what}; the spatial dimension must be an integral 1, 2 or 3`);
  if (m.shape !== null && m.shape.reduce((a, b) => a * b, 1) === 0) return null;
  const byType = typeRefusal(m);
  if (byType !== null) throw refused(byType);
  const values = valuesOf(attr.value);
  if (values.length === 0) return null;
  if (m.type === H5T_STRING && !m.vlen) {
    for (const v of values) {
      const why = typeof v === "string" ? fixedStringRefusal(m, v) : null;
      if (why !== null) throw refused(why);
    }
  }
  const n = values.length === 1 ? floatOf(values[0]) : NaN;
  if (n === 1 || n === 2 || n === 3) return n;
  throw refused(values.length === 1 ? shown(values[0]) : `[${values.map(shown).join(", ")}]`);
}
