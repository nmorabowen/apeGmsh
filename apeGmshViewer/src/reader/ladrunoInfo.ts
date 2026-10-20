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
// throws a plain Error), so any other width is refused here by name. A
// fixed-length string reaches Python as bytes: any byte outside ASCII is refused,
// where h5wasm decodes ASCII as windows-1252 (so b'\xa02' read " 2", then 2).
// h5wasm also cuts the string at its first NUL and exposes no bytes past it. For
// a NULLTERM string that cut is HDF5's own, as for Python. For a NULLPAD or
// SPACEPAD string Python keeps the bytes after the NUL (b'2\x003' is refused,
// b'2\x00' reads as 2), which this reader cannot see, so any NUL there is refused.
// h5py decodes a variable-length string as UTF-8 whatever its cset; for an ASCII
// cset h5wasm gives one char per byte, so the bytes are decoded here the same way.
//
// Known, intentional divergences (this reader refuses, Python reads 2): a float16
// 2.0, a NULLPAD b'2\x00', an HDF5 array type (1,) of int32, a bitfield, and a
// variable-length sequence of int32. No writer produces any of them (#2324).
// Known gap: a variable-length UTF-8 string goes through h5wasm's lax UTF-8
// decoder, which reads the overlong b'\xc0\xb2' as "2"; Python refuses it. The
// raw bytes are out of this module's reach.
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
 * `strpad`, `cset` and `enum_type` are optional only because the caller's
 * structural type leaves some out. h5wasm always reports them, and a missing one
 * that this rule needs is refused, naming the file, instead of skipping the check.
 */
export interface SpatialDimMetadata {
  type: number;
  vlen: boolean;
  shape: number[] | null;
  size?: number;
  strpad?: number;
  cset?: number;
  enum_type?: { type: number };
}

type NeedKey = "size" | "strpad" | "cset" | "enum_type";
type Need = <K extends NeedKey>(key: K) => NonNullable<SpatialDimMetadata[K]>;

// HDF5 H5T_class_t, H5T_str_t and H5T_cset_t values, as h5wasm reports them.
const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;
const H5T_ENUM = 8;
const H5T_STR_NULLTERM = 0;
const H5T_CSET_ASCII = 0;
const CLASS_NAMES: Record<number, string> = {
  2: "a time value",
  4: "a bitfield",
  5: "an opaque value",
  6: "a compound",
  7: "a reference",
  9: "a variable-length sequence",
  10: "an array type",
};

/** Why Python refuses this attribute's HDF5 type, or why h5wasm cannot decode it; null when the value may be read. */
function typeRefusal(m: SpatialDimMetadata, need: Need): string | null {
  if (m.type === H5T_INTEGER || m.type === H5T_STRING) return null;
  if (m.type === H5T_ENUM) return need("enum_type").type === H5T_INTEGER ? null : "an enum over no integer type";
  if (m.type === H5T_FLOAT) {
    const size = need("size");
    return size === 4 || size === 8 ? null : `a ${size * 8}-bit float, which this reader cannot decode`;
  }
  return `${CLASS_NAMES[m.type] ?? "a value"} (HDF5 type class ${m.type}), not a number or a string`;
}

/** Why Python reads this h5wasm-decoded fixed-length string otherwise, or null when it reads the same text. */
function fixedStringRefusal(s: string, need: Need): string | null {
  // windows-1252 and UTF-8 both decode every byte outside ASCII to a code point outside it.
  if (/[^\x00-\x7f]/.test(s)) return `the fixed-length string ${shown(s)}, which holds a byte outside ASCII`;
  // All ASCII, so one char per byte: a shorter string was cut at a NUL.
  const size = need("size");
  if (s.length < size && need("strpad") !== H5T_STR_NULLTERM) {
    return `a ${size}-byte fixed-length string with a NUL after ${shown(s)}, past which this reader cannot see`;
  }
  return null;
}

// ignoreBOM: h5py keeps a leading U+FEFF, which float() refuses; the default drops it.
const UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/**
 * The text h5py reads from a variable-length string of ASCII cset, or null when
 * h5py reads no text. h5py decodes the bytes as UTF-8 (invalid bytes become lone
 * surrogates, which `float()` refuses); h5wasm's `AsciiToString` gives one char
 * per byte, so the bytes are exactly recoverable.
 */
function vlenAsciiText(s: string): string | null {
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c > 0xff) return null;
    bytes[i] = c;
  }
  try {
    return UTF8.decode(bytes);
  } catch (e) {
    if (e instanceof TypeError) return null;
    throw e;
  }
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
  const need: Need = (key) => {
    const v = m[key];
    if (v === undefined || v === null) {
      throw refused(`of HDF5 type class ${m.type}, and its h5wasm metadata has no ${key}, which this reader needs to read it`);
    }
    return v as NonNullable<SpatialDimMetadata[typeof key]>;
  };
  if (m.shape !== null && m.shape.reduce((a, b) => a * b, 1) === 0) return null;
  const byType = typeRefusal(m, need);
  if (byType !== null) throw refused(byType);
  let values = valuesOf(attr.value);
  if (values.length === 0) return null;
  if (m.type === H5T_STRING) {
    values = values.map((v) => {
      if (typeof v !== "string") throw refused(`a string that h5wasm returned as a ${typeof v}`);
      if (!m.vlen) {
        const why = fixedStringRefusal(v, need);
        if (why !== null) throw refused(why);
        return v;
      }
      if (need("cset") !== H5T_CSET_ASCII) return v;
      const text = vlenAsciiText(v);
      if (text === null) throw refused(`the variable-length ASCII string ${shown(v)}, whose bytes are not UTF-8, as h5py decodes them`);
      return text;
    });
  }
  const n = values.length === 1 ? floatOf(values[0]) : NaN;
  if (n === 1 || n === 2 || n === 3) return n;
  throw refused(values.length === 1 ? shown(values[0]) : `[${values.map(shown).join(", ")}]`);
}
