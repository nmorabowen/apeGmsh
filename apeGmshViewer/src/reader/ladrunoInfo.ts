// `INFO@SPATIAL_DIM` of a `.ladruno`, read under Python's rule (#2205).
//
// Python `_ladruno_element_io._spatial_dim_attr` and `_file_spatial_dim` (#2098,
// #2174): a present attribute holding one value is read as `float(value)`, and
// only 1.0, 2.0 or 3.0 is accepted (so `2.0` reads as 2). Any other value (7, 0,
// -1, a truncated 2.9, an array of several values) raises `SpatialDimInvalid`,
// naming the file and the value. An absent attribute, or one holding no value,
// is unknown: the caller infers the dimension from MODEL/NODES/COORDINATES.
//
// Browser-safe: no Node import. The caller passes its own error factory, as
// reader/ladrunoElements.ts does, so this module imports nothing from ladruno.ts.

/** Python `float()` of a str: one decimal literal, `_` only between digits, surrounding whitespace ignored. */
const DIGITS = String.raw`\d(?:_?\d)*`;
const PY_FLOAT = new RegExp(String.raw`^[+-]?(?:${DIGITS}(?:\.(?:${DIGITS})?)?|\.${DIGITS})(?:[eE][+-]?${DIGITS})?$`);

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
    const s = x.trim();
    return PY_FLOAT.test(s) ? Number(s.replace(/_/g, "")) : NaN;
  }
  return NaN;
}

const shown = (x: unknown): string => (typeof x === "string" ? JSON.stringify(x) : String(x));

/**
 * `INFO@SPATIAL_DIM` as 1, 2 or 3; null when absent or empty. Any other value
 * throws `refuse(message)`, the message naming `path`, the attribute and the value
 * (Python `SpatialDimInvalid`).
 */
export function spatialDimOf(attr: { value: unknown } | undefined, path: string, refuse: (message: string) => Error): number | null {
  if (attr === undefined) return null;
  const values = valuesOf(attr.value);
  if (values.length === 0) return null;
  const n = values.length === 1 ? floatOf(values[0]) : NaN;
  if (n === 1 || n === 2 || n === 3) return n;
  const value = values.length === 1 ? shown(values[0]) : `[${values.map(shown).join(", ")}]`;
  throw refuse(`${path}: INFO@SPATIAL_DIM is ${value}; the spatial dimension must be an integral 1, 2 or 3`);
}
