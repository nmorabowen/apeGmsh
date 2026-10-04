// Write a small HDF5 file from a plain object, for the zone readers' tests.
//
//   { attrs: {...}, child: { ... }, column: Int32Array, table: { data, shape } }
//
// `attrs` holds the node's attributes; a typed array or a string array is a
// 1-D dataset; `{data, shape}` is an N-D dataset; any other object is a group.

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Tree = { attrs?: Record<string, any>; [name: string]: any };

export const tree = (t: Tree): Tree => t;

interface H5Writer {
  File: new (path: string, mode: "w") => any;
}

function isLeaf(v: unknown): boolean {
  return ArrayBuffer.isView(v) || (Array.isArray(v) && v.every((x) => typeof x === "string"));
}

function fill(node: any, t: Tree): void {
  for (const [k, v] of Object.entries(t.attrs ?? {})) node.create_attribute(k, v);
  for (const [k, v] of Object.entries(t)) {
    if (k === "attrs" || v === undefined) continue;
    if (isLeaf(v)) node.create_dataset({ name: k, data: v });
    else if (v && typeof v === "object" && "data" in v && "shape" in v) node.create_dataset({ name: k, data: v.data, shape: v.shape });
    else fill(node.create_group(k), v as Tree);
  }
}

export function writeTree(h5: unknown, path: string, t: Tree): void {
  const f = new (h5 as H5Writer).File(path, "w");
  try {
    fill(f, t);
  } finally {
    f.close();
  }
}
