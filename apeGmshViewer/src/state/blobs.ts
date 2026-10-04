// The BlobStore: the large arrays of a loaded artifact, outside the state
// snapshot (D6, decision 15). The state holds a `BlobRef` per array; the
// viewport's GPU cache is keyed by `BlobRef.id`. An array is never mutated
// after it is put, and a ref to an array that is not here is an error, not
// an empty result.
//
// Panels never import this module (the panel-import lint rule).

import type { BlobRef } from "./types.ts";

export type Blob = Float32Array | Float64Array | Int32Array | Uint8Array;

type DType = BlobRef["dtype"];

function dtypeOf(a: Blob): DType {
  if (a instanceof Float32Array) return "f32";
  if (a instanceof Float64Array) return "f64";
  if (a instanceof Int32Array) return "i32";
  return "u8";
}

export class BlobStore {
  private readonly blobs = new Map<string, Blob>();
  private seq = 0;

  /** Store an array under a path-like name; the ref's id is unique for the store's lifetime. */
  put(path: string, array: Blob, shape: readonly number[] = [array.length]): BlobRef {
    const n = shape.reduce((a, b) => a * b, 1);
    if (n !== array.length) throw new RangeError(`blob ${path}: shape [${shape.join(", ")}] has ${n} values, the array ${array.length}`);
    const id = `${++this.seq}:${path}`;
    this.blobs.set(id, array);
    return { id, path, dtype: dtypeOf(array), shape: [...shape] };
  }

  get(ref: BlobRef): Blob {
    const b = this.blobs.get(ref.id);
    if (!b) throw new Error(`blob ${ref.id} is not in the store (released, or from another store)`);
    return b;
  }

  f32(ref: BlobRef): Float32Array {
    const b = this.get(ref);
    if (!(b instanceof Float32Array)) throw new TypeError(`blob ${ref.id} is ${ref.dtype}, not f32`);
    return b;
  }

  i32(ref: BlobRef): Int32Array {
    const b = this.get(ref);
    if (!(b instanceof Int32Array)) throw new TypeError(`blob ${ref.id} is ${ref.dtype}, not i32`);
    return b;
  }

  has(ref: BlobRef): boolean {
    return this.blobs.has(ref.id);
  }

  /** Drop every blob except those in `keep` (the refs the new state still holds). */
  retain(keep: Iterable<BlobRef>): void {
    const ids = new Set<string>();
    for (const r of keep) ids.add(r.id);
    for (const id of [...this.blobs.keys()]) if (!ids.has(id)) this.blobs.delete(id);
  }

  get size(): number {
    return this.blobs.size;
  }
}
