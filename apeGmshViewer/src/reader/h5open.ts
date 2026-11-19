// The one way the viewer opens an HDF5 file with h5wasm (lesson #2260, slice #2261).
//
// h5wasm's Node build (emscripten NODERAWFS) hands HDF5 each open file's id as
// `BigInt(fs.fstatSync(fd).ino)`. `ino` is a JS double, and an NTFS file id
// (a 16-bit sequence number above the MFT record) is >= 2^53 once the sequence
// number reaches 0x20, so two different files can round to the same
// (dev, ino). HDF5's sec2 driver dedupes open files by (dev, ino): opening B
// while A was open returned A's shared handle, and the viewer drew A's data as
// B's with no error (fixtures elem_solid.h5 and elem_mixed_unstamped.results.h5
// on one NTFS volume did exactly that).
//
// `openH5` first replaces, once per h5wasm FS, the `fstat` the HDF5 fstat
// syscall calls with one that reports the exact 64-bit id
// (`fstatSync(fd, {bigint: true}).ino`), so the dedupe compares true ids. When
// it cannot install that (h5wasm not ready, or an h5wasm whose FS lacks the
// members used here) it throws: a refused open, never a silent alias.
//
// scripts/check_app_wall.py W5 keeps every `new <h5wasm>.File(` in src/ and
// scripts/ inside this file.
//
// `createH5` (ADR 0132 D6, VU9a #2803) is the one write: a new file created
// exclusively, `"x"`, so a writer's partial never truncates a file another
// writer made. h5wasm answers an existing name with a negative file id, not a
// throw; `createH5` turns that into a throw.

/** The emscripten FS that h5wasm exports as `FS` once `ready` resolves (NODERAWFS). */
export interface H5FS {
  fstat(fd: number): { ino: number | bigint };
  getStreamChecked(fd: number): { nfd?: unknown };
}

/** What `openH5` needs of the h5wasm module (`h5wasm/node`, after `await h5wasm.ready`). */
export interface H5Openable<F> {
  File: new (filename: string, mode: "r") => F;
  readonly FS: H5FS | null;
}

const exact = new WeakSet<H5FS>();

/** h5wasm's read binding: every `Dataset` read (`value`, `json_value`, `slice`, `to_array`) calls it. */
interface H5ReadBinding {
  get_dataset_data(file_id: bigint, path: string, count: bigint[] | null, offset: bigint[] | null, strides: bigint[] | null, ptr: bigint): number;
  get_dataset_metadata(file_id: bigint, path: string): { size: number; total_size: number };
  readonly HEAPU8: Uint8Array;
}

const checked = new WeakSet<object>();

/**
 * #3010: make a refused read throw, for every reader at once (see "A read that
 * says when HDF5 refused it" below). Idempotent per module. A stand-in h5 with
 * no `Module` (the open tests) has no reads to check.
 */
function checkedReads(mod: H5ReadBinding | null | undefined): void {
  if (mod === undefined || mod === null || checked.has(mod)) return;
  if (typeof mod.get_dataset_data !== "function" || typeof mod.get_dataset_metadata !== "function") {
    throw new Error("h5wasm's Module has no get_dataset_data/get_dataset_metadata, so read errors cannot be checked (#3010); refusing to open HDF5 files (check the pinned h5wasm version)");
  }
  const read = mod.get_dataset_data;
  mod.get_dataset_data = (file_id, path, count, offset, strides, ptr) => {
    const status = read(file_id, path, count, offset, strides, ptr);
    if (status >= 0) return status;
    // h5wasm reclaims a variable-length buffer on the way out, even after a throw: zeroed, it holds nothing to free.
    try {
      const { size, total_size } = mod.get_dataset_metadata(file_id, path);
      const n = count ? count.reduce((a, c) => a * Number(c), 1) : total_size;
      mod.HEAPU8.fill(0, Number(ptr), Number(ptr) + size * n);
    } catch {
      // The dataset no longer opens, so the reclaim finds nothing to walk either.
    }
    throw new H5ReadError(`${path}: HDF5 refused to read this dataset (its chunk index or data does not read)`);
  };
  checked.add(mod);
}

/** What every open and create needs first: exact file ids, and reads that report a refusal. */
function prepare(h5: { readonly FS: H5FS | null }): void {
  exactInodes(h5.FS);
  checkedReads((h5 as { Module?: H5ReadBinding | null }).Module);
}

/** Make `fs.fstat` report the exact 64-bit file id, or throw. Idempotent per FS. */
function exactInodes(fs: H5FS | null): void {
  if (fs === null) throw new Error("h5wasm is not ready: await h5wasm.ready before opening an HDF5 file");
  if (exact.has(fs)) return;
  if (typeof fs.fstat !== "function" || typeof fs.getStreamChecked !== "function") {
    throw new Error(
      "h5wasm's FS has no fstat/getStreamChecked, so the exact file-id fix (lesson #2260) cannot be installed; " +
        "refusing to open HDF5 files (check the pinned h5wasm version)",
    );
  }
  // Not a static `import "node:fs"`: the renderer bundle imports values from
  // the reader modules that import this one, and it has no node:fs.
  const nodeFs = process.getBuiltinModule("node:fs");
  const fstat = fs.fstat.bind(fs);
  fs.fstat = (fd: number) => {
    const st = fstat(fd);
    const nfd = fs.getStreamChecked(fd).nfd;
    if (typeof nfd !== "number") {
      throw new Error(`h5wasm stream ${fd} has no host file descriptor, so its exact file id cannot be read (lesson #2260)`);
    }
    st.ino = nodeFs.fstatSync(nfd, { bigint: true }).ino;
    return st;
  };
  exact.add(fs);
}

// -- A listing or lookup that says when a group index does not read ---------------------
//
// h5wasm's `Group.keys()` calls `Module.get_names`, whose C++ drops H5Literate2's status: when
// a group's B-tree is damaged, the iteration stops and the names found so far come back as the
// whole group, so a stage silently loses its results. `Group.get` calls `Module.get_type`, whose
// C++ ignores both of its statuses: the same damage makes an existing dataset, and a name that
// does not exist, come back as type 0, a group. The only trace is HDF5's error stack, which its
// default handler writes to stderr, through `FS.write` on host fd 2.
//
// `checkedGroups` watches that write around each call. Any diagnostic during a listing is a
// failure. A lookup needs the stack's content, because HDF5 also reports an error on a sound
// file: a path below a dataset, through a dangling soft link or through an external link. Those
// never reach the metadata cache, and a damaged index always fails loading into it. An object
// header that does not load (`H5O_protect`) is not the index: the link was found, `get` answers
// as before, and the object's own reads refuse it (the `.ladruno` torn-stage rule relies on this).
//
// h5wasm's throwing error handler is not used: its throw unwinds the C++ past H5Gclose, the
// leaked group keeps the file open after `close()`, and a later open of that path shares the
// stale file. These checks throw only after the binding has returned.

/** h5wasm's group bindings: `Group.keys()` calls `get_names`, `Group.get` calls `get_type`. */
interface H5GroupBinding {
  get_names(file_id: bigint, path: string, recursive: boolean): string[];
  get_type(file_id: bigint, path: string): number;
}

/** The `FS.write` HDF5's diagnostics pass through. */
type H5FsWrite = (stream: { nfd?: unknown }, buffer: Uint8Array, offset: number, length: number, ...rest: unknown[]) => number;

const watched = new WeakSet<object>();

/** Whether an HDF5 error stack says a group's index could not be loaded. */
function indexDidNotLoad(stack: string): boolean {
  return stack.includes("major: Object cache") && !stack.includes("in H5O_protect()");
}

/**
 * Make a listing HDF5 reported an error on, and a lookup through an index that does not read,
 * throw, for every reader at once. Idempotent per module; a stand-in h5 with no `Module` (the
 * open tests) has no groups. The error is a plain `Error`, what h5wasm raises for an unreadable
 * object, so the readers' rules for a stage that does not read (`isH5Error`) apply to it.
 */
function checkedGroups(h5: { readonly FS: H5FS | null; readonly Module?: H5GroupBinding | null }): void {
  const mod = h5.Module;
  if (mod === undefined || mod === null || watched.has(mod)) return;
  const fs = h5.FS as unknown as { write?: H5FsWrite };
  if (typeof mod.get_names !== "function" || typeof mod.get_type !== "function" || typeof fs.write !== "function") {
    throw new Error("h5wasm has no Module.get_names/get_type/FS.write, so a damaged group index cannot be detected; refusing to open HDF5 files (check the pinned h5wasm version)");
  }
  const write = fs.write.bind(fs);
  const latin1 = new TextDecoder("latin1");
  let diagnosed = false;
  let stack: string | null = null;
  fs.write = (stream, buffer, offset, length, ...rest) => {
    if (stream.nfd === 2) {
      diagnosed = true;
      if (stack !== null) stack += latin1.decode(buffer.subarray(offset, offset + length));
    }
    return write(stream, buffer, offset, length, ...rest);
  };
  const names = mod.get_names;
  mod.get_names = (file_id, path, recursive) => {
    diagnosed = false;
    const out = names(file_id, path, recursive);
    if (diagnosed) throw new Error(`${path}: HDF5 could not list this group (its group index does not read)`);
    return out;
  };
  const type = mod.get_type;
  mod.get_type = (file_id, path) => {
    stack = "";
    try {
      const out = type(file_id, path);
      if (!indexDidNotLoad(stack)) return out;
    } finally {
      stack = null;
    }
    // The damaged index may be any group on the path, so the path does not lead the message.
    throw new Error(`HDF5 could not look up ${path} (the index of a group on this path does not read)`);
  };
  watched.add(mod);
}

/**
 * VU9b (HG9 rule 2 on #2693): every file `openH5` opened and has not closed,
 * by its path as the caller named it. A writer that replaces a file closes
 * this app's own handles on it first (`closeHandlesOn`), so the replace is
 * never blocked by the app itself and no handle reads the file it replaced.
 */
const live = new Map<{ close(): unknown }, string>();

/** The path as the file system compares it: separators unified, and case-folded on Windows and macOS. */
function pathKey(path: string): string {
  const p = path.replace(/\\/g, "/");
  return typeof process !== "undefined" && (process.platform === "win32" || process.platform === "darwin") ? p.toLowerCase() : p;
}

/** How many `openH5` handles on `path` are open now. */
export function openHandlesOn(path: string): number {
  const k = pathKey(path);
  let n = 0;
  for (const p of live.values()) if (p === k) n++;
  return n;
}

/** Close every `openH5` handle on `path`, and say how many were open. */
export function closeHandlesOn(path: string): number {
  const k = pathKey(path);
  const mine = [...live].filter(([, p]) => p === k).map(([f]) => f);
  for (const f of mine) f.close();
  return mine.length;
}

/** Open `path` read-only with h5wasm, with HDF5 comparing exact file ids. */
export function openH5<F>(h5: H5Openable<F>, path: string): F {
  return openTracked(h5 as unknown as H5RuleOpenable<F>, path, "r");
}

/** `new File(path, mode)` with exact file ids, the handle tracked by `path` until its own close; `after` runs after that close. */
function openTracked<F>(h5: H5RuleOpenable<F>, path: string, mode: "r" | "Sr", after?: () => void): F {
  prepare(h5);
  const f = new h5.File(path, mode);
  checkedGroups(h5 as unknown as Parameters<typeof checkedGroups>[0]);
  const handle = f as unknown as { close?: () => unknown; file_id?: bigint | number };
  // A failed open (negative id) holds nothing; a handle is tracked until its own close.
  if (typeof handle.close === "function" && !(Number(handle.file_id) < 0)) {
    const close = handle.close.bind(f);
    const entry = {
      close: () => {
        if (!live.has(entry)) return undefined;
        live.delete(entry);
        try {
          return close();
        } finally {
          after?.();
        }
      },
    };
    handle.close = entry.close;
    live.set(entry, pathKey(path));
  } else {
    after?.();
  }
  return f;
}

// -- A killed fork run: the reader rule's open, steps 1-3 (M2-S7 #2991) ------------
//
// The fork's contract `Ladruno_implementation/232a_mpm_contracts.md` §2.5 at
// `a1d4c7260` ("The reader rule: opening a particle file"); its reference is
// `ladruno_format.open_ladruno` / `repair_copy`, and the Python twin is
// `src/apeGmsh/results/readers/_ladruno_open.py`. The writer runs HDF5 SWMR from
// the end of each stage's layout to close, so a killed run leaves:
//   1. superblock flags 0x00 (closed, or killed in a layout before sec2 wrote the
//      flags): a plain read-only open;
//   2. flags 0x05 (WRITE_ACCESS | SWMR_WRITE): the plain open refuses; h5wasm opens
//      it SWMR-read ("Sr"), as h5py does, never after an h5clear. If that refuses,
//      the contract lets a reader repair a copy instead; the step stays 2;
//   3. flags 0x01 (WRITE_ACCESS only, killed while a layout was written): neither
//      open accepts it, so a COPY is repaired (flags 0x00; EOA = max(EOA, size),
//      zero-padded; lookup3 over bytes 0-43 at 44) and opened plainly. The copy is
//      removed when the handle closes; the user's file is never written.
// h5wasm reports no HDF5 error text (a refused open is a negative id), so the
// flags read from the superblock choose the step. A live writer's lock refuses the
// raw superblock read (Windows: EBUSY / EACCES / EPERM): `H5BusyError`, and
// nothing is copied. (h5wasm itself takes no file lock: emscripten's flock is a stub.)

// -- A read that says when HDF5 refused it ---------------------------------------------
//
// h5wasm's `Dataset.slice` and `.value` ignore H5Dread's status: when HDF5 cannot follow a
// damaged chunk index it prints its error stack and h5wasm returns the uninitialised buffer as
// data. Every one of those reads goes through one binding, `Module.get_dataset_data`, which does
// return the status. `openH5` and its siblings wrap that binding once per module (`checkedReads`,
// #3010), so any read of any dataset throws `H5ReadError` on a refusal, and a damaged dataset is
// never read as numbers. A reader needs nothing more than a file opened here. `readRows` is the
// typed, allocation-owning read the particle reader uses on the same binding.

/** The h5wasm module export `readRows` reads through (`h5wasm.Module`, after `ready`). */
export interface H5ReadModule {
  get_dataset_data(file_id: bigint, path: string, count: bigint[] | null, offset: bigint[] | null, strides: bigint[] | null, ptr: bigint): number;
  _malloc(size: number): number;
  _free(ptr: number): void;
  readonly HEAPU8: Uint8Array;
}

/** A dataset as `readRows` reads it (an h5wasm `Dataset`). */
export interface H5RowsDataset {
  file_id: bigint;
  path: string;
  shape: number[] | null;
  metadata: { type: number; size: number; signed: boolean };
}

/** HDF5 refused a read (its status was negative); nothing it returned is data. */
export class H5ReadError extends Error {}

/**
 * Rows `a..b` (every column) of a 1-D or 2-D integer or float dataset, as the typed array of
 * its stored type (int32, int64, float32, float64; the wasm heap is little-endian, as the
 * callers require of the file). Throws `H5ReadError` when HDF5 refuses the read.
 */
export function readRows(mod: H5ReadModule, ds: H5RowsDataset, a: number, b: number): Int32Array | BigInt64Array | Float32Array | Float64Array {
  const shape = ds.shape ?? [];
  if (shape.length !== 1 && shape.length !== 2) throw new Error(`${ds.path}: readRows reads a 1-D or 2-D dataset, not rank ${shape.length}`);
  const { type, size, signed } = ds.metadata;
  const make =
    type === 0 && signed && size === 4 ? (buf: ArrayBuffer) => new Int32Array(buf)
    : type === 0 && signed && size === 8 ? (buf: ArrayBuffer) => new BigInt64Array(buf)
    : type === 1 && size === 4 ? (buf: ArrayBuffer) => new Float32Array(buf)
    : type === 1 && size === 8 ? (buf: ArrayBuffer) => new Float64Array(buf)
    : null;
  if (make === null) throw new Error(`${ds.path}: readRows reads int32, int64, float32 or float64 data (class ${type}, size ${size})`);
  const cols = shape.length === 2 ? shape[1]! : 1;
  const n = b - a;
  if (n <= 0) return make(new ArrayBuffer(0));
  const nbytes = n * cols * size;
  const count = shape.length === 2 ? [BigInt(n), BigInt(cols)] : [BigInt(n)];
  const offset = shape.length === 2 ? [BigInt(a), 0n] : [BigInt(a)];
  const strides = shape.length === 2 ? [1n, 1n] : [1n];
  const ptr = mod._malloc(nbytes);
  if (!ptr) throw new Error(`${ds.path}: could not allocate ${nbytes} bytes to read rows ${a}..${b}`);
  try {
    const status = mod.get_dataset_data(ds.file_id, ds.path, count, offset, strides, BigInt(ptr));
    if (status < 0) throw new H5ReadError(`${ds.path}: HDF5 refused to read rows ${a}..${b} (its chunk index or data does not read)`);
    return make(mod.HEAPU8.slice(ptr, ptr + nbytes).buffer);
  } finally {
    mod._free(ptr);
  }
}

/** What `openH5ByRule` needs of the h5wasm module: `File` read-only (`"r"`) and SWMR-read (`"Sr"`). */
export interface H5RuleOpenable<F> {
  File: new (filename: string, mode: "r" | "Sr") => F;
  readonly FS: H5FS | null;
}

/** A live writer holds the file; nothing was copied or repaired. */
export class H5BusyError extends Error {}

/**
 * How to follow a run that is still writing (the fork's schema §7b.2, "To read a live file"):
 * the SWMR tail read with HDF5 file locking off, which never blocks the writer. Every
 * `H5BusyError` names it; Python's `_ladruno_open.SWMR_TAIL_READ` is the same text.
 */
export const SWMR_TAIL_READ = 'h5py.File(path, "r", swmr=True, locking=False)';

/** The step of the reader rule that opened a file (232a §2.5). */
export type OpenStep = 1 | 2 | 3;

export interface RuleOpened<F> {
  file: F;
  step: OpenStep;
  /** the user's file's superblock flags at the open; null for a version-0/1 superblock */
  flags: number | null;
  /** whether `file` is a repaired copy (step 3, or step 2 when SWMR-read refused) */
  repaired: boolean;
}

const SB_SIG = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/** HDF5's metadata checksum, `H5_checksum_lookup3` (Bob Jenkins' hashlittle), as an unsigned 32-bit number. */
export function lookup3(data: Uint8Array, initval = 0): number {
  const rot = (x: number, k: number) => ((x << k) | (x >>> (32 - k))) >>> 0;
  const word = (o: number) => (data[o]! | (data[o + 1]! << 8) | (data[o + 2]! << 16) | (data[o + 3]! << 24)) >>> 0;
  let n = data.length;
  let a = (0xdeadbeef + n + initval) >>> 0;
  let b = a;
  let c = a;
  let k = 0;
  while (n > 12) {
    a = (a + word(k)) >>> 0;
    b = (b + word(k + 4)) >>> 0;
    c = (c + word(k + 8)) >>> 0;
    a = (a - c) >>> 0; a = (a ^ rot(c, 4)) >>> 0; c = (c + b) >>> 0;
    b = (b - a) >>> 0; b = (b ^ rot(a, 6)) >>> 0; a = (a + c) >>> 0;
    c = (c - b) >>> 0; c = (c ^ rot(b, 8)) >>> 0; b = (b + a) >>> 0;
    a = (a - c) >>> 0; a = (a ^ rot(c, 16)) >>> 0; c = (c + b) >>> 0;
    b = (b - a) >>> 0; b = (b ^ rot(a, 19)) >>> 0; a = (a + c) >>> 0;
    c = (c - b) >>> 0; c = (c ^ rot(b, 4)) >>> 0; b = (b + a) >>> 0;
    n -= 12;
    k += 12;
  }
  if (n === 0) return c;
  const tail = new Uint8Array(12);
  tail.set(data.subarray(k, k + n));
  const tw = (o: number) => (tail[o]! | (tail[o + 1]! << 8) | (tail[o + 2]! << 16) | (tail[o + 3]! << 24)) >>> 0;
  a = (a + tw(0)) >>> 0;
  b = (b + tw(4)) >>> 0;
  c = (c + tw(8)) >>> 0;
  c = (c ^ b) >>> 0; c = (c - rot(b, 14)) >>> 0;
  a = (a ^ c) >>> 0; a = (a - rot(c, 11)) >>> 0;
  b = (b ^ a) >>> 0; b = (b - rot(a, 25)) >>> 0;
  c = (c ^ b) >>> 0; c = (c - rot(b, 16)) >>> 0;
  a = (a ^ c) >>> 0; a = (a - rot(c, 4)) >>> 0;
  b = (b ^ a) >>> 0; b = (b - rot(a, 14)) >>> 0;
  c = (c ^ b) >>> 0; c = (c - rot(b, 24)) >>> 0;
  return c;
}

/** Whether `sb` (at least 48 bytes) is a version-2/3 superblock with 8-byte offsets. */
function isV23(sb: Uint8Array): boolean {
  return sb.length >= 48 && SB_SIG.every((x, i) => sb[i] === x) && (sb[8] === 2 || sb[8] === 3) && sb[9] === 8;
}

/** The raw read the open uses; tests replace it to stand in for a lock. */
export const superblockIo = {
  readHead(path: string): Uint8Array {
    const fs = process.getBuiltinModule("node:fs");
    const fd = fs.openSync(path, "r");
    try {
      const buf = new Uint8Array(48);
      const n = fs.readSync(fd, buf, 0, 48, 0);
      return buf.subarray(0, n);
    } finally {
      fs.closeSync(fd);
    }
  },
};

/** The first 48 bytes of `path`; a read the OS refuses for a lock is `H5BusyError`. */
function readSuperblock(path: string): Uint8Array {
  try {
    return superblockIo.readHead(path);
  } catch (err) {
    const code = (err as { code?: unknown }).code;
    if (code === "EBUSY" || code === "EACCES" || code === "EPERM" || code === "EAGAIN") {
      throw new H5BusyError(
        `${path} cannot be read (${String(code)}): a live writer holds it; open it again when the run has stopped. ` +
          `To follow the run while it writes, use the SWMR tail read, ${SWMR_TAIL_READ}, and re-open it to refresh (schema §7b.2). Nothing was copied or repaired.`,
      );
    }
    throw err;
  }
}

/** The consistency flags of a version-2/3 superblock (0x00 after a clean close), or null for any other superblock. */
export function superblockFlags(path: string): number | null {
  const sb = readSuperblock(path);
  return isV23(sb) ? sb[11]! : null;
}

/**
 * The reader rule's repair, on a COPY: `dst` gets `src`'s bytes with flags 0x00,
 * EOA = max(stored EOA, file size) (zero-padded), and a new lookup3 at byte 44.
 * `src` is never written. Refuses `dst` = `src`, another superblock, or a bad checksum.
 */
export function repairCopy(src: string, dst: string): void {
  const fs = process.getBuiltinModule("node:fs");
  const nodePath = process.getBuiltinModule("node:path");
  if (pathKey(nodePath.resolve(src)) === pathKey(nodePath.resolve(dst))) throw new Error("repairCopy repairs a copy: dst must differ from src");
  const sb = readSuperblock(src);
  if (!isV23(sb)) throw new Error(`${src}: not a version-2/3 HDF5 superblock with 8-byte offsets, so the reader rule's repair does not apply`);
  const view = new DataView(sb.buffer, sb.byteOffset, 48);
  if (lookup3(sb.subarray(0, 44)) !== view.getUint32(44, true)) throw new Error(`${src}: the superblock checksum does not match`);
  fs.copyFileSync(src, dst);
  const size = BigInt(fs.statSync(dst).size);
  const stored = view.getBigUint64(28, true);
  const eoa = stored > size ? stored : size;
  const fixed = new Uint8Array(sb.subarray(0, 48));
  const fv = new DataView(fixed.buffer);
  fixed[11] = 0x00;
  fv.setBigUint64(28, eoa, true);
  fv.setUint32(44, lookup3(fixed.subarray(0, 44)), true);
  const fd = fs.openSync(dst, "r+");
  try {
    fs.writeSync(fd, fixed, 0, 48, 0);
    if (size < eoa) fs.ftruncateSync(fd, Number(eoa));
  } finally {
    fs.closeSync(fd);
  }
}

/** Repair a copy of `path` in a new temp directory and open it plainly; the directory goes when the handle closes. */
function openRepaired<F>(h5: H5RuleOpenable<F>, path: string): F {
  const fs = process.getBuiltinModule("node:fs");
  const os = process.getBuiltinModule("node:os");
  const nodePath = process.getBuiltinModule("node:path");
  const dir = fs.mkdtempSync(nodePath.join(os.tmpdir(), "apegmsh-ladruno-repair-"));
  const drop = () => fs.rmSync(dir, { recursive: true, force: true });
  try {
    const copy = nodePath.join(dir, nodePath.basename(path));
    repairCopy(path, copy);
    return openTracked(h5, copy, "r", drop);
  } catch (err) {
    drop();
    throw err;
  }
}

/**
 * Open `path` read-only by the reader rule (above), with exact file ids. A file no
 * step opens comes back as the plain open's File, its id negative, for the caller
 * to refuse as it refuses any unreadable file.
 */
export function openH5ByRule<F extends { file_id: bigint | number }>(h5: H5RuleOpenable<F>, path: string): RuleOpened<F> {
  const sb = readSuperblock(path);
  const flags = isV23(sb) ? sb[11]! : null;
  const plain = openTracked(h5, path, "r");
  if (!(Number(plain.file_id) < 0)) return { file: plain, step: 1, flags, repaired: false };
  // Step 2 is the SWMR-read open, for a plain open's "already open for write" (flags 0x05) or
  // "truncated file" (the stored EOA past the file's end), which an SWMR-read open skips; as
  // the fork's open_ladruno, it applies to flags 0x00 too (SWMR-read accepts 0x00 and 0x05).
  const truncated = flags !== null && new DataView(sb.buffer, sb.byteOffset, 48).getBigUint64(28, true) > BigInt(process.getBuiltinModule("node:fs").statSync(path).size);
  if (flags === 0x05 || (flags === 0x00 && truncated)) {
    const swmr = openTracked(h5, path, "Sr");
    if (!(Number(swmr.file_id) < 0)) return { file: swmr, step: 2, flags, repaired: false };
    if (flags === 0x00) return { file: plain, step: 1, flags, repaired: false };
    return { file: openRepaired(h5, path), step: 2, flags, repaired: true };
  }
  if (flags === 0x01) return { file: openRepaired(h5, path), step: 3, flags, repaired: true };
  return { file: plain, step: 1, flags, repaired: false };
}

/** What `appendH5` needs of the h5wasm module. */
export interface H5Appendable<F> {
  File: new (filename: string, mode: "a") => F;
  readonly FS: H5FS | null;
}

/**
 * VU9b (ADR 0130 A4, HG9 rule 1): open a writer's own partial for appending
 * (`"a"`). Only a `.partial` is ever opened for writing: the live `.user.h5`
 * is copied, and the copy is appended to, never the original.
 */
export function appendH5<F extends { file_id: bigint | number }>(h5: H5Appendable<F>, path: string): F {
  if (!path.endsWith(".partial")) throw new Error(`appendH5 opens only a writer's own .partial for writing, never ${path} (HG9 rule 1: the live file is never opened for writing)`);
  prepare(h5);
  const f = new h5.File(path, "a");
  if (Number(f.file_id) < 0) throw new Error(`HDF5 could not open ${path} for appending`);
  return f;
}

/** What `createH5` needs of the h5wasm module. */
export interface H5Creatable<F> {
  File: new (filename: string, mode: "x") => F;
  readonly FS: H5FS | null;
}

/** Create `path` for writing, exclusively (`"x"`: refused when the name exists), with HDF5 comparing exact file ids. */
export function createH5<F extends { file_id: bigint | number }>(h5: H5Creatable<F>, path: string): F {
  prepare(h5);
  const f = new h5.File(path, "x");
  if (Number(f.file_id) < 0) throw new Error(`HDF5 could not create ${path} (it exists, or its folder refuses a new file)`);
  return f;
}
