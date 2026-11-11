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
  exactInodes(h5.FS);
  const f = new h5.File(path, "r");
  const handle = f as unknown as { close?: () => unknown; file_id?: bigint | number };
  // A failed open (negative id) holds nothing; a handle is tracked until its own close.
  if (typeof handle.close === "function" && !(Number(handle.file_id) < 0)) {
    const close = handle.close.bind(f);
    const entry = { close: () => (live.has(entry) ? (live.delete(entry), close()) : undefined) };
    handle.close = entry.close;
    live.set(entry, pathKey(path));
  }
  return f;
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
  exactInodes(h5.FS);
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
  exactInodes(h5.FS);
  const f = new h5.File(path, "x");
  if (Number(f.file_id) < 0) throw new Error(`HDF5 could not create ${path} (it exists, or its folder refuses a new file)`);
  return f;
}
