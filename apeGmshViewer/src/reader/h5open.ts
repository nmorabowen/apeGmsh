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

/** Open `path` read-only with h5wasm, with HDF5 comparing exact file ids. */
export function openH5<F>(h5: H5Openable<F>, path: string): F {
  exactInodes(h5.FS);
  return new h5.File(path, "r");
}
