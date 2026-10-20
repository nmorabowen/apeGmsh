### FIXED — apeGmshViewer reads `.ladruno` `INFO@SPATIAL_DIM` by its HDF5 type, as Python does (#2324)

The viewer read `INFO@SPATIAL_DIM` from the value h5wasm decodes. That value
hid what Python sees, so values that Python refuses were read as 2:

- a fixed-length ASCII string `b'\xa02'`, which h5wasm decodes as windows-1252
  to `" 2"`;
- a variable-length ASCII string `b'\xa02'`, which h5wasm decodes one char per
  byte;
- a fixed-length string `b'2\x003'`, which h5wasm cuts at its first NUL;
- a one-member compound `(2,)`, which h5wasm returns as its member;
- an opaque `b'\x02'`.

A float16 or float128 value threw a plain `Error` that named no file.

The viewer now reads the attribute's HDF5 type first. Each refusal names the file
and the reason, as a `layout` refusal at open. The viewer refuses:

- any class other than an integer, an integer enum, a float or a string;
- a float other than float32 or float64, which h5wasm cannot decode;
- a fixed-length string with a byte outside ASCII;
- a NULLPAD or SPACEPAD fixed-length string that holds any NUL, because h5wasm
  shows no byte past it (a NULLTERM string ends at its NUL, as in Python);
- a variable-length ASCII string whose bytes are not UTF-8. h5py decodes these
  bytes as UTF-8, and the viewer now does the same, so `b'\xc2\xa02'` reads as 2.
  A leading UTF-8 BOM is kept, as h5py keeps it, so `b'\xef\xbb\xbf2'` is
  refused.

If h5wasm's metadata lacks a field the rule needs, the open is refused the same
way, naming the file.

These are known, intentional divergences from Python: the viewer refuses each of
them, and Python reads it as 2. Each has a test that pins it.

- a float16 `2.0`;
- a NULLPAD `b'2\x00'`;
- an HDF5 array type `(1,)` of int32;
- a bitfield;
- a variable-length sequence of int32.

No writer, current or past, produces any of them. The fork and apeGmsh write an
int32 with dataspace `(1,)`, which still reads.

One known gap remains, pinned by a test. h5wasm's UTF-8 decoder reads a
variable-length UTF-8 string with the overlong `b'\xc0\xb2'` as `"2"`, and
Python refuses that string. Every committed `.ladruno` opens as before.
