### FIXED — apeGmshViewer reads `.ladruno` `INFO@SPATIAL_DIM` by its HDF5 type, as Python does (#2324)

The viewer read `INFO@SPATIAL_DIM` from the value h5wasm decodes. That value
hid what Python sees, so four values that Python refuses were read as 2. h5wasm
decodes a fixed-length ASCII string as windows-1252, so `b'\xa02'` became
`" 2"`. It cuts a fixed-length string at its first NUL, so `b'2\x003'` became
`"2"`. It returns a one-member compound as its member, so `(2,)` became `[2]`.
An opaque `b'\x02'` was read as 2 in the same way. A float16 or float128 value
threw a plain `Error` with no file named.

The viewer now reads the attribute's HDF5 type first. It refuses a compound, an
opaque value or any other class that is not a number or a string. It refuses a
float other than float32 or float64, which h5wasm cannot decode, and a
fixed-length string with a byte outside ASCII. It also refuses a NULLPAD or
SPACEPAD string with a NUL before its last byte, because h5wasm shows no byte
past the NUL. Each refusal names the file and the reason as a `layout`
refusal at open.

The viewer is now stricter than Python in two cases, and the tests pin both:
a float16 `2.0`, and a NUL-padded `b'2\x00'`. Python reads both as 2. Every
committed `.ladruno` stores the attribute as int32 and opens as before.
