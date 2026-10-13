### CHANGED — `element_index_space` verifies only as a scalar vlen UTF-8 string (program slice B5-h, #1811)

`architecture/h5-schema.md` now states the stamp's on-disk form: a scalar,
variable-length, UTF-8 string holding exactly `fem`, which is what every
writer (the recorder transcoder and `DomainCapture`, through
`NativeWriter.stamp_element_index_space`) emits. `NativeReader.
element_index_verified` now checks that datatype before the text. A
fixed-length string (padded or not), an ASCII-charset string, bytes, an
array or a non-string value reads as unverified, even when its text is
`fem`. Before, the reader rejected most of these only because `str()` of
a bytes value is `"b'fem'"`, and it verified a variable-length ASCII
`"fem"`. Another HDF5 reader that strips fixed-length padding would have
compared such files equal.
