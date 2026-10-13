### CHANGED — `element_index_space` verifies only as a scalar vlen UTF-8 string (program slice B5-h, #1811)

`architecture/h5-schema.md` now states the rule in one sentence: a file is
verified if and only if `element_index_space` is a variable-length UTF-8
string exactly equal to `"fem"`. It also states the stamp's full on-disk
form: a scalar, variable-length, UTF-8 string, which is what every writer
(the recorder transcoder and `DomainCapture`, through
`NativeWriter.stamp_element_index_space`) emits.

**Changed (behaviour):** a variable-length **bytes** (ASCII charset)
`"fem"` stamp no longer verifies. `NativeReader.element_index_verified`
used to accept it, because h5py decodes it to the same `str`. The reader
now checks the datatype before the text. A fixed-length string or bytes
value (padded or not), an array, or a non-string value reads as
unverified too, even when its text is `fem`. Before, the reader rejected
those only because `str()` of a bytes value is `"b'fem'"`, and another
HDF5 reader that strips fixed-length padding would have accepted them. No
apeGmsh writer ever wrote any of these forms.
