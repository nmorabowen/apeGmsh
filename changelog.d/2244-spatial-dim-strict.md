### FIXED — `.ladruno` `INFO/SPATIAL_DIM` refuses a complex value and a non-ASCII digit, as the viewer does (#2244)

`_file_spatial_dim` read the attribute through `float()`. A complex
`2+5j` was read as 2, with only a `ComplexWarning`, and a string of any
Unicode decimal digit (`"２"`, `"٢"`, `"𝟐"`) was read as its value. Both now
raise `SpatialDimInvalid`, naming the file and the value. The rule is now the
viewer's (`apeGmshViewer/src/reader/ladrunoInfo.ts`, #2265). A real integer,
float or bool, or an ASCII decimal string, reads when it equals 1, 2 or 3.
Around the string, only the whitespace `float()` itself skips is allowed, so
U+FEFF and U+001C stay refused. Fixed-length bytes read as their ASCII text.
No committed fixture carries such a value.
