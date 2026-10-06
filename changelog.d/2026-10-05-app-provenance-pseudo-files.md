### FIXED — apeGmshViewer opens /provenance written from `python -c`, stdin or a notebook (#1435)

The app refused every `/provenance` with an empty `sha256`. As
h5-schema.md specifies, the writer records `""` for a pseudo-file source
(`<string>`, `<stdin>`, `<ipython-input-…>`) and for a source it could not
read, such as a current Jupyter cell's never-written
`<tmp>/ipykernel_<pid>/<hash>.py`. The app read that as "malformed", so
go-to-source was off for the whole file.

- `""` is now accepted for any path. Only a non-empty value that is not 64
  hex digits is refused.
- A source with no digest opens when its path is on disk. Otherwise, and
  always for a pseudo-file, its go-to-source button is disabled and the
  row says "source not recorded".
- A pseudo-file path is no longer joined to `@base_dir`.
- The Sources header's "not marked" notice now depends on whether the
  file has the `records/origin` column, not on its version.
