### FIXED — apeGmshViewer opens /provenance written from `python -c`, stdin or an IPython cell (#1435)

The app refused every `/provenance` whose source was a pseudo-file. The
writer records such a source under its `<...>` name (`<string>`, `<stdin>`,
`<ipython-input-…>`) with an empty `sha256`, as h5-schema.md specifies,
and the app read that as "malformed", so go-to-source was off for the
whole file. An empty digest is now accepted for a `<...>` path and still
refused for any other path, and a pseudo-file path is no longer joined to
`@base_dir`. The Sources header's "not marked" notice now follows the
`records/origin` column rather than the version.
