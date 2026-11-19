### TESTS — the study e2e draws and refuses a compare variant's user result in render mode (#2960)

`apeGmshViewer/test/spec/study.e2e.ts` runs a second `--render` over the
VU10 study, with VU19's `.user.h5` beside the cracked model and a third
row, `Remeshed` (the plate on another mesh, new fixture
`apeGmshViewer/test/fixtures/compare_render/`). A shot naming the Cracked
row's `footfall` is drawn: the manifest names it
`user.compare: {variant, entry}`, its recorded extrema are VU19's closed
form, and its still is written. A shot naming the Remeshed row's
`footfall` is refused with both `fem_hash` values, writes no PNG, and the
run exits 1. Test-only: no app behaviour changes.
