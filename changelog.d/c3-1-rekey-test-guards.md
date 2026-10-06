### CHANGED — path-bound test guards re-keyed by symbol (C3.1, #1428)

The dock-invariant, dim-filter-key and `ModelData` AST guards now find their
target by `class`/`def` name (AST walk) plus package siblings, and the
viewer-state contract resolves each guarded hub as a module or a package with
one budget over the whole package. A missing hub, an ambiguous symbol or an
allowlist key that matches nothing now fails loudly instead of passing over
zero files. No budget was changed.
