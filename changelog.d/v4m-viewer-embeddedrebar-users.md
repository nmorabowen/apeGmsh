### FIXED — apeGmshViewer: the materials window counts embeddedRebar tie users (V4m, #1846)

A bond material used only by `g.reinforce` `LadrunoEmbeddedRebar` ties no
longer reads "used by nothing". The reader now reads the argument columns of
`/opensees/constraints/embeddedRebar` (opensees 2.29.0) and takes each tie's
bond tag from the integer after its one `-bond` token, by the element's
command grammar. The window shows "used by N embeddedRebar ties" beside the
element users. A tie whose bond tag cannot be read (no tag, a non-integer
one, `-bond` twice, a missing column, or a tag that matches no uniaxial
material) makes the window say "users unknown: referenced by embeddedRebar
ties", never 0. A file without the group reads as before.
