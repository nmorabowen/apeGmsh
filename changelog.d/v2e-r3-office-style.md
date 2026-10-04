### CHANGED — apeGmshViewer: the office graphic style (ADR 0112, V2e round 3)

One theme-token module, `apeGmshViewer/src/theme/tokens.ts` (citing
apeGraphStyle `__init__.py` v0.1.0, 2026-10-03), is the only source of
colour and type in the app: a test greps the rest of `src/` for colour
literals. Group colours (R2, revised) are the office `main_colors`, adapted to the
dark theme in OKLCH: sixteen slots (the eight colours and the same hues one
lightness step away, the latter with a striped legend chip as the second
cue), no pair of which is told apart by a red/green difference alone under
Machado 2009 protanopia and deuteranopia simulations. Groups that are
neighbours in the view (an element of each shares a node; the adjacency is
computed at load and kept in the state) are assigned slots greedily so that
every adjacent pair is apart under both deficiencies by a lightness step or a
hue gap, and the office orange never sits beside the dark gold; the same
file always gets the same colours. A colour-by-role mode draws
column, wall, beam, concrete and void in the office colours from a `role`
the file records and from nothing else; a file without one (every file
today) is said so in the legend and drawn as unassigned. Archivo Narrow
(SIL OFL 1.1, licence bundled) is the app's type. Every colour clears WCAG
3:1 (graphics) or 4.5:1 (text) on the background; the test prints the
ratios. The selection core (the picked element's outline and fill,
`src/renderer/selection.ts`) is the office accent #E69F00; the white
fixed-width halo and the pulse from round 2 are unchanged.
