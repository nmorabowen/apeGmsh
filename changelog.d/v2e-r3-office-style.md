### CHANGED — apeGmshViewer: the office graphic style (ADR 0112, V2e round 3)

One theme-token module, `apeGmshViewer/src/theme/tokens.ts` (citing
apeGraphStyle `__init__.py` v0.1.0, 2026-10-03), is the only source of
colour and type in the app: a test greps the rest of `src/` for colour
literals. Group colours (R2, revised) follow the office `main_colors` order,
adapted to the dark theme in OKLCH with alternating lightness: legend
neighbours step by at least 0.10 L, no pair of the sixteen colours is told
apart by a red/green difference alone under a Machado 2009 protanopia
simulation, and past eight groups the hue repeats one lightness step away
with a striped legend chip as the second cue. A colour-by-role mode draws
column, wall, beam, concrete and void in the office colours from a `role`
the file records and from nothing else; a file without one (every file
today) is said so in the legend and drawn as unassigned. Archivo Narrow
(SIL OFL 1.1, licence bundled) is the app's type. Every colour clears WCAG
3:1 (graphics) or 4.5:1 (text) on the background; the test prints the
ratios. The selection core takes the office accent in a later slice
(after #1318).
