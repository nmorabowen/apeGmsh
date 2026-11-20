### ADDED — apeGmshViewer draws crack ticks on layered shells (crack plots, slice 2)

The Cracks card now covers layered shells: `section.fiber.crackPattern` on ASDShellQ4 / ASDShellT3 with a LayeredShell section. At each Gauss point, each crack in the outer concrete layer of a face is drawn as a short tick along its trace on the shell surface (t = e3 × n). The crack vectors are rotated out of the shell's own frame, which comes from `MODEL/LOCAL_AXES` (written by the fork since nmorabowen/OpenSees#1133). An element with no recorded frame is never drawn with a guessed one; the card names how many there are.
- **Faces:** a tick sits at its layer's face, t/2 off the mid-surface, which is on the slab's face when the shells are drawn extruded. It is red on the face toward the positive global axis the shell's normal is most along (a slab's top, a wall's +x or +y side) and blue on the other. A face picker shows both faces or one.
- **Controls:** the minimum opening and the size apply as for the discs.
- **Reads:** main reads only each layer group's two outer layers per step. Each element's frame and section thickness are sent once, at open.
- **Stills:** the capture harness adds `shellcracks-*` stills.
