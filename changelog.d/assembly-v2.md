### ADDED — Assembly v2: instances of model files, tied by label, built into one bridge (ADR 0117 P1, AS1)

`from apeGmsh.assembly import Assembly` now gives `instance`, `tie` and
`bridge`. `asm.instance("pier_1", "pier.h5", translate=..., rotate=((ax, ay,
az), theta))` places a saved `model.h5`; every instance is namespaced
(`pier_1.top`, `pier_1.steel`) and the same file may be placed any number of
times (it is read once). `asm.tie("pier_1.top", "pier_2.bot",
enforce="equation")` declares an assembly-level tie. `asm.bridge(ndm, ndf)`
returns one forward `apeSees` over the flat FEM, with the ties resolved and
each instance's materials, sections and element specs rehydrated from its
`/opensees` zone under `{instance}.{name}`; declare fixes, loads, recorders
and the analysis on it (model content travels, analysis content does not).
Element tags default to the relocated FEM ids (`element_tags="fem"`).

P1 rehydrates `ElasticIsotropic`, `ElasticMembranePlateSection`, `stdBrick`
and `ShellMITC4` with one element spec per physical group; any other
carried content raises `AssemblyError` naming it. A bad label or port, and a
tie that resolves no record, raise `AssemblyError` before anything is
recorded or built. `src/apeGmsh/assembly.py` became the package
`src/apeGmsh/assembly/`; the v1 `add` / `couple` / `materialize` still work
unchanged (moved to `_v1.py`) until their removal slice.
