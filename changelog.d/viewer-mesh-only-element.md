### FIXED — apeGmshViewer shows a mesh cell the bridge did not emit as mesh only (#2672)

A cell in `/elements` with no `/opensees/element_meta` row is legitimate: a helper
or edge line in a group that was given no element. The inspector now shows
"none (mesh only, not emitted)" for it and no longer lists it as a problem.
