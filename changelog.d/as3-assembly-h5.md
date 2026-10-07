### ADDED — Assembly archives: the `/assembly` zone, `Assembly.h5` and `Assembly.from_h5` (ADR 0117 P3, AS3)

After `ops = asm.bridge(ndm, ndf)` and the analysis declarations,
`asm.h5("stack.h5")` writes exactly what `ops.h5` writes plus one optional
root zone, `/assembly`, stamped `/meta/assembly_schema_version = 1.0.0` (its
own key; floor 1.0.0). `/assembly/instances` records each instance's label,
source path, source `fem_hash` and `model_hash`, translation, rotation,
relocated FEM-id window and rank hint; `/assembly/ties` records each tie's
name, kind, ports, options and the number of records it resolved to. An
assembly with no ties writes an empty `/ties` table, writing again replaces
the zone, and a refused row leaves the file unchanged.
`Assembly.from_h5("stack.h5")` re-lists the declared instances and ties
without opening any instance file; the flat zones stay authoritative, so
every model reader opens the archive as it opens a plain `model.h5`, and a
plain file opens exactly as before (`Assembly.from_h5` refuses it).
`/provenance` gains `assembly/instances/<label>` and
`assembly/ties/<name|#k>` records at the declaring lines. The neutral,
opensees and provenance schema versions do not change.
