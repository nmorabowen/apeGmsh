### FIXED — an MPCO or `.ladruno` element filter whose id names both a paired element and a sentinel element is refused instead of read silently

An element the bridge emits without a mesh id (a node-pair `ZeroLength`
spring, an interface `zeroLength`) is addressed by its OpenSees tag; a
paired element by its fem id. Under the default
`apeSees(fem, element_tags="sequential")` the two can coincide in an
ordinary model: a 4x4x4 `stdBrick` block with surface physical groups has
its bricks at fem 33..96 (ops 7..70), and five ground springs take ops
71..75. Id 71 then names brick fem 71 and spring 71. A filter `ids=[71]`
returned the brick with no warning, and an unfiltered read labelled both
elements 71.

For a bound `model.h5` that describes the result files, the reader now:

- warns `WarnElementIdCollision` once, on the first element read, naming
  the shared ids;
- raises `ValueError` for a filter that holds a shared id, naming it and
  the paired element's ops tag;
- warns `WarnElementIdCollision` on each unfiltered read whose index
  carries a shared id.

The model still opens and every other id reads as before. Emitting with
`element_tags="fem"` keeps the id spaces apart (the springs move to ops
217..221). A stub `model_h5=` that does not describe the files stays
silent.
