### ADDED — MPI for spring beds and composed models: routed element-less nodes, `FEMData.repartition`, hosted `Assembly` (ADR 0120)

- **Node-pair elements emit partitioned.** `apeSees` no longer refuses
  `ops.element.ZeroLength(nodes=...)` (and the other node-pair forms) on an
  OpenSeesMP deck. Every element-less node (a decoupled ground or side
  node) takes the lowest rank of the mesh node it is tied to; the spring,
  its nodes, the ground `fix`, `ndf`, `mass` and a side node's `equalDOF`
  are written on that one rank. A node-pair between mesh nodes held by no
  common rank raises `BridgeError`. The snapshot given to `apeSees` is not
  modified.
- **`FEMData.repartition(n_parts, *, weights=None)`** splits a snapshot
  into MPI ranks as one graph (recursive coordinate bisection of the element
  centroids, weighted by node count or by `weights`), whatever module it
  came from: a composed model is no longer one rank per module.
- **`Assembly(name, host=fem_or_model_h5)`** grafts instances onto an
  existing snapshot whose ids and bare names stay; a bare port names a
  host group. **`Assembly.fem()`** returns the merged snapshot without
  building a bridge; a hosted one is unpartitioned, so
  `apeSees(asm.fem().repartition(16))` is a 16-rank deck. A hosted
  assembly refuses reference nodes, `partition_rank`, `bridge()` and `h5()`.
- New ADR 0120 (Proposed). Numeric twin (serial vs 1 / 2 / 4 ranks, eigen
  and transient): `tests/opensees/subprocess/test_partitioned_spring_bed_compose_twin.py`.
