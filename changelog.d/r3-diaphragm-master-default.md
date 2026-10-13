### CHANGED — rigid_diaphragm takes its master from a one-node label (R3-S4, #1786; closes #1778)

**Changed (behaviour):** `g.constraints.rigid_diaphragm(master, slaves, ...)` and
`Assembly.rigid_diaphragm` default to `master_point=None`. With the point omitted,
the master label or port must resolve to exactly one node; that node is the
master, and the diaphragm plane passes through it. A master that resolves to zero
or to several nodes is refused (`ValueError` at resolution, `AssemblyError` from
`bridge()`), naming the count and `master_point`, and so is a plane through the
master that holds no slave node. A given `master_point` keeps the old rule: the
plane passes through the point and the master is the plane node nearest to it.
That plane, too, is now refused (`ValueError` naming the point, the plane and the
node count) when it collects no master or no slave. On main, such a plane gave a
master-less record and the deck line `rigidDiaphragm 3 0` on the direct
`g.constraints` route, while the chain-phase route (`from_h5` sessions,
`_chain_phase_router.py`) silently dropped it. Both routes now raise at
declaration.

- **Fixes #1778, a silent wrong deck.** The old default point was the origin, so
  a floor above z = 0 declared without `master_point` collected no node: the
  record was `(3, 0, [])`, the deck line `rigidDiaphragm 3 0`, and the live
  `eigen` raised `OpenSeesError`. That floor now resolves on its labelled master.
- **The master node can change with no error.** On a floor at z = 0 with
  `master_point` omitted and a joint at the origin, main made the origin joint
  the master and turned the labelled one-node master into a slave, for example
  `(3, 1, [2, 3, 4, 5])`. It now resolves `(3, 5, [1, 2, 3, 4])`: the master is
  the declared node, which is correct.
- **Code that relied on the origin default** (a floor at z = 0 whose master
  label holds several nodes) now raises; pass `master_point=(0, 0, 0)` to keep
  the old resolution.
- **Assembly archives** store an omitted `master_point` as `null`. A
  one-node instance port can now be the master without a point.
- **`p.load` on a labelled node:** there is no `label=` on `p.load`; the
  docstring and the point-load how-to show
  `p.load(node=ops.nodes.one(label="tip"), forces=...)`.
