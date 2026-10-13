### FIXED — rigid_diaphragm with an explicit master_point keeps the declared master (B5, #1817; closes #1804)

**Changed (behaviour):** with `master_point` given, `g.constraints.rigid_diaphragm`
(and the chain-phase route for `from_h5` sessions) no longer picks any plane node
nearest the point as master. A master label that resolves to one node is the
master, and the point only sets the plane. With several master nodes, the plane
node nearest the point is the master, and on a tie a master-label node wins over a
coincident slave-set node.

- **Fixes #1804, a silent wrong answer.** In the reference frame
  (`frame_modal.py`), the floor-2 point (6, 6, 6.4) coincides with joint
  `J11_L2`. Before this fix that joint became the master and the declared,
  fixed and massed `floor_2_master` became a zero-offset slave. A `fix` on its
  ux was dropped by the Transformation handler, so a floor meant to be held
  slid 0.48 mm under a 100 kN load. The declared node is now the master, and
  the fix holds.
- **New refusals (`ValueError` at resolution):** a one-node master that lies off
  the plane through the point, naming the node, its coordinates, its distance
  from the plane and the plane; and a multi-node master label with no node in
  that plane.
- **Code that relied on the old rule** (a one-node master label, with a point
  nearer to another node) now gets the labelled node as master. If the old
  pick was intended, label that node as the master.
