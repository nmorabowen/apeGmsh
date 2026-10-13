### FIXED — rigid_diaphragm never promotes a slave node to master (B5-j, #1838; closes #1829)

**Changed (behaviour):** with an explicit `master_point` and a master label that
resolves to several nodes, the master is now the plane node of the **master
label** nearest the point. A node that belongs only to the slave label is never
the master, however near the point it lies. #1826 covered only a slave node that
is coincident with a master node; a strictly nearer slave still became the master.
A master label with no node in the plane through the point is still refused
(`ValueError`, naming the label and the plane).

- **Fixes #1829, a silent wrong answer.** In the frame_modal frame with floor 2's
  master label widened to {`floor_2_master`, `J22_L2`} and the point 0.14 m from
  slave joint `J00_L2`, that joint became the master and `floor_2_master` became a
  slave. A `fix` on its ux was then dropped, so a floor meant to be held slid
  0.48 mm under a 100 kN load. The master is now `floor_2_master`, and the fix
  holds.
- **Code that relied on the old rule** (a point placed on a slave node to make it
  the master) now gets the nearest master-label node instead. Add that node to the
  master label if it was meant to be the master.
