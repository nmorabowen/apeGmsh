### FIXED — `.ladruno` reads a 2-D `localForce` spelling only behind the 2-D gate, and the 2-D disp beams read their line stations (#2013)

- **The 2-D names are gated.** A `localForce` bucket tagged with the 2-D names
  `N_1,V_1,M_1,N_2,V_2,M_2` used to read `V` as `shear_y` and `M` as
  `bending_moment_z` for every class. ComponentElement3d and
  ForceBeamColumnCBDI3d tag a 12-value 3-D vector with those six names, and
  the recorder keeps six filler columns per row, so their line stations read
  as NaN (zeros on older builds) with no warning. Now the names apply only when
  the bucket's DATA holds exactly the tagged columns, no 3-D base appears
  beside them, the file is 2-D, and the class is no `...3d` class, the gate
  MPCO applies. Any other such bucket lists no line-station component and
  warns, and a read of its elements raises a `ValueError` naming the bucket
  and its class. Reads of other elements go on.
- **The underscore-less spelling reads.** DispBeamColumn2d and
  LadrunoDispBeamColumn2d tag their `localForce` `N1,V1,M1,N2,V2,M2`. The
  reader skipped those names, so these elements had no line stations. Behind
  the same gate, they now read both stations with the one sign table, and they
  meet the force beams at a junction to `rtol=1e-10`.
