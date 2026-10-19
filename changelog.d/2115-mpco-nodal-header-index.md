### FIXED — MPCO reads nodal-force buckets with a header index above 0 (#2115)

MPCO opens a second response header (`<tag>-<Class>[<rule>:<cust>:1]`) when
elements of one class write a different META. The nodal-force reader
(`nodal_resisting_force_*`, `nodal_resisting_moment_*` and their `_local_`
forms, from `globalForce` and `localForce`) kept header 0 only and named every
other header in `NodalForceBucketSkippedWarning`, so those elements read no
rows. A header above 0 is now read once its own META spells the catalog layout:
`NUM_COLUMNS` and the single META block, and the `META/COMPONENTS` names
(`Px_1`, `Mz_2`; `N_1`, `Vy_1`, the 2-D `V_1` or `V1`) at each of the
layout's nodes, in catalog order. A header whose META does not spell it (other
names, another node count or order, MPCO's unnamed `C1..Cn`, or no
`COMPONENTS`) is still left unread and named in the warning, now with the
mismatch; header 0 reads as before.
