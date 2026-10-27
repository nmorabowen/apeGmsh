### FIXED — `LadrunoMembrane` refuses an empty `NodeSet` for `support_nodes=` and `diaphragm_nodes=`

An empty `NodeSet`, such as a group that matched no node, used to drop
`-supportNodes` or `-diaphragmNodes` from the deck without a word. Both
options now refuse it with the message `()` gets
(`... must not be empty`), whether the set is passed bare or inside a
sequence. `support_nodes=()` is now refused too, as `diaphragm_nodes=()`
already was.

The text of `MembraneHoleNodeFreeWarning` is corrected. From fork WP-204
the membrane condenses its in-plane stiffness onto hole nodes as well, but
in the panel's plane only. An unframed hole node still has free
out-of-plane DOFs, so the warning stays; it no longer says the fork
ignores hole nodes.

New tests pin the staged and partitioned build sites that count
`diaphragm_nodes=` (stage ownership and the membrane's MPI rank). The ndf
inference step dropped its named-node block: LadrunoMembrane is
ndf-adaptive, so the block never changed an inferred ndf.
