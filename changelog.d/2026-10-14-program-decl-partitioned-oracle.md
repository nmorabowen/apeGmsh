### TESTS — `program.decl` side-channel oracle under partitioned and staged-partitioned emits (#1912)

The A7 table-join oracle for the six side-channel verbs gains two
two-rank models, `partitioned` and `staged_partitioned`: two initial
stresses in one scope (the global one, or one stage's), each over bars of
both ranks, plus an absorbing flip and two zeroings. Each rank's block
writes both records' `addToParameter` calls and every call must join its
own record's declaration, so a decl carried over from a scope's first
record now fails these cases, not only the flat one. Tests only; no
behaviour or schema change.
