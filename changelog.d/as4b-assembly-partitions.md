### FIXED — An assembly's default deck is serial; `Assembly.instance(partition_rank=)` ranks it (ADR 0117 D6, AS4-b, #1530)

`Assembly.bridge()` merged its instances onto an empty broker, and the merge
engine kept rank 0 for a host the assembly does not have, so the default
`ops.tcl()` came out partitioned with an empty `if {[getPID] == 0} {}` block:
single-process OpenSees saw no element, and OpenSeesMP with one rank per
instance dropped the last one. The merge engine now reserves rank 0 for the
host only when the host owns an element. An assembly whose instances carry
no rank is therefore unpartitioned, and `tcl()` writes the serial deck
(`tcl(flat=True)` is no longer needed). `instance(..., partition_rank=k)`
places an instance on OpenSeesMP rank `k` (ADR 0038 Layer 2): every instance
carries a rank or none does, one rank holds one instance, and `bridge()`
requires the ranks to run `0 .. n-1`; each refusal raises `AssemblyError`
before anything is recorded. Reference nodes (`node()`) live on rank 0 and
are declared on another rank before a coupling there uses them; each
cross-instance MP line is written on every rank that owns one of its nodes
(INV-9). A ranked instance of an assembly archive is refused. The rank
round-trips through `/assembly/instances` (`-1` for none) and
`Assembly.from_h5`. `AssemblyRankWarning` is removed. A host-less
`FEMData().compose(...)` chain without rank hints is also unpartitioned now.
