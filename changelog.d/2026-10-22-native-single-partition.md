### CHANGED — `NativeWriter` refuses a second partition per stage (ADR 0123 slice 2, #2453)

A native `results.h5` stage now holds at most one partition (ADR 0123 D5).
Every `NativeWriter.write_*` call passes through `_require_partition`, which
raises `NativeSecondPartition` (a `ValueError`) when the stage already holds a
partition under another id. The message names the file, the stage, the
existing id and the refused id, and nothing of the refused write reaches the
file. The rule counts partitions and ignores their names: a stage whose only
partition is `partition_3` is legal, a stage's writes re-enter its partition
freely, and each stage has its own. Every writer in apeGmsh already wrote one
partition, so no in-repo caller changes. A third-party script that wrote
several partitions through `NativeWriter` must record a partitioned run as a
`.ladruno` or MPCO part set instead, whose readers reduce across ranks.

Legacy multi-partition files keep reading under the #2360 rule: agreeing node
copies read once, disagreeing ones refuse with
`NativePartitionCopiesDisagree`. Their tests now build such files with raw
h5py (`tests/results/_native_legacy.py`). The viewer's `results.ts` keeps
refusing a stage with several partitions; its header no longer calls merging
a follow-up (D6). A new unit test pins `ops.results`'s refusal of a
partitioned deck's text recorders (ADR 0123 slice 3).
