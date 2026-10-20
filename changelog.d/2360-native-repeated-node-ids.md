### FIXED — native results: a node id repeated in one partition refuses at open, and partition copies that differ refuse on read (#2360)

`NativeReader` merges the partitions of a native `results.h5` and keeps the
first copy of a node that several partitions hold. Two inputs read wrong
without a word.

- **A node id listed twice in one partition's `nodes/_ids`** read as the
  first copy. The file now refuses at open with `NativeRepeatedNodeId`,
  which names the file, the `_ids` dataset and each repeated id with its
  rows. This matches the `.ladruno` refusal (#2300, #2319).
- **A boundary node whose partition copies differ**, such as a per-rank
  partial reaction of 3.0 and 4.0 whose true value is 7.0, read as 3.0.
  Native files carry no partition-reduction rule (MPCO and `.ladruno` have
  `PARTITION_REDUCTION`), so the reader cannot tell a partial value from a
  replicated one. `read_nodes` now keeps the first copy only when every copy
  is bit-for-bit equal over the whole stage. Otherwise it raises
  `NativePartitionCopiesDisagree`, naming the stage, the component, and each
  node with the partitions that hold it.

Replicated copies that agree, such as displacements, read as before. Every
native writer in apeGmsh writes a single partition, so only a file written
through the public `NativeWriter` API with several partitions can meet the
second refusal.
