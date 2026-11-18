### CHANGED — Elements without a mesh id take OpenSees tags above every FEM element id under the default `element_tags="sequential"` (#2926)

Under the default `element_tags="sequential"`, an element that only
OpenSees knows about used to take the next tag after the physical-group
elements. That covers node-pair springs, MP elements (rigid bodies,
couplings, ties, rebar cells) and interface `zeroLength` elements. Such a
tag could equal the FEM id of a paired element. For example, a 4x4x4
`stdBrick` block with surface groups and five ground springs gave the
springs tags 71..75, which are also FEM ids of five bricks. A results read
through the model archive then warned `WarnElementIdCollision` and refused
a filter that held those ids.

Now these elements are numbered above the snapshot's largest element id,
the rule `element_tags="fem"` already used. Physical-group elements keep
their tags, unless a node-pair spec precedes them, in which case they close
up the gap it used to take. A deck with no such elements is byte-identical. In a deck with
them, their tags change, and so do the tags of a physical-group element
declared after a node-pair spec, because the physical-group run no longer
skips the spring's tag. Partitioned and unpartitioned emits give the same
tags. Code that hard-codes the old sentinel tags must read them from the
model archive instead.
