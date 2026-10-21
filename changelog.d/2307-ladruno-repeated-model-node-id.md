### FIXED — a `.ladruno` whose model lists one node id twice is refused when it opens, by name (#2307)

`LadrunoReader` checked the node `ID` of every result group for a repeated
id (#2300) but not the model's own node list, `MODEL/NODES/ID`. A file whose
model listed a node id twice opened without complaint, and the FEMData built
from it held that id twice, at two coordinates, while the element that named
the missing node found none.

Opening such a file now raises `LadrunoRepeatedNodeId` with `model=True`,
naming the file, the `MODEL_STAGE[k]/MODEL/NODES/ID` dataset and each
repeated id with its rows. `Results.from_ladruno` refuses a single file and a
part set alike; a node held by two part files is still legal.

The check also costs less at open: a strictly increasing id list, which is
what the fork writes, is accepted after one pairwise comparison of
neighbouring ids, and only any other list is sorted. The comparison never
subtracts, so ids near the int64 limits cannot wrap into a false "increasing".
On a synthetic 2M-node file with 6 results and 4 stages the check went from
1.0 s to 0.18 s, most of it reading the ids. The refusals are unchanged.
