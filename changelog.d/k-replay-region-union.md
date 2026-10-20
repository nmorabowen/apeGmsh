### FIXED — a partitioned archive whose recorder region lists elements now replays: `OpenSeesModel.build` unions the per-rank `-node … -ele …` rows (#2254)

A partitioned emit writes a recorder's filter region once per rank that owns
a member, each row that rank's share: `-node ids`, `-ele ids` or
`-node ids -ele ids`. The flat deck declares the tag once, as
`region <tag> -node <all> -ele <all>`. `OpenSeesModel.from_h5(...).build()`
unioned only plain `-node` rows and raised `ValueError` on any `-ele` row, so
no partitioned archive with an element-scoped recorder (an `Element` or `MPCO`
recorder on a group) could be replayed.

Replay now unions rows of those three shapes into `-node <union> -ele <union>`,
each list in first-occurrence order with duplicates dropped, the order the
flat bridge deck writes. Any other pair of rows that share a tag still raises
`ValueError` naming the tag: a `-rayleigh` or `-damp` tail, `-eleRange`, a
flag with no ids, or a flag repeated or out of order. A flag with no ids now
refuses for `-node` as well; the bridge never writes one.

The H5 writer is unchanged: what an archive stores and its `model_hash` are
the same as before.
