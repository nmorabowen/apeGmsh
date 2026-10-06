### ADDED — declaration provenance in session-written `model.h5` (ADR 0112 V2c, #1306)

A session now records where each declaration was made in the user's source,
and `g.save()`, `save_to=` and `FEMData.to_h5` write it as the `/provenance`
zone. The records are keyed by declaration path `<zone>/<family>/<name|#k>`,
for example `neutral/labels/deck` or `neutral/loads/#2`. Each record points at
the first line outside apeGmsh (`site`, which is inside a helper when the call
came through one) and at the calling line of the running script (`script`). The
source file's sha256 is stored with it, so a reader can tell that the file has
been edited since. Capture happens at the declaration verbs (`g.loads`,
`g.constraints`, `g.masses` and the other declaration composites), at
`g.labels.add`, at `g.physical.add`, and at every geometry registration. Each
user call makes one record. `FEMData.from_h5` reads the zone back. No hash
reads it, so `snapshot_id` and `fem_hash` are unchanged. A value outside int32
is refused before the file is written. There is no opt-out: capture costs about
6–18 ms per 1,000 registrations.
