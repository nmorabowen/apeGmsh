### CHANGED — ADR 0108 D5's live `ladrunoBranch` seam moves to the `.ladruno` material channel (X, #2192)

ADR 0108 D5 named DomainCapture's per-material routing as the future seam
for a live `ladrunoBranch` read. That class is being removed (#1767), so D5
now carries a dated superseded-in-part note: the live read is a `material`
response, recorded as `material.ladrunoBranch` by the Ladruno recorder and
read back with `Results.from_ladruno` through the material-bucket seam
(D2). The reader already names the eight columns. A post-`61b3efa04`
engine writing the bucket end to end is still the live gate's open half.
`internal_docs/guide_ladruno_adr95_druckerprager_fix.md` §3 follows the
note. No code changes.
