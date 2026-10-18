### FIXED — MPCO names the bucket keys it cannot parse; the converter flags custom-rule springs (#2165)

`Results.from_mpco` dropped a bucket without a word when its key did not
parse as `<tag>-<Class>[<rule>:<cust>:<hdr>]`, at the Gauss,
material-state, spring, fiber and line-station levels. Each such bucket is
still left unread, and is now named, by path and file, in one
`MPCOBucketSkippedWarning` per key, the contract #2159 and #2164 set. A
key under two alias groups of one token is named once. The layer and
nodal-force levels already named such a key in their own warnings.

`convert_mpco` no longer warns that a custom-rule ZeroLength bucket is
"left unread", since it copies that bucket verbatim, and it now lists
`springs` in the converted file's `unsupported` provenance for such a
bucket, as it did for the spring buckets `from_mpco` reads.
