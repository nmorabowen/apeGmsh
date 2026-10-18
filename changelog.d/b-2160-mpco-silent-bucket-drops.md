### FIXED — MPCO names the Gauss, material-state and spring buckets it leaves unread (#2160)

`Results.from_mpco` still dropped three kinds of bucket without a word. At the
Gauss and material-state levels, a bucket on a standard rule whose class and
rule have no response-catalog row (`FourNodeTetrahedron` at `Tet_GL_2`, or a
class the catalog does not hold) was skipped, because the catalog is what
places its Gauss points. At the spring level, a ZeroLength-family bucket with a
nonzero custom-rule index was skipped. Each one is still left unread, and is
now named, by path, class, rule and file, in one `MPCOBucketSkippedWarning` per
bucket, the contract #2115 set. A key that appears under two alias groups of
one token (`stresses` and `material.stress`) is named once. Beam buckets under
`basicForce` belong to the line-station level and stay quiet at the spring
level.
