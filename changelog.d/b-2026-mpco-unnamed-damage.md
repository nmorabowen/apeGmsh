### FIXED — MPCO reads LadrunoConcrete3D damage as tension and compression (program link B, #2026)

LadrunoConcrete3D names no components for `damage`, so MPCO labels each
Gauss-point block `C1,C2`. The reader named those columns `damage_0` and
`damage_1`, and `read_gauss("damage_tension")` returned an empty slab with
no warning. Such a bucket now takes its names from the RESPONSE_CATALOG
`damage` row for the element class (tension first, the order the fork's
`getResponse` writes). A live LadrunoBrick read now matches
`eleResponse(..., "material", gp, "damage")` column by column. A class with
no catalog row keeps the indexed names, and a `damage_tension` read on it
raises `MaterialStateComponentRefused`, naming the bucket. ASDConcrete3D's
`d+,d-` path is unchanged.
