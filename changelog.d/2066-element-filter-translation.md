### FIXED — an MPCO or `.ladruno` element filter reads each paired id as itself, and warns for ids it cannot return (#2066)

With a bridge `model_h5=`, an element filter that mixed a paired `fem_eid`
with an id the archive does not pair was passed through untranslated, so the
paired ids were read as ops tags. On a two-element cantilever with
(fem, ops) = (1, 2), (2, 3), `ids=[2, 999]` returned fem 1's axial force
labelled 2, and `ids=[1, 999]` returned nothing. A paired id whose element the
recorder `region` left out read back as an empty slab with no warning.

Each paired `fem_eid` now reads its own ops tag, whatever else the filter
holds, and comes back labelled with its own `fem_eid`. An element the
archive emitted without a mesh id (an interface `zeroLength` pair) is read
by its ops tag. An id that is no element of the archive, or an element the
files did not record, is not read, and `WarnElementFilterUnread` names it.
A partition set is judged as one: an element recorded by another partition
does not warn.

The translator now decides once per result set whether the archive describes
the files: every ops tag the files record must be an element of the archive.
An unrelated stub `model_h5=` does not describe them, so its reads pass
through untouched, as before, with `WarnElementTagPairingMissing` from
`Results` when the bound `FEMData` speaks `fem_eid`.
