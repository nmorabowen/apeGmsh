### FIXED — Section-document uniaxial params refuse with `SectionDocumentError` at `to_section` (#1354)

A section document's `uniaxial` material params used to be splatted
unchecked into the bridge constructor at `to_section()`, so an unknown
key (say `fillet` on `ElasticMaterial`) loaded fine and later surfaced
as a raw `TypeError`. The params are now bound against the
constructor's own signature first, and a key the bridge does not take,
or a required one the document lacks, refuses as `SectionDocumentError`
naming the material, the key and the `section_doc_version`, the way RC
template params already did. Only that binding failure is translated: a
`TypeError` a constructor raises for its own reasons still propagates.
The loader stays free of bridge signatures; the factory the caller
hands in is introspected at handoff.
