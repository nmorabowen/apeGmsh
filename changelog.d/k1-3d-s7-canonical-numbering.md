### Changed — partitioned decks renumber once: a tag is rank-invariant (K1-3d S7, #1459)

- **Partitioned and partitioned-staged decks now number every derived tag
  in the flat deck's order.** The bridge's tag plan mints regions, MP
  elements (ties, couplings, rigid bodies, rebar cells), interfaces and
  their `uniaxialMaterial` pairs, contacts and contact planes, and
  `parameter` tags in one canonical walk, the flat emit's, whatever the
  emit mode. A partitioned emit writes the same objects rank by rank with
  those tags, so one owner has one tag on 1, 2 or 4 ranks and in the flat
  deck (ADR 0114 D4, amended, item 4).
- **What you see.** A partitioned deck that holds named or recorder
  regions, MP-constraint elements, interfaces beside an element-minting MP
  pass, contacts, or stage flips and updates renumbers those tags once.
  A stage flip or update now declares its record's one tag on every rank
  that holds its elements. Flat and flat-staged decks are byte-identical,
  and no region declaration moves (a stage's regions still follow its
  `domainChange`).
- The interface + embedded drift that ADR 0093 INV-5 (S8 amendment)
  documented is gone: flat and partitioned interface tags are now equal.
