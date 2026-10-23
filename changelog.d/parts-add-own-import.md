### FIXED — `g.parts.add` binds each instance to its own import: live sub-entity tags after `translate=`/`rotate=`, sidecar labels never claim another instance, and one `add` no longer costs O(model)

- **Stale tags after a transform.** OCC keeps the moved shapes' tags but
  rebinds their faces, edges and points under new tags, so
  `Instance.entities` below the top-level shapes named entities that no
  longer existed, and a later `g.parts.fragment_pair` raised "Unknown
  OpenCASCADE entity of dimension 0". The instance now records the
  boundary closure of its moved shapes, at the dims the import
  recorded (`highest_dim_only=True` keeps only the top dim, so stacked
  storeys placed that way fragment and mesh as before). With the
  default flags, `fragment_pair` without `dim=` on stacked storeys
  still fails with "Could not fix wire in surface N": it hands OCC
  each instance's edges and points as well as its faces, a failure
  older than this fix and not addressed here.
- **Sidecar labels matched the whole model.** The anchor rebind matched
  every entity in the session, so a second placement of a Part at the
  same spot bound `b.<label>` to instance `a`'s coincident entity. It now
  matches only the import's own shapes and their sub-entities.
- **Cost.** That whole-model match ran one Python `norm` per
  (anchor, entity) pair, so stacking N storeys as Part instances cost
  O(N²): 136 s of `add` calls for a 49-storey tower. The match is now
  vectorised over the import's own entities.
