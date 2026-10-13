### FIXED — DomainCapture no longer records other elements under a fem_eid; its files are stamped (program slice B5-f, #1764)

A domain capture asks OpenSees for each element by ops tag and labels the
row with the record's `fem_eid`. When no `{fem_eid: ops_tag}` pairing was
available, the capture queried the `fem_eid` as if it were an ops tag,
without a warning. That happened with no bridge or model, when the bridge
could not archive its sidecar model.h5, on any error reading the pairing
(the error was swallowed), and for a whole record the pairing only partly
covered. Under the bridge's default `element_tags="sequential"` the row
then held another element's response, or the element was dropped. On a 2-D
portal with pairing `{5: 3, 4: 4, 6: 5}`, fem 5's row read Fy = 6257.8,
which is the response of tag 5 (column fem 6); fem 5's own value is
−6257.8.

**Changed (behaviour):** an element-scoped `DomainCapture` (any `elements`,
`gauss`, `line_stations`, `fibers` or `layers` record) now raises a
`ValueError` at `__enter__`, before the run file is written, unless a
pairing covers every recorded `fem_eid`. The message names the cause and
the remedy. It affects:

- code that constructs `DomainCapture(...)` directly, without a bridge:
  `tag_map=None` (the default) now refuses for element records. Pass
  `tag_map=` (the model's `{fem_eid: ops_tag}` pairing), or use
  `ops.domain_capture(...)` or `DomainCapture.from_h5(...)`;
- `ops.domain_capture` runs whose bridge could not archive its sidecar;
- records that name an element the bridge did not build.

`tag_map=` keeps its name. It accepts the `ElementTagTranslator` it took
before, or a plain `{fem_eid: ops_tag}` mapping; an empty mapping is a
pairing that covers nothing, so it refuses too.
`ElementTagTranslator.covers_fem(ids)` is new. Node-only captures are
unchanged.

**Added:** every capture that runs, and `make_demo_results` (nodal rows
only), writes `element_index_space = "fem"` (results 1.2.0). Unstamped
`domain_capture` files are **not** grandfathered, and read unverified: the
capture did not translate in every era (`483633df`→`fbf68c11` never,
`fbf68c11`→`2d159362` only for composed models, then silently not on the
paths above). `architecture/h5-schema.md` now states the rule as verified
if and only if `element_index_space == "fem"`, for every `source_type`,
and gives the era table.
