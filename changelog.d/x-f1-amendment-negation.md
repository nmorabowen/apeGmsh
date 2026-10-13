### FIXED — `land_pr.py` check 9: a negated card amendment no longer whitelists the list it opens (#1872)

Check 9's negation guard read a `**Card amendment` comment one line at a time, so
"Do not touch:" followed by a bulleted `b.py` added `b.py` to the Owned list. Now a
negated line ending in `:`, or a negated bullet that names no path, drops the list after
it, across blank lines. Any other negated prose line drops the rest of its paragraph. A
negated bullet that names a path drops itself, its sub-items and its continuation lines,
so one "do not" bullet in a prose amendment does not drop its siblings. The vocabulary
caught too little: `Don’t` (curly apostrophe), `Do _not_ touch`, `won't`, `cannot`,
`neither`, `excluding`, `avoid`, `untouched`, `forbidden`, `out of scope`, `removed from`,
`leave … alone`, `no changes to` and a bare "No `x`" now negate, both in amendments and in
Owned lines, which refuse as before. Letter boundaries replace `\b`, and words inside
backticks (`src/not_found.py`, a quoted `cannot join:` string) are ignored unless the
line's backticks pair ambiguously. `internal_docs/program/slice_card.md` lists the
vocabulary and the list rules.
