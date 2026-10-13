### FIXED — `land_pr.py` check 9: a negated card amendment no longer whitelists the list it opens (#1872)

Check 9's negation guard read a `**Card amendment` comment one line at a time, so
"Do not touch:" followed by a bulleted `b.py` added `b.py` to the Owned list. A negated
lead-in now drops every line after it up to a blank line, a heading or a bold label, as
an Owned section ends; a negated bullet drops itself and its sub-items, so one "do not"
bullet in a prose amendment does not drop its siblings. The negation vocabulary also
caught too little: `Don’t` (curly apostrophe), `Do _not_ touch`, `won't`, `cannot`,
`leave … alone`, `avoid`, `excluding` and `no changes to` now negate, both in amendments
and in Owned lines, which refuse as before. Letter boundaries replace `\b`, and
negation words inside a backticked path (`src/not_found.py`) are ignored.
`internal_docs/program/slice_card.md` lists the vocabulary and the list rule.
