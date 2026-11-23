### FIXED — the amendment zero-width-space test now fails when `land_pr._text` deletes mid-line marks (#3106)

`test_a_mid_line_invisible_mark_does_not_hide_an_amendment_negation_1891` put
`src/b.py` mid-bullet, a shape an amendment never takes, so it passed even with
`_INVISIBLE.sub("", text)` in place of `sub(" ", text)`. Its two cases are now a
negated lead-in followed by a bullet (`Do<ZWSP>not touch these:` and
`- all<ZWSP>except these:`), which the deleting mutation lets through.
