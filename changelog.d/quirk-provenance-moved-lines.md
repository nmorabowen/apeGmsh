### FIXED — quirk lint `comment-provenance` no longer flags a comment moved verbatim (program slice S1-q, #1470)

A pure move re-adds existing comments, so the rule flagged them as new history. A comment
line added under `src/` whose stripped text also appears among the comment lines the same
diff deletes (as a multiset: one deleted copy covers one added copy) is a move, not new
provenance. Genuinely new provenance comments are still flagged. Self-tests in
`tests/test_check_quirks.py`.
