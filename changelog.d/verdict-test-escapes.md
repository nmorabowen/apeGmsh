### Tests: the review-verdict tests write U+FEFF and U+200B as escapes

The tests behind the `review-verdict` gate typed the invisible characters as raw characters, so an editor that strips them would leave the tests green with the fix reverted. They are now written as the escapes `\ufeff` / `\u200b`, a guard test refuses a raw one, and a new test pins that a bare `\r` starts a verdict line while text before the verdict on the same line still does not count.
