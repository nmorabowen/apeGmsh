### Review-verdict matcher accepts a comment that starts with a BOM

`scripts/program_probe.py::verdicts` now normalises each comment body the way `land_pr.py` does for PR bodies (CRLF to LF, a leading `U+FEFF` or `U+200B` dropped at line start) before matching, so a verdict posted by a Windows tool no longer leaves the `review-verdict` status pending. A mark inside a line is not dropped.
