### CI: the live-stock lane names the test that kills it

The `live-stock` job now runs pytest unbuffered and verbose (`PYTHONUNBUFFERED=1`, `-v`), so when a native `exit(0)` ends the session mid-run the log ends on the dying test's nodeid (#2849).
