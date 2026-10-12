### FIXED — test-order leak of a test-only `Steel01Plus` subclass (#1694)

`test_verbs_lock.py` defined a `Steel01Plus` primitive that stayed in
`Steel01.__subclasses__()` after the test, held only by its reference
cycles, so a later `test_family_completeness.py` in the same process
failed its class walk. The test now undoes its monkeypatch, drops the
class and collects, and asserts the subclass is gone. Tests only.
