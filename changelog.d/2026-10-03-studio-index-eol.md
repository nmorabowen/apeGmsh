### FIXED — studio API index checks out with LF on Windows

`tests/studio/test_lookup.py::test_index_build_is_byte_deterministic` compares the LF output of `serialize_index(build_index())` byte-for-byte with `src/apeGmsh/studio/_api_index.json`, so it failed on Windows clones with `core.autocrlf=true`, where the file checked out as CRLF, and passed on Linux CI. `.gitattributes` now pins the file to `text eol=lf`. Existing Windows checkouts pick it up after deleting the file and running `git checkout -- src/apeGmsh/studio/_api_index.json`.
