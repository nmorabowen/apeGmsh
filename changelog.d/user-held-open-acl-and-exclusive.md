### Fixed: a permissions denial is no longer reported as `held-open`, and a holder that shares nothing is (#3080)

On Windows, `write_user_file`, `write_variants` and `write_workspace` called
a file whose ACL (or its folder's) denies the write `held-open`, telling you
to close a program that did not exist. They now refuse it as
`access-denied`, naming the file. Windows answers both cases with the same
error, so the writer opens the file once more, for the same access with
every share mode: access denied means the permissions, a sharing violation
means a holder.

A holder that shares nothing (`CreateFileW` with share mode 0) used to
escape as a raw `PermissionError` from the read before the replace. It now
refuses as `held-open` on all three writers, including the workspace's
check just before the replace, which called it `revision-moved`. In every
case the old file stays whole and the partial is removed. A read-only file
keeps its own `PermissionError`, and off Windows a `PermissionError` stays
raw.
