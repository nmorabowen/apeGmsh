### Fixed: a held workspace refuses the write by name (#2857)

On Windows, `write_workspace` onto a `<stem>.workspace.h5` that another
program holds open (h5py, h5wasm or a plain file handle) now raises
`WorkspaceWriteRefused` with reason `held-open`, naming the file and saying
another program holds it, instead of a raw `PermissionError [WinError 5]`.
The old file stays whole and the partial is removed, as before. Other
failures keep their own type: a read-only workspace still raises
`PermissionError`.
