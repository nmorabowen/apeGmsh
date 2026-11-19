### FIXED — apeGmshViewer: a page reload drops the old page's offers; a provenance row naming another machine is never looked up; xdg-open gets only source files (#2771)

A page reload now starts main's offer list empty, the same as the page's. An
offer the old page listed can no longer be opened, and neither can the set it
left behind. An offer that arrived while the page was loading still waits for
the new page and still opens.

A `/provenance` row that names another machine (a path starting with two
slashes or backslashes, such as `//host/share/build.py` or a `@base_dir` of
`//host/share`) is no longer looked up on disk. On Windows that lookup alone
connects to the host and can send it the user's credentials. Such a row reads
as not present, and go-to-source refuses it as unavailable. A drive letter
mapped to a share still looks local.

On Linux with no `APEGMSH_EDITOR` and no `code` on PATH, go-to-source hands
`xdg-open` only a regular `.py`, `.pyw`, `.pyi`, `.pyx` or `.tcl` file, never a
link. `xdg-open` picks a program by file type and can run a `.desktop` file.
Set `APEGMSH_EDITOR` to open any other file.
