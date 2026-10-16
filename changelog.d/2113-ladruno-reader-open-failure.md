### FIXED — a `.ladruno` that fails the identity check is closed before the error is raised (#2113)

`LadrunoReader` opened the HDF5 file and then checked `INFO/GENERATOR` and
`FORMAT_VERSION`. When that check failed (for example an `.mpco` or a
foreign HDF5 passed to `from_ladruno`), the file stayed open for as long as
the traceback lived, as it does in a notebook, so on Windows it could not be
rewritten or deleted until a garbage collection.

The reader now closes the file on any failure after the open, including an
interrupt, and re-raises the original exception.
