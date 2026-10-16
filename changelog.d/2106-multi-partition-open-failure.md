### FIXED — a partitioned reader that fails to open closes its parts and raises only the original error (#2106)

`LadrunoMultiPartitionReader` and `MPCOMultiPartitionReader` open one
reader per part file, then read every part to check that they agree. When
a part failed to open, the half-built reader's `__del__` raised
`AttributeError` (`_readers` was missing), which Python printed as
"Exception ignored in" next to the real error. When the consistency read
was interrupted (Ctrl-C), the parts already opened stayed open, and so
stayed locked on Windows, for as long as the traceback lived, as it does in
a notebook.

Both readers now close every part they opened on any failure, including an
interrupt, and re-raise the original exception. `close()` and `__del__`
work on a reader whose construction failed. The empty path list case
raises its `ValueError` with nothing printed after it.
