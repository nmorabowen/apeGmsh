### FIXED — the section builder's live-properties worker can no longer run a Qt finalizer (#1649)

With live properties on, the section builder could segfault on Linux, with no
traceback, shortly after a build finished. The worker thread paused the cyclic
garbage collector itself and switched it back on before it had finished, so a
collection could still run on the worker and free an earlier window's Qt
objects off the UI thread. Any change to the allocation count, such as a new
import, could expose it, and it turned `main`'s qt lane red twice. The UI
thread now pauses the collector before the worker starts and resumes it only
after the worker has exited. A new quirk rule, `qt-worker-gc`, flags a
`Thread(target=...)` in a Qt-importing module that targets a bound method or
can switch the collector on.
