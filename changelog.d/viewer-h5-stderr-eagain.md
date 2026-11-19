### FIXED — apeGmshViewer: a torn stage no longer reads as a damaged group index when stderr is full

`apeGmshViewer/src/reader/h5open.ts` tells a damaged group index from a torn
object header by reading HDF5's error stack as it is written to stderr. When
the host refused a write (EAGAIN from a full non-blocking stderr pipe, as
`node --test` gives each test file on Linux), libc dropped the rest of that
line, the `in H5O_protect()` frame went missing, and the reader raised
"HDF5 could not look up /MODEL_STAGE[1]" instead of its named refusal. This
was the intermittent `app-tests` failure in `ladrunoKilledRun.test.ts` and
the `ladrunoParticles.test.ts` damage sweep. A diagnostic the host refuses is
now reported written, so the stack is always whole. New test:
`test/reader/h5StderrRefused.test.ts`.
