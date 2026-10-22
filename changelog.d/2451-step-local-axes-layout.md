### CHANGED — apeGmshViewer reads the per-step `.ladruno` `LOCAL_AXES` stream in one layout, and refuses the others by name (#2451)

The fork's schema disagreed with itself on the per-step co-rotational stream
`RESULTS/ON_ELEMENTS/LOCAL_AXES`. Its §5 drew a `DATA/STEP_<k>` group of per-step
datasets, but its §7.0 gives every streaming group one chunked `DATA[T, n, c]`
and calls the `DATA/STEP_<k>` form superseded. The fork at `FORK_PIN` writes
neither form. The maintainer ruled on §7.0's chunked shape, in §7.2's per-class
buckets:

```
RESULTS/ON_ELEMENTS/LOCAL_AXES/<classTag>-<ClassName>[<g>:<h>]/
    ID    [nE]
    DATA  [T, nE, 4]   quaternion (w, x, y, z); the reader also takes 3 or 9 wide
    STEP  [T]          optional; when present, one value per DATA step
    TIME  [T]          optional; when present, one value per DATA step
```

The viewer's step read (`readLocalAxes(stage, {step})`, and so
`readLineStationStep`) now refuses these by name as a `LadrunoError` of kind
`layout`:

- an `ID`, `DATA`, `STEP` or `TIME` on `LOCAL_AXES` itself (the flat form,
  which this reader used to take, and §5's form);
- a `DATA` group of `STEP_<k>` datasets in a class bucket, named as §5's
  superseded form;
- a `STEP` or `TIME` that is not a dataset, or that does not hold one value
  per `DATA` step.

The fork never wrote the stream, and no committed `.ladruno` holds one, so no
existing file changes how it reads. A step-less read does not touch the stream.
Python reads only `MODEL/LOCAL_AXES`, as before, and its docstring now says so.
The fork's §5 fix, and the writer, are the `chain:F` feature request.
