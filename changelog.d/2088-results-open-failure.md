### FIXED — `Results.from_ladruno` names a model-less file, and both readers free the file when a later step fails (#2088, #2113)

`Results.from_ladruno` on a `.ladruno` file with no `MODEL` group stopped
at a bare `assert`, and under `python -O` failed somewhere less clear.
It now raises a `ValueError` that names the file, says the `MODEL` group is
missing, and points at `model_h5=`. With `model_h5=` the file opens, and the
archive supplies the model.

`Results.from_mpco` and `Results.from_ladruno` open the reader first and
then load the model archive, resolve the FEMData and bind stage names. When
one of those steps raised (or was interrupted), the reader stayed open, and
the file stayed locked on Windows, for as long as the traceback lived. Both
constructors now close the reader on any failure after the open and
re-raise the original exception.
