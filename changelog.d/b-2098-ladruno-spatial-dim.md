### CHANGED — `.ladruno` has one spatial-dimension rule (program slice B, #2098)

`LadrunoReader` and the `localForce` 2-D gate each carried their own copy of the rule that reads a
`.ladruno` file's spatial dimension (`INFO/SPATIAL_DIM`, else the one coordinate width of 1, 2 or 3 all
stages agree on, else unknown). The reader now calls the gate's helper, so the two cannot drift. Before the
merge both copies were run over every input class, including malformed attributes and group or dataset
inputs, and answered the same; no read changes. A table test in `tests/results/readers/test_ladruno_reader.py`
pins the shared answer.
