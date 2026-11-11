### CI: the stage-sweep neighbour test runs its legs in a child

`test_md_stage_all_elements.py::test_the_sweep_leaves_other_elements_alone` now runs each leg in a child interpreter, one neighbour class at a time, so the Fortran `STOP` of stock `StressDensityModel` (#2543) can no longer end the `live-stock` session with status 0 and no report. That `STOP` after the flip is accepted only for `StressDensityModel` on stock openseespy; every other class and backend still meets the numeric assertion (#2881).
