### TESTS — a 3-D trapezoidal `beam_uniform` keeps its tail on every 3-D beam element (#2550)

A new live oracle, `tests/opensees/live/test_beam_uniform_3d_trapezoid_live.py`, puts a trapezoid (`wy`, `wz`, `wx` as `(start, end)` pairs, full span and `span=(0.25, 0.75)`) on a fixed-fixed 3-D beam. It covers each 3-D element the bridge emits: `elasticBeamColumn`, `ElasticTimoshenkoBeam`, `dispBeamColumn`, `forceBeamColumn`, `LadrunoDispBeamColumn`, `LadrunoForceBeamColumn` and `LadrunoIMKBeam`. The support reactions must balance the closed-form resultant and its first moment.

At `FORK_PIN` every element honours the tail. Fork builds older than the pin, such as 79e0623, do not. On those builds `elasticBeamColumn` reads only the start values. `ElasticTimoshenkoBeam`, `dispBeamColumn` and `LadrunoDispBeamColumn` drop the whole load with a "load type unknown" warning, and `analyze` still returns 0. Run 3-D trapezoids on the pinned build or newer.
