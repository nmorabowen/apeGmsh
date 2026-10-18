### Changed

- Internal: the private module-level helpers of `opensees/emitter/h5.py` (write helpers, partition/stage emission blocks, the emit-order program tape) moved to `_h5_write.py`, `_h5_blocks.py` and `_h5_program.py`, a pure move with no body changes; `h5.py` is 5,705 lines (was 6,527). No public import path changes (slice S5-a, #1966).
