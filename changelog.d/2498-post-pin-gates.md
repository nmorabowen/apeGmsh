### FIXED — a live fork element whose option the build rejects says so, not "requires the Ladruno fork"; the post-pin fork tests run unconditionally (#2498, #2503)

A Ladruno build can know an element class but not one of its options. It
prints `unknown option '-shear'` and fails the call. The live emitter's
fork-element gate used to answer with "element X requires the Ladruno fork
build", which is wrong for that build. The gate now reads the build's line,
from `sys.stderr` and from file descriptor 2. It raises "option -shear is not
supported by this build", quoting that line, and the user still sees the
line itself. An unknown class, or output this process cannot read, still
gets the fork-required message.

The gate verifies each element type once per option set, so a second
element that adds an option the first one lacked is checked again.

The four `live-fork` tests that had failed at `ladruno-v20261017`
(`LadrunoIMKBeam -shear` / `-shearHinge`, `LadrunoCohesiveHinge -implex`,
the `LadrunoForceBeamColumn` class) run unconditionally at the new pin,
`ladruno-v20261023`, which has all four features.

The fork rows of `test_md_stage_all_elements.py` now take their backend from
`get_ops()` instead of `importorskip("openseespy")`, so they run on the fork
lane.
