### TESTS — the `.ladruno` live round-trip runs on the fork lane; the MPCO nodal-forces test binds its own model and pins two filter bugs (#1954)

`tests/results/test_ladruno_live_roundtrip.py` moved to
`tests/opensees/integration_ladruno/`. It is marked `ladruno_fork` and drives
the module the bridge drives (`get_ops()`), so the nightly `live-fork` lane
runs its four tests against the pinned build. Before, it was marked `live`
and imported `openseespy.opensees`, so no lane ran it: live-stock has no
`ladruno` recorder, and the fork lane did not select it. A fork build that
writes no `.ladruno` now fails these tests instead of skipping them.

`tests/test_results_mpco_nodal_real.py` failed at its element-id filter
(`ids=[1]` returned no rows). The test paired its hand-written deck with the
session stub `model.h5`, whose element pairing is fem 1 to ops tag 2, while
the deck declared ops tag 1. The filter therefore asked for an element the
file did not hold. The test now writes the cantilever's own archive through
the bridge and declares the element with the paired ops tag. It checks the
fem-to-ops relabel on the unfiltered read and through a filter of paired ids.

The review of this change found that the MPCO reader returns wrong answers
for other filters on a real archive (#2066). A filter that mixes a paired
and an unpaired id is not translated, so `ids=[2, 999]` returns fem 1's row
labelled 2, and `ids=[1, 999]` returns nothing. A paired id that the file
did not record reads back empty with no warning. Three strict xfails in the
same file assert the right answers against statics until #2066 fixes the
reader. These tests still need `APEGMSH_OPENSEES_TCL`, so no CI lane runs
them.
