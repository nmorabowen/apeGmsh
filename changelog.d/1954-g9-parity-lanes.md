### TESTS — the `.ladruno` live round-trip runs on the fork lane; the MPCO nodal-forces test binds its own model (#1954)

`tests/results/test_ladruno_live_roundtrip.py` moved to
`tests/opensees/integration_ladruno/`. It is marked `ladruno_fork` and drives
the module the bridge drives (`get_ops()`), so the nightly `live-fork` lane
runs its four tests against the pinned build. Before, it was marked `live`
and imported `openseespy.opensees`, so no lane ran it: live-stock has no
`ladruno` recorder, and the fork lane did not select it. A fork build that
writes no `.ladruno` now fails these tests instead of skipping them.

`tests/test_results_mpco_nodal_real.py` failed at its element-id filter
(`ids=[1]` returned no rows). The MPCO reader was right. The test paired its
hand-written deck with the session stub `model.h5`, whose element pairing is
fem 1 to ops tag 2, while the deck declared ops tag 1, so the filter asked
for an element the file did not hold. The test now writes the cantilever's
own archive through the bridge and declares the element with the paired ops
tag. It checks the fem-to-ops relabel on the unfiltered read and through the
filter. It still needs `APEGMSH_OPENSEES_TCL`, so no CI lane runs it.
