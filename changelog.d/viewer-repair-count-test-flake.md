### FIXED — viewer reader test no longer fails when other test runs share the temp folder

`apeGmshViewer/test/reader/ladrunoParticles.test.ts` counted repaired-copy folders in
the machine-wide temp folder, so any other process repairing a `.ladruno` file at the
same moment changed the count and failed the test intermittently. The file now gives
its repairs a private temp root. Test only; no change to the viewer.
