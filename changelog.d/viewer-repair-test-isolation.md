### FIXED — apeGmshViewer's repair-copy tests no longer fail at random in a parallel run

`ladrunoParticles.test.ts` counted repair folders in the machine's temp directory,
so another test file's repair could change the count. It now uses its own temp root.
