### ADDED — apeGmshViewer can run the Ladruno fork on one declared uniaxial material (no UI yet)

ADR 0135 amends ADR 0112 D4: the app may start the fork's `OpenSees.exe`, found
through `$APEGMSH_OPENSEES` or the default install. The runner tests a declared
material and its dependencies under a strain protocol and stores the curves. A
model's values are treated as untrusted: only finite numbers and plain words reach
the Tcl script. The screen follows once the shared tool host is decided.
