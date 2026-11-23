### ADDED — the material tester can change a material's parameters for one test

Under the tester's protocol in the Materials & sections window, each numeric argument of the
material's command can be changed for the test only (ADR 0135 D6, D8). Slots are named by the file's
`params_names` where it has them, otherwise by the preceding flag and position (`-iso voce · 2`), and
a slot holding a dependency's tag is locked. A run with changes overlays the declared one, and its
legend line names what changed. The model is never edited: the changed values show as text, with
**Copy** and **Reset**, to carry into the script by hand.
