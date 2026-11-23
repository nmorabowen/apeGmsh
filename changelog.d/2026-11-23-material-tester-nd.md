### ADDED — the material tester runs nD materials

**Test** in the Materials & sections window now opens on nD materials too (ADR 0135 D9, S5). The form
adds the view (3D or plane stress) and the loading: uniaxial stress, triaxial at a confining pressure
p, uniaxial strain (oedometer) or simple shear, each driving one strain component through a
monotonic, cyclic or pasted history while the fork's new `model testNDMaterial` (nmorabowen/OpenSees
WP-248) holds the others by mixed control. The plot shows the driven component (ε11 and σ11, or γ12
and τ12). The tangent is the fork's own with the held components condensed out. ASDConcrete3D, the
first material chosen, peaks at fc in uniaxial compression and higher when confined. Because the
tester has no element, a material's `-autoRegularization` scales its softening to an element of
length 1, and the window says so, with the slot to change to see the declared curve. A material with
many parameters opens with its parameter grid folded. nD runs need a fork build with `ndTest`.
