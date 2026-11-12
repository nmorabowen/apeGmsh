### apeGmshViewer: element contours of `von_mises_stress` and `von_mises_strain` (VA6d, #2766)

An element stress or strain field whose Gauss components make a complete
tensor now offers its von Mises equivalent beside the stored components:
`von_mises_stress` for a stress, `von_mises_strain` for a strain. The window's
contour picker, `--describe` and an `apegmsh-view/1` spec offer the same list,
so a shot such as `contour: {field: "stress", component: "von_mises_stress",
source: "element"}` now draws instead of being refused. The value is computed
at each Gauss point as Python's `_derived` computes it (a strain's engineering
shear halved; a 2-D element's out-of-plane zz taken as 0), then averaged or
carried to the nodes like any element component. The history range is
scanned the same way. `--describe` says what each derived component is
computed from, and the D3 manifest names those components under
`contour.derived.from`.

A field whose tensor is incomplete or ambiguous is refused by name, with the
reason. That covers a missing in-plane xx, yy or xy, some but not all of the
out-of-plane components, or one component named twice. The value is never
drawn from partial components.
