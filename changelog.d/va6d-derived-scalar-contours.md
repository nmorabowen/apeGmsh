### apeGmshViewer: element contours of `von_mises_stress` and `von_mises_strain` (VA6d, #2766)

An element stress or strain field whose Gauss components make a complete
tensor now offers its von Mises equivalent beside the stored components:
`von_mises_stress` for a stress, `von_mises_strain` for a strain. The window's
contour picker, the command palette, `--describe` and an `apegmsh-view/1`
spec offer the same list. A shot such as `contour: {field: "stress",
component: "von_mises_stress", source: "element"}` now draws instead of
being refused.

The value is computed at each Gauss point as v1's default read computes it
(`plane="auto"`). A strain's engineering shear is halved. A recorded zz is
used as recorded. A 2-D element's zz that is not recorded is recovered per
element from its plane type and Poisson's ratio, read from the model's
`/opensees` records:

- σzz = ν(σxx + σyy) on plane strain;
- εzz = -ν/(1-ν)(εxx + εyy) on plane stress;
- 0 on the other idealization;
- 0 where the model does not classify the element. The D3 manifest counts
  those elements under `contour.derived.plane` and warns about them.

The derived value is then averaged or carried to the nodes like any element
component, and the history range is scanned the same way. `--describe` says
what each derived component is computed from.

A field whose tensor is incomplete or ambiguous is refused by name, with the
reason. That covers a missing in-plane xx, yy or xy, one out-of-plane shear
recorded without the other, or one component named twice. The value is never
drawn from partial components.
