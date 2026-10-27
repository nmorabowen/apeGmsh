### CHANGED — joint offsets go at each gmsh curve's two ends; a curved curve with end offsets now refuses (#2551 S2)

A frame member is the gmsh curve, as in ETABS. When a `geomTransf`
`jnt_offset=` is end-sensitive (end lengths, an axial component, or
different offsets at the two ends), the bridge now splits it over each
curve the transform's elements were meshed on, using the per-element
source entity that `FEMData` records since neutral schema 2.36. The curve's
first element takes the i-offset, its last element the j-offset, and each
interior node takes only the transverse part, interpolated along the
curve. A curve meshed into n elements now behaves as the same member as one
element with the same offsets. What frames in along the curve (a secondary
beam, a spring) does not split it: to put a joint mid-span, split the curve.
A constant pair that splits this way fans out into one `geomTransf` line per
distinct tail, like an `InsertionPoint` rule. Line loads over the end zones
follow the split.

- **Refused, naming the curve:** end-sensitive offsets on a curve that is
  not straight. Before this change, #2470 accepted a coarse arc whose turns
  read as frame corners. Split the arc into straight curves, or drop the end
  offsets. Straightness allows round-off relative to the coordinates' size,
  so a straight curve at UTM coordinates still splits.
- **Part of a curve:** a transform with end-sensitive offsets that covers
  only some of a curve's elements keeps each element's own offset. It is
  refused, naming the curve, its missing elements and the node, wherever
  that offset puts an end zone or any other component along the element's
  axis at a node interior to the curve. A part whose interior nodes carry
  only transverse offsets is accepted: it is how an Assembly rehydrate
  re-declares a split curve, one line per element. For example, a lone
  element of a meshed curve with end zones at both its ends refuses, and so
  do two of its three elements. Earlier, #2470 accepted these when the
  elements did not meet in a straight run.
- Also refused: a curve whose elements run both ways, a closed curve, end
  lengths with unequal transverse ends, and an end element no longer than
  its end zone.
- **No source curves:** a FEM without the entity stream (a model.h5
  written before neutral schema 2.36, or a FEM not extracted from gmsh)
  keeps the earlier refusal wherever two of the transform's elements meet
  without a clear corner. The message now says the snapshot carries no
  source curves.
- **Unchanged:** offsets that are the same transverse shift at both ends,
  curves meshed as one element, and every other model the earlier rule
  accepted, apart from the two refusals above, emit byte for byte as before.
  The emit goldens do not move.
