### apeGmshViewer: a User Combination shown as a contour, and refused as a deformed shape (ADR 0130 slice 7b-2)

The phase bar's User Combinations family can now be clicked. It shows the first
combination of the set's `.user.h5` as the current state, coloured as a contour.
A picker in the bar chooses another, and states its rule and whether it is signed.
The app reads the cases a combination reaches through main's `user:readCase` and
evaluates them with the slice 4 evaluator. On San Ramón, a spectrum case's CQC and
SRSS match Python's `Results` to the last digit on all 13,459 nodes. The picker
offers each nodal component on its own. It offers no magnitude, because the CQC of
a magnitude is not the magnitude of the CQCs.

An unsigned combination is a contour only. A modal rule, an envelope, or a `linear`
over one of them is unsigned. The Deformed card disables the toggle and gives the
reason. If the toggle is turned on anyway, nothing deforms and the banner gives the
same sentence. A `--render` shot that asks such a combination for a deformed shape
is refused with that sentence too. A stale `/spectra`, a combination the evaluator
refuses, and a case it cannot read are each stated on the contour card and the
banner, never left as a blank view.

The view spec `apegmsh-view/1` gains an optional shot key, `combination`. It names
the combination to show in place of a stage, and takes no `step`. Its
`manifest.json` entry gives the contour's `source` as `combination` and adds the
combination's name and sign. Three gaps between the app and Python, from the slice
7b-1 review, are closed:

- a `/spectra` that cannot be read is refused before staleness is judged;
- `Ux` and `ux` in one result are refused as ambiguous;
- a stage or mode term in a combination is refused, and the message says that Python
  reads these terms and the app does not yet.
