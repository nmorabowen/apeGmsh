### apeGmshViewer: the modal table beside the mode list, and the Spectra family lists its cases (ADR 0130 slice 7a)

When the viewport shows a mode, a Modal table card in the right rail shows the
set's `.user.h5` `/modes` table. Each row gives the mode, its period, the effective
mass ratio and the cumulative ratio for each component. Each row also has its own
verdict: a row is current only when its eigenvalue equals the one in the `.ladruno`
of that mode, to 1e-9 relative. Otherwise it is stale, and the row's tooltip says
why. The row of the mode shown is marked. A mode that the table has no row for is
shown without one. The Modes button counts the stale rows; before this, it read
"rows not judged".

The app now reads `/spectra`, except the values of each case's results. The
Spectra button counts the cases. Its tooltip names each case with its direction,
scale, modes, damping and results, and each refused case with the reason. A case
holding a family other than `ON_NODES` and `ON_DOMAIN` is refused, as are two
cases with the same name. A refused case does not refuse the zone, and another
major of the zone refuses it. No rule is evaluated over a case yet, and nothing
of it is drawn: the CQC contour is slice 7b.
