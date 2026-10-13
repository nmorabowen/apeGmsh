### `CurvesSkippedWarning` importable from `apeGmsh.mesh`

`from apeGmsh.mesh import CurvesSkippedWarning` now works, so the warning
from `g.mesh.structured.one_element_per_curve()` can be filtered by class.
The method's docstring also states that the by-name refusal applies to
ties into a surface or volume host; a curve tied only to another curve is
accepted by name, and the no-argument form warns and skips it.
