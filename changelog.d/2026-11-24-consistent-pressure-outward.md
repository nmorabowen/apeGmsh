### FIXED — consistent pressure pushes into the body on cavity walls and crack faces

`g.loads.surface.pressure(..., reduction="consistent")` took its direction
from the face connectivity, which points into the body on a cavity wall, a
hole or one side of a crack: positive pressure pulled there, while the
tributary reduction pushed. Both now use the same outward normal, away from
the adjacent volume element. **Behaviour change** for those faces only; a
convex boundary and a model without volumes resolve as before.
