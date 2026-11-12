### FIXED — `s.update_parameter` refuses a name that no targeted element or material takes

`s.update_parameter(name, value, pg=...)` emits `addToParameter $pid element
$tag <name>`. When nothing in the element takes `<name>`, OpenSees prints
`no objects were able to identify parameter` and keeps running, so the update
is lost. `s.update_parameter("xPerm", 1e-5, pg=...)` on a `FourNodeQuad` over
`ElasticIsotropic` used to emit, then silently dropped the permeability
change.

The name is now checked against the element classes it targets, using a
catalogue in `stage/_parameter_names.py`. Each entry cites the C++
`setParameter` it was read from. The rule follows the owner ruling, (b) +
(c):

- A catalogued element that does not take the name and does not forward
  names to its materials is refused (`elasticBeamColumn` forwards nothing).
- A forwarding element (`FourNodeQuad`, `Truss`, `SSPquadUP`, `LadrunoUP`) is
  refused only when every material it carries is catalogued and none of them
  takes the name. A material that matches only its own tag takes the name only
  when `material=` names it.
- An uncatalogued element or material class raises
  `UncataloguedParameterWarning` once per class, and the update emits.

The check runs at the call, before anything is recorded, and again at stage
close, so an element registered later in the stage block is also checked.
`Truss` `E`/`A`, `elasticBeamColumn` `E` and `FourNodeQuad` `E`/`nu`/`pressure`
still emit. An element registered after the stage closes, and the `model.h5`
replay, are not checked.
