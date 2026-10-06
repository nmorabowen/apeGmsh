### FIXED — `apeSees.register` is a no-op on an instance the bridge already holds (program slice B1-7, #1409)

Registering the same primitive twice (a second `ops.register(t)`, or
`ops.register(ops.timeSeries.Linear())` on a handle the namespace already
registered) appended it to the bridge's primitive list again. The deck stayed
single, but `element_tags="fem"` refused the model as an element fanned out
twice, `all_recorder_specs` listed a recorder twice, and the list order the
`analyze` strategy reads as "last algorithm registered" could name a different
algorithm from the one the deck runs. The repeat call now returns the instance
unchanged.
