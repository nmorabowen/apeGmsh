### FIXED — MPCO and `.ladruno` warn alike on an unread layered-shell bucket and on an unusable stored shell frame (#2090)

The MPCO layer reader used to drop, with no warning, a layered-shell bucket whose class has no `LAYER_CATALOG` row,
whose header index is above 0, or whose integration rule differs from the class's surface rule. Each one now warns
`LayerBucketUnreadWarning` naming the bucket and its class, the warning `.ladruno` has raised since #1995 (the two
readers now share one warning class). Beam fiber-section buckets in the shared `section.fiber.<X>` group stay with the
fiber level. ShellMITC3 is the layered bridge shell with no row: the pinned fork (`c744d6e70`) has no ShellMITC3
element, so no file of either format can hold its layers, and both readers warn by class. The fork lane now pins that
refusal, so a fork build that adds the element fails it and calls for the row. On `.ladruno`, a stored
`MODEL/LOCAL_AXES/<classTag>-<Class>` group without the `ID` + `FRAME` pair, or whose `FRAME` size is not four per `ID`,
now warns `ShellFrameUnavailableWarning` before the frames are computed from node coordinates, as MPCO does.
