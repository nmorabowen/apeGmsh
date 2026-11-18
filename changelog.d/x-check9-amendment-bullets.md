### CHANGED — land_pr check 9 reads card amendments by path-opening bullets (#3025)

A `**Card amendment` comment now adds a path only when its backtick span opens a list item; a path in prose adds nothing, and a negation word in a path-opening bullet's trailing prose no longer drops it. `--check-card` prints, per amendment, the paths added and each path ignored with the reason.
