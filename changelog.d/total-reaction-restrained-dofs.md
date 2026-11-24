### FIXED — the spectrum driver's base shear counts only restrained DOFs, so the `-increment` route no longer reads zero (#3148)

`spectrum_after_gravity`'s built-in `base_shear` extractor summed
`nodeReaction` over every DOF of each fixed node, free ones included. In
equilibrium a free DOF's reaction is zero, so the static route was right. On
the fork's `-increment` route each leg is an imposed state, and a free DOF
carries its out-of-balance force. On a shear building whose floors are
fixed in y and rotation only, that force is minus the floor's inertia, and
the floors cancelled the base reaction exactly. The per-mode base shear
read zero, and `check_inertia_balance` raised `InertiaBalanceWarning` with
a gap of 1.000.

The extractor now sums each fixed node's reaction over its restrained DOFs
only (`getFixedDOFs`). The storey-force gates run on both routes again, and
a unit test holds the partly fixed case.
