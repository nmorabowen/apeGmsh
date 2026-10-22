### CHANGED — FEMData built from a results file warns each element dim fact once per session, and a TOPOLOGY that contradicts the catalog has its own warning (#2312)

After #2309 every read of a results file warned `ElementDimUnknownWarning`
for each element class outside the response catalog (ShellMITC3,
TwoNodeLink, MVLEM_3D, the bearings), even when the class-name guess was
right, so opening the same file again repeated the same warning.

The warning now fires once per class and guessed dim per Python session. A
later read warns only for classes not named yet. A class whose dim is still
a guess is always named the first time. A warning that an `error` filter
turned into an exception does not count as given, so the next read raises
again.

A `.ladruno` BASIS `TOPOLOGY` that contradicts the catalog's parent dim used
to reuse `ElementDimUnknownWarning`, although that dim is known, and fired
once per element group. It is now `ElementDimContradictionWarning`, which
names each class once per session with both dims. The catalog's dim still
wins.
