### CHANGED — apeGmshViewer deforms a `.ladruno` by the fork schema's names, and draws its mode shapes (#2128)

The deformed shape of a paired `.ladruno` now reads only the groups that fork
schema §7.1.1 names. A results stage reads `DISPLACEMENT` and a mode stage
reads `MODES_OF_VIBRATION(U)`, with the labels `Ux`, `Uy` and `Uz`. Any other
label under those groups refuses the deformed shape, and the refusal names the
label. This includes `X`, `Y` and `Z`, which the viewer used to accept. A
missing `Uz` still reads as 0 only in a model declared 2-D. The deformed shape
never reads `ROTATION` or `MODES_OF_VIBRATION(R)`. Schema §7.1.1 notes that a
2-D u-p node keeps its pore pressure in `ROTATION/Rz`, and that a 1-D model's
`ROTATION` holds three columns of zeros.

A mode stage of a `.ladruno` now deforms as a native mode stage does. The
legend names its eigenvalue from the file's `LAMBDA`, and also its frequency
and period when the eigenvalue is positive. The contour offers the magnitude of
`MODES_OF_VIBRATION(U)`. When a rotation group holds only `Rz` (a 2-D file),
the contour shows it as its own component, labelled `Rz`, and never as rotation
z, because the file cannot say whether it holds a rotation or a pore pressure.

The hand-built `ladruno_frame3d.ladruno` fixture now has a separate
`DISPLACEMENT` (`Ux,Uy,Uz`) group and a separate `ROTATION` (`Rx,Ry,Rz`) group,
as the fork writes them. It used to have one 6-component `DISPLACEMENT` group.
A new fork-written `ladruno_portal3d.ladruno` fixture covers a 3-D frame with
rotations and modes. A 1-D model still cannot be deformed.
