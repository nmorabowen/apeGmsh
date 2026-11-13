### CHANGED — BezierTri6 Gauss stills ask `cell_averaged`, and a non-float native `_natural_coords` warns on open (#2950)

A `.ladruno` BezierTri6 writes its Gauss points as barycentric weights
(PARAM_DOMAIN `"bary"`), so apeGmshViewer offers them no natural coordinates.
A `node` contour of such a group already draws each element's mean, per
group and labelled, as for BezierTet10 (#2940). The Python stills
(`Results.render`, `view="contour"`) now ask `cell_averaged` for a set that
holds only BezierTri6 or BezierTet10 groups, so their manifest reads
`element_mean_averaged` with no fallback line. A native results file whose
BezierTri6 `_natural_coords` holds the catalog's gmsh rows is still
extrapolated. A native Gauss group whose `_natural_coords` is not floating
point now warns when the file opens, naming the group: its values still read,
but its points are not placed, drawn or extrapolated. The refusal wording
names BezierTri6 beside BezierTet10 and no longer says this case opens with
no warning.
