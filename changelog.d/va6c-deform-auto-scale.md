### apeGmshViewer: a deformed shot with no scale takes the auto scale (VA6c, #2765)

A `deform` with no `scale` in an `apegmsh-view/1` spec no longer means
scale 1. The render mode draws it at the auto scale of the step shown, as
the v1 stills did: the largest finite displacement component becomes 0.12
of the model's bounding-box diagonal, and the scale is 1 when either is
zero. The D3 manifest states the scale it used and `scale_auto: true`, and
refuses the shot when the view is drawn at any other scale.
