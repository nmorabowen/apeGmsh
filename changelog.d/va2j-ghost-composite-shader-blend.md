### FIXED — apeGmshViewer: the reference ghost draws the same app still when model and app shots alternate (VA2j, #2715)

Fixed: in a `--render` run that alternated model-chrome and app-chrome shots,
an app still of the reference ghost could come out 1 level off at about 200
pixels on the ghost's outline, and which shot was hit depended on timing. The
canvas's own blend was not reproducible over pixels whose anti-aliasing samples
differ. The ghost composite now copies the canvas just before it draws, blends
the ghost over that copy in its shader, and writes the result with no blending.
Repeated ghost shots are pixel-identical in both chromes, and the render e2e no
longer excuses any ghost still. The ghost's look is unchanged apart from
rounding: up to 2 levels at about 0.6% of the pixels of a still.
