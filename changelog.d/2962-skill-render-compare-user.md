### DOCS — the viewer skill states how a `--render` spec names a compare user result (#2962)

The user skill's viewer reference said a `--render` shot cannot name a compare variant's own user result, which stopped being true with VF2-d (#2957). It now gives the shipped rule: `"source": "user"` plus `compare`, with `contour.field` as the entry and `contour.component` as the column, and the manifest's `user.compare: {variant, entry}`.
