### FIXED — `reduction="consistent"` on gravity and volume loads is the consistent vector (ADR 0091 Amendment 1)

`g.loads.gravity(..., reduction="consistent")` and `g.loads.volume(...,
reduction="consistent")` returned the equal split for every element type.
They now integrate `∫ N_i dΩ` against the element's shape functions on its
own geometry, curved and distorted elements included. A straight `tet10`
gets `-1/20` of its weight at each corner and `+1/5` at each mid-edge node,
instead of `1/10` everywhere; `tri6`, `quad8`, `quad9`, `hex20`, `hex27`
and distorted `hex8` / `wedge6` / `quad4` change too. **Behaviour change**
for decks that ask for `"consistent"`; the default `"tributary"` is
unchanged.

Both verbs take `basis="lagrange"` (default) or `"bernstein"`. A Ladruno
Bézier region (`BezierTet10`, `BezierTri6`, `LadrunoUP` on tri6/tet10) that
asks for `"consistent"` must now say `basis="bernstein"`, which is the
vector it had, bit for bit; without it `build()` warns
(`WarnLoadBasisMismatch`). The `gravity` docstring said tributary splits
among corner nodes; it splits among all of them.
