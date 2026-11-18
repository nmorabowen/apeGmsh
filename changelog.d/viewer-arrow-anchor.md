### apeGmshViewer: an arrow record can anchor its tail at the point

The instanced arrow record gains an `anchor` field, so `ARROW_STRIDE` goes from 14 to 15.
With `ANCHOR.tip` (0) the record's point is the arrow's tip, as before: loads, reactions
and support glyphs still point at their nodes. With `ANCHOR.tail` (1) the point is where
the shaft starts, and the tip lies one drawn shaft length ahead. Nodal vectors and
principal directions need that. The vertex shader moves the tip by the same pixel-capped
shaft length that `SIZING_EXPR` gives, so the sizing still has one source. The CPU twin
`expandArrows` honours the anchor as well. Every existing builder writes `ANCHOR.tip`, so
nothing on screen changes.
