### Fixed — outline select-all selects only the stage's active elements (#1743)

On a staged model, the session window's outline "Select all nodes" /
"Select all Gauss" selected every element of the category, including
elements not active in the pane's stage and therefore not drawn. Since
X2-f (#1722) a realized pane draws only the stage's active elements;
select-all now narrows through the same filter, so it selects exactly what
the pane draws. A category with no element active in the stage now offers
disabled actions and raises from code, instead of selecting undrawn
elements. A model with no staged program, or a stage that pairs with no
program stage, still selects the whole category, as realize draws it.
