### CHANGED — apeGmshViewer: Display and Deformed cards, a contour scale in the viewport, a one-line banner (#2671)

The left rail gains a Display card (mesh, frames, the contour colormap, and greyed rows for nodes and supports)
and a Deformed card, both moved out of the legend. Every left-rail card is 292 px wide and starts collapsed.
The line-diagram colormap stays in its own card.

A contour now draws a scale in the viewport whether or not its card is open: horizontal by default, top left of
the free space, with a button for vertical, and draggable. The colour bar, the field title, the units note (as a
tooltip on the title) and the colour range controls moved from the Contour card onto it. Its orientation and place are remembered per machine. An agent run has its own profile, so `--render`
stills always draw the default.

A line diagram gets the same scale, stacked under the contour's: its colour bar, title and range controls moved off
the Line Diagram card, and its position is remembered separately. Its colormap stays on the card.

The banner is one line, a count of each kind, and a click opens the full list over the rail. Refused fields in
the Contour card collapse the same way, so a long list no longer pushes out the field readout. Read the full
text with one click; the count stays visible.
